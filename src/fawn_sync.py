"""Build data/fawn_sync.json from FAWN yearly QAQC zip archives."""

from __future__ import annotations

import argparse
import json
import re
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.schema import (
    DATA_DIR,
    FAWN_SYNC_PATH,
    FawnScenarioSync,
    FawnSync,
    Scenario,
    load_scenarios,
)

RAW_DIR = DATA_DIR / "fawn_raw"
STATIONS_PATH = DATA_DIR / "fawn_stations.yaml"
EXTRACT_DIR = RAW_DIR / "extracted"

RAIN_COL = "rain_2m_inches"
RAIN_FLAG_COL = "quality_flag_rain_2m_inches"
BACKUP_COL = "rain_backup_2m_inches"
BACKUP_FLAG_COL = "quality_flag_rain_backup_2m_inches"
USECOLS = ["ID", "UTC", RAIN_COL, RAIN_FLAG_COL, BACKUP_COL, BACKUP_FLAG_COL]
INTERVAL_HOURS = 0.25
MAX_MISSING_FRAC = 0.05
MIN_CLEAN_YEARS = 3
YEAR_RE = re.compile(r"(20\d{2})")
CHUNK_SIZE = 500_000


def load_station_catalog(path: Path | None = None) -> dict[str, str]:
    path = path or STATIONS_PATH
    with path.open(encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise ValueError(f"Expected mapping in {path}")
    return {str(k): str(v) for k, v in raw.items()}


def list_stations() -> None:
    catalog = load_station_catalog()
    for sid in sorted(catalog, key=lambda x: int(x)):
        print(f"{sid:>3}  {catalog[sid]}")


def _parse_year(path: Path) -> int | None:
    match = YEAR_RE.search(path.stem)
    return int(match.group(1)) if match else None


def discover_zip_files(raw_dir: Path) -> list[tuple[int, Path]]:
    search_roots = {raw_dir.resolve()}
    if raw_dir.name == "fawn_raw" and raw_dir.parent.exists():
        search_roots.add(raw_dir.parent.resolve())

    zips: list[tuple[int, Path]] = []
    seen: set[Path] = set()
    for root in sorted(search_roots):
        if not root.exists():
            continue
        for path in sorted(root.rglob("*.zip")):
            if path.name.startswith("._"):
                continue
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            year = _parse_year(path)
            if year is not None:
                zips.append((year, path))

    by_year: dict[int, Path] = {}
    for year, path in zips:
        existing = by_year.get(year)
        if existing is None or path.stat().st_size > existing.stat().st_size:
            by_year[year] = path
    return sorted((year, by_year[year]) for year in by_year)


def _flag_is_good(value) -> bool:
    if pd.isna(value):
        return False
    try:
        return float(value) == 0.0
    except (TypeError, ValueError):
        return str(value).strip() in {"0", "0.0"}


def _resolve_rain(df: pd.DataFrame) -> pd.Series:
    primary = pd.to_numeric(df[RAIN_COL], errors="coerce")
    backup = pd.to_numeric(df[BACKUP_COL], errors="coerce")
    primary_ok = df[RAIN_FLAG_COL].map(_flag_is_good)
    backup_ok = df[BACKUP_FLAG_COL].map(_flag_is_good)
    rain = primary.where(primary_ok)
    rain = rain.where(rain.notna() | ~backup_ok, backup)
    return rain


def _clean_chunk(chunk: pd.DataFrame, station_ids: set[int] | None) -> pd.DataFrame:
    chunk = chunk[chunk["ID"] != "ID"].copy()
    chunk["ID"] = pd.to_numeric(chunk["ID"], errors="coerce")
    chunk = chunk.dropna(subset=["ID"])
    chunk["ID"] = chunk["ID"].astype(int)
    if station_ids is not None:
        chunk = chunk[chunk["ID"].isin(station_ids)]
    if chunk.empty:
        return chunk
    chunk["timestamp"] = pd.to_datetime(chunk["UTC"], errors="coerce")
    chunk = chunk.dropna(subset=["timestamp"])
    chunk["year"] = chunk["timestamp"].dt.year
    chunk["rain_inches"] = _resolve_rain(chunk)
    return chunk[["ID", "year", "rain_inches"]]


def read_year_zip(
    zip_path: Path,
    *,
    station_ids: set[int] | None = None,
    extract_dir: Path | None = None,
) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []

    if extract_dir is not None:
        extract_dir.mkdir(parents=True, exist_ok=True)
        csv_path = extract_dir / f"{zip_path.stem}.csv"
        if not csv_path.exists():
            with zipfile.ZipFile(zip_path) as zf:
                member = next(n for n in zf.namelist() if n.endswith(".csv") and not n.startswith("__"))
                csv_path.write_bytes(zf.read(member))
        source = csv_path
        reader = pd.read_csv(source, usecols=USECOLS, chunksize=CHUNK_SIZE, low_memory=False)
        for chunk in reader:
            cleaned = _clean_chunk(chunk, station_ids)
            if not cleaned.empty:
                parts.append(cleaned)
    else:
        with zipfile.ZipFile(zip_path) as zf:
            member = next(n for n in zf.namelist() if n.endswith(".csv") and not n.startswith("__"))
            with zf.open(member) as handle:
                for chunk in pd.read_csv(handle, usecols=USECOLS, chunksize=CHUNK_SIZE, low_memory=False):
                    cleaned = _clean_chunk(chunk, station_ids)
                    if not cleaned.empty:
                        parts.append(cleaned)

    if not parts:
        return pd.DataFrame(columns=["ID", "year", "rain_inches"])
    return pd.concat(parts, ignore_index=True)


def annual_totals(intervals: pd.DataFrame) -> pd.DataFrame:
    grouped = intervals.groupby(["ID", "year"]).agg(
        total=("rain_inches", "sum"),
        n=("rain_inches", "count"),
        missing=("rain_inches", lambda s: s.isna().sum()),
    )
    grouped["missing_frac"] = grouped["missing"] / grouped["n"]
    return grouped[grouped["missing_frac"] <= MAX_MISSING_FRAC]


def distribution_stats(values: list[float]) -> dict[str, float]:
    arr = np.array(values, dtype=float)
    return {
        "p10": float(np.percentile(arr, 10)),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
    }


def load_from_year_zips(
    raw_dir: Path,
    *,
    station_ids: set[int],
    year_min: int | None,
    year_max: int | None,
    extract_zips: bool,
) -> tuple[dict[int, list[float]], set[int]]:
    zips = discover_zip_files(raw_dir)
    if not zips:
        raise FileNotFoundError(
            f"No yearly zip files found in {raw_dir}. "
            "Download YYYY.csv.zip files from "
            "https://fawn.ifas.ufl.edu/data/fawn_data_qaqc_pub/ "
            "and place them in data/fawn_raw/."
        )

    selected = [
        (year, path)
        for year, path in zips
        if (year_min is None or year >= year_min) and (year_max is None or year <= year_max)
    ]
    if not selected:
        raise ValueError("No zip files matched the requested year range")

    per_year: dict[tuple[int, int], float] = {}
    years_seen: set[int] = set()

    for year, zip_path in selected:
        print(f"Reading {zip_path.name} ...")
        intervals = read_year_zip(
            zip_path,
            station_ids=station_ids,
            extract_dir=EXTRACT_DIR if extract_zips else None,
        )
        if intervals.empty:
            continue
        totals = annual_totals(intervals)
        for (sid, yr), row in totals.iterrows():
            sid_int = int(sid)
            yr_int = int(yr)
            per_year[(sid_int, yr_int)] = float(row["total"])
            years_seen.add(yr_int)

    station_values: dict[int, list[float]] = {sid: [] for sid in station_ids}
    for (sid, _yr), total in sorted(per_year.items()):
        if sid in station_values:
            station_values[sid].append(total)

    return station_values, years_seen


def _prune_station_values(
    station_values: dict[int, list[float]],
    *,
    required: set[int],
) -> dict[int, list[float]]:
    pruned: dict[int, list[float]] = {}
    for sid, values in station_values.items():
        if len(values) >= MIN_CLEAN_YEARS:
            pruned[sid] = values
        elif sid in required:
            raise ValueError(
                f"Station {sid}: only {len(values)} clean years after filtering "
                f"(need >= {MIN_CLEAN_YEARS}). Download more yearly zips."
            )
    return pruned


def required_station_ids(scenarios: list[Scenario]) -> set[int]:
    ids: set[int] = set()
    for sc in scenarios:
        if sc.fawn_station_id:
            ids.add(int(sc.fawn_station_id))
    return ids


def all_catalog_station_ids() -> set[int]:
    return {int(sid) for sid in load_station_catalog()}


def build_sync(
    scenarios: list[Scenario],
    *,
    raw_dir: Path,
    year_min: int | None,
    year_max: int | None,
    extract_zips: bool,
    pool_all_stations: bool,
) -> FawnSync:
    scenario_station_ids = required_station_ids(scenarios)
    if pool_all_stations:
        station_ids = all_catalog_station_ids()
    else:
        station_ids = scenario_station_ids

    station_values, years_seen = load_from_year_zips(
        raw_dir,
        station_ids=station_ids,
        year_min=year_min,
        year_max=year_max,
        extract_zips=extract_zips,
    )
    station_values = _prune_station_values(
        station_values,
        required=scenario_station_ids,
    )

    pooled: list[float] = []
    for sid in sorted(station_values):
        pooled.extend(station_values[sid])

    if not pooled:
        raise ValueError("No station-year totals loaded")

    scenario_rows: dict[str, FawnScenarioSync] = {}
    for sc in scenarios:
        if sc.fawn_station_id:
            sid = int(sc.fawn_station_id)
            if sid not in station_values:
                raise ValueError(f"Scenario {sc.id} references unknown station {sid}")
            ref = station_values[sid]
        else:
            ref = pooled
        stats = distribution_stats(ref)
        scenario_rows[sc.id] = FawnScenarioSync(
            scenario_id=sc.id,
            fawn_station_id=sc.fawn_station_id,
            reference_years=ref,
            target_p10_curator=float(sc.target_p10),
            target_p50_curator=float(sc.target_p50),
            target_p90_curator=float(sc.target_p90),
            **stats,
        )

    return FawnSync(
        schema_version=1,
        generated_at=datetime.now(timezone.utc).isoformat(),
        method="fawn_annual_totals",
        method_note=(
            "Read yearly QAQC zip archives (all stations per file), skip repeated header rows, "
            "sum rain_2m_inches to annual totals per station (values are inches per 15-min interval). "
            "Primary rain when quality flag is 0; otherwise backup when backup "
            f"flag is 0. Exclude station-years with >{int(MAX_MISSING_FRAC * 100)}% missing intervals."
        ),
        region="US-FL",
        years=sorted(years_seen),
        stations_included=sorted(str(sid) for sid in station_values),
        scenarios=scenario_rows,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Sync FAWN annual rainfall into data/fawn_sync.json")
    parser.add_argument("--scenarios", type=Path, default=DATA_DIR / "scenarios.yaml")
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--output", type=Path, default=FAWN_SYNC_PATH)
    parser.add_argument("--year-min", type=int, default=2005)
    parser.add_argument("--year-max", type=int, default=2025)
    parser.add_argument(
        "--extract-zips",
        action="store_true",
        help="Also unzip yearly archives to data/fawn_raw/extracted/ (faster re-runs)",
    )
    parser.add_argument(
        "--pool-all-stations",
        action="store_true",
        help="Load every catalog station when building the pooled random-FL distribution",
    )
    parser.add_argument(
        "--list-stations",
        action="store_true",
        help="Print FAWN station IDs/names and exit",
    )
    parser.add_argument(
        "--list-zips",
        action="store_true",
        help="Print discovered yearly zip files and exit",
    )
    args = parser.parse_args(argv)

    if args.list_stations:
        list_stations()
        return 0

    if args.list_zips:
        zips = discover_zip_files(args.raw_dir)
        if not zips:
            print(f"No yearly zip files found under {args.raw_dir} or {args.raw_dir.parent}")
            return 1
        for year, path in zips:
            print(f"{year}  {path}")
        print(f"Found {len(zips)} zip(s)")
        return 0

    try:
        scenarios = load_scenarios(args.scenarios)
        sync = build_sync(
            scenarios,
            raw_dir=args.raw_dir,
            year_min=args.year_min,
            year_max=args.year_max,
            extract_zips=args.extract_zips,
            pool_all_stations=args.pool_all_stations,
        )
    except Exception as exc:
        print(f"fawn_sync failed: {exc}", file=sys.stderr)
        return 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(sync.model_dump_json(indent=2), encoding="utf-8")
    print(
        f"Wrote {args.output} ({len(sync.scenarios)} scenarios, "
        f"{len(sync.stations_included)} stations, years {sync.years[0]}-{sync.years[-1]})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
