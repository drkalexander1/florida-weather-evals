"""Build data/fawn_sync.json from FAWN yearly QAQC zip archives.

Aggregates 15-minute rainfall to monthly totals, then assembles seasonal
totals (annual, wet Jun-Sep, dry Dec-Feb) per station and season-year.
"""

from __future__ import annotations

import argparse
import calendar
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
    SEASON_MONTHS,
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
INTERVALS_PER_DAY = 96  # 15-minute intervals
MAX_MISSING_FRAC = 0.05
MIN_CLEAN_YEARS = 3
YEAR_RE = re.compile(r"(20\d{2})")
CHUNK_SIZE = 500_000

# (station_id, season) -> list of seasonal totals, one per clean season-year
StationSeasonValues = dict[tuple[int, str], list[float]]


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
    """Reduce a raw CSV chunk to monthly sums of good rainfall intervals."""
    chunk = chunk[chunk["ID"] != "ID"].copy()
    chunk["ID"] = pd.to_numeric(chunk["ID"], errors="coerce")
    chunk = chunk.dropna(subset=["ID"])
    chunk["ID"] = chunk["ID"].astype(int)
    if station_ids is not None:
        chunk = chunk[chunk["ID"].isin(station_ids)]
    if chunk.empty:
        return pd.DataFrame(columns=["ID", "year", "month", "total", "good_n"])
    chunk["timestamp"] = pd.to_datetime(chunk["UTC"], errors="coerce")
    chunk = chunk.dropna(subset=["timestamp"])
    chunk["year"] = chunk["timestamp"].dt.year
    chunk["month"] = chunk["timestamp"].dt.month
    chunk["rain_inches"] = _resolve_rain(chunk)
    return (
        chunk.groupby(["ID", "year", "month"])
        .agg(total=("rain_inches", "sum"), good_n=("rain_inches", "count"))
        .reset_index()
    )


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
        return pd.DataFrame(columns=["ID", "year", "month", "total", "good_n"])
    combined = pd.concat(parts, ignore_index=True)
    # Chunk boundaries can split a month; re-sum.
    return (
        combined.groupby(["ID", "year", "month"], as_index=False)
        .agg(total=("total", "sum"), good_n=("good_n", "sum"))
    )


def _expected_intervals(year: int, month: int) -> int:
    return calendar.monthrange(year, month)[1] * INTERVALS_PER_DAY


def seasonal_totals(monthly: pd.DataFrame) -> StationSeasonValues:
    """Assemble per-station seasonal totals from monthly aggregates.

    A season-year counts as clean when every month of the window is present
    and the combined good-interval coverage is at least 1 - MAX_MISSING_FRAC.
    December is assigned to the following dry-season year.
    """
    by_key: dict[tuple[int, int, int], tuple[float, int]] = {}
    for row in monthly.itertuples(index=False):
        by_key[(int(row.ID), int(row.year), int(row.month))] = (
            float(row.total),
            int(row.good_n),
        )

    station_ids = sorted({int(sid) for sid, _, _ in by_key})
    year_lo = min(yr for _, yr, _ in by_key)
    year_hi = max(yr for _, yr, _ in by_key)

    values: StationSeasonValues = {}
    for season, months in SEASON_MONTHS.items():
        for sid in station_ids:
            totals: list[float] = []
            for season_year in range(year_lo, year_hi + 2):
                total = 0.0
                good = 0
                expected = 0
                complete = True
                for month in months:
                    cal_year = season_year - 1 if (season == "dry_season" and month == 12) else season_year
                    entry = by_key.get((sid, cal_year, month))
                    if entry is None:
                        complete = False
                        break
                    total += entry[0]
                    good += entry[1]
                    expected += _expected_intervals(cal_year, month)
                if not complete or expected == 0:
                    continue
                if 1.0 - good / expected > MAX_MISSING_FRAC:
                    continue
                totals.append(round(total, 4))
            if totals:
                values[(sid, season)] = totals
    return values


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
) -> tuple[StationSeasonValues, set[int]]:
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

    monthly_parts: list[pd.DataFrame] = []
    years_seen: set[int] = set()
    for year, zip_path in selected:
        print(f"Reading {zip_path.name} ...")
        monthly = read_year_zip(
            zip_path,
            station_ids=station_ids,
            extract_dir=EXTRACT_DIR if extract_zips else None,
        )
        if not monthly.empty:
            monthly_parts.append(monthly)
            years_seen.add(year)

    if not monthly_parts:
        raise ValueError("No station data loaded from zips")

    all_monthly = (
        pd.concat(monthly_parts, ignore_index=True)
        .groupby(["ID", "year", "month"], as_index=False)
        .agg(total=("total", "sum"), good_n=("good_n", "sum"))
    )
    return seasonal_totals(all_monthly), years_seen


def _prune_station_values(
    station_values: StationSeasonValues,
    *,
    required: set[int],
) -> StationSeasonValues:
    pruned: StationSeasonValues = {}
    for (sid, season), values in station_values.items():
        if len(values) >= MIN_CLEAN_YEARS:
            pruned[(sid, season)] = values
        elif sid in required:
            raise ValueError(
                f"Station {sid} {season}: only {len(values)} clean season-years after "
                f"filtering (need >= {MIN_CLEAN_YEARS}). Download more yearly zips."
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
    if not station_values:
        raise ValueError("No station-season totals loaded")

    pooled: dict[str, list[float]] = {season: [] for season in SEASON_MONTHS}
    for (sid, season) in sorted(station_values):
        pooled[season].extend(station_values[(sid, season)])

    scenario_rows: dict[str, FawnScenarioSync] = {}
    for sc in scenarios:
        if sc.fawn_station_id:
            sid = int(sc.fawn_station_id)
            ref = station_values.get((sid, sc.season))
            if not ref:
                raise ValueError(
                    f"Scenario {sc.id}: no clean {sc.season} data for station {sid}"
                )
        else:
            ref = pooled[sc.season]
            if not ref:
                raise ValueError(f"Scenario {sc.id}: empty pooled {sc.season} distribution")
        stats = distribution_stats(ref)
        scenario_rows[sc.id] = FawnScenarioSync(
            scenario_id=sc.id,
            fawn_station_id=sc.fawn_station_id,
            season=sc.season,
            reference_years=ref,
            target_p10_curator=float(sc.target_p10),
            target_p50_curator=float(sc.target_p50),
            target_p90_curator=float(sc.target_p90),
            **stats,
        )

    return FawnSync(
        schema_version=2,
        generated_at=datetime.now(timezone.utc).isoformat(),
        method="fawn_seasonal_totals",
        method_note=(
            "Read yearly QAQC zip archives, aggregate 15-min rainfall to monthly totals, then "
            "assemble seasonal totals per station (annual Jan-Dec, wet Jun-Sep, dry Dec-Feb with "
            "December assigned to the following season-year). Primary rain when quality flag is 0; "
            "otherwise backup when backup flag is 0. Exclude station-season-years with "
            f">{int(MAX_MISSING_FRAC * 100)}% missing intervals or missing months."
        ),
        region="US-FL",
        years=sorted(years_seen),
        stations_included=sorted({str(sid) for sid, _ in station_values}),
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
