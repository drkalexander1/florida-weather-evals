"""Validate weather scenario dataset structure."""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

from src.schema import (
    FAWN_SYNC_PATH,
    SCENARIOS_PATH,
    SEASON_MONTHS,
    Scenario,
    load_fawn_sync,
    load_scenarios,
)

STRATA = ["specific_station", "regional_inference", "underspecified"]
SEASONS = list(SEASON_MONTHS)
PROMPT_VARIANTS = {"natural", "statistical"}
EXPECTED_TOTAL = len(STRATA) * len(SEASONS) * len(PROMPT_VARIANTS)


def validate_scenarios(scenarios: list[Scenario]) -> list[str]:
    errors: list[str] = []
    if len(scenarios) != EXPECTED_TOTAL:
        errors.append(f"scenarios: expected {EXPECTED_TOTAL} rows, got {len(scenarios)}")

    ids = [s.id for s in scenarios]
    if len(ids) != len(set(ids)):
        errors.append("scenarios: duplicate id values")

    cells = Counter((s.stratum, s.season, s.prompt_variant) for s in scenarios)
    for stratum in STRATA:
        for season in SEASONS:
            for variant in PROMPT_VARIANTS:
                if cells[(stratum, season, variant)] != 1:
                    errors.append(
                        f"scenarios: expected exactly 1 row for "
                        f"({stratum}, {season}, {variant}), got {cells[(stratum, season, variant)]}"
                    )

    # Within each season, curator interval width should widen as specificity drops.
    for season in SEASONS:
        widths: dict[str, list[float]] = {}
        for s in scenarios:
            if s.season == season:
                widths.setdefault(s.stratum, []).append(s.target_p90 - s.target_p10)
        means = {k: sum(v) / len(v) for k, v in widths.items() if v}
        for i in range(len(STRATA) - 1):
            a, b = STRATA[i], STRATA[i + 1]
            if a in means and b in means and means[a] >= means[b]:
                errors.append(
                    f"scenarios[{season}]: curator interval width should widen down "
                    f"specificity ({a}={means[a]:.1f} >= {b}={means[b]:.1f})"
                )

    for s in scenarios:
        if s.target_p90 - s.target_p10 <= 0:
            errors.append(f"scenarios[{s.id}]: target interval width must be positive")
        if not s.location_description.strip():
            errors.append(f"scenarios[{s.id}]: empty location_description")

    # Paired variants must share ground truth.
    by_cell: dict[tuple[str, str], list[Scenario]] = {}
    for s in scenarios:
        by_cell.setdefault((s.stratum, s.season), []).append(s)
    for (stratum, season), pair in by_cell.items():
        targets = {(s.target_p10, s.target_p50, s.target_p90, s.fawn_station_id) for s in pair}
        if len(targets) > 1:
            errors.append(
                f"scenarios: ({stratum}, {season}) prompt variants have mismatched targets/stations"
            )

    return errors


def validate_fawn_sync(scenarios: list[Scenario], sync_path: Path) -> list[str]:
    errors: list[str] = []
    try:
        sync = load_fawn_sync(sync_path)
    except Exception as exc:
        return [f"fawn_sync: {exc}"]

    if sync is None:
        return errors

    scenario_ids = {s.id for s in scenarios}
    sync_ids = set(sync.scenarios.keys())
    missing = sorted(scenario_ids - sync_ids)
    extra = sorted(sync_ids - scenario_ids)
    if missing:
        errors.append(f"fawn_sync: missing scenarios: {', '.join(missing)}")
    if extra:
        errors.append(f"fawn_sync: stale scenario ids: {', '.join(extra)}")

    season_by_id = {s.id: s.season for s in scenarios}
    for row in sync.scenarios.values():
        if not (row.p10 <= row.p50 <= row.p90):
            errors.append(f"fawn_sync[{row.scenario_id}]: quantiles not ordered")
        expected_season = season_by_id.get(row.scenario_id)
        if expected_season and row.season != expected_season:
            errors.append(
                f"fawn_sync[{row.scenario_id}]: season '{row.season}' != scenario "
                f"season '{expected_season}' — regenerate with python -m src.fawn_sync"
            )

    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate Florida weather eval dataset")
    parser.add_argument("--scenarios", type=Path, default=SCENARIOS_PATH)
    args = parser.parse_args(argv)

    errors: list[str] = []
    scenarios: list[Scenario] = []

    try:
        scenarios = load_scenarios(args.scenarios)
        errors.extend(validate_scenarios(scenarios))
        if FAWN_SYNC_PATH.exists():
            errors.extend(validate_fawn_sync(scenarios, FAWN_SYNC_PATH))
    except Exception as exc:
        errors.append(f"scenarios: {exc}")

    if errors:
        print("Validation FAILED:", file=sys.stderr)
        for err in errors:
            print(f"  - {err}", file=sys.stderr)
        return 1

    strata = Counter(s.stratum for s in scenarios)
    seasons = Counter(s.season for s in scenarios)
    sync = load_fawn_sync(FAWN_SYNC_PATH)
    sync_msg = (
        f"fawn sync present ({len(sync.scenarios)} scenarios)"
        if sync is not None
        else "fawn sync not present (using curator targets)"
    )
    print(
        f"Validation OK: {len(scenarios)} scenarios, strata {dict(strata)}, "
        f"seasons {dict(seasons)}, {sync_msg}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
