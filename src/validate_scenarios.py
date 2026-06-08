"""Validate weather scenario dataset structure."""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

from src.schema import (
    FAWN_SYNC_PATH,
    SCENARIOS_PATH,
    Scenario,
    load_fawn_sync,
    load_scenarios,
)

STRATUM_COUNTS = {
    "specific_station": 2,
    "regional_inference": 2,
    "underspecified": 2,
}
PROMPT_VARIANTS = {"natural", "statistical"}


def validate_scenarios(scenarios: list[Scenario]) -> list[str]:
    errors: list[str] = []
    expected = sum(STRATUM_COUNTS.values())
    if len(scenarios) != expected:
        errors.append(f"scenarios: expected {expected} rows, got {len(scenarios)}")

    ids = [s.id for s in scenarios]
    if len(ids) != len(set(ids)):
        errors.append("scenarios: duplicate id values")

    by_stratum = Counter(s.stratum for s in scenarios)
    for stratum, count in STRATUM_COUNTS.items():
        if by_stratum[stratum] != count:
            errors.append(
                f"scenarios: stratum '{stratum}' expected {count} rows, got {by_stratum[stratum]}"
            )

    by_variant = Counter(s.prompt_variant for s in scenarios)
    for variant in PROMPT_VARIANTS:
        if by_variant[variant] != 3:
            errors.append(
                f"scenarios: prompt_variant '{variant}' expected 3 rows, got {by_variant[variant]}"
            )

    for stratum in STRATUM_COUNTS:
        variants = {s.prompt_variant for s in scenarios if s.stratum == stratum}
        if variants != PROMPT_VARIANTS:
            errors.append(f"scenarios: stratum '{stratum}' missing a prompt variant")

    widths = [s.target_p90 - s.target_p10 for s in scenarios]
    by_stratum_width = {
        stratum: [s.target_p90 - s.target_p10 for s in scenarios if s.stratum == stratum]
        for stratum in STRATUM_COUNTS
    }
    mean_widths = {k: sum(v) / len(v) for k, v in by_stratum_width.items() if v}
    if mean_widths:
        ordered = ["specific_station", "regional_inference", "underspecified"]
        for i in range(len(ordered) - 1):
            a, b = ordered[i], ordered[i + 1]
            if a in mean_widths and b in mean_widths and mean_widths[a] >= mean_widths[b]:
                errors.append(
                    f"scenarios: curator interval width should widen down specificity "
                    f"({a}={mean_widths[a]:.1f} >= {b}={mean_widths[b]:.1f})"
                )

    if min(widths) <= 0:
        errors.append("scenarios: target interval widths must be positive")

    for s in scenarios:
        if not s.location_description.strip():
            errors.append(f"scenarios[{s.id}]: empty location_description")

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

    for row in sync.scenarios.values():
        if not (row.p10 <= row.p50 <= row.p90):
            errors.append(f"fawn_sync[{row.scenario_id}]: quantiles not ordered")

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
    sync = load_fawn_sync(FAWN_SYNC_PATH)
    width_msg = (
        f"fawn sync present ({len(sync.scenarios)} scenarios)"
        if sync is not None
        else "fawn sync not present (using curator targets)"
    )
    print(
        f"Validation OK: {len(scenarios)} scenarios, strata {dict(strata)}, {width_msg}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
