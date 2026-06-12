"""Print per-scenario ground-truth quantiles from data/fawn_sync.json."""

import json
from pathlib import Path

sync = json.loads((Path(__file__).resolve().parent.parent / "data" / "fawn_sync.json").read_text())
print(f"{'scenario':<32}{'p10':>8}{'p50':>8}{'p90':>8}{'width':>8}{'n':>5}")
for sid, row in sync["scenarios"].items():
    if sid.endswith("_natural"):
        width = row["p90"] - row["p10"]
        print(
            f"{sid:<32}{row['p10']:>8.1f}{row['p50']:>8.1f}{row['p90']:>8.1f}"
            f"{width:>8.1f}{len(row['reference_years']):>5}"
        )
