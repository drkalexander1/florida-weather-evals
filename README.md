# Florida Weather Rainfall Evals

LLM benchmark for Florida annual rainfall: **specificity gradient** prompts, **natural vs. statistical** phrasing, and dual scoring on **CRPS** (quantile forecasts) and **confidence**.

**Design notes:** [weather-eval-design.md](weather-eval-design.md)

## Why this design

Follow-on to the [Michigan bird eval](https://github.com/). Bird eval tested unknown unknowns; weather eval tests **known unknowns** — some questions have retrievable answers (Miami), others are underspecified by design (random Florida location). Well-calibrated models should give progressively wider intervals as specificity decreases.

| Stratum | N | What models can draw on |
|---------|---|-------------------------|
| `specific_station` | 2 | Likely memorized station/city figures |
| `regional_inference` | 2 | Regional Gulf-coast inference (~100k city) |
| `underspecified` | 2 | Population-level Florida estimate |

Each stratum runs with two **prompt variants**: `natural` (conversational) and `statistical` (explicit quantile request).

## Metrics

| Metric | What it measures |
|--------|------------------|
| **crps** | Primary — CRPS vs. FAWN reference years (or curator targets) |
| **mean_pinball_loss** | Average pinball loss at 10th / 50th / 90th percentiles |
| **interval_width** | p90 − p10 (should widen down the specificity gradient) |
| **ece_confidence** | Self-reported confidence vs. median error |

## Quick start

```bash
pip install -e .
python -m src.validate_scenarios
```

### Demo (no API)

```bash
python scripts/generate_demo_predictions.py
python -m src.score --run results/demo
```

### API eval

```bash
cp .env.example .env
# Add real keys only when running evals.

python -m src.run_eval --models gpt-4o-mini,gpt-4o --output results/my-run
python -m src.score --run results/my-run
```

Interrupted runs **resume automatically** — re-run the same command to continue. Use `--fresh` to start over.

### FAWN ground truth sync (optional)

Download the **yearly QAQC zip files** into `data/fawn_raw/` (no manual unzipping needed):

```bash
python scripts/download_fawn_zips.py          # 2005-2024 (~1 GB total)
python -m src.fawn_sync --list-zips           # verify discovery
python -m src.fawn_sync --pool-all-stations
python -m src.validate_scenarios
```

Or download manually from [FAWN FTP](https://fawn.ifas.ufl.edu/data/fawn_data_qaqc_pub/) into `data/` or `data/fawn_raw/`.

```bash
python -m src.fawn_sync --list-stations   # see valid station IDs/names
```

Use at least **3 calendar years** of zips (recommended: **2005–2024**). Optional `--extract-zips` caches extracted CSVs under `data/fawn_raw/extracted/` for faster re-runs.

Scenario station IDs (from FAWN metadata, not sequential 107/204):

| Scenario | FAWN ID | Name |
|----------|---------|------|
| Miami / specific | **440** | Homestead |
| Gulf ~100k city | **480** | North Port |

Writes `data/fawn_sync.json` with per-scenario reference years and pooled distributions.

## Outputs

Per run directory: `predictions.jsonl`, `summary.json`, `by_scenario.csv`, `by_stratum.csv`, `by_prompt_variant.csv`.

## License

MIT
