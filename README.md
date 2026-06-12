# Florida Weather Rainfall Evals

LLM benchmark for Florida rainfall calibration: a **specificity gradient** crossed with a **seasonal gradient**, **natural vs. statistical** phrasing, and scoring on **CRPS** (quantile forecasts), **confidence calibration**, and **self-consistency checks**.

Follow-on to the [Michigan bird eval](https://github.com/drkalexander1/michigan-bird-evals). Bird eval tested unknown unknowns; weather eval tests **known unknowns** — some questions have retrievable answers (Miami), others are underspecified by design (random Florida location). The design asks whether models widen their intervals as specificity decreases.

**Design notes:** [weather-eval-design.md](weather-eval-design.md) · **Results write-up:** [RESULTS.md](RESULTS.md)

## Why this design

Two orthogonal gradients, 18 scenarios (3 strata × 3 seasons × 2 prompt variants):

| Stratum | What models can draw on |
|---------|-------------------------|
| `specific_station` | Likely memorized city figures (Miami / FAWN 440 Homestead) |
| `regional_inference` | Regional Gulf-coast inference (~100k city / FAWN 480 North Port) |
| `underspecified` | Population-level Florida estimate (pooled FAWN network) |

| Season | Window | Why it matters |
|--------|--------|----------------|
| `annual` | Jan–Dec | Anchor; comparable to earlier runs |
| `wet_season` | Jun–Sep | Most of Florida's rain; sharp climatological signal |
| `dry_season` | Dec–Feb | Low totals, skewed distribution; tests scale-aware uncertainty |

Each cell runs with two **prompt variants**: `natural` (conversational) and `statistical` (explicit quantile / predictive-distribution framing). The prompts deliberately avoid coaching about interval width — whether intervals widen for vaguer questions is the result being measured.

## Metrics

| Metric | What it measures |
|--------|------------------|
| **crps** | Primary — CRPS vs. FAWN reference season-years (or curator targets) |
| **crps_relative** | CRPS / target median — comparable across seasons |
| **mean_pinball_loss** | Average pinball loss at 10th / 50th / 90th percentiles |
| **interval_width** | p90 − p10 (should widen down the specificity gradient within a season) |
| **relative_interval_width** | (p90 − p10) / target median — comparable across seasons |
| **ece_confidence** | Self-reported confidence vs. relative median error |
| **power_analysis** | Paired per-scenario CRPS gaps between models; t-stat and scenarios needed for 80% power (see below) |
| **seasonal_consistency** | Self-consistency: does annual p50 cohere with wet + dry p50s? (no ground truth needed) |

### Power analysis (`summary.json` → `power_analysis`)

Because every model answers the same scenarios, scoring compares **per-scenario differences** in `crps_relative` (shared scenario difficulty cancels). For each model pair the summary reports:

- `delta_crps_relative` — observed mean paired gap
- `t_paired` — paired t at the current scenario count
- `scenarios_needed_80pct_power` — back-of-envelope n for α=0.05, 80% power at the observed gap
- `min_detectable_delta_at_n` — smallest gap detectable at the current n

Treat scenarios as exchangeable for this calculation (generous). With multiple pairs, apply a multiple-comparison correction before claiming significance. **End-to-end tier gaps** (e.g. top vs. bottom model) can be nominally detectable at n=18; **adjacent tiers** generally are not.

### Seasonal consistency (`summary.json` → `seasonal_consistency`)

For each location × prompt-variant cell, compute:

`gap = (p50_annual − p50_wet − p50_dry) / p50_annual`

May sits in neither season window and quantiles are not additive, so **FAWN targets also show a positive gap (~+28% to +32%)** — that is the calibrated reference, not zero. Models with large `gap_spread` across cells give internally incoherent annual vs. seasonal answers even when each cell looks plausible in isolation.

## Quick start

```bash
pip install -e .
python -m src.validate_scenarios
```

### Demo (no API)

```bash
python scripts/generate_demo_predictions.py
python -m src.score --run results/demo
python scripts/inspect_summary.py results/demo
```

### API eval

```bash
cp .env.example .env
# Add real keys only when running evals.

python -m src.run_eval --models gpt-4o-mini,gpt-4o --output results/openai-seasonal
python -m src.run_eval \
  --models claude-haiku-4-5,claude-sonnet-4-6,claude-opus-4-8,claude-fable-5 \
  --output results/anthropic-seasonal
python -m src.score --run results/openai-seasonal
python -m src.score --run results/anthropic-seasonal
python scripts/power_analysis.py results/anthropic-seasonal results/openai-seasonal
```

Interrupted runs **resume automatically** — re-run the same command to continue. Use `--fresh` to start over.

**Optional — enable thinking on Claude models** (Fable/Mythos always think; disables temperature=0 on other Claudes):

```bash
python -m src.run_eval --models claude-sonnet-4-6,claude-opus-4-8 \
  --output results/anthropic-thinking --thinking --fresh
```

Both providers use the same elicitation protocol: structured JSON schema output (OpenAI `response_format`, Anthropic `output_config.format`). No tools, no extra prompt instructions.

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
python scripts/inspect_sync.py            # print target quantiles from fawn_sync.json
```

Use at least **3 calendar years** of zips (recommended: **2005–2025**). Optional `--extract-zips` caches extracted CSVs under `data/fawn_raw/extracted/` for faster re-runs.

The sync aggregates 15-minute data to monthly totals, then assembles **seasonal totals** per station and season-year (annual Jan–Dec, wet Jun–Sep, dry Dec–Feb; December counts toward the following dry-season year). Station-season-years with missing months or >5% missing intervals are excluded.

Scenario station IDs (from FAWN metadata, not sequential 107/204):

| Scenario | FAWN ID | Name |
|----------|---------|------|
| Miami / specific | **440** | Homestead |
| Gulf ~100k city | **480** | North Port |

Writes `data/fawn_sync.json` with per-scenario reference years and pooled distributions.

## Outputs

Per run directory:

| File | Contents |
|------|----------|
| `predictions.jsonl` | One structured prediction per scenario × model |
| `summary.json` | Overall + breakdowns; **`power_analysis`** and **`seasonal_consistency`** when applicable |
| `by_scenario.csv` | Per-prediction scores |
| `by_stratum.csv`, `by_season.csv`, `by_prompt_variant.csv` | Aggregates |
| `interval_width_by_stratum.png` | One panel per season |

Utility scripts: `scripts/inspect_summary.py`, `scripts/power_analysis.py`, `scripts/export_frame.py` (merge scored frames across runs for ad-hoc analysis).

## Limitations (read before citing results)

This is an eval-design portfolio piece, not a research paper. Claims should match what the design can support:

- **Scenario count.** 18 scenarios, one sample per cell. Paired testing helps for model-vs-model comparisons on the same scenarios, but adjacent-tier rankings are usually underpowered; see `power_analysis` in `summary.json`.
- **Specificity gradient.** Each stratum is one geographic instantiation. Seasonal replication adds difficulty variation but does not fully deconfound location idiosyncrasy from stratum effects.
- **Underspecified target width.** Pooled statewide FAWN quantiles are only slightly wider than single-station quantiles (Florida stations are climatologically similar). Flat model intervals may partly reflect the instrument, not only model failure. A stronger design would pool deliberately diverse climates (e.g. panhandle vs. keys).
- **Ground-truth uncertainty.** Target quantiles come from ~20 clean station-years per cell; sampling error on tail quantiles is non-trivial.
- **Protocol history.** Early Anthropic runs used forced tool-choice JSON; current code uses structured outputs for all Claude models. Compare runs only when elicitation protocol matches (check `manifest.json` and provider version).
- **Thinking confound.** `claude-fable-5` has always-on adaptive thinking; other models default to temperature=0 without thinking unless `--thinking` is set.

What *is* defensible without overclaiming: behavioral patterns visible across models (answer recycling, constant GPT confidence, seasonal vs. annual incoherence), tier-level gaps when `power_analysis` supports them, and the documented design iteration (quantile scoring, scale normalization, self-consistency metric, power back-of-envelope).

## License

MIT
