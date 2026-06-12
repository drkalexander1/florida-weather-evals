# Results — seasonal runs (June 2026)

Runs: `results/anthropic-seasonal` (claude-haiku-4-5, claude-sonnet-4-6, claude-opus-4-8, claude-fable-5) and `results/openai-seasonal` (gpt-4o-mini, gpt-4o). 18 scenarios each (3 strata × 3 seasons × 2 prompt variants), FAWN ground truth.

**Protocol note.** The core Anthropic run is the **tool-use trio** (haiku, sonnet, opus): 54 predictions collected first via forced tool-choice JSON. **claude-fable-5** was added afterward (18 predictions) after the initial run failed on Fable's always-on thinking; those 18 used voluntary tool-call elicitation, not the same protocol as the trio. OpenAI models used structured `response_format` throughout. Treat Fable as exploratory; the behavioral findings below are unchanged with or without it.

Read the [Limitations](README.md#limitations-read-before-citing-results) section before citing anything here; n=18 supports tier-level claims only.

## Headline

**No model reliably widens intervals down the specificity gradient** — flatness is partly an instrument limitation (pooled FAWN targets barely widen either), but inverted widths and answer recycling are not. The seasonal gradient, by contrast, produced clean signal: all six models track seasonal difficulty the same way, and the dry season exposes scale-aware uncertainty failures that annual-only evals miss.

## 1. Specificity gradient: no model widens appropriately

A calibrated forecaster should give wider intervals for "a random Florida location" than for "Miami." FAWN annual target intervals are almost flat across strata (~24.5–29.7 in wide; specific_station is actually the widest, not underspecified). The models don't track even that weak target gradient:

| Model | random_fl interval wider than Miami's | Identical answers recycled across locations |
|-------|--------------------------------------|---------------------------------------------|
| claude-opus-4-8 | 3/6 cells | 0/6 cells |
| claude-sonnet-4-6 | 4/6 | 2/6 |
| claude-fable-5 | 2/6 | 1/6 |
| claude-haiku-4-5 | 1/6 | 4/6 |
| gpt-4o | 2/6 | 3/6 |
| gpt-4o-mini | 1/6 | 5/6 |

"Recycled" = the same (p10, p50, p90) tuple appears at two or more of the three locations within a season×variant cell. Opus is the only model with zero recycling.

Two distinct failure modes:

- **Answer recycling.** gpt-4o-mini reuses the same tuple across locations in 5/6 cells (Gulf and random match exactly in 4/6). Haiku recycles in 4/6. Models appear to compute one "Florida rainfall" answer and reuse it.
- **Inverted widths.** Several models give the underspecified question a *narrower* interval than the specific one (haiku annual natural: Miami 33 in wide, random_fl 20 in wide). Flat intervals are partly explained by flat targets; inverted ones are not.

Mean interval width by stratum confirms it in aggregate: Anthropic models actually *narrow* slightly down the gradient (20.2 → 18.4 → 18.1 in); OpenAI is flat (22.8 → 21.5 → 21.6 in). These aggregates are identical for the tool-use trio alone.

## 2. Seasonal gradient: clean, consistent signal

Scale-normalized CRPS (`crps_relative` = CRPS / target median) by season, all models pooled per provider:

| Season | Anthropic (4 models) | Anthropic (trio only) | OpenAI | Target median scale |
|--------|---------------------|----------------------|--------|--------------------|
| annual | 0.117 | 0.118 | 0.118 | ~50–55 in |
| wet (Jun–Sep) | 0.164 | 0.171 | 0.210 | ~28–33 in |
| dry (Dec–Feb) | 0.500 | 0.523 | 0.486 | ~6–8 in |

Dry season is ~4× harder than annual *relative to scale*, uniformly across every model. Anthropic models run wide in relative terms on dry season (1.60 — intervals wider than the median itself). gpt-4o's dry-season intervals are tight in absolute inches on Gulf/random cells (6.5 in vs. 8–10 in targets) while still modestly wide relative to scale (1.21). Annual-only evals would have shown all six models as well-calibrated; the dry season is where calibration actually differentiates.

## 3. Seasonal self-consistency: two models fail in opposite ways

`gap = (p50_annual − p50_wet − p50_dry) / p50_annual` per location. The FAWN reference gap is **+0.28 to +0.32** (May/Oct–Nov fall in neither window; quantiles aren't additive).

| Model | Mean gap | Spread | Verdict |
|-------|----------|--------|---------|
| claude-opus-4-8 | 0.307 | 0.069 | coherent |
| claude-fable-5 | 0.321 | 0.095 | coherent (exploratory) |
| claude-sonnet-4-6 | 0.305 | 0.164 | coherent |
| gpt-4o-mini | 0.322 | 0.213 | coherent on average, noisy |
| **gpt-4o** | **0.120** | 0.121 | systematic: over-allocates to wet season |
| **claude-haiku-4-5** | 0.247 | **0.550** | incoherent: includes a physically impossible cell |

- **haiku-4-5** produced wet=50 + dry=9.5 against annual=55 for the Gulf city (gap −0.08): its seasonal medians *sum past its own annual median*, which no real rainfall distribution can do. Each answer looks plausible alone; jointly they're impossible. This is exactly what the consistency check was built to catch — no ground truth needed.
- **gpt-4o** assigns 73–82% of annual rainfall to the four wet-season months across all three locations (FAWN targets: ~57–63%). Its answers are internally *consistent* but consistently wrong about Florida's seasonal split — a confident, wrong prior rather than incoherence.

## 4. Model ranking and what's statistically supportable

### Primary result: tool-use trio + GPT

Overall `crps_relative` (lower better), **excluding Fable**:

sonnet-4-6 **0.254** < opus-4-8 0.260 < gpt-4o-mini 0.269 < gpt-4o 0.274 < haiku-4-5 **0.298**

Within Anthropic, Sonnet ≈ Opus (t = 0.58, would need ~420 scenarios to resolve). Haiku is clearly worst (vs. Opus: t = 2.31, ~26 scenarios needed). **Both GPT models sit between Sonnet and Opus** — not a clean "Anthropic beats GPT" story for the comparable run.

gpt-4o vs. gpt-4o-mini is a tie (t = 0.22, would need ~2,900 scenarios).

### Exploratory: Fable add-on

Including Fable (different protocol): fable-5 **0.230** < sonnet 0.254 < opus 0.260 < … < haiku 0.298. Fable vs. haiku is nominally detectable at n=18 (Δ = −0.069, t = −2.65) — a single nominal result before any multiple-comparison correction. Fable vs. Sonnet/Opus is borderline underpowered (~30–38 scenarios needed). **Do not rank Fable against the trio or GPT without the protocol caveat.**

On the design doc's hypotheses: the bird-eval "Sonnet > Opus" pattern replicates **directionally** (0.254 vs. 0.260) but is nowhere near significant. Tier stratification holds at the Anthropic extremes (Haiku worst); adjacent Claude tiers and both GPT tiers are indistinguishable at this n.

## 5. Confidence reporting: GPT's is decorative

- **gpt-4o reported confidence = 0.80 for all 18 scenarios.** gpt-4o-mini used 0.80/0.85. The field carries almost no information.
- Claude models vary confidence across scenarios (sonnet uses 7 distinct values, 0.45–0.80) and are directionally sensible — lower confidence on vaguer questions.
- ECE (lower = better): opus 0.61, fable 0.62, sonnet 0.64 < haiku 0.78 < gpt-4o 0.80, gpt-4o-mini 0.84. All values are poor in absolute terms (everyone is overconfident against the relative-error criterion), but the Anthropic models' self-reports at least correlate with their accuracy.

## 6. Prompt variants: statistical phrasing helps, mildly

Statistical phrasing beats natural phrasing on `crps_relative` for both providers (Anthropic 0.248 vs. 0.273; OpenAI 0.261 vs. 0.282). Models do give genuinely different answers per phrasing (5–8 of 9 location×season cells differ per model), most often tightening dry-season quantiles under statistical phrasing. The design doc predicted statistical language would separate epistemic from aleatoric uncertainty better; the data is consistent with a weaker version — statistical phrasing shifts answers toward the target, but doesn't fix the specificity-gradient failure for any model.

## What this eval can and can't claim

**Defensible:** the behavioral patterns above (recycling, constant GPT confidence, haiku's impossible cell, gpt-4o's wet-season over-allocation, the seasonal difficulty ordering) — these are direct observations on the tool-use trio and GPT runs, unchanged by Fable. Haiku vs. Opus/Sonnet tier gap (with power caveat).

**Exploratory only:** Fable's top-line score and fable-vs-haiku gap — different elicitation protocol.

**Not defensible:** adjacent-tier rankings (sonnet vs. opus, 4o vs. 4o-mini), ranking Fable against the trio, any cross-provider ranking finer than "models cluster between 0.25 and 0.30," or generalization beyond Florida rainfall — the specificity gradient has one geographic instantiation per stratum.
