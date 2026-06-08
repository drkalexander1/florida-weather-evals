# Weather eval — design notes

## Core hypothesis

Follow-on to the Michigan bird eval. Primary question: do Anthropic models show less tier stratification than GPT models on calibration tasks, and does that replicate across domains?

Secondary question: can models represent **known unknowns** appropriately — i.e., do they express wider uncertainty when the question is genuinely underspecified, vs. when a specific answer is knowable?

## Key design distinction

**Bird eval** tested unknown unknowns — models either know the ornithology or they don't.

**Weather eval** tests known unknowns — some questions have retrievable answers (Miami annual rainfall), others are underspecified by design (random Florida location). A well-calibrated model should give progressively wider intervals as specificity decreases.

## Uncertainty decomposition

Two distinct things being measured:

- **Epistemic uncertainty** — how confident is the model in its point estimate? Reducible with more information.
- **Aleatoric uncertainty** — natural variance in the phenomenon itself. Irreducible even with perfect knowledge.

Models likely conflate these, but not for the same reasons humans do. Human conflation stems from cognitive heuristics (gambler's fallacy etc.). Model conflation is more likely a training data artifact — natural language doesn't cleanly separate the two, so the linguistic representation is blurred from the start.

**Testable prediction:** asking for uncertainty in statistical language ("report posterior mean and aleatoric variance separately") should produce better separation than natural language ("how confident are you?"). Build both prompt variants in.

## Specificity gradient

| Question | What model can draw on | Expected interval width |
|----------|----------------------|------------------------|
| Annual rainfall in Miami | Likely memorized — specific figure in training data | Narrow |
| Gulf coast Florida city, ~100k pop | Regional inference | Moderate |
| Random Florida location | Population-level Florida estimate | Wide |

Key finding to look for: do models give equally confident answers across the gradient (failure) or do intervals widen appropriately (success)?

## Scoring

Ask models for **10th, 50th, and 90th percentile estimates** rather than point estimate + confidence number. Score with **CRPS** (Continuous Ranked Probability Score) — a proper scoring rule for distributional predictions that penalises both overconfident narrow intervals and unnecessarily wide ones.

Also retain a `confidence` field (matching bird eval structure) for cross-paper comparison.

## Prompt variants

Run each scenario with two prompt phrasings:
1. **Natural language** — "My grandma lives in Florida. How much rain should she expect over the next year?"
2. **Statistical** — "Provide your 10th, 50th, and 90th percentile estimates for annual rainfall at a random Florida agricultural monitoring station."

Compare whether interval width and calibration differ between phrasings.

## Ground truth data — FAWN

**Source:** Florida Automated Weather Network (FAWN) — https://fawn.ifas.ufl.edu  
**Coverage:** 53 stations, ~20 years, 15-minute intervals  
**Download:** Manual CSV download per station  
**Rainfall column:** inches per hour (rate) — multiply by 0.25 to get inches per 15-min interval  
**Quality flags:** Keep flag=0 only. Flag 4 = missing, flags 1-3 = erroneous/suspect.  
**Backup sensor:** Primary rainfall + backup column — fall back to backup when primary is flagged.

**Processing pipeline:**
1. Filter to quality_flag == 0
2. Multiply by 0.25 → actual inches per interval
3. Sum to daily totals
4. Sum to annual totals per station per year
5. Exclude station-years with >5% missing intervals
6. Compute per-station: mean, std, 10th/50th/90th percentiles across clean years
7. Compute cross-station pooled distribution (ground truth for underspecified questions)
8. Write `data/fawn_sync.json` — mirrors structure of `ebird_sync.json`

## Note on reference distributions

"Random Florida location" has a well-defined ground truth: the distribution across all 53 FAWN stations. Expanding to "random US location" would require committing to a weighting scheme (by number of towns, by population, by land area) that significantly changes the correct answer — avoid this scope until the reference distribution question is resolved.

## Relationship to bird eval

- Same framework: stratified scenarios, dual scoring targets, per-model and per-stratum breakdowns
- New element: prompt variant comparison (natural vs. statistical language)
- New scoring: CRPS on quantile estimates rather than log loss on point probability
- Retain `confidence` field for cross-eval comparison
- Writeup should explicitly test whether Sonnet > Opus pattern replicates
