# Adversarial Rebuttal Loop — Round 2 (Author)

We accept the reviewer's central statistical-validity concerns. We did not run
new experiments. Every new quantity below is computed from a committed
publication artifact or from the per-seed ETT values already reported in the
paper. We now distinguish descriptive single-seed evidence, cross-hardware
sensitivity, and internal policy confirmation from statistical replication or
external generalization.

## R2.1 — ACCEPT

The 585-cell A100 matrix uses only seed 2021. Its 562/585 win count and mean
metric reductions are therefore descriptive point estimates with unquantified
seed variance. A strict win only records the sign of each reported metric at
that seed; it is not a cell-level significance test.

We added two evidence-backed bounds:

- From `docs/publication_results/a100_composed_summary.json`, 120 of the 562
  strict wins have `minimum_metric_gain_percent <= 1.0`, and 192 have
  `minimum_metric_gain_percent <= 2.0`. These cells are explicitly identified
  as marginal rather than seed-robust.
- From the values already reported in Appendix Table `tab:ett-seeds`, the
  max-min spread of three-seed MSE reduction across the four ETT datasets is
  0.37-6.29 percentage points on the official test (median 2.06) and
  0.40-3.38 points on the subsequent post-test timeline (median 1.68).

The ETT calculation is not a confidence interval for the broad matrix: it
covers only TimeMixer, horizon 96, four ETT datasets, and different
frequency-specific correction policies. It nevertheless shows that a
single-seed margin such as DE's 1.3-point mean MSE reduction is within the
observed scale of seed sensitivity in the only available multi-seed study. We
therefore do not claim that DE or any individual marginal cell is seed-robust.
No committed artifact supports a multi-seed confidence interval for the
585-cell win rate or its aggregate mean reduction.

Manuscript changes:

- `paper/latex/main.tex` (Abstract): labels the primary matrix as single-seed
  and names seed 2021.
- `paper/latex/sections/04_experimental_design.tex` (Statistical analysis):
  states that the broad matrix has no confidence interval or hypothesis test
  and that strict wins are not significance claims.
- `paper/latex/sections/05_results.tex` (Overall accuracy; Seed and hardware
  sensitivity): rounds aggregate reductions to one decimal, labels them point
  estimates, reports the 120/192 marginal-win counts, and gives the bounded
  ETT seed-sensitivity comparison.
- `paper/latex/sections/appendix_sections/05_ett.tex` (Full ETT prequential
  results): reports the official-test and post-test three-seed ranges and
  limits their interpretation.
- `paper/latex/sections/appendix_sections/07_aggregate.tex` (Aggregate
  results): labels the table single-seed and removes two-decimal precision
  from the overall mean reductions.
- `paper/latex/sections/06_limitations_conclusion.tex` (Limitations): makes
  broad multi-seed uncertainty and marginal-cell reruns explicit future work.

## R2.2 — ACCEPT

The A10G execution is not an exact numerical reproduction of A100. From
`docs/publication_results/hardware_comparison.json`:

- only 472/1326 baseline and 469/1326 corrected values satisfy the frozen
  tolerance `1e-5 + 1e-4 * max(abs(a10g), abs(a100))`;
- the median A100-A10G relative discrepancies are 0.0885% for baseline and
  0.0608% for corrected values; the 90th percentiles are 2.95% and 2.36%, the
  95th percentiles are 5.80% and 5.47%, and the maxima are 32.79% and 28.13%;
- selected methods agree for 569/585 cells and selected parameters for
  529/585;
- strict-win classifications agree for 581/585 cells.

The four verdict changes are
`long_term/ETTh2/TimeMixer/336`,
`long_term/ETTm1/PatchTST/720`, `epf/NP/TimesNet/24`, and
`epf/DE/FEDformer/24`. Every one is an A100 win and A10G non-win, explaining
the 562 versus 558 totals. We now call A10G a cross-hardware sensitivity check,
not numerical replication. A100 remains the reporting default required by the
committed publication snapshot, but that convention does not make its
individual-cell values hardware-invariant.

We also evaluated the hardware coupling directly. Joining the 462 IDs in
`docs/publication_results/confirmation_summary.json` to the cell verdicts in
`hardware_comparison.json` gives 437/462 strict wins on A10G and 440/462 on
A100. Three holdout verdicts change from A10G non-win to A100 win
(`ETTm1/PatchTST/720`, `NP/TimesNet/24`, and `DE/FEDformer/24`); none changes
in the reverse direction. This A100 value is a post hoc cross-tab, not a new
independent confirmation experiment. The manuscript now states plainly that
the historical 437/462 confirmation and 562/585 headline come from
numerically inconsistent hardware executions.

Manuscript changes:

- `paper/latex/sections/04_experimental_design.tex` (Training and selection):
  renames A10G as a sensitivity check and explains the A100 reporting
  convention.
- `paper/latex/sections/05_results.tex` (Seed and hardware sensitivity;
  Frozen policy confirmation): reports the discrepancy distribution,
  agreement rates, four verdict changes, and the A100 440/462 cross-tab.
- `paper/latex/sections/appendix_sections/02_data.tex` (Dataset and model
  scope): states that the strict numerical-consistency gate fails.
- `paper/latex/sections/appendix_sections/07_aggregate.tex` (Aggregate
  results): adds the cross-hardware distribution table and identifies all four
  changed verdicts.
- `paper/latex/sections/06_limitations_conclusion.tex` (Limitations): limits
  the A10G evidence to directional sensitivity rather than exact
  reproducibility.

## R2.3 — ACCEPT

The timestamp boundary is outcome-blind but opportunistic. The exact rule in
`docs/publication_results/confirmation_summary.json` is the first 123
completed cells ordered by `started_unix`; it is not stratified by dataset,
model, task family, or baseline difficulty.

The same artifact and `a10g_composed_summary.json` show material composition
imbalance. All 123 development cells are long-term tasks. The holdout contains
241 long-term, 156 PEMS, and 65 EPF cells. All 52 ETTh1 cells are in
development, while all PEMS and EPF cells are in the holdout. Both partitions
contain all 13 model architectures. Thus the split does include task families
and datasets absent from development, but it is not a clean dataset-family or
model-family holdout and remains entirely within the predefined benchmark
suite.

For a limited post hoc difficulty diagnostic, we computed baseline A10G test
quartiles within the normalized long-term metric space. Development versus
long-term holdout MSE quartiles are 0.335/0.424/0.498 versus
0.264/0.399/0.561; MAE quartiles are 0.347/0.430/0.478 versus
0.316/0.375/0.437. These overlapping distributions do not prove balance:
dataset composition differs, and PEMS and EPF metrics cannot be pooled with
normalized long-term errors. We therefore concede that difficulty balance is
not established.

The confirmation claim is now scoped precisely: it shows that a rule specified
from earlier-started A10G long-term cells remained effective on later-started
model-dataset-horizon combinations and task families in the same benchmark
suite. It does not establish generalization to new model architectures,
external datasets, a prospectively stratified difficulty holdout, or new
training seeds.

Manuscript changes:

- `paper/latex/sections/04_experimental_design.tex` (Internal policy
  holdout): gives the exact timestamp rule, composition, opportunistic nature,
  and bounded interpretation.
- `paper/latex/sections/appendix_sections/03_search.tex` (Internal policy
  holdout): replaces the previous generic independence language with the exact
  unstratified execution-order definition.
- `paper/latex/sections/05_results.tex` (Frozen policy confirmation): limits
  the result to later-started in-suite combinations and rejects external or
  difficulty-balanced interpretation.
- `paper/latex/sections/appendix_sections/07_aggregate.tex` (Aggregate
  results): adds split-composition and baseline-difficulty diagnostics.
- `paper/latex/sections/06_limitations_conclusion.tex` (Limitations): records
  dataset-family and model-family holdouts as future work.
