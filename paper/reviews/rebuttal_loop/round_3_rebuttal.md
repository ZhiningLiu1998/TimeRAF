# Adversarial Rebuttal Loop — Round 3 (Author)

We accept the central request for internal justification and did not run new
experiments. The revision distinguishes selected configurations from causal
component evidence: the committed artifacts support configuration counts,
dataset-level family attribution, and test-side regression characterization,
but not component ablations, per-origin shrinkage diagnostics, or a
validation-to-test calibration curve.

## R3.1 — ACCEPT

The paper did not establish which descriptor and reliability components earn
their place. A selected hyperparameter is not an ablation, and the available
grid couples \(k\), \(\tau\), and \(\lambda\), so even their selection counts
cannot isolate individual effects.

Contrary to our initial expectation, the A100 records in
`docs/publication_results/hardware_comparison.json` do retain the selected
parameters for all 214 retrieval-selected cells. Direct counts from
`cells[].selected_params.a100` are:

- \(\alpha=.05/.1/.2/.3/.5\): 0/1/6/29/178 cells;
- \((k,\tau,\lambda)=(8,.1,1)/(16,.3,4)\): 151/63 cells;
- descriptor tail length 48/96/168: 57/148/9 cells;
- base-forecast features excluded/included: 61/153 cells.

All 214 configurations use first differences, 12 pooled time steps, and a
maximum of 16 channel groups. Calendar features are present in 91 and absent
in 123; the latter are the PEMS configurations for which calendar marks are
unavailable. The concentration at \(\alpha=.5\), the largest searched value,
does not show that shrinkage is inert or active because the actual correction
is also multiplied by \(\boldsymbol{\rho}_t\). Neither
`hardware_comparison.json` nor `a100_composed_summary.json` retains per-origin
\(\boldsymbol{\rho}_t\), candidate-level validation losses, component-removed
predictions, or a wider sensitivity sweep. We therefore cannot measure how
often shrinkage materially departs from one or claim that any descriptor
component improves retrieval.

Manuscript changes:

- `paper/latex/sections/03_method.tex` (Context--forecast descriptor): calls
  16-group pooling potentially lossy and states that descriptor components and
  pooled-versus-channelwise retrieval are not ablated.
- `paper/latex/sections/05_results.tex` (Overall accuracy and selected
  policies): reports the principal selected-configuration counts and limits
  their interpretation.
- `paper/latex/sections/appendix_sections/07_aggregate.tex` (Aggregate
  results): adds the complete selected-configuration table and identifies
  which fixed quantities and diagnostics are absent.
- `paper/latex/sections/06_limitations_conclusion.tex` (Limitations):
  registers descriptor removal, pooling, \(\alpha\)/\(k\) sensitivity, and
  shrinkage diagnostics as required future experiments.

## R3.2 — PARTIALLY ACCEPT

We accept that abstention is available but not demonstrated. Identity and
static bias are selected in 0/585 A100 cells, so every primary test forecast is
revised. We now remove language that could make the unused identity option
sound like observed protection.

We also accept the selection-failure characterization. From
`docs/publication_results/a100_composed_summary.json`, the 23 non-strict cells
select historical residual retrieval in 7 cases, overlap in 11, seasonal in 5,
and causal bias in 0. They occur on Weather (7), DE (7), PEMS04 (3), PEMS07
(2), and ETTh2, Electricity, NP, and BE (one each). Fourteen degrade every
reported metric; nine have mixed signs. Taking, for each cell, the largest
percentage increase among its reported test metrics, degradation ranges from
0.0194% to 36.5015%, with median 0.9727%. The largest failures are all three
PEMS04--TimeMixer horizons under overlap revision; the maximum is a 36.5015%
MAPE increase.

We do not accept “validation-to-test selection overfitting” as an established
cause. The broad matrix uses the conservative-margin family policy, not the
additional-system study's direct global argmin. More importantly, the
committed snapshot lacks candidate-level validation losses, so it cannot
quantify the validation-to-test gap or distinguish selector overfitting from
temporal distribution shift. The observed failures are evidence of
miscalibrated test-side protection and are consistent with either explanation;
we state that directly without assigning an unmeasured cause.

Manuscript changes:

- `paper/latex/sections/01_introduction.tex` (Introduction, practical
  advantages): renames reliability as mechanisms and states that neither
  calibrated abstention nor operational shrinkage is demonstrated.
- `paper/latex/sections/03_method.tex` (Validation-guided revision policy):
  calls identity an abstention candidate and separates capability from
  calibration.
- `paper/latex/sections/05_results.tex` (Abstention and selection failures):
  reports family, dataset, sign-pattern, and degradation statistics for all 23
  non-strict cells and explains the missing validation evidence.
- `paper/latex/sections/appendix_sections/07_aggregate.tex` (Aggregate
  results): adds the eight-dataset regression table.
- `paper/latex/sections/06_limitations_conclusion.tex` (Limitations): records
  abstention calibration and candidate-level validation/test analysis as open.
- `paper/latex/main.tex` (Ethics statement): states that identity was never
  selected and is not demonstrated operational protection.

## R3.3 — ACCEPT

The pooled descriptor's fidelity is unvalidated, and aggregate family counts
hide important dataset differences. From
`docs/publication_results/hardware_comparison.json`, Traffic selects retrieval
in only 8/52 cells, versus 27 overlap, 16 seasonal, and 1 causal-bias
selection. Electricity selects retrieval in 28/52, overlap in 14, seasonal in
8, and causal bias in 2. Thus the reviewer's reading is correct: Traffic's
52/52 strict wins are mostly attributable to temporal corrections, not
retrieval.

The full breakdown adds an important qualification. Retrieval is selected in
35/39, 29/39, 30/39, and 29/39 cells on PEMS03, PEMS04, PEMS07, and PEMS08,
respectively, even though these datasets also have 170--883 channels and use
the same pooling limit. Therefore family selection varies across
high-dimensional datasets; it neither uniformly rejects nor validates the
pooled descriptor. Only a pooled-versus-channelwise or descriptor ablation can
test fidelity.

Manuscript changes:

- `paper/latex/sections/05_results.tex` (Dataset-level family selection):
  reports and interprets the Traffic, Electricity, and PEMS counts while
  explicitly rejecting fidelity claims from selection frequency.
- `paper/latex/sections/appendix_sections/07_aggregate.tex` (Aggregate
  results): adds family counts for every one of the 16 datasets.
- `paper/latex/sections/03_method.tex` and
  `paper/latex/sections/06_limitations_conclusion.tex`: identify 16-group
  compression as potentially lossy and pooled-versus-channelwise evaluation as
  missing.
