# Adversarial Rebuttal Loop — Round 4 (Author)

We accept the reviewer's presentation-honesty concerns. We did not run new
experiments. The compact publication snapshot does not contain the prediction
bundles or isolated timing measurements needed for a new experiment or an
overhead benchmark, so we narrow those claims rather than infer missing
evidence. A separate committed paper extract supports the existing real-data
appendix example. All failure counts below come directly from the committed
A100 per-cell records.

## R4.1 — ACCEPT

Figure 1 is a hand-constructed motivation, not empirical evidence. The phrase
"recovers the depicted truth by construction" made that distinction less
clear and gave disproportionate prominence to residual retrieval. We removed
it and now label the figure explicitly as an illustrative schematic whose
curves depict a hypothesis rather than demonstrate the mechanism.

The compact snapshot's `docs/publication_results/README.md` states that raw
prediction bundles remain outside Git, so those result artifacts do not
support selecting a new real-data case. The manuscript does, however, already
include Appendix Figure `fig:residual-example`, backed by the committed
`paper/data/real_retrieval_case.json` extract and identified as a real
retrieval-selected ETTh1 MULL case from a frozen TimeMixer. We now point to
that empirical case directly from Figure 1.

We also place the schematic in the observed selection context. From
`docs/publication_results/a100_composed_summary.json`, residual retrieval is
selected in 214/585 cells, while temporal or bias corrections are selected in
371/585. The caption now states these counts instead of allowing the opening
illustration to stand in for the portfolio evidence.

Manuscript changes:

- `paper/latex/sections/01_introduction.tex` (Figure 1 caption): labels the
  panel a hand-constructed illustrative schematic, removes "by construction,"
  says that it depicts a hypothesized mechanism, reports the 214/585 minority
  selection count, and points to the real ETTh1 appendix example.

## R4.2 — ACCEPT

The paper did not measure the revision layer's efficiency. In particular,
`elapsed_seconds` in `a100_composed_summary.json` and
`a10g_composed_summary.json` covers complete experimental cells, including
base-model work; it does not isolate memory construction, retrieval, or
revision. The committed artifacts contain no isolated retrieval latency or
memory-footprint measurement. We therefore do not use those elapsed times as
overhead evidence.

We replaced "Interpretability and efficiency" with the narrower,
directly-supported claim "Interpretability and parameter-free revision":
the method updates no base-model parameter and reuses validation forecasts.
We now state immediately that these are qualitative properties and that
memory-construction time, retrieval latency, and storage footprint were not
measured.

Manuscript changes:

- `paper/latex/sections/01_introduction.tex` (practical advantages): removes
  the efficiency label, retains only the parameter-update and forecast-reuse
  facts, and discloses the unmeasured overheads.
- `paper/latex/sections/06_limitations_conclusion.tex` (Limitations): states
  that complete-cell elapsed times cannot isolate revision overhead and names
  memory construction, retrieval latency, and storage as unmeasured.

## R4.3 — ACCEPT

The recurrence statement is explanatory after the fact, not a validated
pre-deployment rule. The committed outputs do not retain candidate-level
validation losses, so they cannot support a calibration curve, quantify the
validation-to-test gap, or derive a validation-time warning for a new cell.
We now state that limitation in the Method, Results, Limitations, and
Conclusion, and we identify prospective recurrence or shift diagnostics as
future work.

We added the bounded descriptive risk map supported by
`docs/publication_results/a100_composed_summary.json`:

- TimeMixer accounts for 7/23 non-strict cells and is strict in 38/45 cells
  overall. Six of those seven failures use overlap revision.
- Weather and DE account for 14/23 non-strict cells, seven each. Weather's
  failures comprise six retrieval and one overlap selection; DE's comprise
  five overlap and two seasonal selections.
- All five PEMS failures are TimeMixer--overlap combinations: three PEMS04
  horizons and two PEMS07 horizons. The three PEMS04 failures are the
  largest-magnitude failures, with the worst a 36.5% MAPE increase.
- By selected family, the non-strict counts are 11/159 for overlap, 7/214 for
  retrieval, 5/156 for seasonal, and 0/56 for causal bias.

These are post-hoc single-seed frequencies, not estimated deployment
probabilities. They identify combinations that deserve caution without
claiming an a priori classifier that the evidence cannot support.

Manuscript changes:

- `paper/latex/sections/03_method.tex` (Validation-guided revision policy):
  states that aggregate validation selection neither estimates recurrence
  probability nor supplies a validated regression warning.
- `paper/latex/sections/05_results.tex` (Abstention and selection failures):
  adds the backbone--dataset--family concentrations, family denominators, and
  the explicit boundary between an observed risk map and a predictive rule.
- `paper/latex/sections/06_limitations_conclusion.tex` (Limitations and
  Conclusion): records the missing candidate-level evidence, makes prospective
  recurrence/shift diagnostics future work, and calls recurrence an
  explanatory hypothesis rather than an a priori deployment rule.
