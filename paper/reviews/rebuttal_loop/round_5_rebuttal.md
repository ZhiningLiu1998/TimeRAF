# Adversarial Rebuttal Loop - Round 5 (Author)

We accept the reviewer's final coherence request. We did not run new
experiments. All numerical claims below come from the committed publication
snapshot, whose `SHA256SUMS` verification passes. We use one thesis throughout:

> On our benchmark suite, validation-guided additive revision broadly improves
> heterogeneous frozen forecasters without backbone retraining, while
> validation should choose among residual memory and simpler corrections
> because the useful family depends on the forecasting system.

## R5.1 - ACCEPT

The reviewer is correct that the title had moved to portfolio framing while
parts of the narrative still introduced retrieval as if it were the complete
method. We now state the thesis above in the abstract, introduction, and
conclusion. The central question asks whether additive revision broadly
improves heterogeneous frozen forecasters and which family validation selects.

Residual retrieval remains the principal novel candidate, so its technical
development remains detailed. However, it is now introduced from the outset as
one member of a portfolio containing seasonal, overlap, bias, and identity
policies. This distinction now appears before the retrieval discussion in
Related Work and in the first sentences of the Method overview.

Exact manuscript changes:

- `paper/OUTLINE.md` (Central question, new Thesis, and Core argument): aligns
  the planning document with the portfolio thesis and records the different
  roles of the broad matrix and additional-system study.
- `paper/latex/main.tex` (Abstract): opens with post-training correctability,
  states the common thesis, and presents retrieval and simpler corrections as
  alternative validation-selected families.
- `paper/latex/sections/01_introduction.tex` (Introduction): states the thesis
  before the retrieval motivation and aligns the central question with it.
- `paper/latex/sections/02_related_work.tex` (Related Work opening): explicitly
  positions residual retrieval as one candidate in a post-hoc revision
  portfolio.
- `paper/latex/sections/03_method.tex` (Overview): defines the portfolio before
  introducing residual memory.
- `paper/latex/sections/06_limitations_conclusion.tex` (Conclusion): restates
  the same thesis and closes with the evidence for both the portfolio and
  residual-memory parts.

## R5.2 - ACCEPT

The positive contribution is supported and should be stated directly. The
committed evidence establishes the following:

- `a100_composed_summary.json` reports 562/585 strict improvements at seed
  2021 across the 13 primary architectures.
- `key_metrics.json` reports 81/94 strict improvements across the six
  additional forecasting systems.
- The existing three-seed ETT tables report improvement for every dataset,
  metric, and seed on the chronologically subsequent timeline.
- `a100_composed_summary.json` reports residual retrieval as the largest
  individual primary family (214 cells); the additional-system artifact
  reports retrieval selection in 70/94 cells.

We state these results without inserting a caveat into the same sentence. The
Limitations section still bounds their interpretation: the 585-cell matrix is
single-seed, the studies use different selectors, and the paper lacks a
portfolio-minus-retrieval counterfactual. Those qualifications remain
important, but they no longer replace the conclusion.

Exact manuscript changes:

- `paper/latex/main.tex` (Abstract): adds the 13-architecture, six-system, and
  forward-time positive results and states the family attribution.
- `paper/latex/sections/01_introduction.tex` (Contributions): leads with the
  revision portfolio, then the residual candidate, and gives the exact scope
  and outcomes of the three studies.
- `paper/latex/sections/06_limitations_conclusion.tex` (Conclusion): states the
  established result directly before identifying the observed failure regimes.

## R5.3.1 - ACCEPT

The phrase "retrieval-selected ETTh1 case" was imprecise. The committed
`paper/data/real_retrieval_case.json` records a real ETTh1 residual-retrieval
prediction selected by a deterministic visualization screen. That screen does
not make the case part of aggregate method selection. We therefore use the
narrower description requested by the reviewer.

Exact manuscript change:

- `paper/latex/sections/01_introduction.tex` (Figure 1 caption): changes
  "retrieval-selected ETTh1 case" to "ETTh1 case illustrating residual
  retrieval," matching Appendix Figure `fig:residual-example`.

## R5.3.2 - PARTIALLY ACCEPT

We accept that the additional-system study is the strongest retrieval evidence
in the paper and now foreground it. We qualify one numerical implication in
the critique: 81/94 is the strict-win count for the complete selected
portfolio, not for retrieval alone. Direct aggregation of committed
`docs/publication_results/appendix_summary.json` fields
`cell_states[].method` and `cell_states[].all_test_metrics_improve` gives 65
strict wins among 70 retrieval-selected cells. The same artifact reports
81/94 strict wins overall.

Those 65/70 selected-policy outcomes are the cleanest evidence in our studies
that validation finds reusable model-specific residuals. They are not a
causal ablation: the artifacts do not contain the counterfactual result from
removing retrieval and selecting the next candidate. We retain that boundary.
The finding is consistent with the broad matrix, where residual memory is the
largest individual family but temporal families dominate collectively.

Exact manuscript change:

- `paper/latex/sections/05_results.tex` (Does revision transfer beyond
  individual backbones?): reports 65/70 retrieval-selected strict wins,
  distinguishes them from the 81/94 portfolio total, identifies this as the
  strongest residual-memory evidence, and explains its consistency with the
  broad matrix.

## R5.3.3 - ACCEPT

The diagnostics added in earlier rounds need explicit artifact provenance.
The reproducibility statement now names the committed files and maps each
diagnostic to its source:

- `a100_composed_summary.json`: primary per-cell outcomes and the non-strict
  risk map;
- `a10g_composed_summary.json`: secondary full-matrix result;
- `appendix_summary.json`: additional-system outcomes and family attribution,
  including the 65/70 retrieval-selected strict-win count;
- `hardware_comparison.json`: A100--A10G discrepancy distributions, selected
  hyperparameters, and per-dataset family counts;
- `confirmation_summary.json`: the fixed 462-cell mask and confirmation
  result;
- `key_metrics.json`: the machine-readable headline index.

It also names `docs/publication_results/SHA256SUMS`, which independently
authenticates these files in the committed snapshot.

Exact manuscript change:

- `paper/latex/main.tex` (Reproducibility statement): adds the artifact names,
  diagnostic-to-source mapping, and checksum manifest.
