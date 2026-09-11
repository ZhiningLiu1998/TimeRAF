# TimeRaf Paper Narrative

## Title

**Retrieve the Error, Not the Input: Model-Adaptive Residual Retrieval for
Time-Series Forecasting**

## Central question

When we augment a time-series forecaster with retrieval, what should we
retrieve?

## Thesis

A retrieval-augmented forecaster should retrieve its own past errors, not
historical input windows and the futures that followed them. The residual is
the part of the signal a trained forecaster misses, it is model-specific, and
transferring it preserves the level, phase, and rhythm the base forecast
already had right.

## Core argument

1. A modern forecaster is strong at routine level, phase, and rhythm; what it
   misses is a small, structured, recurring, model-specific residual.
2. Whole-window input similarity is dominated by the routine structure the base
   forecast already predicts, so a rare-event prefix rarely decides the
   neighbourhood (signal dilution).
3. A retrieved future is an absolute trajectory and carries its own level,
   phase, and covariate response, so injecting it overwrites correct structure
   (object mismatch).
4. An input-similarity neighbourhood is identical whichever forecaster is being
   augmented, so it cannot adapt to that forecaster's blind spot (model
   blindness).
5. Storing residuals fixes all three: contrast amplification, additive
   compatibility, and model adaptivity. The query descriptor includes the base
   forecast, which is what makes the index model-conditioned.
6. Retrieved errors help only where errors recur, so reliability shrinkage and
   a validation gate over simpler corrections and identity are part of the
   method, not a hedge.

## Evidence and its honest boundary

1. Fixed-forecast comparison, 585 settings, identical base tensors: strict wins
   are Residual-kNN 490, TimeRaf 533, Analog-kNN 478, RAFT 375, SARAF 384.
2. Residual retrieval beats the two published input-retrieval operators by a
   wide margin, including PEMS medians of ~10-12% against ~0%.
3. The endpoint-aligned Analog-kNN control is strong. It wins the long-term and
   PEMS \emph{absolute} aggregates, driven by weak backbones.
4. Spearman rho between backbone base MSE and the residual-minus-analog
   advantage is -0.94 (long-term), -0.70 (PEMS), -0.27 (EPF): the error object
   wins for accurate forecasters and loses for poor ones. Post-hoc analysis.
5. Restricted to the five most accurate backbones (225 settings): Residual-kNN
   170, TimeRaf 185, Analog-kNN 157, RAFT 77, SARAF 80.
6. Primary matrix breadth: 562/585 strict wins across 13 architectures;
   437/462 on the frozen held-out complement.
7. Additional systems (ensembles, AutoML, Chronos-Bolt): 81/94 strict wins,
   with error retrieval selected in 70 and winning 65. Strongest direct
   evidence for the error object.
8. Primary-matrix attribution: retrieval selected in 214/585 and responsible
   for 207 of 562 strict wins; simpler corrections carry the remaining 355. The
   claim is that error retrieval is the right default, not the sole cause.

## Main-paper structure (follows TimeFuse, arXiv 2505.18442)

1. **Introduction:** domain, architectural progress, the residual that remains,
   the three failure modes of input retrieval (Figure 1), the central question,
   the method, four advantages, three contributions.
2. **Preliminaries:** notation, model-relative residual, causality boundary,
   Problem 1.
3. **Method:** why errors instead of inputs (three drawbacks mapped 1-1 to
   three benefits), chronological error memory, context-forecast descriptor,
   retrieval and reliability shrinkage, validation gate and selection rule.
4. **Experiments:** setup; what should we retrieve (Table 1 absolute, Table 2
   strict wins, Figure 3 strength dependence); breadth over 13 backbones and
   six additional systems (Table 3); analysis of selection, native systems,
   forward transfer, and failures.
5. **Related work:** forecasting architectures, retrieval-augmented
   forecasting, forecast combination and post-hoc correction.
6. **Conclusion** with one consolidated scope-and-limitations paragraph.

## Figure roles

1. **Figure 1 (teaser):** periodic load plus rare fault; input retrieval copies
   a future at the wrong level; residual retrieval transfers only the error.
2. **Figure 2 (method):** validation-time calibration versus forecast-time
   revision, with the frozen backbone and the identity fallback.
3. **Figure 3 (finding):** residual-minus-analog advantage per backbone,
   ordered by base accuracy, with Spearman rho per task family.
4. **Appendix:** measured ETTh1 residual-retrieval example, Algorithm 1, and
   the selected-family composition by backbone and dataset.

## Writing rules for this paper

1. State results confidently; keep caveats in the single scope-and-limitations
   paragraph rather than after every claim.
2. Never hide the Analog-kNN control or the regimes where it wins.
3. Every number traces to `docs/publication_results/`; table and figure sources
   are the generators in `paper/tools/`.
4. Main text occupies exactly nine full pages (see `AGENTS.md`).
