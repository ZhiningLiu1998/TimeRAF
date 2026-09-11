# Adversarial Rebuttal Loop — Round 5 (Reviewer, final)

Four rounds of honest re-scoping have fixed the major integrity problems. My
final-round job is the opposite of the earlier ones: make sure the paper still
**coheres** and still asserts a **clear, defensible positive contribution**
rather than dissolving into caveats.

## R5.1 — Narrative coherence: is this now two papers stitched together?

The title is now "Validation-Guided Revision of Frozen Forecasts" (portfolio
framing), but the abstract's opening, the introduction's blind-spot motivation,
and the entire Related Work section still lead with the **retrieval** story
(RAFT/Cross-RAG/CRAFT/SARAF, "retrieve what the model missed"). Method emphasis
is likewise retrieval-first, with seasonal/overlap/bias presented as
"deliberately simple alternatives." After the re-scoping, a reader can perceive
a mismatch: retrieval-centric setup, portfolio-centric contribution.

Please do a coherence pass so the framing is consistent end to end:
- State the paper's thesis in **one sentence** and make the abstract,
  introduction, and conclusion all serve that same sentence.
- In Related Work and Method, make explicit *early* that residual retrieval is
  one candidate within a revision portfolio (you now say this in the intro and
  results; the related-work/method framing should not still read as if
  retrieval is the whole method).
- Verify the OUTLINE, abstract, intro central question, and conclusion state
  the **same** re-scoped thesis (they were edited in different rounds and may
  have drifted).

## R5.2 — Do not let the caveats swallow the contribution

Rounds 1–4 added many honest limitations. That is correct, but a paper must
still stand for something. State — crisply, in the abstract and conclusion, and
without hedging that immediately undercuts it — exactly what IS established:

> A validation-guided additive revision layer, requiring no backbone retraining,
> broadly improves heterogeneous frozen forecasters on this benchmark suite
> (single seed), across 13 architectures, six additional forecasting systems,
> and a forward-time prequential study; residual memory is the largest single
> correction family and the dominant one on the additional systems.

Make sure the contribution list and conclusion assert this cleanly. A reader
should finish the paper knowing what is true, not only what is uncertain. Check
that no sentence in the conclusion hedges the core positive result into
meaninglessness.

## R5.3 — Precision and constructive rebalancing

1. **Fix an imprecise claim.** The introduction now says
   Appendix Figure~\ref{fig:residual-example} "shows a real retrieval-selected
   ETTh1 case," but that figure's own caption says the case was "screened solely
   for visualization" and "does not enter method selection." "Retrieval-selected"
   implies the validation selector chose retrieval for that cell. Correct the
   intro wording to match the appendix (e.g. "a real ETTh1 case illustrating
   residual retrieval," not "retrieval-selected"), or confirm and state that the
   cell's selected family was indeed retrieval.

2. **Foreground your strongest retrieval evidence.** The additional-systems
   study is where the retrieval thesis is genuinely supported: retrieval is
   selected in 70/94 cells and delivers 81/94 strict wins, unlike the broad
   matrix where temporal families dominate. After four rounds of deflating the
   retrieval story, do not now *undersell* it — explicitly note that the
   additional-systems result is the cleanest evidence that model-specific
   residual memory is reused, and that this is consistent with (not contradicted
   by) the broad matrix where simpler corrections often suffice.

3. **Reproducibility close-out.** Several new diagnostics (A100↔A10G discrepancy
   distribution, selected-hyperparameter counts, per-dataset family selection,
   non-strict risk map) were added from `hardware_comparison.json` and the
   per-cell composed summaries. Ensure the reproducibility statement or an
   appendix names the committed artifacts these are computed from, so a reader
   can reproduce every added number.

---

Summary: This final round is about a coherent, honest, and still-confident
paper. Keep the integrity you have gained, unify the framing around one thesis,
and let the additional-systems result carry the retrieval claim it legitimately
supports. Where the earlier rounds said "concede," this one says: state what you
have actually earned.
