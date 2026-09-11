# Adversarial Rebuttal Loop — Round 1 (Reviewer)

Role: Reviewer (NeurIPS/ICML-style, borderline reject leaning). I have read the
full main text (intro, related work, method, experiments, results, conclusion)
and the OUTLINE. This round targets the three most load-bearing structural
concerns. Rebut or revise; do not fabricate numbers.

## R1.1 — Framing vs. evidence mismatch (most serious)

The paper's stated novelty and title ("Retrieve What the Forecaster Missed")
center on **residual retrieval** from a chronological error memory. Yet the
headline result is that **562/585 (96.1%) cells improve**, and by your own
Section 5 the correction family actually *selected* is:

- residual retrieval: 214 cells
- seasonal: 156, overlap: 159, causal bias: 56
- identity / static bias: 0

So on the flagship benchmark, the flagship mechanism is chosen in **~37%** of
cells, and the majority of the reported 96.1% "improvement" is delivered by
deliberately simple temporal baselines (seasonal + overlap = 315 cells) that
have nothing to do with retrieval. A reader could reproduce most of the
headline gain with a seasonal-naive + overlap-averaging portfolio and **no
retrieval at all**.

This is a serious framing problem. The contribution as marketed (model-relative
residual retrieval) is not the contribution the main experiment actually
demonstrates. Either (a) the central claim must be re-scoped to "a
validation-guided revision portfolio, of which residual retrieval is one
member," which substantially deflates novelty, or (b) you must isolate the
*marginal* value of residual retrieval — e.g., report the headline with the
retrieval candidate removed from the portfolio, so we can see how much of the
96.1% survives. Right now neither is done, and the abstract/title over-attribute
the result to retrieval.

**Action:** Report a portfolio-minus-retrieval ablation (computable from the
per-cell selection data you already have), and align the title/abstract/intro
claims with what the majority-selected families actually contribute.

## R1.2 — No empirical comparison to any retrieval-augmented forecasting baseline

Related Work positions TimeRAF against RAFT, Cross-RAG, CRAFT, SARAF, TS-RAG,
RATD, TimeRAG, etc. But **every experiment compares corrected-vs-uncorrected
base forecasts only**. There is not a single head-to-head against a competing
retrieval-augmented method. For a paper whose identity is "a better way to do
retrieval for forecasting," this is a critical missing baseline: you show you
beat the *un-retrieved* base, never that your residual formulation beats
*analog-future* retrieval, which is precisely the strawman the intro attacks
(the "two mismatches" argument).

The whole motivating premise — that analog-future retrieval entangles old
level/phase/trend and gets dominated by common structure — is asserted and
illustrated with a synthetic figure, but never demonstrated empirically against
an actual analog-future retriever on your benchmark.

**Action:** At minimum, add a controlled comparison against a naive
analog-future retriever (retrieve top-k similar windows, copy their futures /
blend them) as an additional candidate, on the same cells, so the residual
formulation's advantage over analog-future retrieval is measured rather than
argued. If infeasible to run at full scale, run it on a defensible subset and
report it as such — do not leave the core claim unbenchmarked.

## R1.3 — "One framework" claim undermined by three different selectors

Section 4 concedes the broad matrix, the additional-systems study, and the ETT
prequential study each use a **different** selection protocol (preregistered
conservative-margin family order; global validation-argmin; and a bespoke ETT
"causal stack" with rolling ridge weights). The ETT study in particular
hand-builds a frequency-specific corrector (linear + overlap + daily profile)
that is not the residual retriever at all.

This reads as per-experiment tuning. A skeptical reviewer concludes the
"framework" is really a family of hand-selected pipelines, one per experiment,
which threatens both the generality claim and reproducibility. Why three
selectors? What breaks if the *single* global-argmin selector (used for the
additional systems) is applied uniformly to all three studies? If it
underperforms, that is important negative evidence; if it works, the paper is
much stronger with one selector.

**Action:** Justify each selector's necessity with evidence, or unify. State
explicitly, up front (not buried in Section 4), that these are distinct
protocols and quantify the cost of unifying them.

---

Summary for the author: Round 1 is about honesty of attribution and complete
baselines. The single biggest risk to acceptance is that the title promises
retrieval while the evidence credits simple temporal heuristics. Address R1.1
head-on.
