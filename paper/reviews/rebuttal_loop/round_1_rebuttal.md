# Adversarial Rebuttal Loop — Round 1 (Author)

We agree that the original framing attributed too much of the paper's evidence
to residual retrieval. We verified the committed publication snapshot before
revising the manuscript. No new experiment was run, and we do not infer
unobserved candidate performance from validation selections.

## R1.1 — ACCEPT

The reviewer is correct that the 562/585 headline supports the complete
revision portfolio, not residual retrieval alone. Residual retrieval is the
largest individual selected family (214 cells), but non-retrieval policies are
selected in 371 cells.

We added a selection-stratified analysis computed from the committed A100
per-cell records. It shows 207 strict wins among the 214 retrieval-selected
cells and 355 strict wins among the 371 non-retrieval-selected cells (seasonal
151/156, overlap 148/159, and causal rolling bias 56/56). This directly
attributes the observed selected-policy outcomes and confirms that most strict
wins occur under non-retrieval corrections.

We cannot report the requested portfolio-minus-retrieval ablation from the
committed result snapshot. The per-cell records contain test metrics only for
the selected candidate; they do not contain test predictions or metrics for
the next-ranked non-retrieval candidate in the 214 retrieval-selected cells.
Validation selection data cannot determine those counterfactual test outcomes.
We therefore do not label the new analysis as an ablation and explicitly state
that the exact result requires a new evaluation.

Manuscript changes:

- `paper/latex/main.tex` (title and Abstract): changed the title to
  "TimeRaf: Validation-Guided Revision of Frozen Forecasts"; recast TimeRaf as
  a revision portfolio; and disclosed the 214 retrieval versus 371
  non-retrieval selections.
- `paper/latex/sections/01_introduction.tex` (Introduction): changed the
  central question and contributions from retrieval-first claims to
  validation-guided revision with residual retrieval as one candidate.
- `paper/latex/sections/05_results.tex` (Overall accuracy and selected
  policies): added selected-family strict-win attribution and explicitly
  distinguished it from a portfolio-minus-retrieval ablation.
- `paper/latex/sections/appendix_sections/07_aggregate.tex` (Aggregate
  results): added the full selected-family attribution table.
- `paper/latex/sections/06_limitations_conclusion.tex` (Limitations and
  Conclusion): made the missing ablation explicit, identified it as future
  work, and revised the conclusion to attribute the broad result to the
  portfolio.
- `paper/OUTLINE.md`: aligned the working title, central question, argument,
  and section plan with the revised scope.

## R1.2 — ACCEPT

The reviewer is correct: the experiments compare each selected correction
with its unchanged frozen base, not with an analog-future retriever or a
retrieval-augmented forecasting method. The original text used the two
analog-retrieval mismatches too strongly given that evidence.

We did not run a new analog-future or recent-RAG baseline in this round. Such a
comparison requires producing candidate predictions under the same splits and
selection protocol; no corresponding outputs exist in the committed
publication artifacts. We now present the analog-future discussion as a
motivating hypothesis and do not claim empirical superiority for residual
retrieval.

Manuscript changes:

- `paper/latex/main.tex` (Abstract): explicitly states that the paper does not
  establish superiority over analog-future or retrieval-augmented baselines.
- `paper/latex/sections/01_introduction.tex` (Introduction): changes the two
  "mismatches" to potential failure modes and states that they are not
  empirically ranked here.
- `paper/latex/sections/02_related_work.tex` (Retrieval-augmented
  forecasting): states that the taxonomy is conceptual and that no
  head-to-head comparison is reported.
- `paper/latex/sections/03_method.tex` (Residual versus analog-future
  retrieval): replaces the rhetorical "why not" heading with a descriptive
  comparison.
- `paper/latex/sections/04_experimental_design.tex` (Correction candidates):
  identifies the unchanged frozen forecast as the experimental comparator.
- `paper/latex/sections/appendix_sections/03_search.tex` (Exact broad search
  space): changes the retrieval-example caption from a superiority question
  to a descriptive statement about the components transferred.
- `paper/latex/sections/06_limitations_conclusion.tex` (Limitations): adds a
  controlled analog-future comparison under the same selector as explicit
  future work.

## R1.3 — ACCEPT

We accept that the original paper did not foreground the selector differences
enough. The three studies share an additive-revision interface, but they do not
test one universal selection rule. The broad matrix uses a preregistered
conservative-margin family policy, the additional-system study uses direct
global validation argmin, and the ETT study uses a frequency-specific
prequential policy. Their purposes differ, but that does not establish that
the distinct selectors are necessary or interchangeable.

We did not apply global argmin uniformly in this round and therefore cannot
quantify the cost of unification. Reporting such a number without re-evaluating
the candidates would be unsupported. We have narrowed the claim to a shared
revision interface and made selector unification an open empirical question.

Manuscript changes:

- `paper/latex/main.tex` (Abstract and Reproducibility statement): says that
  the later studies use study-specific selection protocols and replaces the
  singular protocol claim with study-specific protocols.
- `paper/latex/sections/01_introduction.tex` (Introduction): names all three
  protocols before the contribution list and removes the implication of one
  selector.
- `paper/latex/sections/04_experimental_design.tex` (Protocol scope): adds an
  up-front statement of the three selectors, their study roles, and the
  unmeasured cost of unification.
- `paper/latex/sections/06_limitations_conclusion.tex` (Limitations): limits
  the evidence to the shared revision interface, not selector invariance, and
  identifies uniform global-argmin evaluation as missing.
- `paper/OUTLINE.md`: records the three protocols and selector scope in the
  paper narrative.
