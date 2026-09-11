# Adversarial Rebuttal Loop — Round 3 (Reviewer)

Rounds 1–2 substantially improved honesty of attribution and statistical
framing. Round 3 turns to the **method's own internal justification**: does each
mechanism earn its place, and does the reliability/abstention story hold up?

## R3.1 — No ablation of the method's components

The method has many hand-designed moving parts: the fixed context–forecast
descriptor (normalized shape bins, first differences, endpoint/summary stats,
linear slope, base-forecast shape stats, calendar marks), deterministic channel
pooling to ≤16 groups, and the reliability-shrinkage machinery (top-$k$,
temperature $\tau$, shrinkage $\lambda$, scale $\alpha$, $\epsilon$). **Not one
of these is ablated.** A method paper is expected to show which components
matter — e.g. descriptor-with/without-shrinkage, retrieval with $\alpha$ swept,
$k$ sensitivity. As written, a reader cannot tell whether the shrinkage factor
$\boldsymbol{\rho}_t$ ever meaningfully departs from 1, whether $k$ matters, or
whether the elaborate descriptor beats a trivial one.

At minimum, report the **distribution of selected hyperparameters** ($\alpha$,
$k$, $\tau$, $\lambda$) across the 214 retrieval-selected cells — if $\alpha$
clusters near 1 and $\rho_t$ near 1, the shrinkage is largely inert and should
be presented as such; if they vary widely, that is evidence the reliability
machinery is active. If the committed artifacts do not contain per-cell
selected hyperparameters (I suspect they do not), concede this explicitly,
report what you *can* (the family distribution you already have), and register
the full ablation as a required experiment in Limitations — do not leave the
impression the components are validated.

## R3.2 — The abstention/reliability claim is contradicted by the selections

The intro sells "Reliability: … validation can retain the base," and lists
**Identity** as "explicit abstention." Yet identity and static bias are selected
in **0 of 585 cells**. The advertised safety valve is never used. Two issues:

1. As written, the reliability/abstention capability is *aspirational, not
   demonstrated*. The paper must not imply the mechanism protected any cell when
   it never fired.
2. More concerning: the selector **always acts**, even on the hardest cells. Of
   the 23 non-strict cells, 7 were assigned residual retrieval, 11 overlap, 5
   seasonal — i.e. on these the validation-argmin chose to correct and then
   *lost* on at least one test metric. That is the signature of
   **validation→test selection overfitting**: validation predicted a gain that
   did not transfer, and no abstention caught it. Please characterize the
   regression cells (which families, how large the test-side degradation, which
   datasets), and confront directly whether an argmin that never abstains is
   mis-calibrated on hard regimes. If the committed data lacks the
   validation-side scores needed to show the val↔test gap quantitatively,
   concede that and still report the regression-cell characterization you can
   compute.

## R3.3 — The hand-crafted descriptor's fidelity is unvalidated and confounded by pooling

High-dimensional datasets (traffic ≈862 channels, electricity ≈321) are
deterministically pooled to ≤16 channel groups before retrieval — a heavy,
lossy compression of exactly the datasets where the descriptor should matter
most. The paper never checks whether retrieval on a ≤16-group summary is even
meaningful there.

This is testable with data you already have: report, per dataset, which
correction family was selected. My read of the selection counts is that
**traffic leans on overlap+seasonal (retrieval a minority) while electricity
leans on residual retrieval** — despite both being heavily pooled. If so, that
(a) further tempers the retrieval narrative (traffic's strong wins are largely
temporal, not retrieval), and (b) shows the pooled descriptor's role is
inconsistent across the very datasets that stress it. Report this breakdown and
interpret it honestly; do not let the aggregate hide that the flagship
mechanism is a minority contributor on the largest-channel dataset.

---

Summary: Round 3 asks the method to justify itself internally. Where the
committed artifacts cannot support an ablation, concede and register it as
required future work rather than implying the components are validated. Where
they can (family-by-dataset, regression-cell characterization), report it and
follow the evidence even when it further deflates the retrieval story.
