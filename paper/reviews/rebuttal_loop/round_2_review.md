# Adversarial Rebuttal Loop — Round 2 (Reviewer)

The Round-1 revisions are accepted: the re-scoped title, the honest abstract
disclosure of 214 vs. 371 selections, and the attribution table materially
improve the paper's integrity. Round 2 now stresses the **statistical validity
and reproducibility** of the quantitative claims that remain.

## R2.1 — The headline has no uncertainty quantification

The primary 585-cell matrix is run at a **single seed (2021)**. The paper
reports point estimates (96.1% strict wins, 8.48% mean MSE reduction, etc.) with
**no confidence intervals, no significance test, and no seed variance** for the
main result. Only the small ETT study has a bootstrap CI.

Two consequences:
1. For any individual cell, a "strict win" (all metrics decrease) at one seed
   can be within run-to-run noise. Some of the 562 wins — especially the
   marginal ones (e.g. DE at 1.3% MSE, the near-boundary Weather cells) — are
   plausibly seed artifacts. The paper cannot currently distinguish a real
   improvement from a lucky seed at the cell level.
2. The aggregate 8.48% mean reduction is presented as if precise, but its
   sampling variability across seeds is unknown.

I am not asking you to rerun 585 cells at multiple seeds if that is infeasible.
But you must (a) state clearly that the headline is single-seed and therefore
carries unquantified seed variance, (b) avoid over-precise language ("8.48%")
that implies more certainty than a single seed supports, and (c) if any
subset already has multi-seed data (the ETT study does), use it to *bound* the
typical per-cell seed noise so the reader can judge which margins are safe.
Where the improvement margin is smaller than plausible seed noise, say so.

## R2.2 — A100 vs. A10G numerical inconsistency is unresolved for the reader

AGENTS.md and the hardware comparison record that the **strict A10G/A100
numerical consistency gate is false** — the two hardware runs do not match
numerically. The paper designates A100 as authoritative and keeps A10G as
"replication evidence," but:

- If the two runs disagree beyond a strict tolerance, in what sense is either
  "reproducible"? A reader on different hardware will get yet another set of
  numbers. What is the *magnitude* of the A100↔A10G disagreement (per-cell,
  distributional), and does it ever flip a strict win to a loss?
- More seriously, the "frozen policy confirmation" (437/462) is computed on the
  **A10G** run, while the headline 562/585 is **A100**. The paper explicitly
  says the 462 is "not a subset of the 562." So the two central numbers live on
  two numerically-inconsistent hardware runs. This coupling should be stated
  plainly, and you should report how many confirmation-set conclusions would
  change if evaluated on A100 (or concede you cannot).

**Action:** Quantify the A100↔A10G disagreement from the committed
hardware_comparison artifact, report whether it ever changes a win/loss
verdict, and make the cross-hardware coupling of the two headline numbers
explicit rather than a footnote.

## R2.3 — "Confirmation" is internal, and the dev/holdout split may be easy/hard-confounded

You already concede the 462-cell confirmation shares model and dataset families
with development, so it is internal not external validation. Push this further:

1. The dev/confirmation boundary is defined by `started_unix` order (first 123
   cells = development). Wall-clock start order is arbitrary with respect to
   cell *difficulty* — there is no guarantee the 462 holdout is not
   systematically easier or harder than the 123 dev cells. A cleaner protocol
   would hold out by dataset or model family. Please report whether the dev and
   holdout partitions are balanced in difficulty (e.g. baseline error
   distribution, task-family composition), or acknowledge the boundary is
   opportunistic.
2. Because the selector was specified on cells drawn from the *same 15 datasets
   and 13 models*, the confirmation tests only that a frozen rule generalizes to
   new (model, dataset, horizon) combinations within the same families — not to
   new datasets or models. The paper should not let "437/462 held-out" read as
   evidence of external generalization. State the exact scope of what is being
   confirmed.

**Action:** Report difficulty balance across the dev/holdout partition (or
concede the boundary is opportunistic), and precisely bound what "confirmation"
demonstrates versus what it does not.

---

Summary: Round 2 is about not letting single-seed point estimates,
hardware-inconsistent headline numbers, and an internal split masquerade as
strong statistical evidence. Where you cannot add experiments, add honesty and
bounds.
