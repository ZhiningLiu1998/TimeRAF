# Adversarial Rebuttal Loop — Round 4 (Reviewer)

The method-internal disclosures in Round 3 are strong. Round 4 addresses
**presentation honesty of the motivation, unquantified claims, and the failure
profile** — the gap between how the paper opens and what it actually shows.

## R4.1 — Figure 1 is a synthetic cartoon that recovers truth "by construction"

The entire opening argument rests on the stadium-traffic motivation figure,
whose caption states that "adding only that residual preserves the current base
and recovers the depicted truth **by construction**." This is circular: the
figure is hand-built so the mechanism looks perfect, then used to motivate the
mechanism. After Rounds 1–3 we have established that (a) retrieval is a
*minority* contributor to the headline, and (b) on the largest-channel dataset
(Traffic) retrieval is selected in only 8/52 cells. A dramatic synthetic figure
that showcases residual retrieval therefore over-sells exactly the component the
evidence credits least.

Two acceptable fixes:
1. Replace or supplement it with a **real-data instance** — an actual
   retrieval-selected cell where a recurring model error was corrected, plotted
   from real predictions — so the reader sees the mechanism on real data, not a
   constructed ideal.
2. If the prediction bundles needed to plot a real example are not available
   here, then clearly **label the figure as an illustrative schematic**, remove
   or heavily qualify "by construction," and add a sentence noting that the
   figure depicts the *hypothesized* mechanism, whose empirical contribution is
   quantified (and found to be a minority of selections) in Section 5.

Do not leave a by-construction cartoon as the paper's primary intuition pump
without this caveat.

## R4.2 — "Efficiency" is claimed but never measured

The introduction lists "**Interpretability and efficiency**: every revision …
updates no model parameter, and reuses validation forecasts." Efficiency is
asserted with zero measurement. There is no number for: the cost of building
the error memory, retrieval latency at a new origin, or the storage footprint
of the memory — which for high-dimensional datasets (Traffic ≈862 channels over
the full validation timeline) is precisely where cost could bite, even with the
≤16-group pooling.

Either (a) quantify the revision layer's added cost and memory footprint (note
that the committed `elapsed_seconds` is total cell time including base-model
training, so it does **not** isolate the layer's overhead — do not misuse it),
or (b) drop the efficiency claim to what you can support: the layer trains no
parameters and reuses forecasts, which is a *qualitative* efficiency argument,
not a measured one. State explicitly that wall-clock/memory overhead of the
revision layer is not measured here.

## R4.3 — No a-priori failure rule; the "recurrence" story is post-hoc

The conclusion's thesis — "error is useful memory only when the blind spot
recurs" — is diagnosed *after* seeing test results. A practitioner cannot use
it: nothing tells them, before deployment, when the layer will hurt. The
failures are also not random. From the committed data:
- The **largest-magnitude** failures are concentrated: the three
  PEMS04–TimeMixer horizons under overlap revision, including a 36.5% MAPE
  increase.
- TimeMixer is the single most failure-prone backbone (7 of 23 non-strict
  cells), and DE/Weather (volatile or low-signal regimes) account for 14 of 23.

Please add a **bounded risk characterization**: state which (backbone, dataset,
family) combinations carry elevated regression risk, so the "broad
applicability" claim is qualified by a stated failure profile rather than a
post-hoc slogan. If no *validation-time* signal predicting these regressions is
retained in the artifacts (I believe candidate-level validation losses are not
retained), concede that a predictive early-warning rule is not yet available and
mark it as future work — but still give the practitioner the observed risk map.

---

Summary: Round 4 closes the gap between the paper's confident opening/claims and
its actual, now-honest evidence. Align the motivation figure, the efficiency
claim, and the failure discussion with what the data supports.
