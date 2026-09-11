# Claude Review Round 3

- Model: `claude-fable-5`
- Session: `98c13fcc-28f3-43c5-89a7-7705e101e6f7`
- Scope: statistical interpretation and evidence tiers

## Findings and decisions

1. **Bootstrap intervals may treat the 585 benchmark cells as independent.**
   Rejected. The broad matrix has no bootstrap interval. The only bootstrap is
   the ETT prequential analysis: it first averages aligned per-origin losses
   across the three model seeds and then resamples chronological blocks of
   length 96. Appendix `app:ett` already states that the interval measures
   temporal uncertainty of the seed-averaged predictor, does not estimate the
   training-seed distribution, and does not treat overlapping origins or
   seeds as independent observations.
2. **The later ETT timeline should not be called independent confirmation.**
   Rejected because the manuscript does not make that claim. Experimental
   design calls it a separate temporal evaluation, and the results explicitly
   identify official-test diagnostics as development evidence. The conclusion
   is limited to forward temporal survival of post-hoc revision, not
   independent confirmation or residual retrieval in isolation.

## Verification

No manuscript change was warranted. The previously built PDF remains 22
pages, with no overfull boxes, undefined references, or undefined citations.
