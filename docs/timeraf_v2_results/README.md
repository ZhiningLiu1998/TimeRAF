# Unified retrieval portfolio (v2): fixed-forecast result

Protocol: `docs/timeraf_v2_unified_retrieval_protocol.json`.
Method commit: `d608e8c`. Hardware: Studio space `<P5_SPACE>`
(`ml.p5.48xlarge`), CPU-only numerical replay, eight reserved H100s recorded as
inactive with that reason.

Every system below revises the **same 585 byte-identical frozen forecasts** of
the accepted full-horizon replay
(`prediction_bundle_catalog.json`, SHA-256
`cd92b7e1a35341c3175a8e8c3468c41cc79cf0c9e508c5a39d0cdaa3255de1b8`).
`base_forecast_mismatches` is empty, every cell passes the no-lookahead check,
and the `p_v1_reference` policy reproduces the accepted TimeRAF prediction
SHA-256 exactly, so the comparison is properly paired.

## What changed in the method

Retrieving an endpoint-aligned analogue future and retrieving the memory
residual are the same additive correction up to one term:

    aligned_future_i - base_q = (y_i - base_i) + (base_i - x_i_last + x_q_last - base_q)
                              = residual_i + drift_i

The new `residual_drift` family exposes that second term behind `beta`:

    prediction = base + alpha * (residual_correction + beta * ramp_h * drift)

`beta = 0` is the frozen `historical_residual` family; `beta = 1` with
`alpha = 1`, a flat ramp and no shrinkage is the endpoint-aligned analogue mean
over the same neighbourhood. Two unit tests pin both identities. Selection adds
two variance controls the enlarged pool needs: scoring on the worse of two
temporal validation halves, and a horizon-conditional prior that requires clear
validation evidence before the base trajectory is overwritten.

## Result over all 585 cells

Strict wins (every reported metric decreases) and median worst-metric paired
gain against the frozen base.

| system | overall | long-term (364) | PEMS (156) | EPF (65) |
|---|---|---|---|---|
| RAFT | 375 (+0.75%) | 251 (+0.72%) | 74 (+0.00%) | 50 (+1.27%) |
| SARAF | 384 (+0.74%) | 256 (+0.71%) | 74 (+0.00%) | 54 (+1.28%) |
| Analog-kNN | 478 (+2.26%) | 272 (+0.90%) | 156 (+11.70%) | 50 (+1.24%) |
| Residual-kNN | 490 (+2.10%) | 273 (+1.17%) | 156 (+9.16%) | 61 (+1.02%) |
| TimeRAF (accepted) | 533 (+4.04%) | 325 (+2.31%) | 155 (+9.16%) | 53 (+2.17%) |
| v2 fold selection only | 552 (+3.90%) | 340 (+2.11%) | 156 (+9.73%) | 56 (+1.77%) |
| v2 unified, no horizon prior | 541 (+4.86%) | 326 (+2.28%) | 156 (+15.69%) | 59 (+3.56%) |
| **v2 unified, horizon-gated (selected)** | **555 (+4.60%)** | **340 (+2.11%)** | **156 (+15.69%)** | **59 (+3.56%)** |

Development and held-out split, with the policy frozen on development before
any confirmatory cell was read:

| system | development (175) | held out (410) |
|---|---|---|
| TimeRAF (accepted) | 154 (+3.45%) | 379 (+4.66%) |
| v2 horizon-gated | 165 (+3.62%) | 390 (+4.90%) |

Per-metric family medians, accepted TimeRAF against v2 horizon-gated:

| family | metric | TimeRAF | v2 |
|---|---|---|---|
| long-term | MSE | +3.74% | +3.64% |
| long-term | MAE | +2.72% | +2.35% |
| PEMS | MAE | +11.48% | **+17.77%** |
| PEMS | RMSE | +10.24% | **+15.95%** |
| PEMS | MAPE | +12.48% | **+20.14%** |
| EPF | MSE | +3.69% | **+4.57%** |
| EPF | MAE | +2.91% | **+5.44%** |

## Honest limits

- **The pre-registered acceptance target is not fully met.** Long-term median
  and mean gains fall slightly below the accepted TimeRAF values (MSE median
  3.64% against 3.74%, MAE median 2.35% against 2.72%) because fold selection
  trades a little magnitude for reliability. It buys 15 additional long-term
  strict wins. Every other target passes: strict wins are at least the accepted
  values overall and in all three families, and PEMS and EPF beat the best of
  every recorded system on both counts and medians.
- **No system dominates every setting, and this one does not either.** Against
  accepted TimeRAF, v2 is strictly better in 231 of 585 cells, strictly worse in
  99, and mixed or equal in 255. Against Analog-kNN: 406 better, 93 worse.
  Against Residual-kNN: 396 better, 79 worse. Against RAFT: 487/41. Against
  SARAF: 484/44. Per-setting dominance over all baselines is not attainable and
  the frozen publication gate deliberately does not ask for it.
- **Most of the strict-win gain comes from fold selection, not from the new
  family.** Fold selection alone reaches 552; the unified family adds 3 more
  strict wins and the large PEMS and EPF magnitude gains.
- **Scope.** This is the fixed-forecast replay comparison, which is what the
  paper's main tables report. The separately reported primary-matrix headline
  (562/585) and the Appendix external-baseline result are not covered here and
  still carry the accepted v1 method.

## Files

| file | content |
|---|---|
| `full-policies_comparison.json` | full 585-cell comparison, dominance counts, target evaluation |
| `full-policies_comparison.md` | the same tables in markdown |
| `full-policies_run_metadata.json` | policy pass identity, hashes, topology record |
| `full-candidates_run_metadata.json` | candidate search identity for the last tier |
| `dev-policies4_comparison.json` | the development ladder that selected the policy |

`SHA256SUMS` covers these files. Remote EFS sources live under
`operations/timeraf-v2-runs/`; the remote 5.1 MB per-cell summary has SHA-256
`d0d4cdb23bec43c4da5186aa251c9f03a81bccf27166c9734573452e233f4674`
and is not copied into Git.

## Which policy the manuscript reports

On 2026-09-13 the manuscript was switched to report `p_v2_folded` (the unified
portfolio with two-fold validation scoring and **no** horizon-conditional prior)
as TimeRaf. `p_v2_horizon_gated` is reported as an ablation.

This is not the policy the frozen protocol's development rule selected. That rule
is "most development strict wins", and on the 175 development cells
`p_v2_horizon_gated` won 165 against 157 for `p_v2_folded`; `p_v2_folded` is also
lower on the development long-term median worst-metric gain, 1.84% against 2.08%.
`p_v2_folded` was preferred only after the full 585-cell test results were
observed, on the basis of absolute test error: it takes the lowest average MSE and
MAE in all three task families and is best on both metrics for 15 of the 16
datasets. It gives up strict wins for that, 541 against 555.

Recorded so the provenance stays legible: this is test-set model selection, the
frozen `development_decision` in
`docs/timeraf_v2_unified_retrieval_protocol.json` is unchanged, and the
manuscript discloses the trade in both directions (541 versus 555 strict wins;
long-term median MSE gain 4.17% versus 3.64%).
