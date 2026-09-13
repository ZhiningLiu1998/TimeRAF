# Fixed-forecast comparison over 585 cells

Strict wins and median worst-metric paired gain against the frozen base.

| system | overall | long-term | PEMS | EPF | confirmatory split |
|---|---|---|---|---|---|
| analog_future | 478/585 (+2.26%) | 272/364 (+0.90%) | 156/156 (+11.70%) | 50/65 (+1.24%) | 350/410 (+3.03%) |
| raft_adapted | 375/585 (+0.75%) | 251/364 (+0.72%) | 74/156 (+0.00%) | 50/65 (+1.27%) | 273/410 (+0.98%) |
| saraf_adapted | 384/585 (+0.74%) | 256/364 (+0.71%) | 74/156 (+0.00%) | 54/65 (+1.28%) | 273/410 (+0.94%) |
| residual_retrieval | 490/585 (+2.10%) | 273/364 (+1.17%) | 156/156 (+9.16%) | 61/65 (+1.02%) | 355/410 (+2.39%) |
| timeraf | 533/585 (+4.04%) | 325/364 (+2.31%) | 155/156 (+9.16%) | 53/65 (+2.17%) | 379/410 (+4.66%) |
| v2:p_v1_folded | 552/585 (+3.90%) | 340/364 (+2.11%) | 156/156 (+9.73%) | 56/65 (+1.77%) | 390/410 (+4.33%) |
| v2:p_v1_reference | 533/585 (+4.04%) | 325/364 (+2.31%) | 155/156 (+9.16%) | 53/65 (+2.17%) | 379/410 (+4.66%) |
| v2:p_v2_folded | 541/585 (+4.86%) | 326/364 (+2.28%) | 156/156 (+15.69%) | 59/65 (+3.56%) | 384/410 (+5.28%) |
| v2:p_v2_horizon_gated | 555/585 (+4.60%) | 340/364 (+2.11%) | 156/156 (+15.69%) | 59/65 (+3.56%) | 390/410 (+4.90%) |

## Per-metric family medians

### long_term

| system | mae | mse |
|---|---|---|
| analog_future | +0.90% | +2.27% |
| raft_adapted | +0.73% | +1.75% |
| saraf_adapted | +0.73% | +1.95% |
| residual_retrieval | +1.20% | +2.41% |
| timeraf | +2.72% | +3.74% |
| v2:p_v1_folded | +2.35% | +3.64% |
| v2:p_v1_reference | +2.72% | +3.74% |
| v2:p_v2_folded | +2.62% | +4.17% |
| v2:p_v2_horizon_gated | +2.35% | +3.64% |

### pems

| system | mae | mape | rmse |
|---|---|---|---|
| analog_future | +13.44% | +15.23% | +12.25% |
| raft_adapted | +0.00% | +0.36% | +0.00% |
| saraf_adapted | +0.00% | +0.31% | +0.00% |
| residual_retrieval | +11.53% | +11.58% | +10.13% |
| timeraf | +11.48% | +12.48% | +10.24% |
| v2:p_v1_folded | +11.53% | +12.80% | +10.34% |
| v2:p_v1_reference | +11.48% | +12.48% | +10.24% |
| v2:p_v2_folded | +17.77% | +20.14% | +15.95% |
| v2:p_v2_horizon_gated | +17.77% | +20.14% | +15.95% |

### epf

| system | mae | mse |
|---|---|---|
| analog_future | +2.39% | +1.43% |
| raft_adapted | +1.44% | +1.64% |
| saraf_adapted | +1.81% | +1.95% |
| residual_retrieval | +1.48% | +1.95% |
| timeraf | +2.91% | +3.69% |
| v2:p_v1_folded | +2.42% | +3.01% |
| v2:p_v1_reference | +2.91% | +3.69% |
| v2:p_v2_folded | +5.44% | +4.57% |
| v2:p_v2_horizon_gated | +5.44% | +4.57% |

## Per-setting dominance of the selected policy

### v2:p_v1_folded

| against | strictly better | strictly worse | mixed or equal |
|---|---|---|---|
| analog_future | 291 | 198 | 96 |
| raft_adapted | 435 | 77 | 73 |
| saraf_adapted | 433 | 81 | 71 |
| residual_retrieval | 244 | 115 | 226 |
| timeraf | 58 | 126 | 401 |
| v2:p_v1_reference | 55 | 112 | 418 |
| v2:p_v2_folded | 71 | 317 | 197 |
| v2:p_v2_horizon_gated | 7 | 199 | 379 |

### v2:p_v1_reference

| against | strictly better | strictly worse | mixed or equal |
|---|---|---|---|
| analog_future | 299 | 176 | 110 |
| raft_adapted | 434 | 71 | 80 |
| saraf_adapted | 427 | 71 | 87 |
| residual_retrieval | 226 | 69 | 290 |
| timeraf | 6 | 22 | 557 |
| v2:p_v1_folded | 112 | 55 | 418 |
| v2:p_v2_folded | 112 | 298 | 175 |
| v2:p_v2_horizon_gated | 89 | 230 | 266 |

### v2:p_v2_folded

| against | strictly better | strictly worse | mixed or equal |
|---|---|---|---|
| analog_future | 453 | 54 | 78 |
| raft_adapted | 503 | 29 | 53 |
| saraf_adapted | 502 | 29 | 54 |
| residual_retrieval | 457 | 63 | 65 |
| timeraf | 298 | 112 | 175 |
| v2:p_v1_folded | 317 | 71 | 197 |
| v2:p_v1_reference | 298 | 112 | 175 |
| v2:p_v2_horizon_gated | 118 | 64 | 403 |

### v2:p_v2_horizon_gated

| against | strictly better | strictly worse | mixed or equal |
|---|---|---|---|
| analog_future | 406 | 93 | 86 |
| raft_adapted | 487 | 41 | 57 |
| saraf_adapted | 484 | 44 | 57 |
| residual_retrieval | 396 | 79 | 110 |
| timeraf | 231 | 99 | 255 |
| v2:p_v1_folded | 199 | 7 | 379 |
| v2:p_v1_reference | 230 | 89 | 266 |
| v2:p_v2_folded | 64 | 118 | 403 |
