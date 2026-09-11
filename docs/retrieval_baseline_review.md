# Retrieval Baseline Review

Date: 2026-08-31

## Decision

The paper reports five external retrieval-augmented forecasting baselines:
RAFT, SARAF, RAF, TS-RAG, and RATD. They appear in two comparisons because no
single protocol can run all five methods on the same 13 base forecasts without
changing the published systems.

The original fixed-forecast comparison reused the 208-cell Appendix replay
subset: all 13 frozen backbones on all 16 datasets, but only horizon 96 for
long-term forecasting and horizon 24 for PEMS and EPF. That reuse, rather than
a methodological reason, caused the missing horizons. It cannot support the
cross-horizon averages requested for the main tables.

The replacement comparison covers the same 585 cells as the primary matrix:
long-term horizons 96, 192, 336, and 720; PEMS horizons 6, 12, and 24; and EPF
horizon 24. Its hash-verified prediction bundles replay the accepted A10G
checkpoints on P5 H100 hardware. Their
recomputed base metrics are the authoritative fixed forecasts for this
comparison. They are not required to equal the original A10G metrics: the
project's accepted hardware audit already establishes that replay values can
change across hardware. Every difference from the A10G snapshot is
recorded as provenance. These are also not the primary full-matrix A100
forecasts. The paper will label this comparison as checkpoint replay and will
not merge it with the primary A100 table.

The fixed-forecast comparison contains four methods:

1. unchanged base forecast;
2. a fixed-forecast reimplementation of the RAFT retrieval operator;
3. a fixed-forecast reimplementation of the SARAF retrieval operator; and
4. TimeRAF.

RAFT and SARAF are the two published retrieval-augmented forecasting
baselines. Their displayed rows are named `RAFT` and `SARAF`; the frozen
protocol retains the internal IDs `raft_adapted` and `saraf_adapted`. The
paper must not present these rows as executions of the original end-to-end
models: the original methods train their own forecaster and fusion layers,
whereas this experiment fixes each of the 13 base forecasts and
reimplements only their retrieval rules.

Base, Analog-kNN, Residual-kNN, and TimeRAF have no external baseline paper.
Base is the unchanged control, Analog-kNN is a controlled analog-future
baseline designed for this study, Residual-kNN is a TimeRAF ablation, and
TimeRAF is the proposed method. Giving any of these rows an unrelated
citation would misstate their provenance.

The native comparison runs RAF, TS-RAG, and RATD against each method's own
released no-retrieval base. Its canonical manifest contains 52 paired
configurations: 44 RAF pairs, seven TS-RAG pairs, and one RATD pair. These
methods keep their released backbones, datasets, horizons, splits, and
metrics, subject only to the compatibility repairs frozen before execution.
Their absolute metrics are not ranked across methods.

## Candidate Audit

| Method | Venue/status | Public code | License | Directly wraps the 13 frozen forecasts? | Decision |
|---|---|---|---|---|---|
| RAFT | ICML 2025 | `archon159/RAFT` at `bfe2320f5d719d1e2a024330de3410b370bbb2a9` | MIT | No. It trains a shallow MLP and learned fusion around multiscale retrieved futures. | Reimplement its retrieval operator under the fixed-forecast protocol. |
| TS-RAG | NeurIPS 2025 | `UConn-DSIS/TS-RAG` at `73ac807789d2e61b8a3dfc8514e3fc947fe185cc` | MIT | No. The released implementation supports Chronos-Bolt and MOMENT and trains an Adaptive Retrieval Mixer. | Run the released Chronos-Bolt path in the native comparison. |
| RAF | arXiv 2024 | `kutaytire/Retrieval-Augmented-Time-Series-Forecasting` at `425e2af35797d0d32b63cfc554c7d97b3c54b390` | Apache-2.0 | No. It augments Chronos input and fine-tuning data. | Run its released benchmark grids in the native comparison. |
| RATD | NeurIPS 2024 | `stanliu96/RATD` at `719e4008f72e89544a621d97b5d1a69164fc14d3` | MIT | No. It is a diffusion forecaster; the released repository describes itself as incomplete and supplies an Electricity experiment. | Run its released Electricity setting in the native comparison after the frozen compatibility repairs. |
| SARAF | KDD 2026 | `ShiqiaoZhou/SARAF` at `d3eee1b9542af29799847c2718ec7fc5ae7e07ea` | No repository license | No. It trains its own linear forecaster and fusion projection. | Reimplement the published retrieval equations under the fixed-forecast protocol without copying source. |
| TimeRAG | ICASSP 2025 | No official repository found from the paper page | N/A | No. It prompts an LLM with retrieved series. | Exclude. |
| TimeRAF foundation model | arXiv 2024 | No official repository linked by the paper | N/A | No. It is tied to a foundation-model prompting/fusion path. | Exclude. |

## Reported-Row Provenance and Implementation

The table below is the authoritative mapping for every reported row. A dash in
the reference column means that the row is defined in this project rather than
claimed as a published external method.

| Paper row | Status in this study | Implementation and frozen evidence | External reference |
|---|---|---|---|
| Base | Unchanged fixed-forecast control | `ts_rag/retrieval_baselines.py::run_retrieval_baseline_cell` | -- |
| Analog-kNN | Controlled analog-future baseline designed for this study | `ts_rag/retrieval_baselines.py::_analog_descriptor`, `_analog_search`, `_analog_apply` | -- |
| RAFT | Fixed-forecast reimplementation of the published retrieval operator; excludes the original forecaster and learned fusion | `ts_rag/retrieval_baselines.py::_raft_descriptor`, `_raft_retrieval`, `_raft_search`, `_raft_apply` | Han et al., ICML 2025; BibTeX key `han2025raft` |
| SARAF | Fixed-forecast reimplementation of the published retrieval operator; excludes the original forecaster and learned projection | `ts_rag/retrieval_baselines.py::_stationarity`, `_saraf_select`, `_saraf_search`, `_saraf_apply` | Zhou et al., KDD 2026; BibTeX key `zhou2026saraf` |
| RAF | Native end-to-end system paired with Chronos | `scripts/run_native_raf_baseline.py`; upstream commit and archive hash in `docs/native_retrieval_baseline_protocol.json`; P5 run identity in `docs/native_retrieval_baseline_p5_execution.json` | Tire et al., arXiv 2024; BibTeX key `tire2024raf` |
| TS-RAG | Native end-to-end system paired with Chronos-Bolt | `scripts/run_native_ts_rag_baseline.py`; upstream commit and archive hash in `docs/native_retrieval_baseline_protocol.json`; checkpoint assets in `docs/native_ts_rag_assets.json` | Ning et al., NeurIPS 2025; BibTeX key `ning2025tsrag` |
| RATD | Native end-to-end system paired with CSDI | `scripts/run_native_ratd_baseline.py` with exact origin-sharded evaluation in `scripts/run_native_ratd_sharded_eval.py`; upstream commit and archive hash in `docs/native_retrieval_baseline_protocol.json`; TCN/data assets in `docs/native_ratd_assets.json` | Liu et al., NeurIPS 2024; BibTeX key `liu2024ratd` |
| Residual-kNN | Retrieval-only ablation of TimeRAF | `ts_rag/retrieval_baselines.py::_residual_search`, `_residual_apply`; index in `ts_rag/historical_retrieval.py::HistoricalResidualIndex` | -- |
| TimeRAF | Proposed full validation-selected revision portfolio | `ts_rag/retrieval_baselines.py::run_retrieval_baseline_cell`; selection in `ts_rag/benchmark_rag.py::run_validation_selected_rag` | -- |

The fixed-forecast rows share `scripts/run_retrieval_baseline_cell.py` as the
cell entry point and `scripts/run_retrieval_baseline_matrix.py` as the
585-cell matrix orchestrator. Their frozen definitions and search spaces are
recorded in `docs/retrieval_baseline_protocol.json`; that hash-bound file uses
the historical internal IDs `raft_adapted` and `saraf_adapted`. The native
rows use the hash-bound 52-pair manifest generated from
`docs/native_retrieval_baseline_protocol.json`.

## Why Published Numbers Are Not Imported

The primary TimeRAF protocol uses input length 96 for long-term and PEMS and
168 for EPF. RAFT and SARAF report their main long-term experiments with input
length 720 and train a method-specific forecaster. TS-RAG and RAF use
foundation-model backbones and different retrieval databases or data splits.
RATD reports a diffusion protocol. These results do not match the frozen
backbones, input lengths, task families, or paired test predictions used here.
Copying those numbers into one table would not be a controlled comparison.

## Reporting Boundary

The paper will report:

- all 585 accepted Base and +TimeRAF absolute results, with no averaging over
  backbone or horizon in the complete tables;
- Base, RAFT, SARAF, and TimeRAF in the 585-cell full-horizon fixed-forecast
  main tables;
- Analog-kNN and Residual-kNN only in the ablation study;
- all 52 native RAF, TS-RAG, and RATD pairs against their own released bases;
  and
- the precise distinction between reimplemented retrieval operators and
  original end-to-end published systems.

Official end-to-end results may be added only in a separate table whose
backbone, split, input length, horizon, and metric are stated explicitly.

## Completed Full-Horizon Execution

The replacement P5 run completed all 585 cells with zero failed, incomplete,
pending, or running cells. Every cell passed finite-metric,
Base-recomputation, exact system-coverage, and no-lookahead checks.

- method/source revision:
  `e97746ed847f25d1f11f47242bc04db2b48b83ef`;
- compatibility-recovery launcher revision:
  `cc80cabd363ac136c5d2e9284529e912b601544e`;
- protocol SHA-256:
  `2d3a9a6c66ae80808c0d734d48996341cb2d1fcb123cf04e81ab8dc1713a0bc6`;
- replay catalog SHA-256:
  `cd92b7e1a35341c3175a8e8c3468c41cc79cf0c9e508c5a39d0cdaa3255de1b8`;
- result summary SHA-256:
  `a9cdf53dc50ad7b3200da49c8e9bc16db8dd48af41b3fed33513033449c44c2a`;
- completion receipt SHA-256:
  `baedd1c56cb0a71adc764a46de68bbb14cf3790f2cf0bb6f56b5299f6daf26ff`.

The first attempt stopped after 432 completed cells when RAFT period 4 did
not divide PEMS horizon 6. Before observing a result for that cell, the
shape-only recovery protocol froze a rule that retains a period only when it
divides both context and prediction length. The completed run preserves the
432 unaffected cells, evaluates 153 cells with the recovery launcher, and
retains the original failure receipt. The rule uses no metric quality.

The recovery startup evidence records eight distinct positive worker PIDs
bound to physical GPU IDs 0--7. Strict wins over the fixed Base predictions
are:

| System | Long-term | PEMS | EPF | Overall |
|---|---:|---:|---:|---:|
| Analog-kNN | 272/364 | 156/156 | 50/65 | 478/585 |
| RAFT | 251/364 | 74/156 | 50/65 | 375/585 |
| SARAF | 256/364 | 74/156 | 54/65 | 384/585 |
| Residual-kNN | 273/364 | 156/156 | 61/65 | 490/585 |
| TimeRAF | 325/364 | 155/156 | 53/65 | 533/585 |

TimeRAF has 43 more strict wins than Residual-kNN and larger median paired MSE
and MAE reductions (3.72% and 4.65% versus 2.23% and 2.33%). Analog-kNN and
Residual-kNN remain controls and ablations, not external baselines.

## Superseded Single-Horizon Execution

The earlier run completed all 208 cells with zero failed, incomplete, pending,
or running cells. It remains historical evidence for the original 96/24
scope, but it is not the result source for the replacement main tables.

- source and launcher revision:
  `964433ca271bd93aaac091feb18f565f27b13045`;
- protocol SHA-256:
  `7f561754a6f7b79e7fb89396e46f235a6220544cbbb16ebab7ad619907125647`;
- replay catalog SHA-256:
  `2e072fc9a7a13cb6f0b0fabc323dba61bc91bf77ff60c17990c2fcc466347dac`;
- result summary SHA-256:
  `d2fb2d8f3d4d79b79e74192bc865a468bf5b9b838beada79dd96f6c4f2b2cd6f`.

The run observed four distinct positive worker PIDs bound to physical GPU IDs
0--3, with no inactive reserved GPU. Every cell passed finite-metric,
Base-recomputation, exact system-coverage, and no-lookahead checks.

Strict wins over the fixed Base predictions are:

| System | Long-term | PEMS | EPF | Overall |
|---|---:|---:|---:|---:|
| Analog-kNN | 71/91 | 52/52 | 50/65 | 173/208 |
| RAFT | 65/91 | 17/52 | 50/65 | 132/208 |
| SARAF | 69/91 | 17/52 | 54/65 | 140/208 |
| Residual-kNN | 81/91 | 52/52 | 61/65 | 194/208 |
| TimeRAF | 84/91 | 52/52 | 53/65 | 189/208 |

Residual-kNN has five more strict wins than the full portfolio. TimeRAF and
Residual-kNN are identical in the 106 cells where the portfolio selects
historical residual retrieval. In the remaining 102 cells, TimeRAF is lower
on every metric in 68, Residual-kNN is lower on every metric in 19, and 15
are mixed or tied. TimeRAF therefore has larger median paired MSE and MAE
reductions, but seasonal-family selection loses more strict wins than it
rescues in this fixed single-horizon comparison.

## Completed Native Execution

The P5 native run completed all 52 canonical pairs with finite metrics and
zero missing pairs. The canonical artifact is
`docs/publication_results/native_retrieval/native-52-schema-v2.json` at
SHA-256
`72db2f50aadd3f373e773bd4bcab4f107aa5b9a41ec45502964eed82b2b23b58`.
It contains 44 RAF, seven TS-RAG, and one RATD pair.

| Method | Strict wins | Base metrics | Retrieval metrics | Mean delta |
|---|---:|---|---|---|
| RAF | 33/44 | WQL 0.1666; MASE 3.0852 | WQL 0.1425; MASE 2.8322 | WQL -0.0242; MASE -0.2530 |
| TS-RAG | 7/7 | MSE 0.2007; MAE 0.2524 | MSE 0.1939; MAE 0.2491 | MSE -0.0068; MAE -0.0033 |
| RATD | 0/1 | RMSE 0.3891; MAE 0.2326 | RMSE 0.6374; MAE 0.4637 | RMSE +0.2483; MAE +0.2311 |

RAF and TS-RAG improve their average metrics, while RATD is materially worse
than CSDI on its released Electricity setting. The methods remain
incomparable across rows because their native protocols differ.

The RATD finalizer verifies 5,093 test origins for both RATD and CSDI. Its
training topology uses GPU IDs 0--1, and exact origin-sharded evaluation uses
all eight H100 GPUs for RATD and seven for CSDI. Separate two-origin checks
prove byte-exact inference equivalence across physical GPUs before sharding.
