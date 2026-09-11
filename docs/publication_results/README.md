# TimeRAF Publication Results Snapshot

This directory is the compact, Git-versioned local snapshot of the accepted
publication results as of 2026-09-01. It contains numerical summaries and
provenance receipts only. Raw logs, prediction bundles, checkpoints, datasets,
and other large artifacts remain on the canonical Studio EFS or in the
project's Greenland S3 prefix.

## Reporting Default

Use `a100_composed_summary.json` as the default primary full-matrix result for
all new tables, plots, analyses, and narrative claims. The A10G summary remains
secondary hardware-comparison and historical audit evidence. Do not average or
merge the two primary summaries. Confirmation, Appendix, and final-audit
artifacts retain their own recorded source identities.

## Main Results

| Evidence | Strict improvements | Rate | Gate |
| --- | ---: | ---: | --- |
| Primary full matrix, 4x A10G | 558 / 585 | 95.38% | Pass |
| Primary full matrix, 8x A100 | 562 / 585 | 96.07% | Pass |
| Frozen confirmatory subset | 437 / 462 | 94.59% | Pass |
| Appendix paper baselines | 81 / 94 | 86.17% | Pass |
| Final publication audit | - | - | `publication_ready=true` |

Appendix task-family results are 33/40 for long-term forecasting, 24/24 for
PEMS, and 24/30 for EPF. The two paper-declared out-of-table cells remain
reported but are excluded from the 94-cell numerical denominator.

## Retrieval-Operator Comparison

The separate full-horizon checkpoint-replay comparison fixes all 585 Base
forecasts and evaluates Analog-kNN, the RAFT and SARAF retrieval operators,
Residual-kNN, and the full TimeRAF portfolio. Strict improvements over Base
are 478/585, 375/585, 384/585, 490/585, and 533/585, respectively. The RAFT
and SARAF rows apply their retrieval operators to fixed forecasts; they do
not execute the original learned forecasters or fusion layers.

All 585 cells complete with finite metrics and exact long-term
96/192/336/720, PEMS 6/12/24, and EPF 24 coverage. The first attempt stopped
after 432 completed cells because RAFT period 4 did not divide PEMS horizon 6.
The frozen compatibility recovery drops only periods that do not divide both
context and prediction length; it uses no observed metric. The completed run
retains those 432 unaffected cells and evaluates 153 cells with launcher
revision `cc80cabd363ac136c5d2e9284529e912b601544e`. Startup evidence records
eight distinct positive worker PIDs bound to GPU IDs 0--7 on the P5 space.

## Native Retrieval Systems

The native end-to-end comparison contains all 52 frozen pairs: 44 RAF, seven
TS-RAG, and one RATD. RAF improves both reported metrics in 33/44 pairs, and
TS-RAG does so in 7/7. RATD is worse than its CSDI base on the Electricity
pair: RMSE changes from 0.3891 to 0.6374 and MAE from 0.2326 to 0.4637.

All native results were run on the eight-H100 P5 space. The canonical
`native_retrieval/native-52-schema-v2.json` artifact validates finite metrics,
exact manifest coverage, source identities, startup topology, finalizer
receipts, and RATD/CSDI cross-GPU equivalence evidence.

## Cross-Hardware Result

The strict-improvement classification agrees for 581/585 cells (99.32%).
Observed full-matrix makespan is 3.209x shorter on 8x A100 than on 4x A10G,
although the A10G envelope includes Studio interruptions and 55 loaded
checkpoints. On the 530 cells trained on both systems, the ratio of summed
cell time is 1.574x and the median per-cell A100 speedup is 1.001x.

The strict numerical consistency gate **does not pass**: baseline metrics are
within the frozen tolerance for 472/1326 values and corrected metrics for
469/1326 values. Selected methods agree for 569/585 cells and selected
parameters for 529/585. Do not describe the two hardware runs as numerically
identical. This hardware comparison outcome is separate from the independently
passing publication audit.

## Files

- `key_metrics.json`: small machine-readable index of the headline results.
- `a10g_composed_summary.json`: recovery-composed 585-cell A10G result.
- `a100_composed_summary.json`: recovery-composed 585-cell A100 result.
- `confirmation_summary.json`: frozen 462-cell confirmatory evaluation.
- `appendix_summary.json`: accepted 96-row/94-evaluable Appendix result.
- `hardware_comparison.json`: cell-level A10G/A100 numerical and runtime
  comparison.
- `hardware_comparison.md`: concise generated hardware comparison.
- `publication_audit.json`: final independently recomputed acceptance report.
- `retrieval_baseline_summary.json`: complete six-system, 585-cell
  full-horizon checkpoint-replay result.
- `retrieval_baseline_run_metadata.json`: source, protocol, catalog, and
  requested-topology identity for that run.
- `retrieval_baseline_startup_topology.json`: observed eight-worker GPU
  bindings for that run.
- `retrieval_baseline_completion_receipt.json`: schema-v2 receipt binding the
  full-horizon inputs, result, topology, and compatibility recovery.
- `retrieval_baseline_bundle_catalog.json`: hash-verified index of all 585
  replay prediction bundles.
- `retrieval_baseline_recovery_failure_receipt.json`: preserved first-attempt
  failure evidence.
- `retrieval_baseline_materialization_receipt.json`: branch-free launcher
  materialization identity.
- `native_retrieval/native-52-schema-v2.json`: canonical 52-pair native
  end-to-end retrieval result.
- `native_retrieval/`: accepted RAF, TS-RAG, and RATD summaries and compact
  RATD finalization, topology, and equivalence evidence.
- `provenance/`: numerical-recovery, finalization, and Appendix completion
  receipts.
- `SHA256SUMS`: integrity hashes for every other file in this directory.

Paths embedded in copied artifacts intentionally retain their original EFS
locations. Verify the snapshot from this directory with:

```bash
sha256sum --check SHA256SUMS
```
