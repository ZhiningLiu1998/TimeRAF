# TimeRAF Research Extension

This extension reuses Time-Series-Library's TimeMixer training path and adds
reproducible historical forecast correction, causal online stacking, and
time-aware significance evaluation.

## Architecture

- `ts_rag/config.py`: matched dataset and TimeMixer presets.
- `ts_rag/exp_wrapper.py`: deterministic prediction-bundle export.
- `ts_rag/historical_retrieval.py`: vectorized residual retrieval and
  neighbor-disagreement shrinkage.
- `ts_rag/online_ensemble.py`: NLinear auxiliary forecast, overlapping forecast
  revisions, seasonal forecast, and causal rolling stacking.
- `ts_rag/post_test.py`: isolated loader for canonical ETT rows after the
  repository's official split.
- `ts_rag/significance.py`: paired single- and multi-seed moving-block
  bootstrap.
- `ts_rag/matrix.py` and `ts_rag/appendix_matrix.py`: frozen primary,
  confirmatory, and Appendix publication gates.
- `ts_rag/publication.py`: fail-closed joint evidence and artifact audit.
- `scripts/timefuse_post_review_pipeline.py`: review-gated, resumable
  Greenland replay, import, Appendix, and final-audit orchestration.

The final phase-1 method is frequency-aware:

- hourly ETTh1/ETTh2 use the frozen causal online ensemble;
- 15-minute ETTm1/ETTm2 use validation-selected historical residual retrieval.

Every online contribution is indexed by target timestamp. A query at time `t`
can read prefix statistics only through `t - 1`.

## Reproduction

Create the research environment from `requirements-research.txt`, then train a
matched baseline:

```bash
PYTHONPATH=. .venv/bin/python scripts/run_ett_baseline.py \
  --dataset ETTh1 --seed 2021 --model_id ts_rag \
  --output_dir ./ts_rag_outputs/baselines \
  --checkpoints ./checkpoints/baselines --use_gpu false
```

Run validation-selected historical correction:

```bash
PYTHONPATH=. .venv/bin/python scripts/run_historical_rag.py \
  --dataset ETTm1 --seed 2021 --model_id ts_rag \
  --output_dir ./ts_rag_outputs/baselines --use_gpu false
```

Run the frozen hourly online ensemble:

```bash
PYTHONPATH=. .venv/bin/python scripts/run_online_ensemble.py \
  --dataset ETTh1 --seed 2021 --model_id ts_rag \
  --output_dir ./ts_rag_outputs/baselines --use_gpu false \
  --online_policy fixed_robust
```

Export and evaluate the unused post-test timeline:

```bash
PYTHONPATH=. .venv/bin/python scripts/export_ett_post_test.py \
  --dataset ETTh1 --seed 2021 --model_id ts_rag \
  --output_dir ./ts_rag_outputs/baselines \
  --checkpoints ./checkpoints/baselines --use_gpu false --train_model false

PYTHONPATH=. .venv/bin/python scripts/run_post_test_method.py \
  --dataset ETTh1 --seed 2021 --model_id ts_rag \
  --output_dir ./ts_rag_outputs/baselines --use_gpu false
```

After all four datasets and three seeds are present:

```bash
PYTHONPATH=. .venv/bin/python scripts/summarize_ett_research.py
PYTHONPATH=. .venv/bin/python scripts/summarize_ett_post_test.py
PYTHONPATH=. .venv/bin/pytest -q
```

## Results

Both official-test and untouched post-test summaries pass all phase-1 gates:
MSE and MAE improve for every dataset and seed, every per-seed MSE gain exceeds
1%, and every aligned three-seed MSE bootstrap interval is below zero.

- Official test: `ts_rag_outputs/baselines/ett_multiseed_summary.json`
- Confirmatory post-test: `ts_rag_outputs/baselines/ett_post_test_summary.json`
- Protocol and leakage controls: `docs/research_protocol.md`
- Chronological decisions and rejected experiments: `docs/research_log.md`

The official ETTh2 test informed development of the online regularization, so
the post-test timeline is the confirmatory result. This limitation is retained
in the log rather than hidden.

## Publication-Scale Execution

The broad reproduction targets publishable utility rather than universal
per-cell dominance: at least 80% strict improvement overall, at least 70% per
task family, positive family means, and positive overall means and medians.

Interactive matrix work uses four independent A10G workers in the EFS-backed
`<DEV_SPACE>` space. Very large replay jobs may use one Greenland
`ml.p4de.24xlarge` host with eight independent workers. Acceptance requires
eight distinct CUDA-bound PIDs and observed nonzero utilization on all eight
A100s; durable Greenland artifacts use
`s3://<DEV_BUCKET>/timeraf/greenland/`.

After the 208-checkpoint catalog is reviewed and committed, the staged
post-review runner verifies its commit and reviewed hash before upload or
submission. Final claims are emitted only by
`scripts/audit_timefuse_publication.py`, which jointly audits primary,
confirmation, replay/topology, and Appendix evidence.
