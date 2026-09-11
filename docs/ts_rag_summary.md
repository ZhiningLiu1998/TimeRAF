# Residual Event RAG MVP Summary (Historical)

This document describes the first implementation only. It is retained to
explain the original event-memory code, but that method did not pass the final
ETT research gate. The current architecture, commands, and results are in
`README_ts_rag.md`; experimental decisions and rejected candidates are in
`docs/research_log.md`.

## Reused Time-Series-Library modules

- `data_provider/data_factory.py`
- `data_provider/data_loader.py`
- `exp/exp_basic.py`
- `exp/exp_long_term_forecasting.py`
- `models/TimeMixer.py`
- `utils/metrics.py`

The new wrapper class `ts_rag/exp_wrapper.py` inherits the long-term forecasting experiment class and only adds deterministic split export for `X`, `Y`, and `Y_base`.

## Added modules

- `ts_rag/config.py`: dataset presets and shared CLI configuration
- `ts_rag/exp_wrapper.py`: train/load/export wrapper on top of Time-Series-Library
- `ts_rag/residual_utils.py`: proxy residuals and score helpers
- `ts_rag/event_detection.py`: transparent multivariate event detector
- `ts_rag/memory_bank.py`: training-event memory bank build/save/load/stats
- `ts_rag/retriever.py`: brute-force residual-event retrieval and raw-window retrieval
- `ts_rag/correction.py`: retrieval-based future correction
- `ts_rag/evaluation.py`: overall and event-subset metrics
- `ts_rag/visualization.py`: dataset and case-study plotting
- `ts_rag/pipeline.py`: end-to-end orchestration

## Current multivariate residual-event definition

- For correction, the memory bank stores `future_residual = Y_future - Y_base`.
- For event detection on the input window, the current MVP uses a moving-average proxy residual:
  - `R_input = X - moving_average(X)`
- Event score is `||R_t||_2`.
- Events are detected using percentile thresholding plus simple non-maximum suppression.
- Retrieval objects remain multivariate patches with shape `[W_event, D]`; they are not collapsed to a scalar.

## Retrieval and correction

- Residual-event retrieval:
  - detect a query event patch on the current input window
  - flatten the multivariate residual patch
  - compute cosine similarity or negative Euclidean distance
  - optionally add active-variable mask similarity
  - return top-k historical events from the training-only memory bank
- Retrieval-only baseline:
  - KNN on raw input windows from the train split
  - average the corresponding future targets
- Correction:
  - preferred mode is weighted averaging of retrieved future residuals
  - final forecast is `Y_base + correction`
  - uniform and similarity-weighted aggregation are both supported

## Current tradeoff and rationale

This first version intentionally does not implement rolling fitted residuals from TimeMixer over the encoder window. That would require a more invasive change to the original modeling pipeline. Instead, this MVP keeps the training/forecasting path unchanged and uses:

- TimeMixer future residuals for correction
- a simple moving-average proxy residual for event localization on the input window

This keeps the system runnable and modular while preserving the core hypothesis test: whether residual-event retrieval can help TimeMixer on event-heavy windows.

## Experimental outputs

Running `scripts/run_residual_rag.py` produces:

- overall metrics: MAE / MSE / RMSE
- event-heavy subset metrics
- non-event subset metrics
- side-by-side results for:
  - TimeMixer only
  - Retrieval only
  - TimeMixer + residual event RAG correction
  - TimeMixer + raw window retrieval correction

## Initial result status

This code delivers the end-to-end MVP pipeline, but it is not the final
validated method. After running the legacy scripts, inspect:

- `ts_rag_outputs/<setting>/evaluation.json`
- `ts_rag_outputs/<setting>/memory_bank_stats.json`
- `ts_rag_outputs/<setting>/case_studies/`

## Improvements explored after this MVP

Later experiments replaced brute-force event retrieval with vectorized
historical residual retrieval, chronological validation, disagreement
shrinkage, and a causal online ensemble. See `docs/research_log.md` for both
successful and rejected variants.
