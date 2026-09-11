# Retrieval-Augmented Time-Series Forecasting: Literature Notes

This review was assembled from arXiv metadata and the downloaded LaTeX sources
under `paper/reference/literature/`. It focuses on forecasting methods that
retrieve numerical time-series evidence, rather than text-only RAG systems
that merely happen to consume temporal data.

## Development of the area

| Work | Retrieval unit | Integration | Training regime | Main limitation relative to this work |
| --- | --- | --- | --- | --- |
| MQRetNN (Yang et al., 2022) | Encoded contexts from other entities | Cross-entity attention after a pretrained MQCNN encoder | Retriever/attention trained with the forecaster | Population-specific architecture; retrieves representations rather than a frozen model's errors |
| RATD (Liu et al., 2024) | Similar historical series | Retrieved reference guides diffusion denoising | End-to-end diffusion model | Backbone-specific and computationally heavy |
| RAF (Tire et al., 2024/2026) | Historical chunks for a TSFM | Prompt/context augmentation | Zero-shot TSFM | Focuses on foundation models and direct retrieved values |
| TimeRAG (Yang et al., 2024) | DTW-nearest historical sequences | Serializes references into an LLM prompt | Training-free LLM forecasting | Text interface and direct analog futures |
| TimeRAF (Zhang et al., 2024) | Task-specific knowledge-base series | Learned retriever plus channel prompting | End-to-end TSFM | Requires a retrieval-aware foundation model; exact name conflicts with this repository |
| TS-RAG (Ning et al., 2025) | Encoder embeddings from a knowledge base | Adaptive Retrieval Mixer | Retrieval module around a TSFM | Zero-shot TSFM setting, not arbitrary frozen forecasters |
| RAFT (Han et al., 2025) | Input-similar train windows and their futures | Joint input/reference forecaster | Learned forecaster | Direct future retrieval; retraining required |
| Cross-RAG (Lee et al., 2026) | Retrieved examples from several RAG methods | Query-reference cross-attention | TSFM-specific module | Learns to reject irrelevant references but still changes the backbone path |
| CRAFT (Kang et al., 2026) | Channel-specific windows | Sparse time-domain pruning and spectral ranking | Retrieval-aware forecaster | Higher channel-wise index cost; no model-error memory |
| SARAF (Zhou et al., 2026) | Similar historical windows | Stationarity-controlled diversity and Gaussian aggregation | Plug-in retriever plus forecaster | Dataset-level stationarity; retrieves futures instead of residuals |
| SpecReTF (Nguyen et al., 2026) | Windowed spectra | Amplitude/phase similarity plus recency | Retrieval-based forecaster | Frequency-specific similarity; no causal error correction |

## Shared findings

1. **Selective history can be more useful than indiscriminate long context.**
   Retrieval provides an explicit route to distant analogs without asking a
   parametric model to retain every historical regime.
2. **Input similarity is not equivalent to future usefulness.** Cross-RAG
   reports degradation from irrelevant neighbors; SARAF links this mismatch to
   non-stationarity; CRAFT and SpecReTF expose channel and frequency mismatch.
3. **Fusion needs reliability control.** Recent methods add learned mixers,
   cross-attention, diversity, or domain-specific similarity. This supports
   TimeRAF's disagreement shrinkage and conservative identity fallback.
4. **Most methods retrieve targets or latent representations.** Few methods ask
   the more model-specific question: *given what this frozen forecaster just
   predicted, how did it err in comparable historical contexts?*
5. **Evaluation leakage is easy to hide in overlapping windows.** A forecast
   at origin `t` must not consume residuals whose targets extend to or beyond
   `t`. The codebase therefore indexes all online evidence by target timestamp
   and tests future-target perturbations.

## Positioning of this paper

TimeRAF is a post-hoc correction layer. It stores out-of-sample residual
trajectories `y - y_hat` together with descriptors of the observed context,
the frozen base forecast, and optional calendar information. At a new origin,
it retrieves analogous descriptors and adds a reliability-shrunk weighted
residual. This differs from direct analog forecasting in three ways:

1. the memory is conditional on the base model and therefore corrects its
   systematic local bias instead of replacing its learned prior;
2. the backbone remains frozen and can be any forecasting architecture; and
3. the identity forecast is an explicit candidate, so weak validation evidence
   can abstain from correction.

The broad selector also includes label-free overlap and seasonal candidates
and causal rolling bias. Their inclusion tests whether retrieval is genuinely
needed for a cell and protects against validation-to-test residual shift.

## Naming issue

Zhang et al. already published a method named **TimeRAF** in 2024
(arXiv:2412.20810). The repository name can remain TimeRAF, but a distinct
method name is strongly recommended before ICLR submission. The current paper
title therefore avoids making the colliding name its primary title.

