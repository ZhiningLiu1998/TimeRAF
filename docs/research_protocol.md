# TimeRAF ETT Research Protocol

## Objective

Develop a causal historical correction method that improves a matched
TimeMixer baseline on ETTh1, ETTh2, ETTm1, and ETTm2. The correction may use
retrieved residuals, prior forecast revisions, and simple auxiliary forecasts,
but it must never read a target that is unavailable at the forecast origin. Phase 1 uses
`seq_len=96` and `pred_len=96`. Longer standard horizons are evaluated after
the method passes the four-dataset phase-1 gate.

## Leakage Controls

- Train TimeMixer only on the official training split.
- Use validation labels for method selection and correction calibration.
- Do not select hyperparameters, examples, or checkpoints using test metrics.
- Fit the final test-time retrieval memory using data whose targets occur
  before the test forecast origin.
- Report train, validation, and test metrics separately.
- Average paired loss differences over model seeds at each aligned forecast
  origin before temporal block bootstrap; do not treat seeds or overlapping
  windows as independent observations.

The original MVP stores in-sample training residuals. Those residuals are not
considered reliable correction targets because a model generally underestimates
its deployment error on its own training windows. New methods must use
out-of-sample or strictly historical residual targets.

## Baseline

The reference baseline is the repository's TimeMixer configuration:

| Dataset | d_model | Batch | Channel independent |
| --- | ---: | ---: | ---: |
| ETTh1 | 16 | 128 | 1 |
| ETTh2 | 16 | 32 | 1 |
| ETTm1 | 16 | 16 | 1 |
| ETTm2 | 32 | 128 | 1 |

Shared settings are two PDM layers, three average-pooling downsampling layers,
downsampling factor two, MSE loss, Adam, learning rate `0.01`, ten epochs,
and patience ten. Final evidence uses seeds `2021`, `2022`, and `2023`.

## Success Gate

A candidate passes only when:

1. MSE and MAE are lower than the matched TimeMixer prediction on all four ETT
   test sets.
2. The MSE reduction is at least 1% on every dataset.
3. A paired moving-block bootstrap over chronological forecast origins gives a
   95% confidence interval below zero for the candidate-minus-baseline MSE.
4. The result survives at least three model seeds.

Overlapping forecast windows invalidate an IID per-window significance test.
Bootstrap blocks must preserve temporal adjacency and use a block length no
shorter than the forecast horizon.

## Confirmatory Post-Test

The repository's ETT loaders stop after 20 synthetic 30-day months, while each
canonical CSV contains roughly four additional months. Once the method is
frozen, those unused rows form a confirmatory post-test timeline:

- hourly: forecast origins 14,400 through 17,324, 2,925 windows;
- 15-minute: forecast origins 57,600 through 69,584, 11,985 windows.

The training scaler and TimeMixer checkpoints remain unchanged. At each
post-test origin, a causal method may use validation and official-test targets
that have already occurred, but not the current or future post-test targets.
The same four success gates apply to this timeline.

## Protocol Deviation

ETTh2 official-test diagnostics were inspected while developing the online
ensemble regularization. Consequently, the official-test result is development
evidence rather than a pristine final test. The configuration was frozen before
the post-test timeline was exported or evaluated. Confirmatory claims therefore
use the post-test result; official-test metrics remain reported for benchmark
comparability.

## Artifact Contract

Each run keeps:

- exact arguments and data SHA-256;
- checkpoint and deterministic train/validation/test prediction bundles;
- baseline and corrected metrics;
- validation search table;
- elapsed time and software versions;
- significance report for final candidates.

Binary artifacts live under `ts_rag_outputs/`. Conclusions, failed experiments,
and design changes are appended to `docs/research_log.md`.

## Full-Matrix Execution Topology

All full-matrix GPU work runs in the persistent `<DEV_SPACE>` SageMaker
Studio space on one `ml.g5.12xlarge` host. The host reserves four NVIDIA A10G
GPUs. The matrix launcher uses four independent cell workers, assigns them to
CUDA devices 0 through 3, and records `processes_per_host=4`, `total_gpus=4`,
and `world_size=4`. This is cell-level parallelism; each baseline model remains
a single-GPU run under the paper's training protocol.

The canonical remote project root is
`/home/sagemaker-user/user-default-efs/workspace/TimeRAF`. Source, Python
environments, Git metadata, project caches, datasets, checkpoints, generated
outputs, logs, migration staging, and temporary intermediates remain under that
EFS-backed root. Instance EBS and `/tmp` do not hold project artifacts.

The parent sees all four CUDA devices. Before a child initializes CUDA, the
launcher restricts `CUDA_VISIBLE_DEVICES` to that child's assigned physical GPU;
the child then uses local device `cuda:0`. This keeps saved checkpoint device
metadata valid while preserving one distinct physical GPU per worker.

The launcher holds one matrix lock, updates one source-revision-filtered
summary from its main process, and gives each cell its own status, result, and
log files. A resumed run skips only completed cells from the requested source
revision. Run metadata records the method revision and launcher revision
separately. Runtime verification requires four distinct active cell PIDs on
four distinct GPUs whenever at least four cells remain queued.

The local App supervisor uses `scripts/sagemaker_app_supervisor.sh`. It may
recreate only a missing, deleted, or failed App, using `ml.g5.12xlarge` and
SageMaker Distribution GPU alias `4.2.2`; configuration drift on a live App is
reported rather than changed automatically.

The supervisor and post-primary worker use the refreshable
`timeraf-modeldev` AWS profile by default. That profile resolves through an
Ada `credential_process` configured for Conduit account `<AWS_ACCOUNT_ID>` and
role `<ADMIN_ROLE>`; static credentials in the default
profile are not an acceptable dependency for a long-running experiment.
The corresponding versioned matrix health worker is
`scripts/timefuse_matrix_monitor.sh`; it verifies the four-process topology,
EFS placement, progress and failures, and emits the completion event consumed
by the post-primary finalizer.

## Greenland Large-Scale Execution

Experiments with at least eight independent runnable cells may use one
Greenland `ml.p4de.24xlarge` host. The matrix launcher must reserve all eight
A100 GPUs, launch eight independent cell workers on physical devices zero
through seven, and record `instance_count=1`, `processes_per_host=8`,
`total_gpus=8`, `world_size=8`, and zero inactive GPUs. Each child sees only
its assigned physical GPU and addresses it as local `cuda:0`; this preserves
the paper's single-GPU model protocol while parallelizing across cells.

The launcher refuses a p4de topology with fewer than eight workers and refuses
an initial queue shorter than eight cells. Submission evidence must contain
the Greenland scheduler manifest and its p4de/eight-GPU request. Startup
evidence must contain eight worker PIDs, eight distinct device bindings, and
sampled `nvidia-smi` utilization while the queue has at least eight cells.
Successful completion is not a substitute for this topology evidence.
Import and final audit independently validate the PID and utilization maps;
they do not accept producer-written topology pass booleans as proof.

The persistent Studio App remains EFS-only at
`/home/sagemaker-user/user-default-efs/workspace/TimeRAF`. Greenland does not
mount that EFS, so durable Greenland artifacts use
`s3://<DEV_BUCKET>/timeraf/greenland/`; node-local storage is disposable
cache only. Every submission records immutable source and launcher revisions,
image identity, S3 input/output locations, initiative, region, job UUID, and
console URL.

The only approved Greenland runtime image is the digest in the committed
`docs/greenland_approved_image.json` allowlist. Input preparation requires the
allowlist bytes to match the selected source revision, then validates its
image source revision, 13 model imports, dependency and codec checks,
checkpoint preflight, removed-package check, and completed zero-finding ECR
scan. Submission repeats the allowlist validation. A zero-scan image is not
eligible merely because it is immutable; the approval must explicitly cover
the selected job kind.

### Cross-Hardware Full-Matrix Replication

The A10G primary matrix continues without interruption. In parallel, one
Greenland `ml.p4de.24xlarge` run repeats the exact 585-cell manifest at
SHA-256
`45830d23f3b017c15d3680c88f837f84a9441eef9a6ceaaa5370c14872660c2a`,
seed 2021, and frozen method revision `d9be338`. Eight independent workers
train and evaluate cells on A100 devices zero through seven. The job does not
consume replay checkpoints and rejects no-train, export-only, force-train,
array-export, filtering, truncation, topology overrides, and early-stop-on-cell
failure options. `--max-failures=0` keeps all 585 cells schedulable, while
`--num-workers=4` preserves the data-loader configuration used by the A10G
run.

Source and dataset inputs are content addressed in the project S3 prefix.
Results, logs, topology samples, and generated checkpoints are uploaded with
per-object SHA-256 metadata. The runtime writes `upload_complete.json` only
after every output succeeds; an upload exception instead writes a best-effort
`upload_failed.json`. The EFS importer requires the successful marker to cover
the exact non-control object set, validates all metadata, recomputes the
585-cell summary from imported statuses and results, and independently checks
eight distinct worker PIDs plus positive utilization on all eight A100s.
Generated checkpoints remain durable in S3 and are omitted from the default
EFS import because they are reproducible and not needed for comparison.

Cross-hardware numerical consistency compares every baseline and corrected
paper metric with the symmetric rule
`abs(a10g-a100) <= 1e-5 + 1e-4 * max(abs(a10g), abs(a100))`. It also reports
exact selected-method agreement, selected-parameter agreement, and
strict-improvement classification agreement. Runtime evidence includes summed
cell-worker time, observed cell envelope, matrix elapsed time, cells/hour, and
per-cell speedup distributions. The primary hardware recommendation uses the
subset trained from scratch on both systems. The raw A10G envelope is retained
but explicitly includes Studio interruption time and cells that may load
release checkpoints.

When numerical recovery is needed, consistency fields come from the
recovery-composed summaries, while every runtime field comes from the original
terminal first-pass summaries. Numerically incomplete first-pass attempts
remain valid runtime observations. This keeps both per-cell timing and the
A10G envelope aligned with the A100 first-pass matrix timer instead of mixing
recovery jobs into only one side. The report records this source split and
fails its runtime-evidence gate unless all 585 first-pass timings are finite
and positive and the A100 matrix timer is present. The numerical tolerances
are immutable constants rather than tunable report arguments.

The detached `scripts/timefuse_greenland_replication_worker.sh` owns
post-submission polling and finalization. It imports the completed Greenland
run below `outputs/greenland_runs/<run-id>/`, then independently waits for the
A10G runner to exit after 585 completed cells before writing JSON and Markdown
reports below `outputs/greenland_comparisons/<run-id>/`. If either first-pass
matrix has non-finite results, set
`TIMERAF_GREENLAND_POST_IMPORT_ACTION=import-only`; the worker then exits
successfully after the hash-verified import, and the final comparison is
deferred until numerical recovery has produced explicit composed summaries.
Terminal S3 identity is checked with the immutable launcher source revision,
while import and optional comparison use a separately pinned controller
revision. This permits fail-closed importer repairs without changing the
submitted method or launcher identity. Import completion additionally requires
the EFS `recomputed_matrix_summary.json` size and SHA-256 to match its fetch
receipt. Import-only mode accepts a terminal non-finite first pass only when
all 585 raw attempts are terminal and every independently recomputed
`incomplete` row is a completed non-finite result inside the frozen recovery
cohort. The receipt binds those cell IDs, status/result hashes, and topology
evidence; missing artifacts, runtime failures, and out-of-cohort cells fail
the import.

Every Greenland importer writes an EFS-local
`recomputed_matrix_summary.json` and records its path, size, and SHA-256 in
the fetch receipt. Downloaded runtime summaries retain p4de node-local
artifact paths and therefore are not valid composition inputs. Numerical
recovery composition requires the two A10G recovery metadata files and the
two A100 fetch receipts in addition to the four summaries. It verifies the
frozen profile, cohort, protocol, topology, EFS summary hash, and consistent
exact/fallback revision within each hardware system. A10G and A100 may record
different recovery revisions when the difference is infrastructure-only; the
composition receipt preserves both.

Composition retains each selected cell's first-pass `started_unix` so the
pre-registered 123/462 development-confirmation boundary cannot move during
numerical recovery. The recovery attempt's own timestamp remains available as
`numerical_recovery.recovery_started_unix`. A final publication audit of a
composed primary summary requires `--recovery-receipt`; the audit rehashes and
reloads all six summaries and four provenance inputs, reruns profile selection
and composition, and verifies that the primary summary is the recomputed A10G
output.

Once both A100 recovery profiles have been hash-verified and imported, run
`scripts/finalize_timefuse_numerical_recovery.py` from a pinned EFS source
snapshot. The finalizer accepts the two first-pass summaries, four recovery
summaries, two A10G metadata files, two A100 fetch receipts, and the A100
first-pass final status. It then executes the existing composition,
confirmation, comparison, and checkpoint-indexing entry points. It requires a
complete composed 585-cell scope and a complete hash-bearing 208-checkpoint
catalog, while preserving a valid below-threshold publication or hardware
consistency result as an experimental outcome. Its immutable receipt binds all
input and output paths, sizes, and SHA-256 values and is rehashed rather than
overwritten on resume. Before writing that receipt, finalization independently
requires the exact 585-cell/1,326-metric comparison scope, frozen tolerances,
both first-pass runtime scopes, all 585 finite positive timings, and the A100
matrix timer; a false numerical-consistency gate remains valid evidence only
after those structural checks pass.

The finalizer invokes research Python through the canonical EFS
`.venv-gpu312/bin/python` entry. A Python venv normally links that entry to
the base interpreter under `/opt`; validation therefore binds the containing
EFS venv and its `pyvenv.cfg` while retaining the EFS invocation path. Resolving
the final symlink and then requiring the target itself to reside on EFS would
incorrectly reject the supported Studio environment.

The detached recovery lifecycle uses one screen rather than independent
pollers. `scripts/timefuse_numerical_recovery_worker.sh` first completes both
A10G profiles; only after it exits successfully does
`scripts/timefuse_greenland_recovery_worker.sh` take over in the same session.
The A10G runner executes matrix and cell entry points from its pinned source,
but sets their working directory to the canonical EFS project root so relative
dataset paths resolve against durable project data. It also keeps the pinned
source first on `PYTHONPATH`. The exact profile may terminate cells only with
an explicit `NumericalIntegrityError`; any other failed-cell error blocks the
fallback transition.
The second worker independently verifies the two first-pass matrices, released
A10G lock, both local recovery profiles, EFS control environment, committed
dry-run audit, and a fresh initiative/pool capacity snapshot. It submits each
profile at most once, refuses to resubmit when S3 contains an untracked
submission receipt, verifies all eight runtime worker bindings from
CloudWatch, waits for hash-bound terminal markers, and imports both profiles
below EFS. Before submission it also rehashes the primary A100 recomputed
summary and requires its receipt counts and non-finite cell IDs to match that
summary's complete 585-cell terminal scope.

The independent EKS progress monitor records the latest CloudWatch event time
in addition to result counts and topology headers. While fewer than 585 result
records are visible, more than 30 minutes without a log event is a one-shot
stale-activity alert; a later event emits a recovery record. This heartbeat is
diagnostic only and cannot create, stop, import, or resubmit a Greenland job.

After both imports, the same screen runs
`scripts/timefuse_recovery_finalization_worker.sh`. This worker writes one
stable authoritative A10G primary summary, then invokes the scheduler-free
finalizer to compose both hardware results, rebuild the 462-cell confirmation,
compare all 585 cells, and stage exactly 208 replay checkpoints. It exits at a
`CATALOG_READY` event. Catalog review, copying the accepted bytes into the
repository, and a local Git commit remain mandatory before input preparation
or any replay submission.

Greenland scheduler control is isolated from the research runtime. Run
`prepare`, `dry-run`, and `submit` with the EFS-resident control Python at
`<artifact-root>/.venv-greenland-control/bin/python`, containing
`amzn-greenland-torchx-launcher==1.0.47` and
`torchx-nightly==2026.7.30`. Its registered scheduler backends must include
`greenland`. The post-review pipeline records this executable and package
identity during `prepare`, then rejects path, version, or scheduler drift
before either scheduler operation. Research stages continue to use
`.venv-gpu312`; no environment involved in this workflow may live on
instance-local storage.

## Full-Matrix Publication Gate

The 585-cell expansion is intended to establish broad, publishable utility,
not universal per-cell dominance. The gate is frozen before the full first pass
is used for method development:

1. At least 80% of all cells strictly improve every metric reported by the
   paper.
2. At least 70% of cells strictly improve every metric within each task family.
3. Mean gain is positive for every metric within every task family.
4. Overall mean and median gains are positive for every reported metric.
5. Regressions and runtime failures are reported rather than removed from the
   denominator.
6. After the method is frozen, additional seeds or held-out combinations
   confirm positive aggregate gains.

The release-checkpoint matrix and any full-matrix cells inspected during
development are development evidence. Confirmatory evidence must not tune the
selection policy.

### Frozen Confirmatory Complement

The publication gate was frozen only after 123 matrix cells had completed.
To avoid retroactively treating those observed outcomes as held out,
`docs/timefuse_confirmation_protocol.json` fixes them by an
outcome-independent runtime boundary: sort cell states by `started_unix`,
break ties by cell ID, and classify the first 123 as development. The sorted
development IDs hash to
`d7f9de51cbe7489d74c9c254797c78bc2342f4475630b3ea04e5a19440b73328`.
The last development start at `2026-07-30T19:02:13Z` and first confirmatory
start at `2026-07-30T20:23:43Z` are separated by 4,889.7 seconds.

The complement contains 462 cells spanning all 13 models, all three task
families, and 15 datasets. `scripts/summarize_timefuse_confirmation.py`
verifies the source revision, boundary timestamps, cell-ID hash, exact
585=123+462 partition, and complete cell-state identity before computing any
gate. It applies the same 80% overall, 70% per-family, positive per-family
metric means, positive overall means and medians, and no-runtime-failure
criteria to the 462-cell denominator. Incomplete cells stay in that
denominator. No confirmatory outcome may change the frozen `d9be338` method.

When the primary runner has exited, regenerate the final 585-cell summary from
its immutable status/result artifacts with
`scripts/summarize_timefuse_matrix.py --source-revision d9be338
--require-publication-scope`, writing through an explicit separate
`--summary` path. The standalone summarizer does not acquire GPUs or alter the
original matrix output tree. Do not repurpose the matrix runner's dry-run path
for this step, because it writes run metadata. The regenerated summary is
authoritative for both the full publication gate and the frozen confirmatory
complement.

The detached `scripts/timefuse_post_primary_worker.sh` automates this
transition. It waits for the versioned monitor's exact 585-cell completion
event, verifies EFS storage, confirms the runner is absent and its lock is
free, then expands a pinned Git revision below the EFS project root. It writes
the primary and confirmatory summaries, stages and hashes exactly 208 replay
checkpoints, and records artifact hashes in a receipt. It stops before
Greenland input preparation or submission so the generated catalog can first
be copied into Git, reviewed, and committed as part of the declared source
revision.

After that explicit review and commit,
`scripts/timefuse_post_review_pipeline.py` resumes through durable `prepare`,
`dry-run`, `submit`, `import`, `appendix`, and `audit` stages. Every invocation
reuses an EFS-resident state file and the same run ID, full source commit, and
reviewed catalog SHA-256. Before upload, `prepare` compares the catalog in the
materialized tree byte for byte with `git show <revision>:<catalog>` and the
reviewed digest. It also reloads the frozen replay manifest and numerical
recovery protocol, validates exact 208-cell catalog coverage and per-record
hash metadata, and requires the six replay cells in the recovery cohort to
retain their selected recovery profiles and receipt provenance. The prepared
state retains the catalog-bound composition receipt record. Before the final
audit, the pipeline rechecks that EFS file's size and SHA-256 and supplies it
to `audit_timefuse_publication.py --recovery-receipt`; no independent manual
receipt path can replace the reviewed identity. An existing state file cannot
be replaced by another `prepare`; later stages resume it. A real scheduler
action requires a completed `dry-run` stage and `--confirm SUBMIT`.

The post-review `appendix` stage is durable only after it independently
reloads the generated schema-v2 bundle catalog and RAG summary. It requires
the frozen source and method revisions, 96 unique reported states, 94
completed evaluable states, two explicit paper OOT states, and no failed,
running, pending, or incomplete state. The pipeline records catalog and
summary SHA-256 values in its stage evidence. Scientific gate failure is not a
structural failure: a complete below-threshold outcome proceeds to the final
audit with `publication_ready=false`.

The publication auditor's exit status is part of this contract: zero means a
valid publishable report, one means a valid report whose frozen scientific
gate did not pass, and two or greater means malformed evidence or execution
failure. The post-review pipeline records both zero and one as completed audit
stages with the report hash and explicit `publication_ready` value. It rejects
missing, invalid, contradictory, or structurally failed reports.

All post-review research-side commands use the canonical EFS
`<artifact-root>/.venv-gpu312/bin/python`. `prepare` verifies the venv marker
and executable and persists the lexical invocation path; import, Appendix RAG,
and final audit repeat that check and reject state or CLI path drift. The venv
entry may symlink to the SageMaker base interpreter, but an arbitrary `/opt`,
local-worktree, or instance-storage Python is not accepted.

## Appendix Baseline-System Scope

The 585-cell matrix covers the 13 models that the paper explicitly calls its
base forecasting models. The Appendix Additional Baselines table reports six
other systems at prediction length 96 for long-term datasets and 24 for PEMS
and EPF: Forward Selection, Portfolio Ensemble, ZeroShot Ensemble, AutoGluon
with 24 models, and finetuned and zeroshot Chronos-Bolt-Base. Across 16
datasets this is 96 reported cells. The paper itself marks AutoGluon on
Electricity and Traffic as out-of-time, so 94 cells have numerical baselines.

This scope is tracked separately in
`docs/timefuse_appendix_experiment_manifest.jsonl`. RAG correction uses the
same dataset splits, metric spaces, test stride, and validation-only selection
as the primary matrix. The appendix gate requires at least 80% strict
all-metric improvement overall, at least 70% within every task family and
baseline system, positive per-family metric means, and positive overall metric
means and medians. The two paper-declared OOT cells remain visible but are
excluded from numerical gain denominators.

Each evaluable appendix baseline exports one NPZ prediction bundle containing
validation/test `x`, `y`, `y_base`, `x_mark`, and `y_mark` arrays. Arrays are in
the task's reported metric space; PEMS bundles are therefore already
unnormalized. `scripts/run_appendix_rag_cell.py` validates shapes and finite
values, records the bundle SHA-256, performs the same validation-only method
selection as the primary matrix, and may save corrected predictions. Test
labels never enter static candidate selection; causal online candidates may
use only targets strictly earlier than the current forecast origin.

The paper and its public repository do not release Appendix ensemble code or
the Forward Selection iteration count and Portfolio subset size. The
pre-registered reproduction therefore uses standard Caruana selection with
replacement for 50 additions, minimizing validation MSE for long-term/EPF and
validation MAE for PEMS. Portfolio Ensemble ranks all 13 base models by that
same validation loss, evaluates equal-weight top-k prefixes for
`k=1,...,13`, and selects k on validation only. Every output records the full
selection trace, model order, weights, and SHA-256 of all base prediction
bundles. These explicit assumptions remain visible when comparing against the
paper's reported baseline values.

ZeroShot Ensemble is leave-one-dataset-out within each task family. It derives
23 deterministic statistical, lag, change, spectral, autoregressive, and
cross-channel features from at most 256 evenly spaced validation windows and
64 evenly spaced channels. Target features are robust-scaled using source-task
medians and median absolute deviations. A fixed RBF kernel with source-task
median pairwise distance as bandwidth weights the other datasets; each source
dataset contributes normalized reciprocal ranks of its 13 validation losses.
The target dataset's validation/test labels and model losses are excluded from
its weights. Feature sampling, source distances, similarities, ranks, and
final weights are recorded.

### AutoGluon and Chronos Prediction Export

The external baseline environment pins `autogluon.timeseries==1.4.0` and
`torch==2.7.1`. The official AutoGluon wheel supports Python 3.9 through 3.12
and declares the Transformers, Accelerate, GluonTS, and statistical-model
dependencies used by the `high_quality` and Chronos implementations. The
Python-3.12/Linux-x86_64 resolution is frozen in
`requirements-appendix.lock`. AutoGluon uses the versioned `high_quality`
preset. Chronos-Bolt-Base uses the versioned `Chronos` hyperparameters with
`model_path=bolt_base`; zero-shot disables model selection, and finetuning
defaults to 1,000 update steps and records the inference and finetuning batch
sizes.

Studio uses the isolated EFS environment
`<artifact-root>/.venv-appendix140`, created by
`scripts/provision_appendix_environment.py`. Provisioning rejects a non-NFS4
project root, an existing unmarked environment, dependency conflicts, wrong
package versions, and anything other than four visible CUDA devices. Its
receipt binds the source revision and lock hash to the complete installed
package-inventory hash. The post-review Appendix stage repeats the import,
version, CUDA, inventory, and receipt checks before launching work; the
primary `.venv-gpu312` environment remains unchanged.

Before an external phase, including a dry run, the pipeline requires an exact
installed-version match before acquiring the output lock or writing run
metadata. Each exporter CLI and direct exporter API call repeats the check
before writing status, loading a dataset, or creating an artifact. Pipeline
and worker provenance record both the installed and required versions.
Existing external outputs are resumable only when their run identity and
predictor metadata both record `1.4.0`. Catalog records repeat that identity,
and catalog construction and publication audit reject drift in any copy.

PyTorch 2.7 defaults unspecified `torch.load` calls to
`weights_only=True`, while AutoGluon 1.4.0's locally generated deep-model
predictor checkpoints require the legacy full-object loader. During an
`autogluon_high_quality` export only, the worker therefore sets
`TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` inside a scoped context that covers fit,
model selection, predictor reload, and prediction. This exception is approved
only for predictor artifacts generated locally by the frozen Appendix
environment and dataset; it must not be used to load an external predictor or
an untrusted checkpoint. A conflicting `TORCH_FORCE_WEIGHTS_ONLY_LOAD`
setting is a hard failure, and the prior environment is restored when the
export ends. Chronos-only exporters do not enable the exception. The exact
policy is recorded independently in run identity, predictor metadata, and
top-level bundle metadata. Resume and catalog validation reject a
high-quality bundle if any of those three records is missing or differs.

Training input is the train-only TSLib target timeline in the same normalized
space seen by the paper baselines. For each validation and test split, every
TSLib forecast origin is retained. A bounded window-to-item adapter maps each
`(forecast origin, output variate)` pair to one temporary univariate item,
forecasts the complete horizon, and restores arrays to
`[origin, horizon, variate]`. The input lookback is therefore exactly the
TSLib lookback, rather than an unreported terminal full-history forecast.
ETTm, PEMS, and hourly datasets use synthetic regular indices with their
actual 15-minute, 5-minute, and hourly cadence, respectively.

Prediction arrays are staged as memmaps and finalized into the common Appendix
NPZ schema. In Studio all predictor, staging, output, status, model-cache,
package-cache, and temporary paths remain below
`/home/sagemaker-user/user-default-efs/workspace/TimeRAF`. The launcher
explicitly redirects XDG, Hugging Face, Transformers, Torch, Ray, Joblib,
Matplotlib, and temporary-file roots there. The exporter records the installed
AutoGluon version, fit recipe, actual model names and count, split sizes,
effective batch sizes, bundle hash, and elapsed time. The actual model list is
authoritative because the paper labels AutoGluon as "24 models" without
releasing its environment.

For Greenland, only eight simultaneously runnable GPU-backed Chronos cells
establish the required p4de topology. AutoGluon `high_quality` includes
CPU-bound statistical models and cannot be used as evidence that an A100 is
active. A p4de launch therefore starts with eight Chronos workers bound to
distinct GPUs; AutoGluon cells run on a resource shape appropriate to their
measured CPU/GPU profile.

### Appendix Matrix Orchestration

`scripts/build_appendix_bundle_catalog.py` records each prediction bundle's
cell ID, path, size, and SHA-256. `scripts/run_appendix_matrix.py` requires a
catalog entry and an existing file for every selected evaluable cell, runs
independent RAG cells with a resumable source-revision filter, and retains one
matrix lock and per-cell logs. Run metadata records both the catalog hash and
the bundle-producing source revision.

The Appendix summary always reports all 96 rows and identifies the 94
paper-evaluable rows separately. The two AutoGluon OOT rows must finish in the
explicit `paper_oot` state; they never enter metric denominators. Publication
acceptance requires all 94 evaluable rows to complete without failures, at
least 80% strict all-metric improvement overall, at least 70% within every
task family and each of the six baseline systems, positive per-family metric
means, and positive overall metric means and medians. Filtered development
runs cannot satisfy the fixed 96/94 full-scope criterion.

The production bundle path is
`scripts/run_appendix_bundle_pipeline.py`. Before any generation it requires
the frozen 208-row replay manifest SHA-256 and recomputes every replay bundle
size and hash. It deterministically maps the 13-model libraries to all 16
Appendix datasets, builds 32 Forward Selection/Portfolio bundles and 16
leave-one-dataset-out ZeroShot bundles, and resumes only artifacts whose
source revision and input-catalog identity still match.

The same pipeline schedules the 46 evaluable AutoGluon/Chronos exports with
one physical GPU isolated per worker and records source, launcher, predictor,
status, and bundle provenance. Both AutoGluon OOT cells remain explicit and
produce no prediction bundle. The final schema-v2 catalog is written only
when exactly 94 evaluable bundle files and metadata files pass SHA-256
verification.

The schema-v2 catalog records the absolute EFS project root defining all
relative artifact paths. Consumers rehash the Appendix manifest, 208-entry
replay catalog, all 94 bundles, and all 94 metadata files. Metadata must agree
on cell identity, bundle hash, and source revision; static ensemble bundles
must also identify the replay-catalog hash. Appendix RAG records frozen
`method_revision=d9be338` separately from later source/launcher revisions.
Both bundle generation and the RAG matrix receive the same canonical EFS
project root explicitly; using the materialized source directory as the
catalog root is invalid.

The first complete Appendix RAG matrix under `distribution_robust_v2` is
immutable development evidence. Full 94-cell method-leader diagnostics then
froze `validation_argmin_v3` in
`docs/timefuse_appendix_selector_v3_protocol.json`. The selector minimizes
the worst candidate-to-baseline validation metric ratio over the unchanged
candidate grid. Exact ties prefer identity, then method name and canonical
parameter JSON. It has no fixed gain threshold or method-family priority and
does not read test metrics during production selection. Test outcomes were
used to design this generic policy and are therefore disclosed as development
evidence. A new full-scope Appendix run must use a fresh output root and a
separate Appendix method revision while primary, recovery, confirmation, and
replay retain `d9be338`.

On p4de, the initial external queue must include at least eight pending
Chronos cells. Eight workers are fixed to physical GPU IDs zero through seven,
and a two-second sampler records CUDA process PIDs and utilization. A
nominally successful export is rejected unless one sample proves eight
distinct bindings and cumulative samples show nonzero utilization on every
A100. AutoGluon-only residual work must run on a more appropriate resource
shape rather than occupying p4de without eight GPU-active workers.

### Base-Prediction Replay

The three advanced ensembles require aligned bundles from all 13 paper base
models. After the 585-cell training pass completes, its checkpoint catalog is
replayed only at the Appendix horizons: long-term 96, PEMS 24, and EPF 24.
This is 208 checkpoint-inference cells. The replay command uses
`--no-train --export-only`; it loads the original checkpoint, predicts every
validation and test window, writes the common NPZ bundle, and skips RAG search.

Export-only attempts have their own source-revision filter and a separate
output root. Matrix summaries distinguish them from ordinary evaluation
attempts, preventing a newer export result from shadowing a primary result.
`scripts/index_prediction_bundles.py` selects the 208-cell scope, verifies
optional hashes against the recorded digest, and records missing bundles.

The checkpoint catalog is generated with
`scripts/index_matrix_checkpoints.py`, using the 208-cell `--manifest`, a
dedicated EFS `--stage-root`, and `--hash`. This unifies checkpoints reused
from the release root and checkpoints trained in the resume root. Staging uses
hardlinks when both sources share the EFS filesystem and copies only when
hardlinks are unavailable. Catalog paths are relative to that staged archive
root rather than the whole project. The catalog and replay manifest must be
committed in the declared source revision before input preparation. Both
input preparation and the Greenland runtime independently require all 208
manifest cells to map to unique regular checkpoint files whose sizes and
SHA-256 values match the catalog. The content-addressed checkpoint archive
contains only those 208 verified files.

For a composed recovery summary, catalog generation also requires
`--recovery-receipt`. It rehashes the receipt, verifies that its A10G output
path and hash identify the supplied summary, and records the recovery profile
on each selected checkpoint. Six replay cells are in the recovery cohort:
Electricity horizon 96 and the five EPF cells. Their staged checkpoints must
come from the exact/fallback profile selected by the cross-hardware numerical
validity rule.

Catalog generation itself is also fail closed. When a selected manifest is
provided, `scripts/index_matrix_checkpoints.py` checks exact cell-ID equality,
all-usable and non-empty files, method-source revision consistency, unique
archive-safe relative paths, and valid SHA-256 values before writing the JSON.
Staging requires `--hash`. An incomplete intermediate staging tree may be
resumed, but no partial catalog is emitted or eligible for commit.

Greenland input preparation uses two EFS roots without modifying the completed
experiment worktree. A fixed-commit Git workspace below the canonical
project's `operations/` directory is `--project-root`; the canonical TimeRAF
EFS root is `--artifact-root`. Git revision checks and the source archive use
the former. Dataset, staged replay checkpoints, and input staging use the
latter. Both roots therefore remain on `user-default-efs`, while archive names
remain rooted at `dataset/` and `checkpoints/replay/` exactly as expected by
the p4de runtime.

The fixed source workspace is materialized from an EFS-resident Git bundle by
`scripts/materialize_greenland_source.py`. It fetches the requested bundle ref
into a bare repository's `FETCH_HEAD`, verifies the exact full commit SHA,
requires `git for-each-ref` to remain empty, and expands the tree without a
checkout. A receipt records the bundle hash and confirms no branch was
created. The materializer has only standard-library dependencies and can run
as a single copied tool without loading code from the active worktree. Input
preparation uses the materialized `source/` directory and must receive the full
commit SHA explicitly for source and launcher revisions.

On p4de, replay runs as eight single-GPU cell workers. Checkpoints, data, and
source are immutable S3 inputs under
`s3://<DEV_BUCKET>/timeraf/greenland/`; bundle outputs and logs are uploaded
to the same project prefix. Node-local files are disposable staging. The run
is accepted only after scheduler evidence and runtime evidence show eight
active ranks on eight distinct A100s. The immutable replay manifest is
`docs/timefuse_checkpoint_replay_manifest.jsonl` (208 rows, SHA-256
`447391d99c46bc1dd4170e71a8388bad5edb0a4c48a3048e230e404b96a0a77a`).
The runtime validates its row count, unique cell IDs, family horizons, and
hash evidence before GPU work starts. The replay uses `--max-failures=0` so
one failed export is reported without suppressing the remaining cells.

The EFS control-wheel supply chain is fixed to
`amzn_greenland_torchx_launcher-1.0.47-py3-none-any.whl` at SHA-256
`ed72caf99711d5fbe1370b3b2fefff3316038fe2c4ef97b995f29a600439d19d`
and `torchx_nightly-2026.7.30-py3-none-any.whl` at SHA-256
`56d2342879c37a746a64e6a65b9e537fe2bcdac8ef1fb464b23229493d4578a9`.
They are retained under the canonical project's
`operations/control-wheels/` directory.

Import a completed run into the Studio EFS project with
`scripts/fetch_greenland_outputs.py`. It assumes the validated Greenland
service role, requires the destination filesystem to be `nfs4`, and verifies
every downloaded object's S3 metadata SHA-256. It then requires successful
final status, matching source/launcher revisions, the eight-distinct-binding
and all-eight-utilized topology evidence, and exactly 208 completed
export-only cells with no failures. The generated replay catalog must contain
all 208 usable bundles and verified hashes. Because result files record
node-local absolute paths, relocation is permitted only to
`prediction_bundle.npz` beside the corresponding downloaded `result.json`,
with the originally recorded bundle hash unchanged.

## Final Evidence Audit

Final acceptance uses `scripts/audit_timefuse_publication.py`. It recomputes
the 585-cell and 94-cell gates from cell states, rebuilds the frozen 462-cell
confirmatory complement, validates every Greenland receipt object and replay
bundle, and independently checks the eight-PID/eight-A100 topology evidence.
Primary, recovery, confirmation, and replay retain method revision
`d9be338`; Appendix is validated independently against the committed
`validation_argmin_v3` method revision. The numerical-recovery protocol path
is always explicit and resolves from the pinned materialized source. The
canonical Appendix RAG replay records four reserved but inactive A10Gs and
four CPU workers because no model inference occurs.
The result is publishable only when primary, confirmation, complete 208-cell
replay, eight-A100 topology, and Appendix criteria all pass. Structural drift
is an error; honest below-threshold results remain in a valid report with
`publication_ready=false`.
