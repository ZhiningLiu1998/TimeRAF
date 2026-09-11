# TimeFuse A10G vs A100 Full-Matrix Comparison

- Method revision: `d9be338`
- Manifest SHA-256: `45830d23f3b017c15d3680c88f837f84a9441eef9a6ceaaa5370c14872660c2a`
- Numerical tolerance: `atol=1e-05`, `rtol=0.0001`
- Overall consistency gate: **FAIL**
- Runtime evidence gate: **PASS**
- Runtime source: `first_pass_summaries`

## Consistency

| Check | Result |
|---|---:|
| Complete 585-cell scope | True |
| Baseline metric values within tolerance | 472/1326 |
| Corrected metric values within tolerance | 469/1326 |
| Selected method agreement | 569/585 |
| Selected parameter agreement | 529/585 |
| Strict-improvement classification agreement | 581/585 |

## Runtime

| Measure | 4x A10G | 8x A100 |
|---|---:|---:|
| Runtime cell attempts | 585 | 585 |
| Observed cell envelope (hours) | 73.760 | 22.981 |
| Reported matrix time (hours) | n/a | 22.984 |
| Throughput basis (hours) | 73.760 | 22.984 |
| Cells/hour | 7.931 | 25.452 |
| Sum cell-worker hours | 235.194 | 161.639 |

Observed matrix makespan speedup: **3.209x**.
Both-trained subset: 530 cells; ratio of summed cell time **1.574x**; median per-cell A100 speedup **1.001x**.

Runtime uses the first-pass 585-cell artifacts on both systems; recovery-composed artifacts are used only for numerical consistency. The A10G cell envelope includes any Studio interruption. Use the both-trained subset for hardware guidance because loaded release checkpoints do not perform the same work as fresh training.
