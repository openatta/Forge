# P1 Report

> **MOCK TRAINING NOTICE**: the `base`/`baseline_random`/`value_selected` arms below share the exact
> same (untrained) MockStudent weights -- `train/sft.py` is in mock mode this round, so their eval
> numbers are expected to come out numerically identical. This validates that the full pipeline
> (generation -> dedup -> collection -> selection -> compilation -> "training" -> evaluation ->
> reporting) works end to end. It does NOT show whether value-selected data beats random data --
> that question can only be answered once `mode="real"` training runs on actual GPU compute.
> The **teacher** arm is a real number (the teacher LLM itself, evaluated on the same holdout set).

## Per-cell success rate (21 cells)
| cell | base | value_selected | baseline_random | teacher |
|---|---|---|---|---|
| arithmetic/easy | 100% (n=1, CI [21%,100%]) | 100% (n=1, CI [21%,100%]) | 100% (n=1, CI [21%,100%]) | 100% (n=1, CI [21%,100%]) |
| arithmetic/medium | 0% (n=1, CI [0%,79%]) | 0% (n=1, CI [0%,79%]) | 0% (n=1, CI [0%,79%]) | 0% (n=1, CI [0%,79%]) |
| arithmetic/hard | 0% (n=1, CI [0%,79%]) | 0% (n=1, CI [0%,79%]) | 0% (n=1, CI [0%,79%]) | 100% (n=1, CI [21%,100%]) |
| algebra/easy | 100% (n=1, CI [21%,100%]) | 100% (n=1, CI [21%,100%]) | 100% (n=1, CI [21%,100%]) | 100% (n=1, CI [21%,100%]) |
| algebra/medium | 100% (n=1, CI [21%,100%]) | 100% (n=1, CI [21%,100%]) | 100% (n=1, CI [21%,100%]) | 100% (n=1, CI [21%,100%]) |
| algebra/hard | n/a | n/a | n/a | n/a |
| geometry/easy | 100% (n=1, CI [21%,100%]) | 100% (n=1, CI [21%,100%]) | 100% (n=1, CI [21%,100%]) | 100% (n=1, CI [21%,100%]) |
| geometry/medium | 100% (n=1, CI [21%,100%]) | 100% (n=1, CI [21%,100%]) | 100% (n=1, CI [21%,100%]) | 100% (n=1, CI [21%,100%]) |
| geometry/hard | 0% (n=1, CI [0%,79%]) | 0% (n=1, CI [0%,79%]) | 0% (n=1, CI [0%,79%]) | 0% (n=1, CI [0%,79%]) |
| word_problems/easy | 100% (n=1, CI [21%,100%]) | 100% (n=1, CI [21%,100%]) | 100% (n=1, CI [21%,100%]) | 100% (n=1, CI [21%,100%]) |
| word_problems/medium | 100% (n=1, CI [21%,100%]) | 100% (n=1, CI [21%,100%]) | 100% (n=1, CI [21%,100%]) | 100% (n=1, CI [21%,100%]) |
| word_problems/hard | 100% (n=1, CI [21%,100%]) | 100% (n=1, CI [21%,100%]) | 100% (n=1, CI [21%,100%]) | 100% (n=1, CI [21%,100%]) |
| fractions_decimals/easy | 100% (n=1, CI [21%,100%]) | 100% (n=1, CI [21%,100%]) | 100% (n=1, CI [21%,100%]) | 100% (n=1, CI [21%,100%]) |
| fractions_decimals/medium | 100% (n=1, CI [21%,100%]) | 100% (n=1, CI [21%,100%]) | 100% (n=1, CI [21%,100%]) | 100% (n=1, CI [21%,100%]) |
| fractions_decimals/hard | n/a | n/a | n/a | n/a |
| ratios_proportions/easy | 100% (n=1, CI [21%,100%]) | 100% (n=1, CI [21%,100%]) | 100% (n=1, CI [21%,100%]) | 100% (n=1, CI [21%,100%]) |
| ratios_proportions/medium | 100% (n=1, CI [21%,100%]) | 100% (n=1, CI [21%,100%]) | 100% (n=1, CI [21%,100%]) | 100% (n=1, CI [21%,100%]) |
| ratios_proportions/hard | 100% (n=1, CI [21%,100%]) | 100% (n=1, CI [21%,100%]) | 100% (n=1, CI [21%,100%]) | 100% (n=1, CI [21%,100%]) |
| exponents_roots/easy | 100% (n=1, CI [21%,100%]) | 100% (n=1, CI [21%,100%]) | 100% (n=1, CI [21%,100%]) | 100% (n=1, CI [21%,100%]) |
| exponents_roots/medium | 100% (n=1, CI [21%,100%]) | 100% (n=1, CI [21%,100%]) | 100% (n=1, CI [21%,100%]) | 100% (n=1, CI [21%,100%]) |
| exponents_roots/hard | 0% (n=1, CI [0%,79%]) | 0% (n=1, CI [0%,79%]) | 0% (n=1, CI [0%,79%]) | 0% (n=1, CI [0%,79%]) |
| **macro avg** | 79% | 79% | 79% | 84% |

## Worst 3 cells (ranked by `value_selected` arm)
- arithmetic/medium: 0%
- arithmetic/hard: 0%
- geometry/hard: 0%

## Ledger traceability (one sample per arm)
- **value_selected**: question:q-10da41663c52 -> attempt:att-q-10da41663c52-teacher-hi-0-ec4248 -> sample:smp-q-10da41663c52-sft-batch-d84e89e3f3-a9f619
- **baseline_random**: question:q-4bae40b4f47e -> attempt:att-q-4bae40b4f47e-teacher-hi-0-ec4248 -> sample:smp-q-4bae40b4f47e-sft-baseline-d84e89e3f3-f36d96
