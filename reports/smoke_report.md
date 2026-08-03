# Smoke Report

## Question generation
- Seed lines read: 7
- Exact-dedup dropped: 1
- Train questions: 5
- Holdout questions: 1

## Teacher collection
- Teacher attempts total (this run's train pool): 15 (newly collected: 0)
- Student(weak) attempts total: 5 (newly collected: 0)
- Teacher cost total: $0.0000
- Cost per question: $0.0000

## Verification / selection
- Teacher pass rate (p_T, high-temp attempts): 100.00%
- Student(weak) pass rate (p_S): 100.00%
- SFT samples selected (rejection sampling, 1 per solved question): 5

## Holdout eval (MockStudent)
| cell | question_id | result |
|---|---|---|
| arithmetic/medium | q-68ff8bf0b712 | PASS |

## Ledger traceability example
Sample `smp-q-b9f80495a743-sft-batch-31218c7873-8688a0` traced root-first:

- `question` **q-b9f80495a743** (upstream: none)
- `attempt` **att-q-b9f80495a743-teacher-hi-0-e1a376** (upstream: ['q-b9f80495a743'])
- `sample` **smp-q-b9f80495a743-sft-batch-31218c7873-8688a0** (upstream: ['att-q-b9f80495a743-teacher-hi-0-e1a376'])
