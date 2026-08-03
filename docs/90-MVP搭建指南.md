# 90｜MVP 搭建指南：多卡服务器上的最小蒸馏闭环

> 系列：[总览](00-总览-蒸馏关键环节与学习路线.md)
> 阅读时间：约 25 分钟
> 前提：多卡服务器（按 8×A100/H100 级别写，4 卡可缩配）；目标任务 = Agent/工具轨迹（优先）+ 代码/数学可验证任务 + 领域问答
> 本篇集中全系列所有 MVP 内容；各环节的原理见对应编号文档。

---

## 一、搭建思路：骨架先行 + 冒烟直通

三条实施原则：

1. **框架全量占位**：仓库目录结构按最终形态一次建好，MVP 不用的模块保留空实现（raise NotImplementedError + docstring 说明未来职责），后续填肉不动骨架；
2. **MVP 路径写到可直接动手**：本篇第五、六节是操作手册级别，照着敲即可；
3. **配好 API 就能跑通一个用例**：`smoke` 命令把"出题→教师作答→验证→选择→编译→评测报告"整条链在**分钟级、几毛钱 API 费**内跑一遍；学生模型用 mock 接口顶替，不需要 GPU 就能联调全管线。

**mock 的边界（重要）**：mock 学生只验证*管线正确性*（数据流、schema、断点续跑、报告生成），不产出任何*训练结论*。难度坐标 p_S、师生差距选择、训练对比、评测数字，全部要等真学生（vLLM 起服务）接入后才有意义。文档在每个步骤标注了 `[mock可跑]` / `[需真学生]` / `[需GPU训练]`。

### MVP 要验证什么

> 跑通"问题生成 → 教师作答/走轨迹 → 沙盒验证 → 过滤选择 → SFT → 隔离评测 → 诊断补题"的完整闭环，并证明**闭环选出的数据在固定预算下稳定优于分层随机数据**。

三阶段路线（P1 是 P2 的基础设施联调，与"Agent 优先"的目标不冲突）：

| 阶段 | 任务域 | 新增能力 | 周期估计 |
|---|---|---|---|
| P0 | 冒烟用例 | 骨架 + 全链路直通（mock 学生） | 2–3 天 |
| P1 | 代码+数学（可验证任务） | 真学生接入、拒绝采样、SFT、对照评测 | 1–2 周 |
| P2 | Agent 工具轨迹 | 沙盒环境、harness、轨迹 schema、终态验证 | 2–3 周 |
| P3 | 领域问答 + on-policy KD | 文档反推出题、证据验证、GKD | 2 周 |

刻意不做的：多教师路由、token-KD、RLVR、跨 tokenizer、K8s 分布式编排、自动课程。这些都等闭环先赢了随机基线再说（原理见各文档"常见坑"）。

## 二、技术栈选型（含开源实现分析）

### 2.1 一览表（推荐项加粗）

| 层 | 推荐 | 备选 | MVP阶段 |
|---|---|---|---|
| 教师（API） | 你常用的前沿模型 API，经 **LiteLLM** 统一协议 | — | P0 起 |
| 教师（本地） | **Qwen3-32B 级开源强模型 @ vLLM** | DeepSeek 蒸馏系列 | P1 起 |
| 学生 | **Qwen3-8B**（P0 用 mock 顶替） | Llama 3.x 8B | P1 起 |
| 推理服务 | **vLLM**（offline batch + server 两用） | SGLang（多轮共享前缀强，P2 可换） | P1 起 |
| 问题生成管道 | **Bespoke Curator** | distilabel | P0 起 |
| 沙盒 | P2 先**进程内工具**（无需 Docker）→ 后升级 **Docker 容器池** | E2B/Modal（省运维） | P2 |
| Agent 环境抽象 | **verifiers** 的接口规范（reset/step/verify） | 自定义 Gym 风格 | P0 定接口 |
| Agent harness | **自写 tool-loop（~200 行，OpenAI 兼容协议）** | 改造 SWE-agent/OpenHands | P2 |
| 训练 | **TRL**（SFT→DPO→GKD 同框架递进） | LLaMA-Factory / verl（大规模 OPD） | P1 起 |
| 评测 | **自建隔离集 harness + lm-eval-harness 回归** | lighteval / Inspect | P0 定格式 |
| 数据存储 | **jsonl/parquet + DuckDB 查询** | 数据库（后期） | P0 起 |
| 实验追踪 | W&B 或 MLflow + 内容哈希 | — | P1 起 |

### 2.2 关键开源实现分析

**训练框架**：

- [TRL](https://github.com/huggingface/trl)——SFTTrainer/DPOTrainer/GKDTrainer 齐备，P1→P3 不换框架。GKDTrainer 实现了 on-policy 混合比例 λ 和 JSD 插值 β。缺点：大规模 RL/OPD 吞吐一般。**MVP 首选**。
- [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)——配置驱动零代码 SFT/DPO，适合当"第二意见"快速复现；深度定制（自定义 loss mask、分段加权）要翻源码。
- [verl](https://github.com/volcengine/verl)——生产级 RL/OPD：教师独立 GPU 资源池、GKD 式 top-k 前向 KL 与 PG 式反向 KL、多教师路由（[OPD 文档](https://verl.readthedocs.io/en/latest/algo/opd.html)）。**P3 认真做 on-policy 时迁入**。
- [DistillKit](https://github.com/arcee-ai/DistillKit)——token-KD 细节的读源码材料，不当生产框架。

**数据生成管道**：

- [Bespoke Curator](https://github.com/bespokelabsai/curator)——自动缓存（改代码重跑不重复扣费）、失败重试、结构化输出。**MVP 首选**：蒸馏管线 80% 的代码本质是"可靠地大量调模型"。
- [distilabel](https://github.com/argilla-io/distilabel)——现成组件多（EvolInstruct/Magpie 等），抽象层厚；想套现成策略时选它。
- [synthetic-data-kit](https://github.com/meta-llama/synthetic-data-kit)——P3 文档→QA 可直接用或抄 prompt。
- [NeMo Curator](https://docs.nvidia.com/nemo/curator/latest/)——单独用它的去重模块也值。

**Agent 环境/轨迹（P2 核心）**：

- [verifiers](https://github.com/PrimeIntellect-ai/verifiers)——环境标准化 + Rubric + OpenAI 兼容 rollout；同一环境可被 prime-rl/TRL 训练复用。**接口规范直接采用**，避免自创 schema 后重构。
- [SWE-smith](https://github.com/SWE-bench/SWE-smith) + [SWE-ReX](https://github.com/SWE-agent/SWE-ReX)——代码仓库类 Agent 场景直接复用。
- [tau-bench](https://github.com/sierra-research/tau-bench)——模拟用户型多轮环境参考。
- [Agent Data Protocol](https://arxiv.org/html/2510.24702v1)——轨迹 schema 对齐它。

**端到端参考（整体抄作业对象）**：

- [open-r1](https://github.com/huggingface/open-r1)——vLLM 批量生成→过滤→SFT→评测全套，**结构最值得抄**。
- [OpenThoughts](https://github.com/open-thoughts/open-thoughts)——数据管线消融的方法论。
- [Thinking Machines on-policy distillation](https://thinkingmachines.ai/blog/on-policy-distillation/)——P3 GKD 实验设计参照。

### 2.3 GPU 分配（8 卡示例；P0 不需要 GPU）

```text
数据生产期：4 卡 vLLM 教师(32B, TP=4) + 2 卡 vLLM 学生(评测/难度实测) + 2 卡机动/探针训练
训练期：    4-8 卡 学生全参 SFT（8B, FSDP/DeepSpeed ZeRO-2, bf16）
GKD 期(P3)：4 卡教师推理池 + 4 卡学生训练
CPU 侧：    沙盒容器池吃 CPU/内存，与 GPU 任务错峰或分机
```

## 三、仓库骨架（一次建全，标注实现状态）

图例：✅ = P0 冒烟即实现；🔶 = P1/P2/P3 实现；⬜ = 占位空实现（NotImplementedError + docstring）

```text
distill-mvp/
├── README.md                    ✅ 快速开始 + 本图例
├── .env.example                 ✅ TEACHER_API_KEY / TEACHER_MODEL / STUDENT_MODE=mock|vllm
├── pyproject.toml               ✅ 依赖：litellm, bespokelabs-curator, duckdb, datasketch, pydantic, typer
├── run.py                       ✅ CLI 入口：smoke / gen / collect / verify / select / compile / eval / report
├── configs/
│   ├── smoke.yaml               ✅ 冒烟配置（5 题、k=2、mock 学生）
│   ├── p1_math_code.yaml        🔶 P1 正式配置
│   ├── p2_agent.yaml            ⬜
│   └── p3_domain.yaml           ⬜
├── core/
│   ├── schemas.py               ✅ 全部 pydantic 模型：Question/Attempt/Trajectory/Sample/EvalResult
│   ├── model_client.py          ✅ 统一模型接口 + MockStudent + LiteLLM 教师实现
│   ├── ledger.py                ✅ ID 体系与追溯：问题↔轨迹↔样本↔批次↔checkpoint↔评测
│   └── hashing.py               ✅ 配置/prompt/工具 schema 的内容哈希
├── taxonomy/
│   ├── matrix.yaml              ✅ 覆盖矩阵定义（冒烟版只有 2×2 格子）
│   └── quota.py                 ✅ 配额分配
├── datagen/
│   ├── seeds/                   ✅ 种子题目录（冒烟自带 5 道数学题）
│   ├── generate.py              ✅ 出题管道（冒烟=直读种子；P1 接 Curator 生成）
│   ├── dedup.py                 🔶 MinHash+语义+模板族去重（冒烟只做精确去重）
│   └── evolve.py                ⬜ Evol 难度爬坡
├── collect/
│   ├── sample.py                ✅ 每题 k 采样编排（asyncio + 断点续跑 + 全量落盘）
│   ├── harness/
│   │   ├── tool_loop.py         🔶 P2: OpenAI tools 协议循环，师生共用
│   │   └── budget.py            🔶 P2: 步数/token/超时预算
│   └── sim_user.py              ⬜ P2+: 模拟用户
├── envs/
│   ├── base.py                  ✅ Environment 抽象（reset/step/verify/snapshot）——只定接口
│   ├── toy_calc/                🔶 P2 冒烟环境：进程内计算器+文件工具，无需 Docker
│   ├── sandbox/pool.py          ⬜ Docker 容器池管理
│   └── registry.py              ✅ 环境注册表
├── verify/
│   ├── base.py                  ✅ Verifier 抽象 + 验证结果 schema
│   ├── math_answer.py           ✅ 数学答案检查（math-verify 封装）
│   ├── code_exec.py             🔶 P1: 单测执行（subprocess 起步，P2 换沙盒）
│   ├── env_state.py             🔶 P2: 环境终态检查（转调 env.verify）
│   ├── evidence.py              ⬜ P3: 证据定位
│   └── judge.py                 ⬜ P3: 校准 LLM Judge
├── select_/
│   ├── rejection.py             ✅ 拒绝采样：k 候选→验证过滤→选最短正确解
│   ├── value.py                 🔶 P1: 师生差距+覆盖配额综合选择
│   └── baselines.py             🔶 P1: 分层随机对照组导出
├── compile_/
│   ├── to_sft.py                ✅ → SFT jsonl（学生模板渲染；P2 加 obs mask）
│   ├── to_dpo.py                ⬜
│   └── to_gkd_prompts.py        ⬜
├── train/
│   ├── sft.py                   🔶 P1: TRL SFTTrainer 脚本（多卡 accelerate 配置在 configs/）
│   ├── dpo.py                   ⬜
│   └── gkd.py                   ⬜ P3
├── evals/
│   ├── holdout/                 ✅ 隔离集目录（冒烟自带 5 题；入库即冻结）
│   ├── run_eval.py              ✅ 跑评测：模型客户端×题集×验证器→分格结果
│   ├── report.py                ✅ 分格报告 + 对比表（markdown 输出到 reports/）
│   └── regression/              🔶 P1: lm-eval-harness 配置
├── reports/                     ✅ 每轮产物：评测报告/错误分类账/下轮数据订单
└── data/                        ✅ 运行时产物（jsonl/parquet），.gitignore
```

两个命名注意：`select`/`compile` 是 Python 关键字/内建，目录名加下划线。`core/model_client.py` 是整个骨架的枢纽，见下节。

## 四、接口契约（先定死，各模块围绕它开发）

### 4.1 统一模型接口（mock 的关键）

教师、真学生、mock 学生实现同一个接口，管线其余部分完全无感：

```python
# core/model_client.py
class ModelClient(Protocol):
    model_id: str          # 含版本，进 ledger
    async def chat(self, messages: list[dict], *, tools: list[dict] | None = None,
                   temperature: float = 0.7, max_tokens: int = 4096) -> ChatResult: ...
    # ChatResult: {content, tool_calls, finish_reason, usage, logprobs|None}

class TeacherClient(ModelClient):   # LiteLLM 封装：重试/限流/落盘/计费
class VLLMClient(ModelClient):      # 指向 vLLM OpenAI server（教师本地版 & 真学生共用）
class MockStudent(ModelClient):     # P0：三种可配模式
    # mode="echo"     返回固定模板答案（测数据流）
    # mode="weak"     调一个便宜 API 小模型冒充弱学生（测评测/差距逻辑，花几分钱）
    # mode="replay"   从指定 jsonl 回放答案（测确定性、CI 可用）
```

`STUDENT_MODE=mock|vllm` 环境变量切换，代码零改动——这就是你第 3 点的实现机制。

### 4.2 Environment 抽象（P0 只定接口，P2 实现）

```python
# envs/base.py —— 对齐 verifiers 的语义，方便后期直接迁移
class Environment(ABC):
    def reset(self, task_spec: TaskSpec) -> Observation: ...
    def step(self, action: ToolCall) -> tuple[Observation, bool]: ...
    def verify(self) -> RewardInfo: ...        # 终态可执行判据
    def snapshot(self) -> bytes: ...           # P2 重放用
    version: str                               # 镜像/工具schema哈希
```

### 4.3 核心数据 schema（pydantic，P0 全部定义）

```python
Question(id, text, cell, family_id, source, verify_method, gold=None)
Attempt(id, question_id, actor_model, sampling, messages, usage, ts)      # 一次作答
VerifyResult(attempt_id, passed, detail, verifier_id)
Sample(id, question_id, attempt_id, kind="sft|dpo|gkd_prompt", payload, batch_id)
EvalRecord(model_id, question_id, passed, cell, run_id)
```

ledger 规则：任何产物必带上游 ID + 生成配置哈希。这 100 行是后期归因（`06`）和资产复用（`07`）的地基。

## 五、P0 冒烟用例：配好 API 直接跑（2–3 天搭完，跑一次几分钟）

### 5.1 你要做的

```bash
git init distill-mvp && cd distill-mvp   # 按第三节建骨架
cp .env.example .env                     # 填 TEACHER_API_KEY、TEACHER_MODEL
pip install -e .
python run.py smoke                      # 一条命令
```

### 5.2 `smoke` 命令内部依次执行（全部 [mock可跑]）

```text
1. gen      读 datagen/seeds/ 的 5 道数学题（附金标准答案）→ data/questions.jsonl
            （其中 1 道故意重复，验证 dedup 拦截；1 道进 evals/holdout/ 不进训练池）
2. collect  教师 API 每题 k=2 采样（T=0.8）+ 1 次低温 → data/attempts.jsonl
            同时 MockStudent(mode=weak) 每题 1 答 → 学生 attempts
3. verify   math_answer 验证所有 attempts → data/verify.jsonl
4. select   拒绝采样：每题选 1 条最短正确教师解；报告 p_T、p_S
5. compile  → data/sft.jsonl（学生 chat 模板渲染，可直接喂 TRL）
6. eval     MockStudent 跑 holdout → evals 结果
7. report   → reports/smoke_report.md：
            题数/去重数/教师通过率/每题成本/样本数/学生 holdout 分格成功率/全链路 ledger 追溯样例
```

**通过标准**：报告生成且数字自洽（如教师 5 题 k=2 共 10 次作答、验证通过≥8、sft.jsonl 4 条——holdout 那题不在内）；随便挑一条 sft 样本，能用 ledger 从 sample_id 追回 question → attempt → verify 记录；中途 Ctrl-C 再跑 `smoke` 能从断点续（落盘幂等）。

### 5.3 P2 的第二个冒烟（骨架期可先不做，接口留好）

`python run.py smoke --env toy_calc`：教师走 3 步工具轨迹（计算器+写文件），`toy_calc` 是**进程内** Python 工具（不需要 Docker），终态验证=检查文件内容。它验证 tool_loop、Trajectory schema、obs mask 编译三件事。Docker 沙盒等 P2 正式开始再上。

## 六、P1 实施步骤（真学生接入，1–2 周）

1. **D1 起服务** [需GPU]：2 卡 vLLM 起 Qwen3-8B OpenAI server；`.env` 切 `STUDENT_MODE=vllm`。跑 `python run.py eval --model student` 确认与 mock 路径无缝切换；
2. **D1–2 隔离集**：领域内 300–500 题（模板族隔离），底座跑 pass@1/pass@8 基线 + lm-eval 回归基线。**没有基线前不生产训练数据**；
3. **D3–4 出题**：`datagen/generate.py` 接 Curator；覆盖矩阵先粗（代码 10 知识点×3 难度，数学同理）；来源=公开集迁移（污染检查）+ Evol 变体 + 程序化生成；目标 5k–20k 题全带验证方式；
4. **D5–7 教师采集** [需GPU或纯API]：本地 32B 教师（4 卡 TP=4）offline batch，每题 k=8+1 低温；难题（通过率<25%）升级 API 教师；同步实测学生 p_S；
5. **D8 选择**：拒绝采样→师生差距+配额选出训练集（3k–10k）；`baselines.py` 同规模分层随机对照集；
6. **D9–10 训练** [需GPU训练]：`train/sft.py`（TRL 全参，accelerate/ZeRO-2），两个数据集各训（相同 token 预算、相同超参搜索、3 seeds）；
7. **D11–12 评测判定**：分格报告+回归；"闭环选择>分层随机"成立→P1 达标，产出错误分类账与下轮订单；不成立→按 `04` 消融排查（最常见：验证器漏放、去重不彻底、隔离集泄漏）。

## 七、P2 要点（Agent 轨迹，2–3 周）

填肉 `envs/`、`collect/harness/`：

1. **环境**：toy_calc 之后，挑 2–3 类贴业务的工具环境（文件/数据处理、内部 API 模拟、检索）；任务 spec 程序化生成保证判据可执行（`02` 3.3）；镜像/工具 schema 哈希版本化；此时上 Docker 容器池（预热池+断网+资源限额，`03`）；
2. **harness**：tool_loop 师生共用；预算：15 步/64k token/单工具 30s；失败轨迹全留；
3. **采集**：每任务教师 k=4 rollout；终态验证通过进拒绝采样池；抽 5% 重放质检；
4. **编译**：obs 打 loss mask（TRL `assistant_only_loss` 或自定义 collator）——**拿一条样本人工数 token 核对一次**，错了整批报废；
5. **训练评测**：轨迹数据 + 30% P1 数据混训（防单轮回归）；同环境终态成功率（每任务 4 次）+ 步数/无效调用率 + 异常切片（工具报错注入、不需工具的题）；
6. 产能：单机 64 并发容器一天数千条轨迹；瓶颈通常在教师 API 限流（多 key+错峰）。

## 八、P3 要点（领域问答 + on-policy，2 周）

1. **领域问答**：文档反推出题（synthetic-data-kit 或 Curator 管道）；验证=证据定位（答案在源文档可定位支撑段落）+ 校准 Judge（`verify/judge.py` 此时实现）；
2. **on-policy KD**：P1/P2 失败题做 prompt 集，TRL GKDTrainer（λ=0.5 起，教师=本地 32B）；对照=同预算继续 SFT。GKD 组 pass@1 显著更优而 pass@8 不变 → exposure bias 被修复，符合预期（`05`）；
3. 吞吐不够或上 RLVR 时迁 verl（环境按 verifiers 规范写，迁移成本低）。

## 九、预算粗算（量级参考）

```text
P0 冒烟：API 几毛钱，无 GPU
P1 教师侧：本地 32B 为主 ≈ GPU 电费；API 难题补充 1–3k 美元量级
P2 轨迹：2 万条生成、~30% 入库，API 教师为主 3–10k 美元量级（本地教师走量可大幅压缩）
训练：8B 全参 SFT(10k×3seeds×2组) ≈ 8 卡×数小时/次
人力才是大头：P0–P3 全程 1–2 人 × 6–8 周
```

## 十、成功标准

```text
P0：smoke 全链路直通，报告数字自洽，ledger 可追溯，断点可续
硬标准：
1. 闭环选择组在隔离集上稳定优于分层随机组（3 seeds，宏平均+最差格）
2. Agent 任务终态成功率相对底座显著提升，效率指标（步数/无效调用）不劣化
3. 回归集降幅在预算内
软标准：
4. 评测报告 → 下一轮训练数据就绪 ≤ 3 天
5. 任一训练样本 5 分钟内追溯到轨迹、环境版本、验证记录
```

达标后的扩展（按 ROI 排序）：on-policy KD 常态化 → 失败-恢复轨迹专项 → 多教师复核 → RLVR（仅高价值域）→ 部署工件化与路由（`07`）。
