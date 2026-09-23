# JEV-Benchmark 设计（当前策略）

> **目标**：在公开金标决策任务上对比 **Jev（System One）** 与 **同档小参数 LLM**，并给出 **Jev ≈ xB–yB** 的能力括号。  
> 本文件与实现一致；旧的「仅 Jev / 七模型 OpenRouter 阵容」已作废。

---

## 1. 评测问题

1. **质量**：Choice / Score / Noul 上谁更准？  
2. **校准**：Brier / ECE 谁更诚实？  
3. **工程**：格式合法率、延迟、成本。  
4. **能力括号（主结论）**：Jev 落在 Qwen3.5 尺寸阶梯的哪一档？

**不测**：生成、长链推理、多跳、数学。

---

## 2. 金标（已固化，seed=42）

| ID | 数据集 | 原语 | 主表 n | 出处 |
|---|---|---|---:|---|
| `banking77` | BANKING77 | Choice 77 类 | **300** | Casanueva et al., EMNLP 2020 |
| `clinc150` | CLINC150 + OOS | Choice 150+oos | **200**（约 15% OOS） | Larson et al., EMNLP-IJCNLP 2019 |
| `sst5` | SST-5 | Score 0–4 | **200** | Socher et al., EMNLP 2013 |
| `boolq` | BoolQ | Noul | **200** | Clark et al., NAACL 2019 |
| `agnews` | AG News | Choice 4 类 | **150** | Zhang et al., NeurIPS 2015 |

**合计 1050**。字段见 `jevbench/schemas.py`；`datasets/*.meta.json` 含 citation / sha256 / source_url。

历史下载源：jsDelivr + hf-mirror。重新下载 BoolQ parquet 需安装 `requirements-fetch.txt` 中的 pyarrow；公开仓库不直接分发原始题目文本。

---

## 3. 模型阵容（当前）

### 基线
| id | API | 说明 |
|---|---|---|
| `jev` | OpenRouter `typesafe/jev-1.13` → `POST /v1/systemone` | System One 原生 choice/score/noul |

### 小参数阶梯（硅基流动，OpenAI 兼容 chat）
| id | model | 定位（官方） |
|---|---|---|
| `qwen3.5-4b` | `Qwen/Qwen3.5-4B` | 4B 稠密，入门 |
| `qwen3.5-9b` | `Qwen/Qwen3.5-9B` | 9B 稠密，小档最强 |
| `qwen3.5-27b` | `Qwen/Qwen3.5-27B` | **27B 稠密**，中档主力 |
| `qwen3.5-35b` | `Qwen/Qwen3.5-35B-A3B` | MoE 35B/**激活 3B**，官方弱于 27B |
| `qwen3.5-122b` | `Qwen/Qwen3.5-122B-A10B` | MoE 122B/**激活 10B**，官方中档最强 |

追加完整上界候选：`qwen3.8-27b`（`Qwen/Qwen3.8-27B`，SiliconFlow，同样关闭思考）。`qwen3.5-397b` 曾在 OpenRouter 尝试运行，但中途因 HTTP 402 余额不足中断，**不纳入完整评测比较**；逐题痕迹仍保留。

模型规模与路由形态不能代替本评测结果；35B/122B 为 MoE，不能把总参数直接与稠密 27B 排序。

密钥：`SILICONFLOW_API_KEY`（`.env`，不入库）；Jev 用 `OPENROUTER_API_KEY`。

---

## 4. LLM 协议（与 Jev 语义对齐）

- `temperature=0`，**`enable_thinking=false`**（一律关思考）  
- `response_format: json_object`  
- 输出契约：`{"answer", "probabilities", "confidence"}`  
- 大选项集允许 **稀疏 probabilities**（缺省填 0 再归一化；answer 无质量则用 confidence 补）  
- **格式一次解析，失败不重试**；网络/429/5xx 重试 ≤4 次并单独计数  
- 选项描述与金标 `labels` / `level_descriptions` 字面一致  
- `oos_tau=0.35` **仅适用于选项集中含 `oos` 的任务**；无 OOS 选项的 Choice 使用模型选择的有效标签。

指标：Acc / Macro-F1 / Score MAE·Exact·±1 / Noul Acc·AUC·F1 / Brier / ECE(10) / Format% / p50·p95 / $/1k。  
端到端质量口径：**format/网络失败计错**；另报合法样本 Acc。

---

## 5. 已完成结果（n=1050，e2e Acc）

| 任务 | Jev | Q3.5 4B | 9B | 27B | 35B | 122B | Q3.8 27B |
|---|---:|---:|---:|---:|---:|---:|---:|
| banking77 | 0.777 | 0.643 | 0.693 | **0.780** | 0.743 | 0.723 | 0.767 |
| clinc150 | **0.905** | 0.735 | 0.830 | 0.885 | 0.840 | 0.865 | 0.885 |
| agnews | 0.867 | 0.813 | 0.773 | **0.873** | 0.827 | 0.860 | 0.860 |
| sst5 | 0.530 | 0.430 | 0.490 | **0.580** | 0.485 | 0.550 | 0.530 |
| boolq | **0.890** | 0.825 | 0.865 | 0.865 | 0.880 | 0.870 | 0.875 |

配对结论：
- 以五任务等权端到端准确率为主指标，Jev 比 Qwen3.5 4B、9B、35B 分别高 **10.23、6.53、3.87 个百分点**；配对置换检验双侧 `p` 分别为 0.0002、0.0002、0.0016。
- Qwen3.5-27B 减 Jev 为 **+0.30 个百分点**，95% 配对 bootstrap 区间 **[−1.70, +2.43]**；当前样本未检出差异，**这不是等价性证明**。
- Qwen3.5-122B 减 Jev 为 **−2.00 个百分点**，区间 **[−4.30, +0.30]**，双侧 `p=0.0876`。
- Qwen3.8-27B 减 Jev 为 **−1.03 个百分点**，区间 **[−3.00, +0.93]**，双侧 `p=0.3245`；单侧“Qwen 胜出” `p=0.8440`。它在五项任务中均未超过 Jev（SST-5 持平）。

**主结论（能力括号）**  
> 在本套五项快速决策任务上，Jev 的观测表现接近 **Qwen3.5-27B**；显著高于 9B，但“等价于 27B”尚未经过预设等价性检验。  
> 完整评测的 Qwen3.5 4B–122B-A10B 和 Qwen3.8-27B 中，**尚未测得统计上胜过 Jev 的模型，因此能力上界仍未定**。

本轮按用户指定到 Qwen3.8-27B 为止。不能把“没有测到上界”写成“Jev 能力没有上界”。逐题哈希和统计方法见 `results/statistics_with_upper.json` 与 `EXPERIMENT_LOG.md`。

---

## 6. 代码与产物

```
JEV-benchmark/
├── DESIGN.md              # 本文件
├── datasets/              # 1050 金标 JSONL + meta + README
├── fetch.py               # 冻结金标
├── jevbench/
│   ├── schemas.py         # Sample / Prediction
│   ├── config.py          # 模型注册表、常量
│   ├── prompts.py         # LLM JSON 契约 / Jev request
│   ├── adapters.py        # Jev systemone + SiliconFlow chat
│   ├── metrics.py         # acc / brier / ece / …
│   ├── runner.py          # 批量评测（增量 checkpoint）
│   ├── repair.py          # 补跑失败槽位
│   └── aggregate.py       # 从 raw 重算 summary_merged.json
├── scripts/run_*_live.command   # 终端可视化跑批
├── results/raw/*.jsonl    # 逐模型×数据集预测
├── results/summary_merged.json
├── results/statistics_with_upper.json  # 逐题配对统计及输入哈希
├── results/runs/                 # 新运行的不可变快照
└── EXPERIMENT_LOG.md             # 实验决策与中断记录
```

常用命令：
```bash
python3 -m jevbench.runner --models jev --datasets banking77 --limit 5
python3 -m jevbench.repair --model qwen3.5-27b --dataset clinc150
python3 -m jevbench.aggregate
```

跑长任务：用 `scripts/*_live.command` 开终端（可见进度）；runner 每 5 条写盘，可断点续跑。

---

## 7. 公平性与局限（报告必写）

1. LLM 概率为 JSON 外挂，非原生校准；Jev 为 System One 原生概率。  
2. 公开分类/情感/问答金标 ≠ 工单生产分布。  
3. 35B/122B 为 MoE，**激活参数**小于总参数；勿把总参数当「稠密等价」。  
4. 硅基可能 429：失败与格式错误分开；必要时 `repair` 补跑。  
5. 延迟为供应商 E2E（含排队），非模型纯算力。  
6. 成本为 token×未独立归档的历史牌价占位值，**非实际账单**；Qwen3.8-27B 的 USD 成本缺失。跨供应商延迟只作客户端观测描述。完整表见 `report/RESULTS.md`。

---

## 8. 状态

| 项 | 状态 |
|---|---|
| 金标 1050 | ✅ |
| Jev + 5 档 Qwen 全量 | ✅ |
| 能力括号 | ✅ **≈27B 级** |
| 配对统计与留痕 | ✅ `statistics_with_upper.json`、运行清单与快照 |
| Qwen3.8-27B 全量 | ✅ 1,050 条；未测得上界 |
| 报告 Markdown | ✅ `report/RESULTS.md` |
| 可复现 SVG 图表 | ✅ `report/figures/`，由 `scripts/build_publication.py` 生成 |
| 公开版离线统计核验 | ✅ `results/reanalysis.csv` 与 `scripts/verify_release.py` |
| fan-out 效率 | ⬜ 未做 |
| 交互对比台 | ⬜ 未做 |
| 更高模型上界 | ⬜ 本轮按用户要求停止扩展 |
