# 金标数据集固化记录

生成命令：`"$MIMO_PYTHON" fetch.py`（seed=42）

| 文件 | 条数 | 任务 | 划分 | 标签 | SHA-256 前 12 位 | 论文 |
|---|---:|---|---|---|---|---|
| `datasets/banking77.jsonl` | 300 | choice | test | 77 意图 | `0862f30acbb3` | Casanueva et al., EMNLP 2020 |
| `datasets/clinc150.jsonl` | 200 | choice | test + oos_test | 150 + **oos**（30 条 OOS，15%） | `aaac80ce9be5` | Larson et al., EMNLP-IJCNLP 2019 |
| `datasets/sst5.jsonl` | 200 | score | test | 档位 0–4 | `74592bb80173` | Socher et al., EMNLP 2013 |
| `datasets/boolq.jsonl` | 200 | noul | validation | true/false（124/76） | `6743afe1653e` | Clark et al., NAACL 2019 |
| `datasets/agnews.jsonl` | 150 | choice | test | World/Sports/Business/SciTech | `b9d554701d43` | Zhang et al., NeurIPS 2015 |

本表记录当前主评测的 1,050 条冻结样本；完整 SHA-256 见对应 `*.meta.json`。早期较大抽样规模已不用于主结果。

**许可修正**：冻结文件中的 `license` 字段是当时的采集记录，不应视为授权声明。CLINC150 上游许可证为 CC BY 3.0，BoolQ 上游声明 CC BY-SA 3.0；SST-5 和 AG News 的文本再分发许可未明确核实。公开仓库不再分发这五份原文，完整来源说明见 [DATA_PROVENANCE.md](../DATA_PROVENANCE.md)。

## 统一样本字段

见 `jevbench/schemas.py`（`Sample`）：`id, dataset, split, text, lang, task_type, label`，Choice 另有 `labels` + `label_descriptions`，Score 另有 `levels` + `level_descriptions`，`meta` 含 `citation` / `source_url` / `source_row`。

## 下载源（本机网络实测）

| 数据集 | 可用源 |
|---|---|
| banking77 | `hf-mirror.com/datasets/mteb/banking77` → `test.jsonl` |
| clinc150 | `cdn.jsdelivr.net/gh/clinc/oos-eval` → `data_full.json` |
| sst5 | `hf-mirror.com/datasets/SetFit/sst5` → `test.jsonl` |
| boolq | `hf-mirror.com/datasets/google/boolq` → validation **parquet**（需 pyarrow） |
| agnews | `cdn.jsdelivr.net/gh/mhjabreel/CharCnn_Keras` → `test.csv` |

注：`raw.githubusercontent.com` / `huggingface.co` 本机不通；BoolQ parquet 依赖 `/tmp/jev_pydeps`（Tsinghua 源安装的 pyarrow，**未写入 MIMO_PYTHON**）。

## 抽样规则

- Choice：按标签分层，比例配额 + seed=42  
- CLINC150：域内分层 170 + 域外固定 30（oos_frac=0.15）  
- Score：按档位分层 200  
- BoolQ：validation 全集 shuffle 后取 200  
- AG News：四类各约 37–38 条  

完整元数据见各 `datasets/*.meta.json`（含 citation、license、source_url、sha256）。
