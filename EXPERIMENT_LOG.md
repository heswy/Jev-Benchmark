# 实验记录

本文件只记录实验决策和可核对的产物；逐题结果保留在 `results/raw/` 和新运行的 `results/runs/`。本项目是在已有 1,050 题、30 份历史预测的基础上继续，不替换旧结果。

公开仓库中完整模型的 `results/raw/` 收于 `results/raw-complete.tar.gz`，已完成的 Qwen3.8 运行快照和旧摘要收于 `results/history.tar.gz`；两个压缩包的 SHA-256 记录于 `results/ARCHIVES.sha256`。397B 中断文件只保留在原本地项目中，不进入公开压缩包或主榜。原始题目文本因上游许可差异不随仓库发布，见 `DATA_PROVENANCE.md`。

## 2026-09-22：历史基线

- 五份冻结金标与各自 `*.meta.json` 的 SHA-256 一致。
- Jev 与 Qwen3.5 4B、9B、27B、35B-A3B、122B-A10B 的 30 份原始预测均覆盖各自全部样本 ID，共 6 × 1,050 条。
- 旧汇总 `results/summary_merged.json` 的质量指标只按有效预测计算。历史运行未留下每次独立的原始快照、供应商版本或完整请求；不可凭后来的源码哈希补称历史运行完全可复现。

## 2026-09-23：冻结统计口径

- 新增 `jevbench.statistics`，直接读取金标与逐题预测，校验哈希、ID 唯一性与完整性。失败预测计错。
- 预先固定主指标：五项任务端到端准确率等权平均。SST-5 以概率最高档位是否命中计算精确准确率；BoolQ 以 `p(true) >= 0.5` 判定；Choice 沿用 `oos_tau=0.35`。
- 逐数据集报告配对 bootstrap 区间和 McNemar 精确检验；主指标报告分层配对 bootstrap 区间和配对置换检验。固定随机种子 42、默认各 5,000 次。
- 预先列出两个首选上界候选：`qwen3.8-27b`（与旧 Qwen 同用 SiliconFlow），`qwen3.5-397b`（OpenRouter，同代更大但供应商不同）。在两者全量结果揭晓前，追加备用候选 `qwen3.8-2.4t`（OpenRouter），仅当前两者未胜出才运行。只在主指标高于 Jev 且一侧置换检验经三个候选 Bonferroni 校正后 `p < 0.05` 时称为统计上界。其他新模型若在看到结果后加入，应标为探索性。

## 2026-09-23：上界运行

- SiliconFlow 模型只读列表确认 `Qwen/Qwen3.8-27B` 可用。对四类题型做接口探针，均得到可解析预测。全量命令与旧版源码哈希记录于 `results/manifests/20260923T005308Z_qwen3.8-27b.json`。该运行使用旧 runner，因此命令中的 `--tag` 受旧版优先级错误影响，在摘要中显示 `full`；命令原文保留在清单中。运行中新增了归档与标签修复代码，不改变该已启动进程的调用协议。
- OpenRouter 的 `qwen/qwen3.5-397b-a17b` 初始接口探针沿用 `reasoning.effort=low`，四类题中三类把 800 输出 token 耗在思考内容而没有 JSON。此探针不计入正式评测。随后对该候选显式设为 `reasoning.effort=none`；四类探针均有效，另一次 BoolQ 探针返回 `reasoning_tokens=0`。正式运行使用无思考设置。供应商不同及 OpenRouter 未固定具体承载商，结论需注明。
- 397B 正式运行在 CLINC150 中途收到 OpenRouter `HTTP 402` 余额不足，随后 SST-5 的多数已提交请求也返回 402。已停止运行并把已落盘的文件归档于 `results/runs/20260923T010052Z_upper-qwen35-397b-v1/`，清单标记为 `interrupted_insufficient_credits`。已完成的 BANKING77 为 300 条；CLINC150 为 200 条（其中 67 条 402）；SST-5 仅落盘 140 条（其中 87 条 402）。这些记录保留作失败审计，不纳入完整五任务主指标，也不把 402 当模型能力错误。
- SiliconFlow 的模型列表中没有原定备用 `Qwen3.8-2.4T-A95B`，但有 `deepseek-ai/DeepSeek-V4-Pro`。在 3.8-27B 全量结果完成前，固定以 DeepSeek V4 Pro 替换不可用的备用候选；仍按三个候选校正。若运行，它是跨家族上界，不能解释成 Qwen 参数规模上界。
- 用户随后明确将本轮工作收敛到 Qwen3.8-27B，并要求不要使用 DeepSeek V4。此前 DeepSeek V4 仅做过四道接口格式探针，**没有全量运行或写入逐题结果**；不再发起其他候选运行。397B 不完整运行同样不参与能力结论。最终报告对完整的 Qwen3.8-27B 使用单候选配对检验，同时保留本日志中计划变更的时间顺序。
- Qwen3.8-27B 完成 1,050 条；有效 1,047 条（BANKING77 有 3 条格式失败），五份 raw 与金标的样本 ID 均完整且唯一。`results/manifests/20260923T005308Z_qwen3.8-27b.json` 和 `results/runs/20260923T005308Z_upper-qwen38-27b-v1/` 保存命令、代码/数据哈希、逐题快照和原生成摘要。旧 runner 把未知 USD 牌价显示为 0；这是成本缺失，不是免费。后来的 `summary_merged.json` 将其记为 `unavailable`，不改写原生成摘要。
- 以预设五任务等权端到端准确率比较，Jev 为 0.7937，Qwen3.8-27B 为 0.7833，差值（Qwen 减 Jev）为 −0.0103，配对 bootstrap 95% 区间 [−0.0300, +0.0093]；单侧 Qwen 胜出置换 `p=0.8440`。本轮**未测得 Jev 的能力上界**。完整逐任务结果与输入哈希见 `results/statistics_with_upper.json`。

## 2026-09-23：OOS 适用范围修正

- 原 `evaluate_dataset` 与配对统计对所有 Choice 都应用 `oos_tau=0.35`，即使选项集中没有 `oos`。现在仅当选项集包含 `oos` 时使用阈值；其余 Choice 按合法的模型 `answer` 评分。
- 不改金标、不改任何逐题预测、不发起新模型请求。重算后 BANKING77 的 Qwen3.5-4B 和 27B 各恢复 1 条正确；其余模型与数据集的正确数未变。Jev 对 Qwen3.8-27B 的主结论和数值均不变。统计与汇总 JSON、设计文档及报告已同步更新；回归测试覆盖含 OOS 和不含 OOS 两种情形。
- Git 仓库于本日就地初始化；未删除原有文件。公开发布前还需确定代码许可并逐项核对原数据再分发许可。
