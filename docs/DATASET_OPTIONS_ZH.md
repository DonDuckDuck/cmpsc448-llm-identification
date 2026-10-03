# 新数据集可行性核查（2026-10-10）

这是一份下一轮实验的选材记录，不是已经完成的新实验。当前报告仍使用原有 UltraFeedback 数据和已验证结果。

## 优先候选：Arena Human Preference 140k

- [官方数据集](https://huggingface.co/datasets/lmarena-ai/arena-human-preference-140k)
- [官方发布说明](https://arena.ai/blog/opendata-july2025)
- 固定版本：`6322995ab34d7c2693e3f47dd13fa5caa0789a74`。
- 当前公开版本实际为 135,634 条对战/投票记录，而非 140,000 条独立问题。数据时间为 2025 年 4—7 月；比现用数据的模型版本新，但不是 2026 年最新模型集合。
- 已核实可匿名读取，包含 model_a/model_b 与 conversation_a/conversation_b。原始文件约 1.61 GB。
- 按官方完整统计（partial=false）汇总双方模型标签：

| 家族 | 模型侧记录数（清洗前） | 实际版本举例 |
|---|---:|---|
| Claude | 57,441 | Sonnet 4、Opus 4、3.7 Sonnet |
| GPT | 21,474 | GPT-4.1、ChatGPT-4o；未包含 o 系列 |
| Grok | 16,393 | Grok 3、Grok 3 mini、Grok 4 |
| DeepSeek | 12,671 | V3-0324、R1-0528 |

总计 107,979 个模型侧记录，不能直接当作独立训练样本数。详细版本计数和可复查端点见 arena_feasibility_audit.json。

### 建议的下一轮设计

保留当前配对实验作为历史模型基线，再做 Claude/GPT/Grok/DeepSeek 四分类扩展。先核实英语、首轮、纯文本、非空回答和去重之后的各类规模，再确定样本量，不预先承诺最终数量或准确率。

1. 优先提取第一轮回答，避免把其他模型写的历史上下文误标为当前模型输出。胜负标签不能替代作者标签。
2. 按 evaluation_session_id 和重复/近重复问题分组划分训练、验证、测试；同一场对战的两侧始终位于同一划分。
3. 控制语言、主题、时间、模型版本及 reasoning 模式差异；去除自报模型身份的直接线索。
4. Arena 通常只有两个模型回答同一个问题，不保证四个家族都有同题回答。因此不能照搬原实验的完整三家族配对结论，也不能宣称 prompt-only 必然为 25%。必须实测 prompt-only 基线。
5. 在未查看新测试结果前冻结划分和模型选择规则。重新训练 CNN、LSTM；报告 macro-F1、balanced accuracy、各类召回率、多种子波动、混淆矩阵和学习曲线。
6. 先完成同版本比较，再考虑版本留出评估；不要把新任务的准确率与原三分类准确率直接解释为性能提升。

可选固定非推理版本在筛选前分别有：GPT-4.1 6,540、Claude Sonnet 4 8,295、Grok 3 preview 6,040、DeepSeek V3-0324 6,117 个模型侧记录。Grok 4 只有 1,561 个，说明“更新版本”和“更大平衡样本”之间存在取舍。

官方数据卡说明用户问题采用 CC-BY-4.0，模型输出另受各提供商条款约束。延续现有项目方式：发布代码、统计和来源记录，原始对话留在本地。

## 其他候选

- [ShareChat](https://huggingface.co/datasets/tucnguyen/ShareChat)：官方列出 142,808 个会话，覆盖 ChatGPT、Claude、Grok、Gemini、Perplexity，没有 DeepSeek。Claude 只有 946 个会话；不同平台的问题和界面特征存在混杂。原始文件需要登录接受数据集条件。本次没有下载或接受条件。适合另行研究跨平台差异，不是此次四家族任务的首选。
- [WildChat-4.8M](https://huggingface.co/datasets/allenai/WildChat-4.8M)：非毒性公开子集为 3,199,860 个会话，来源是 ChatGPT；虽然体量更大，但缺少此次要比较的多个家族。

选择结论：目前核实过的候选里，Arena 140k 最适合扩展本作业的模型家族与样本规模。数据更大或版本更新不保证分类准确率更高。
