"""Refresh the repository landing page and Chinese result sheet from audited metrics."""
from pathlib import Path
import json
ROOT=Path(__file__).resolve().parents[1]
S=json.loads((ROOT/'results/arena/summary.json').read_text());A=json.loads((ROOT/'results/arena/data_audit.json').read_text())
F=json.loads((ROOT/'results/arena_followup/posthoc.json').read_text());C=json.loads((ROOT/'results/arena_followup/curve.json').read_text());D=json.loads((ROOT/'results/arena_followup/domain.json').read_text())['domains']
def pct(x):return f'{100*x:.2f}%'
rows=[]
for arch in ['cnn','lstm']:
 for mode in ['input','output','combined']:
  r=S['core'][arch+'_'+mode];rows.append(f'| {arch.upper()} | {mode} | {pct(r["accuracy_mean"])} | {pct(r["macro_f1_mean"])} | {pct(r["balanced_accuracy_mean"])} |')
table='\n'.join(rows)
text=f'''# Who Wrote It?

### Identifying LLMs from Their Responses

**Haodong Huang · CMPSC 448 · Individual midterm project**

The main study now uses **Arena Human Preference 140k**: **{A['retained_rows']:,} curated English, single-turn responses** from four specific model releases. Required CNN and RNN/LSTM models are trained, selected on validation data, and evaluated across three seeds.

![Arena neural results](results/figures/arena_results.png)

## Main results

Means across seeds 1448, 1449, and 1450 on the same {A['splits']['test']['rows']:,}-response test split. Macro-F1 is the primary metric; per-class recall, seed variation, and group-bootstrap intervals are in the report.

| Model | Input | Accuracy | Macro-F1 | Balanced accuracy |
|---|---|---:|---:|---:|
{table}

The character TF-IDF baseline scores **{pct(S['baselines']['char_output']['accuracy'])} accuracy / {pct(S['baselines']['char_output']['macro_f1'])} macro-F1**. Its point estimate is slightly higher than the CNN mean; it is retained as a strong, inexpensive comparator.

The majority-class accuracy is **{pct(S['majority_accuracy'])}**. Uniform random accuracy and constant-class balanced accuracy are 25%. Prompt-only macro-F1 is not mathematically fixed at 25% in this unpaired four-source subset.

## Follow-up experiments (after the main evaluation)

Run after the frozen main study, under a plan written first ([docs/arena_followup_plan.json](docs/arena_followup_plan.json)); no setting was chosen with test labels. Report Section 11 has details.

| Question | Finding |
|---|---|
| Does more training data help? | CNN macro-F1 {pct(C['summary']['0.125']['cnn']['macro_f1_mean'])} → {pct(C['summary']['1']['cnn']['macro_f1_mean'])} and LSTM {pct(C['summary']['0.125']['lstm']['macro_f1_mean'])} → {pct(C['summary']['1']['lstm']['macro_f1_mean'])} going from {C['rows']['0.125']:,} to {C['rows']['1']:,} training responses (fixed split). |
| Is it just length? | With every answer cut to exactly 128 tokens: CNN {pct(F['equal_length']['cnn']['macro_f1'])}, LSTM {pct(F['equal_length']['lstm']['macro_f1'])} macro-F1. |
| Same prompt, two authors? | On {F['matched_pairs']['pairs']} identical-prompt test pairs the CNN assigns both authors correctly {pct(F['matched_pairs']['cnn']['pairwise_assignment'])} of the time; prompt-only is exactly 50%. |
| Where is the fingerprint? | Opening window ≫ middle; CNN keys on openers like "Certainly! Here's" (GPT), "Below is a detailed" (Grok), "I'd recommend" (Claude), bold numbered headers and "Would you like" (DeepSeek). |
| Transfer to unseen task types (RQ3, 3 seeds)? | Trained without coding, CNN macro-F1 on coding is {pct(D['code']['cnn']['holdout_macro_f1_mean'])} vs. {pct(D['code']['cnn']['control_macro_f1_mean'])} with matched mixed training; trained without creative writing (the assignment's example), {pct(D['writing']['cnn']['holdout_macro_f1_mean'])} vs. {pct(D['writing']['cnn']['control_macro_f1_mean'])} on writing. |
| Ensemble? | Equal-weight CNN+LSTM+char TF-IDF: {pct(F['ensemble']['equal']['macro_f1'])} macro-F1 ({100*F['ensemble']['equal']['minus_char']:+.2f} pp over char TF-IDF). |

![Learning curve](results/figures/followup_learning_curve.png)

## What data are actually used?

| Label | Exact model ID | Retained responses |
|---|---|---:|
'''
for f in S['labels']:text+=f'| {f} | `{next(m for m,v in A["models"].items() if v==f)}` | {A["family_counts"][f]:,} |\n'
text+=f'''
Source: [Arena dataset](https://huggingface.co/datasets/lmarena-ai/arena-human-preference-140k), pinned revision `6322995ab34d7c2693e3f47dd13fa5caa0789a74`. The complete release has 135,634 battles from April-July 2025. The four selected IDs contribute 26,992 sides before filtering. We retain all eligible English, first-evaluation, strict single-turn text answers after length, sensitive-pattern, and duplicate checks. Complete model IDs and family names are masked.

The target is **four fixed releases**, not every member of their families or the latest 2026 models. User prompts are CC-BY-4.0; the source card notes separate provider terms for model outputs. Raw conversations are neither committed nor reproduced in the report.

## Experimental controls

- Train / validation / test: **{A['splits']['train']['rows']:,} / {A['splits']['val']['rows']:,} / {A['splits']['test']['rows']:,}** responses. Sessions, exact/near-duplicate prompts, and exact repeated answers cannot cross splits.
- Training-only vocabularies, TF-IDF, scalers, and inverse-frequency class weights.
- Four validation-screen candidates; selected CNN/LSTM recipes repeated across all three input conditions and three seeds.
- **RQ1:** response-only author classification. **RQ2:** prompt vs. response vs. combined input.
- **RQ3:** non-code training and coding test, compared with a mixed-domain control matched by each training-class count; one seed per diagnostic condition.
- **RQ4:** six lexical/structural baselines and neural word-only normalization ablations.
- Shuffled-training-label CNN negative control; independent validation/test rescoring and saved-model replay.

The current study contains **{S['neural_fits']} neural fits + {len(S['baselines'])} classical fits**, with **{S['test_runs']} test prediction files**. Rejected screen candidates have no test scores. [Complete run ledger](results/arena/run_ledger.csv) · [Verification](results/arena/verification.json)

## Reproduce

Use Python 3.12. Create a virtual environment and install the pinned dependencies:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-lock.txt
python src/arena_data.py download
python src/arena_data.py prepare
python src/arena_train.py train
python src/arena_train.py domain
python src/arena_train.py evaluate
python src/arena_analyze.py
python src/arena_followup.py plan
python src/arena_followup.py posthoc
python src/arena_followup.py curve
python src/arena_followup.py domain
python -m unittest discover -s tests -v
python src/build_lessons_figures.py
python src/build_report.py
python src/build_readme.py
```

Reproduce in a fresh clone. `prepare` validates the published manifest and refuses to overwrite existing split text. The roughly 1.61 GB download and full training checkpoints stay local. Retained final-study fit/validation time was {S['fit_seconds']/60:.1f} minutes on the original machine, excluding downloading, curation, analysis, reporting, and superseded development fits. Different hardware may take longer and produce slightly different results. See [reproducibility details](docs/REPRODUCIBILITY.md).

## Demonstrate a saved model

```sh
python src/predict.py --response 'Your example answer goes here.'
python src/predict.py --checkpoint models/arena_lstm_output_s1448.pt --response 'Your example answer goes here.'
```

Scores are **uncalibrated closed-set outputs**, not reliable probabilities of authorship for arbitrary text. The supplied CNN and LSTM examples use seed 1448; reported main results average three separately trained models, not the demonstration alone.

## Interpret the results carefully

Arena does not give every prompt to all four selected models. Language, generation settings, task mix, and platform conventions can affect attribution. The prompt-only control measures some of that risk. The CNN reads 384 answer tokens; LSTM reads 192, so their comparison concerns complete recipes rather than architecture in isolation. Normalization also changes token composition and truncation. Test-group intervals condition on fitted models and do not capture all training, selection, or distribution-shift uncertainty.

The test set was untouched by model training and selection before the reported evaluation. Any future tuning using these published test scores must be identified as follow-up work and needs a new test set for confirmation.

## Project evolution and dataset-change rationale

We changed datasets for two reasons: **(1) study the more recognizable Claude/GPT/Grok/DeepSeek model families requested for this project; (2) increase the usable sample, hoping that more training evidence would improve accuracy.** The second is a hypothesis, not a demonstrated causal result. Model-family relevance does not imply a measured ranking of dataset popularity; this release contains 2025 data.

| Stage | Dataset / classes | Curated responses | CNN accuracy | LSTM accuracy |
|---|---|---:|---:|---:|
| Initial implementation | UltraFeedback / Falcon, GPT, Llama | 4,254 | 77.52% | 64.01% |
| Same-data optimization | Same UltraFeedback data and split | 4,254 | 80.91% | 77.00% |
| Current study | Arena / Claude, GPT, Grok, DeepSeek | {A['retained_rows']:,} | {pct(S['core']['cnn_output']['accuracy_mean'])} | {pct(S['core']['lstm_output']['accuracy_mean'])} |

All scores are response-only, three-seed means within each stage. Same-data optimization increased CNN/LSTM accuracy by **3.39/12.99 percentage points**, but was exploratory follow-up on an already evaluated test set. Arena provides **2.27 times as many curated responses**, with explicit model-release selection, session/duplicate grouping, frozen validation selection, and matched-size domain controls. Its scores are **not directly comparable** to the earlier three-class task: author classes, distributions, and split designs changed. A fixed-split Arena learning curve would be needed to isolate the effect of training-set size; it has not been run.

[Initial report](report/archive/Who_Wrote_It_Haodong_Huang_v1.pdf) · [Optimized historical report](report/archive/Who_Wrote_It_Haodong_Huang_v2.pdf) · [Historical methods](docs/ITERATION_NOTES_ZH.md)

## Repository guide

| Path | Purpose |
|---|---|
| `src/arena_data.py` | Pinned acquisition, filtering, grouping, and split generation |
| `src/arena_train.py` | Validation selection, core/diagnostic fitting, frozen test evaluation |
| `src/models_v2.py` | CNN and pooled LSTM, with explicit four-class output support |
| `src/arena_analyze.py` | Independent audits, grouped intervals, figures, and checkpoint replay |
| `src/arena_followup.py` | Follow-up: learning curve, length/position/matched-prompt diagnostics, ensemble, CNN n-grams, RQ3 extension |
| `results/arena_followup/` | Follow-up metrics and logs |
| `tests/` | Sixteen checks of leakage, attribution, padding, sequence handling, and follow-up subsetting |
| `results/arena/` | Complete metrics, validation probabilities, non-text test predictions |
| `docs/arena_*.json` | Data provenance, curation repair, plans, selection, and freeze records |
| `data/arena/split_manifest.json` | Non-text published split assignments |
| `models/arena_*.pt` | Two locally usable demonstration checkpoints |
| `report/` | Current report plus archived earlier reports |

Course grounding: Chapter 4 (CNN/pooling/LSTM) and Chapter 9 (loss/mini-batches/Adam). Specific budgets, filters, and hyperparameters are project design decisions. The submission deliverable is this repository URL and its PDF; publishing it does not submit it to Canvas.
'''
(ROOT/'README.md').write_text(text)
zh='# Arena 实验结果速览\n\n作者：Haodong Huang。主报告维持老师原题。以下数字由实际测试结果生成。\n\n'
zh+='数据：9,643 条回答；训练/验证/测试 = 6,735/1,442/1,466；四个固定版本 Claude Sonnet 4、GPT-4.1、Grok 3、DeepSeek V3。\n\n'
zh+='| 模型 | 输入 | 准确率 | Macro-F1 | Balanced accuracy |\n|---|---|---:|---:|---:|\n'+table+'\n\n'
for a in ['cnn','lstm']:
 c=S['contrasts'][a+'_prompt_gain'];zh+=f'{a.upper()} 加入问题后的 macro-F1 差值：{100*c["macro_f1_difference"]:+.2f} 个百分点；95% 组重采样区间 [{100*c["ci95"][0]:+.2f}, {100*c["ci95"][1]:+.2f}]。\n\n'
zh+='以上为三个种子的均值。跨领域、格式消融与打乱标签为单种子诊断。新旧数据集的标签数、版本和划分不同，不要把旧三分类与新四分类的分数直接解释为提升。完整解释与局限见英文报告及 ARENA_PROTOCOL_ZH.md。\n'
zh+=f'''
## 补充实验（主实验评估之后）

计划先写入 docs/arena_followup_plan.json，再运行；没有用测试标签选择任何设置。测试集此前已用过，因此属于后续分析，不是第二次确认。

- 学习曲线（固定划分）：训练数据从 {C['rows']['0.125']:,} 增加到 {C['rows']['1']:,} 条，CNN macro-F1 {pct(C['summary']['0.125']['cnn']['macro_f1_mean'])} → {pct(C['summary']['1']['cnn']['macro_f1_mean'])}，LSTM {pct(C['summary']['0.125']['lstm']['macro_f1_mean'])} → {pct(C['summary']['1']['lstm']['macro_f1_mean'])}。
- 去掉长度信息（每条回答都截成正好 128 token）：CNN {pct(F['equal_length']['cnn']['macro_f1'])}，LSTM {pct(F['equal_length']['lstm']['macro_f1'])}。
- 同一问题、两个作者的 {F['matched_pairs']['pairs']} 对：CNN 正确分配 {pct(F['matched_pairs']['cnn']['pairwise_assignment'])}；只看问题必然是 50%。
- 指纹位置：开头窗口最强，中间最弱；CNN 依赖开场白与 Markdown 习惯。
- 等权融合 CNN+LSTM+字符 TF-IDF：macro-F1 {pct(F['ensemble']['equal']['macro_f1'])}。
'''
(ROOT/'docs/ARENA_RESULTS_ZH.md').write_text(zh)
print('README and Chinese results refreshed from audited Arena metrics')
