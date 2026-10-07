# Who Wrote It?

### Identifying LLMs from Their Responses

**Haodong Huang · CMPSC 448 · Individual midterm project**

The main study now uses **Arena Human Preference 140k**: **9,643 curated English, single-turn responses** from four specific model releases. Required CNN and RNN/LSTM models are trained, selected on validation data, and evaluated across three seeds.

[Read the current PDF](report/Who_Wrote_It_Haodong_Huang.pdf) · [Editable report](report/report.md) · [中文结果](docs/ARENA_RESULTS_ZH.md) · [中文方法与答辩](docs/ARENA_PROTOCOL_ZH.md)

![Arena neural results](results/figures/arena_results.png)

## Main results

Means across seeds 1448, 1449, and 1450 on the same 1,466-response test split. Macro-F1 is the primary metric; per-class recall, seed variation, and group-bootstrap intervals are in the report.

| Model | Input | Accuracy | Macro-F1 | Balanced accuracy |
|---|---|---:|---:|---:|
| CNN | input | 26.94% | 26.31% | 26.69% |
| CNN | output | 86.79% | 86.51% | 86.49% |
| CNN | combined | 86.54% | 86.18% | 86.22% |
| LSTM | input | 25.15% | 24.06% | 25.25% |
| LSTM | output | 83.04% | 82.52% | 82.63% |
| LSTM | combined | 81.20% | 80.79% | 80.95% |

The character TF-IDF baseline scores **87.65% accuracy / 87.33% macro-F1**. Its point estimate is slightly higher than the CNN mean; it is retained as a strong, inexpensive comparator.

The majority-class accuracy is **28.72%**. Uniform random accuracy and constant-class balanced accuracy are 25%. Prompt-only macro-F1 is not mathematically fixed at 25% in this unpaired four-source subset.

## What data are actually used?

| Label | Exact model ID | Retained responses |
|---|---|---:|
| Claude | `claude-sonnet-4-20250514` | 2,774 |
| GPT | `gpt-4.1-2025-04-14` | 2,388 |
| Grok | `grok-3-preview-02-24` | 2,201 |
| DeepSeek | `deepseek-v3-0324` | 2,280 |

Source: [Arena dataset](https://huggingface.co/datasets/lmarena-ai/arena-human-preference-140k), pinned revision `6322995ab34d7c2693e3f47dd13fa5caa0789a74`. The complete release has 135,634 battles from April-July 2025. The four selected IDs contribute 26,992 sides before filtering. We retain all eligible English, first-evaluation, strict single-turn text answers after length, sensitive-pattern, and duplicate checks. Complete model IDs and family names are masked.

The target is **four fixed releases**, not every member of their families or the latest 2026 models. User prompts are CC-BY-4.0; the source card notes separate provider terms for model outputs. Raw conversations are neither committed nor reproduced in the report.

## Experimental controls

- Train / validation / test: **6,735 / 1,442 / 1,466** responses. Sessions, exact/near-duplicate prompts, and exact repeated answers cannot cross splits.
- Training-only vocabularies, TF-IDF, scalers, and inverse-frequency class weights.
- Four validation-screen candidates; selected CNN/LSTM recipes repeated across all three input conditions and three seeds.
- **RQ1:** response-only author classification. **RQ2:** prompt vs. response vs. combined input.
- **RQ3:** non-code training and coding test, compared with a mixed-domain control matched by each training-class count; one seed per diagnostic condition.
- **RQ4:** six lexical/structural baselines and neural word-only normalization ablations.
- Shuffled-training-label CNN negative control; independent validation/test rescoring and saved-model replay.

The current study contains **27 neural fits + 6 classical fits**, with **31 test prediction files**. Rejected screen candidates have no test scores. [Complete run ledger](results/arena/run_ledger.csv) · [Verification](results/arena/verification.json)

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
python -m unittest discover -s tests -v
python src/build_report.py
python src/build_readme.py
```

Reproduce in a fresh clone. `prepare` validates the published manifest and refuses to overwrite existing split text. The roughly 1.61 GB download and full training checkpoints stay local. Retained final-study fit/validation time was 12.5 minutes on the original machine, excluding downloading, curation, analysis, reporting, and superseded development fits. Different hardware may take longer and produce slightly different results. See [reproducibility details](docs/REPRODUCIBILITY.md).

## Demonstrate a saved model

```sh
python src/predict.py --response 'Your example answer goes here.'
python src/predict.py --checkpoint models/arena_lstm_output_s1448.pt --response 'Your example answer goes here.'
```

Scores are **uncalibrated closed-set outputs**, not reliable probabilities of authorship for arbitrary text. The supplied CNN and LSTM examples use seed 1448; reported main results average three separately trained models, not the demonstration alone.

## Interpret the results carefully

Arena does not give every prompt to all four selected models. Language, generation settings, task mix, and platform conventions can affect attribution. The prompt-only control measures some of that risk. The CNN reads 384 answer tokens; LSTM reads 192, so their comparison concerns complete recipes rather than architecture in isolation. Normalization also changes token composition and truncation. Test-group intervals condition on fitted models and do not capture all training, selection, or distribution-shift uncertainty.

The test set was untouched by model training and selection before the reported evaluation. Any future tuning using these published test scores must be identified as follow-up work and needs a new test set for confirmation.

## Historical study

The earlier UltraFeedback experiments used 4,254 responses from Falcon/GPT/Llama. They remain in the repository as a separate historical three-class study. Their 80.91% CNN / 77.00% LSTM optimized accuracies are **not directly comparable** to this four-class Arena task.

[Initial report](report/archive/Who_Wrote_It_Haodong_Huang_v1.pdf) · [Optimized historical report](report/archive/Who_Wrote_It_Haodong_Huang_v2.pdf) · [Historical methods](docs/ITERATION_NOTES_ZH.md)

## Repository guide

| Path | Purpose |
|---|---|
| `src/arena_data.py` | Pinned acquisition, filtering, grouping, and split generation |
| `src/arena_train.py` | Validation selection, core/diagnostic fitting, frozen test evaluation |
| `src/models_v2.py` | CNN and pooled LSTM, with explicit four-class output support |
| `src/arena_analyze.py` | Independent audits, grouped intervals, figures, and checkpoint replay |
| `tests/` | Thirteen checks of leakage, attribution, padding, and sequence handling |
| `results/arena/` | Complete metrics, validation probabilities, non-text test predictions |
| `docs/arena_*.json` | Data provenance, curation repair, plans, selection, and freeze records |
| `data/arena/split_manifest.json` | Non-text published split assignments |
| `models/arena_*.pt` | Two locally usable demonstration checkpoints |
| `report/` | Current report plus archived earlier reports |

Course grounding: Chapter 4 (CNN/pooling/LSTM) and Chapter 9 (loss/mini-batches/Adam). Specific budgets, filters, and hyperparameters are project design decisions. ChatGPT/Codex assisted with code, debugging, analysis, and drafting; the student is responsible for understanding and verifying the submission. The submission deliverable is this repository URL and its PDF; publishing it does not submit it to Canvas.
