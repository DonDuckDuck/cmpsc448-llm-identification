"""Build the PDF and matching Markdown report from measured result artifacts."""
from pathlib import Path
import json,re,html
import numpy as np
from reportlab.pdfgen import canvas
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,Image,PageBreak,KeepTogether
from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from matplotlib.font_manager import findfont,FontProperties
from PIL import Image as PILImage
ROOT=Path(__file__).resolve().parents[1]
S=json.loads((ROOT/'results/summary.json').read_text());A=json.loads((ROOT/'results/data_audit.json').read_text())
P=json.loads((ROOT/'docs/data_provenance.json').read_text())
V=json.loads((ROOT/'results/v2/summary.json').read_text())
def result(t):return json.loads((ROOT/'results'/f'{t}.json').read_text())
def pct(x):return f'{100*x:.1f}%'
def pp(x):return f'{100*x:+.2f}'
def interval(xs):return f'[{100*xs[0]:.2f}, {100*xs[1]:.2f}]'
for n,weight in [('Body','normal'),('Bold','bold')]:
    pdfmetrics.registerFont(TTFont(n,findfont(FontProperties(family='DejaVu Sans',weight=weight))))
pdfmetrics.registerFontFamily('Body',normal='Body',bold='Bold',italic='Body',boldItalic='Bold')
TEAL=colors.HexColor('#0B7774');INK=colors.HexColor('#172C3B');GRAY=colors.HexColor('#526474');PALE=colors.HexColor('#EDF5F5')
styles=getSampleStyleSheet()
styles.add(ParagraphStyle(name='TitleCustom',fontName='Bold',fontSize=29,leading=35,textColor=INK,spaceAfter=15))
styles.add(ParagraphStyle(name='HeadingCustom',fontName='Bold',fontSize=18,leading=23,textColor=INK,spaceAfter=13))
styles.add(ParagraphStyle(name='SubCustom',fontName='Bold',fontSize=11,leading=15,textColor=TEAL,spaceBefore=8,spaceAfter=5))
styles.add(ParagraphStyle(name='BodyCustom',fontName='Body',fontSize=9.4,leading=14,textColor=INK,spaceAfter=8))
styles.add(ParagraphStyle(name='SmallCustom',fontName='Body',fontSize=8,leading=11.3,textColor=GRAY,spaceAfter=6))
styles.add(ParagraphStyle(name='CellCustom',fontName='Body',fontSize=8.1,leading=11,textColor=INK))
styles.add(ParagraphStyle(name='TableHeadCustom',fontName='Bold',fontSize=8,leading=11,textColor=colors.white))
story=[];md=[];W=516

def p(text,small=False):
    story.append(Paragraph(text,styles['SmallCustom' if small else 'BodyCustom']))
    md.append(re.sub('<[^>]+>','',html.unescape(text))+'\n')
def sub(text):
    story.append(Paragraph(text,styles['SubCustom']));md.append('### '+text+'\n')
def page(title):
    if story:story.append(PageBreak())
    story.append(Paragraph(title,styles['HeadingCustom']));md.append('\n## '+title+'\n')
def table(headers,rows,widths=None,small=False):
    widths=[x*W for x in widths] if widths else [W/len(headers)]*len(headers)
    cells=[[Paragraph(html.escape(str(c)),styles['TableHeadCustom']) for c in headers]]
    cells += [[Paragraph(html.escape(str(c)),styles['CellCustom']) for c in row] for row in rows]
    t=Table(cells,colWidths=widths,repeatRows=1,hAlign='LEFT')
    t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),TEAL),('VALIGN',(0,0),(-1,-1),'TOP'),
                          ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,PALE]),
                          ('LEFTPADDING',(0,0),(-1,-1),7),('RIGHTPADDING',(0,0),(-1,-1),7),
                          ('TOPPADDING',(0,0),(-1,-1),4 if small else 6),('BOTTOMPADDING',(0,0),(-1,-1),4 if small else 6),
                          ('LINEBELOW',(0,-1),(-1,-1),.4,colors.HexColor('#CFDEDF'))]))
    story.extend([t,Spacer(1,9)])
    md.extend(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |'])
    md.extend('| '+' | '.join(map(str,row))+' |' for row in rows);md.append('')
def fig(name,caption,maxheight=215):
    path=ROOT/'results/figures'/f'{name}.png'
    with PILImage.open(path) as im:iw,ih=im.size
    factor=min(W/iw,maxheight/ih);story.append(Image(str(path),width=iw*factor,height=ih*factor))
    story.append(Spacer(1,4));p(caption,small=True)
    md.append(f'![{caption}](../results/figures/{name}.png)\n')
def contrast(name):
    c=S['contrasts'][name];return f"{pp(c['accuracy_difference'])} percentage points; paired 95% CI {interval(c['ci95'])}"

story.append(Spacer(1,25))
story.append(Paragraph('WHOSE SLOP IS IT?',styles['TitleCustom']))
p('<b>Identifying LLM families under a matched-prompt evaluation</b>')
p('Haodong Huang | Individual project<br/>CMPSC 448 | October 10, 2026')
md.insert(0,'# Whose Slop Is It?\n')
story.append(Spacer(1,18))
sub('Abstract')
p(f'This project compares a text CNN and an LSTM recurrent neural network (RNN) for identifying Falcon, GPT, and Llama responses. A curated UltraFeedback subset contains {A["selected_prompts"]:,} shared prompts and {A["selected_rows"]:,} responses. Each prompt appears once per family; near-duplicate prompt groups remain within a single data split. Model-name tokens are masked, and no model metadata enter the classifiers. Initial response-only accuracies are {pct(S["core"]["cnn_output"]["accuracy_mean"])} (CNN) and {pct(S["core"]["lstm_output"]["accuracy_mean"])} (LSTM). A bounded validation-only search subsequently yields {pct(V["core"]["cnn_output"]["accuracy_mean"])} and {pct(V["core"]["lstm_output"]["accuracy_mean"])} across three seeds. All configurations are chosen before follow-up test evaluation. Prompt-only accuracy is exactly 33.3% by construction. Both initial and optimized input comparisons are reported. Structural baselines, normalization, and a source-holdout experiment test alternative explanations. These are results for selected historical models and this curated distribution, not a universal detector of contemporary LLMs.')
sub('Main findings')
table(['Question','Evidence and answer'],[
['RQ1: Can responses identify a family?',f'Optimized CNN: {pct(V["core"]["cnn_output"]["accuracy_mean"])}; optimized LSTM: {pct(V["core"]["lstm_output"]["accuracy_mean"])}; chance: 33.3%.'],
['RQ2: Does the prompt help?',f'Optimized combined-minus-response change: CNN {pp(V["contrasts"]["cnn_prompt_gain"]["accuracy_difference"])} pp; LSTM {pp(V["contrasts"]["lstm_prompt_gain"]["accuracy_difference"])} pp. Both intervals include zero.'],
['RQ3: Transfer across sources?', 'Initial-model probe: both exceed chance on a held-out source; differences from matched-size controls are uncertain.'],
['RQ4: What signals matter?', 'Initial-model probes: structural features predict above chance; formatting effects depend on the model and training recipe.']],[.31,.69])
p(f'Scope: 26 initial neural runs, 5 classical baselines, {V["fit_count"]} optimization-stage fits ({V["test_run_count"]} evaluated on test), 2 validation-only timing pilots, and 2,000 cluster-bootstrap resamples. The follow-up reuses the previously reported split; it is not a fresh blind test.',small=True)

page('1. Dataset and problem definition')
p('The target is a three-way family label, not response quality or whether the author is human. UltraFeedback stores an instruction and multiple model-tagged completions [1,2]. The source artifacts are pinned by revision and SHA-256 hash. Only instruction text and response text become features; ratings, critiques, model names, source labels, and system prompts are excluded from predictive inputs.')
table(['Family label','Eligible generator identifiers','Selection per prompt'],[
['Falcon','falcon-40b-instruct','One response'],['GPT','gpt-3.5-turbo; gpt-4','One eligible response, seeded choice'],['Llama','llama-2-7b-chat; llama-2-13b-chat; llama-2-70b-chat','One eligible response, seeded choice']],[.15,.53,.32])
p('If a prompt has multiple eligible responses within a family, a deterministic prompt-specific random choice selects one. Quality scores never guide selection. Llama-derived models such as Vicuna and WizardLM are not mixed into the chosen Llama label. Falcon has one version while GPT and Llama pool multiple versions; this asymmetry limits family-level conclusions.')
table(['Curation step','Prompt records'],[
['Original records scanned','63,967'],['Excluded ShareGPT source (real-user chat)','19,949'],['Excluded: no complete three-family set with usable answers','42,595'],['Excluded: email or API-key-like pattern','5'],['Final complete matched prompts','1,418'],['Final response-level examples','4,254']],[.77,.23],small=True)
p('Usable answers contain at least five whitespace-separated words. Prompts must contain at least three words. Screening is narrow and is not a comprehensive privacy guarantee. Raw and processed text stay out of the public submission bundle; reconstruction scripts and provenance are provided. The dataset card labels the release MIT; upstream source conditions still matter [1].',small=True)
table(['Split','Prompts','Responses','Each family'],[['Train','992','2,976','992'],['Validation','213','639','213'],['Test','213','639','213']],[.28,.24,.24,.24],small=True)
p('Training, validation, and test proportions are approximately 70/15/15. Seed 448 fixes data construction and splitting. All 1,418 eligible prompts are used; the preconfigured cap of 1,500 is not reached.',small=True)

page('2. Leakage controls and the prompt-only test')
sub('Keep dependent observations together')
p(f'Prompts are normalized with Unicode NFKC, lowercase, and collapsed whitespace for identity checks. Character 3-5-gram TF-IDF cosine similarity of at least 0.90 links near duplicates. Connected components define split groups. This detects {A["near_duplicate_prompt_pairs"]} similar prompt pairs and produces {A["duplicate_clusters"]:,} clusters. Long exact shared responses (at least ten words) also link their prompts before splitting. The resulting groups are disjoint across all splits.')
p('Duplicate detection uses the full unlabeled prompt collection solely to protect the split. This is not the predictive vocabulary: that vocabulary is fitted on training text only. A fixed threshold catches obvious near copies, not all semantic paraphrases.')
sub('Remove easy label cues')
p(f'Explicit names such as GPT, OpenAI, Llama, Meta, Falcon, and TII are replaced with the same [MODEL] token in prompts and responses. This affects {A["identity_affected_outputs"]} responses. It prevents direct name lookup but can still preserve cues that an identity was mentioned. Broad masking can also remove legitimate topic words; this is a documented preprocessing tradeoff.')
sub('Why 33.3% is exact here')
p('Every held-out prompt has three identical input texts, one carrying each family label. A deterministic prompt-only classifier must predict the same family for those three rows, so it gets exactly one correct. Its accuracy is therefore 1/3, regardless of which family it chooses. This is a deliberate negative control, not a claim that prompts are always uninformative in natural user traffic. Macro-F1 can differ because predicted family frequencies differ.')
sub('Remaining generation confound')
p('UltraFeedback varies instruction principles and custom system prompts during generation [1]. We do not feed these fields to the classifier, but their effects remain in the generated answer. Matching the user prompt does not make the full generation conditions identical. Consequently, learned signals can reflect the generation protocol as well as model identity.')
principles=sorted(set(k for v in A['principles_by_family'].values() for k in v))
table(['Generation principle','Falcon','GPT','Llama'],[[k]+[A['principles_by_family'][f].get(k,0) for f in ['Falcon','GPT','Llama']] for k in principles],[.46,.18,.18,.18],small=True)
p('Automated checks verify split disjointness, one response per family per prompt, train-only vocabulary construction, preserved response budget in combined inputs, and prediction invariance to extra padding.',small=True)

page('3. Initial models and training procedure')
p('The CNN follows the course treatment of convolution and pooling, adapted to a one-dimensional token sequence (Chapter 4, especially p. 36). LSTM fulfills the required RNN model; it follows the course LSTM material (Chapter 4, pp. 69-79). Mini-batch optimization and Adam follow Chapter 9, pp. 41-48 and 62-65 [3,4]. PyTorch supplies the differentiable layers and optimizer [5].')
table(['Component','Text CNN','Text LSTM'],[
['Token representation','Trainable 64-dimensional embeddings','Trainable 64-dimensional embeddings'],
['Sequence processing','64 filters for each width 3, 4, 5; ReLU','One unidirectional layer; 64 hidden units'],
['Sequence summary','Masked global max pooling; 192 features','Hidden state at the final real token'],
['Prediction','Dropout 0.30; linear layer; 3 logits','Dropout 0.30; linear layer; 3 logits'],
['Output-only parameter count',f'{result("cnn_output_s448")["parameters"]:,}',f'{result("lstm_output_s448")["parameters"]:,}']],[.29,.355,.355],small=True)
p('Text is lowercased and split into word and punctuation tokens; newline markers preserve layout. The vocabulary keeps training tokens occurring at least twice, capped at 12,000 entries including reserved tokens. Unknown tokens map to UNK and PAD has embedding index zero. Input-only uses up to 96 tokens; output-only uses up to 192. Combined input reserves 96 prompt tokens, a separator, and 192 response tokens, so a long prompt cannot eliminate the answer.')
p('Training minimizes multiclass cross-entropy: L = -(1/B) sum log p(correct family | text). Settings: batch size 64, Adam learning rate 0.001, weight decay 0.0001, gradient norm clip 1.0, maximum 8 epochs, and patience 3. The selected checkpoint maximizes validation macro-F1; ties favor lower validation loss. Test performance is computed only after checkpoint selection. No hyperparameter sweep is performed.')
p('Primary seeds are 448, 449, and 450, with a fixed split. Auxiliary runs use seed 448. Python, NumPy, PyTorch, and the data-loader generator are seeded. Hardware: Apple M4 Pro, 24 GB memory, MPS backend. MPS numerical behavior is not guaranteed bitwise identical across systems.',small=True)
fig('learning_curves','Figure 1. Response-only learning curves for the predesignated seed 448; all runs and checkpoints are retained.',maxheight=168)

page('4. Initial experiments: RQ1 and RQ2')
p('The six initial conditions share the same examples and split. Accuracy and macro-F1 are reported as mean plus/minus sample standard deviation across three training seeds. Intervals resample near-duplicate clusters, carrying all matched answers together, and average the metric over the same three fitted models. They quantify held-out-sample uncertainty conditional on these runs, not uncertainty over all datasets or all possible training seeds.')
rows=[]
for model in ['cnn','lstm']:
 for mode in ['input','output','combined']:
    x=S['core'][model+'_'+mode]
    rows.append([model.upper(),{'input':'Prompt','output':'Response','combined':'Both'}[mode],f'{100*x["accuracy_mean"]:.2f} +/- {100*x["accuracy_sd"]:.2f}',f'{100*x["macro_f1_mean"]:.2f} +/- {100*x["macro_f1_sd"]:.2f}',interval(x['accuracy_ci95'])])
table(['Model','Input','Accuracy (%)','Macro-F1 (%)','95% accuracy CI'],rows,[.12,.14,.25,.25,.24],small=True)
fig('main_comparison','Figure 2. Three-seed mean accuracy with 95% cluster-bootstrap intervals. The prompt-only interval collapses by design.',maxheight=214)
p(f'<b>RQ1:</b> Both response-only networks identify families above chance. CNN reaches {pct(S["core"]["cnn_output"]["accuracy_mean"])} versus {pct(S["core"]["lstm_output"]["accuracy_mean"])} for LSTM under this recipe. This comparison does not establish that CNNs always dominate RNNs.')
p(f'<b>RQ2:</b> Adding the prompt changes CNN accuracy by {contrast("cnn_prompt_gain")}; for LSTM the change is {contrast("lstm_prompt_gain")}. Both intervals exclude zero. Irrelevant prompt variation and limited model capacity are plausible explanations, not isolated causal findings. Combined models retain the same 192-token answer budget.')

page('5. Initial-model errors and family variation')
fig('confusion_matrices','Figure 3. Response-only confusion matrices, row-normalized after averaging counts across three seeds. Each family has 213 held-out responses.',maxheight=230)
p('The confusion matrices show which families are mistaken for each other rather than reducing the experiment to one accuracy number. A family-level success rate can hide substantial variation among its underlying model versions. The following diagnostic uses seed 448, designated before testing; it does not select the best seed.')
table(['Generator version','Test responses','Correct family prediction'],[[k,v['n'],pct(v['accuracy'])] for k,v in S['cnn_seed448_version_recall'].items()],[.49,.20,.31],small=True)
p('These are per-version recalls for the family classifier, not version-identification accuracies. The models have not learned separate labels for GPT-4 and GPT-3.5, for example. Unequal sample counts and varying task mixes mean the version differences are descriptive; they should not be interpreted as a ranking of writing quality or model capability.')
p(f'Truncation is substantial: {pct(S["truncation"]["test"]["response_over192"])} of test responses exceed 192 tokens, and {pct(S["truncation"]["test"]["prompt_over96"])} of test prompts exceed 96 tokens. Predictions therefore concern bounded prefixes. The CNN can capture local patterns anywhere within its retained prefix; a final-state LSTM must summarize that prefix in 64 hidden units. This is a plausible source of the architectural gap, but a pooling or bidirectional-LSTM ablation would be needed to test it.')

page('6. Initial baselines and negative controls')
p('A TF-IDF + logistic regression baseline uses the same truncated tokens as the neural models, with unigram/bigram features, sublinear term frequency, minimum document frequency 2, at most 30,000 features, C = 1, and 1,000 maximum iterations. Training-only features prevent vocabulary leakage. This checks whether a simple sparse classifier is already competitive.')
fig('baselines','Figure 4. Simple baselines versus the required response-only networks. Neural bars average three seeds; classical baselines are single fixed runs.',maxheight=205)
rows=[]
for tag,label in [('tfidf_input','TF-IDF: prompt'),('tfidf_output','TF-IDF: response'),('tfidf_combined','TF-IDF: both'),('style_all','Structural features'),('style_no_length','Structural: remove length')]:
 x=S['auxiliary'][tag];rows.append([label,pct(x['accuracy']),pct(x['macro_f1'])])
table(['Baseline','Accuracy','Macro-F1'],rows,[.54,.23,.23],small=True)
p(f'TF-IDF response-only accuracy is {pct(S["auxiliary"]["tfidf_output"]["accuracy"])}, close to the CNN mean. There is no basis here to claim that neural complexity is necessary. Learning discriminative lexical patterns is already effective on this small historical benchmark.')
sub('Shuffled-label diagnostic')
p(f'An additional post hoc diagnostic permutes training labels while preserving validation/test labels and all settings. CNN accuracy falls to {pct(S["auxiliary"]["cnn_output_s448_shuffled"]["accuracy"])} and LSTM to {pct(S["auxiliary"]["lstm_output_s448_shuffled"]["accuracy"])}. These particular runs are below chance, not exactly at chance. They show that corrupted supervision does not retain useful above-chance performance; one permutation does not prove that every form of leakage has been excluded. This diagnostic was not used to tune or replace the primary results.')

page('7. RQ4: initial-model signal probes')
p('The structural baseline uses ten features: log character count, log word count, mean word length, lexical diversity, newline rate, punctuation rate, digit rate, uppercase rate, list-line rate, and code-fence count. Standardization is fitted only on training examples. This baseline sees full responses, unlike the prefix-limited text classifiers, so it is an exploratory signal probe rather than an equal-information leaderboard.')
fig('style_distributions','Figure 5. Full-response structural distributions across the curated corpus. Box plots omit outliers visually, not from calculations.',maxheight=195)
p(f'The structural classifier reaches {pct(S["auxiliary"]["style_all"]["accuracy"])}. Removing the two explicit length features changes accuracy by {contrast("style_length_removal")}. Other features can still correlate with length, so this does not prove length is irrelevant. It does show that explicit length counts alone are not necessary for this feature set to perform above chance.')
sub('Remove punctuation and newline markers, then retrain')
table(['Model (seed 448)','Original accuracy','Normalized accuracy','Change (pp)'],[[m.upper(),pct(result(f'{m}_output_s448')['test']['accuracy']),pct(S['auxiliary'][f'{m}_output_s448_normalized']['accuracy']),pp(S['contrasts'][m+'_normalization_effect']['accuracy_difference'])] for m in ['cnn','lstm']],[.25,.25,.25,.25],small=True)
p(f'CNN change: {contrast("cnn_normalization_effect")}. LSTM change: {contrast("lstm_normalization_effect")}. The CNN interval overlaps zero; the LSTM improvement is positive for this seed. Thus formatting cues are not uniformly beneficial across architectures.')
p('Normalization retains lowercase word tokens and removes punctuation/newline tokens. It also changes token boundaries and allows more words inside the fixed token limit. The intervention bundles formatting removal with altered effective context; it cannot isolate a pure causal effect of punctuation. Auxiliary results use one seed and should be repeated before strong claims.',small=True)

page('8. RQ3: initial-model source transfer')
p('This exploratory experiment holds out Evol-Instruct from training and validation and evaluates on its original test prompts. Source identity is a proxy for distribution shift, not a clean coding-versus-writing task label. The 58 test prompts (174 responses) are identical for both conditions. A mixed-source control randomly subsamples the original training prompts to match the holdout training size.')
table(['Condition','Train rows','Validation rows','Test rows'],[['Evol-Instruct excluded','2,175','441','174'],['Mixed-source, same size','2,175','441','174']],[.46,.18,.18,.18],small=True)
fig('interventions','Figure 6. Auxiliary interventions, seed 448. Left: normalization and retraining. Right: equal-size training controls on the same held-out-source test subset.',maxheight=215)
table(['Architecture','Mixed source','Source held out','Difference (pp)'],[[m.upper(),pct(S['auxiliary'][f'{m}_output_s448_source_control']['accuracy']),pct(S['auxiliary'][f'{m}_output_s448_source_holdout']['accuracy']),pp(S['contrasts'][m+'_source_shift']['accuracy_difference'])] for m in ['cnn','lstm']],[.28,.24,.24,.24],small=True)
p(f'CNN shift effect: {contrast("cnn_source_shift")}. LSTM shift effect: {contrast("lstm_source_shift")}. Both intervals include zero. Each held-out-source classifier remains above chance, but the experiment does not provide clear evidence of either a deterioration or an improvement relative to the equal-size control.')
p('Holding training size constant reduces a simple data-volume confound. It does not make the two training distributions otherwise identical. The validation distribution excludes Evol-Instruct for both conditions. Only one source, one split, and one training seed are examined; the test contains only 58 prompts. Stronger domain-generalization claims would require explicit task annotation, more held-out domains, and repeated runs.')

page('9. Reproducibility, limitations, and lessons')
sub('Reproduce the results')
p('Create a Python 3.12 environment and install requirements-lock.txt. First run the download and preparation scripts, tests, initial experiment runner, baselines, and initial analysis. Then run src/iterate.py with search, confirm, and evaluate in that order, followed by analyze_v2.py, both verification scripts, and build_report.py. The README contains exact commands. The experiment runner resumes completed runs; a fresh result directory is required for a complete retraining. Every neural result records its settings, selected epoch, vocabulary size, parameter count, history, device, runtime, and split-manifest hash.')
p(f'Recorded fitting and validation time summed across the initial neural runs and classical baselines is {S["runtime_seconds"]:.1f} seconds. This excludes process startup, downloads, data preparation, report generation, and the timing pilots. The small networks and short sequences make fitting fast; data curation and analysis remain essential work.',small=True)
sub('Practical lessons supported by the experiments')
p('<b>Evaluation design can matter more than an extra model layer.</b> Matched prompts turn an ambiguous input-only experiment into an exact negative control. Grouped splitting prevents related answers from entering both train and test.')
p('<b>More input is not automatically better.</b> Both initial architectures lose accuracy after adding prompts under the fixed recipe; the optimized comparison is reported separately. Keeping the answer budget unchanged makes this comparison easier to interpret.')
p('<b>A strong baseline improves the conclusion.</b> TF-IDF is competitive with CNN, while structural features alone predict above chance. These findings constrain what can be claimed about learned LLM fingerprints.')
p('<b>Uncertain and negative findings belong in the report.</b> The source-shift intervals overlap zero, and normalization benefits one architecture. Reporting those results is more informative than selecting only favorable experiments.')
sub('Boundaries of the evidence')
p('The labels describe selected historical releases rather than current hosted systems or unseen families. Common-prompt availability creates a selected subset. Custom system prompts and generation principles remain confounds. Near-duplicate detection is incomplete. Short prefixes, small models, fixed budgets (initially 8 epochs, then up to 24), one split, and three primary seeds restrict conclusions. Bootstrap intervals are conditional on fitted models and assume clusters are approximately independent; they are not corrected for multiple exploratory comparisons. The optimization stage is a post hoc follow-up using the same test split, and validation performance is optimistic after model selection. The study is not suitable for accusing an individual author of using a particular AI system.')
sub('AI assistance disclosure')
p('ChatGPT assisted with experiment design, implementation, debugging, analysis, and report drafting. All reported metrics come from locally executed scripts and saved predictions, not invented examples. Automated checks cover important data and model invariants. Haodong Huang is responsible for reviewing, understanding, and explaining the submitted work, as required by the assignment [6].',small=True)

page('10. Optimization: select using validation only')
p('The initial model comparison prompted a bounded follow-up, rather than an open-ended attempt to maximize the test score. Before fitting, docs/iteration_plan.json fixed twelve candidates. Each used output-only training and seed 448. The top two candidates per architecture were repeated with seeds 449 and 450; the winner maximized mean validation macro-F1. Selection was saved and hashed before fitting the other input conditions or evaluating any optimized test predictions.')
p('All candidates use Adam, weight decay 0.0001, batch size 64, gradient clipping 1.0, at most 24 epochs, and validation patience 5. They add embedding dropout 0.10 and allow up to 16,000 training-only vocabulary entries. The search changes multiple settings, so improvement over the initial recipe cannot be attributed to one isolated factor.')
searchrows=[]
for x in V['search_ledger']:
    structure='CNN' if x['architecture']=='cnn' else ('BiLSTM' if x['bidirectional'] else 'LSTM')+(' last' if x['pooling']=='last' else ' pool')
    details=f"{structure}; e{x['emb']}/h{x['hidden']}; {'words' if x['normalized'] else 'raw'}"
    searchrows.append([x['candidate']+(' *' if x['selected'] else ''),details,str(x['max_tokens']),str(x['lr']),pct(x['seed448_val_f1'])])
table(['ID','Structure / dimensions / tokens','Limit','LR','Val. F1 (448)'],searchrows,[.10,.47,.10,.13,.20],small=True)
p('e = embedding size; h = hidden units or CNN filters per kernel; pool = masked mean + max across valid positions; words = punctuation/newline removal. Asterisk marks the final three-seed validation winner, which need not be the seed-448 winner. Candidate details and all results are retained.',small=True)
sub('Selected configurations')
for arch,x in V['selection'].items():
    c=x['config'];p(f"<b>{arch.upper()} {x['candidate']}:</b> {c['max_tokens']}-token response budget, embedding {c['emb']}, hidden/filter width {c['hidden']}, dropout {c['dropout']}, learning rate {c['lr']}; pooling {c['pooling'] if arch=='lstm' else 'max'}, bidirectional {c['bidirectional'] if arch=='lstm' else 'not applicable'}. Mean validation macro-F1: {pct(x['validation_macro_f1_mean'])}.",small=True)
p('The backward LSTM reverses only the valid token prefix. Its valid-position outputs were tested against a packed PyTorch bidirectional LSTM; extra padding and batch composition do not change predictions. These tests guard against a subtle source of invalid performance gains.',small=True)

page('11. Optimized results and the limits of improvement')
rows=[]
for arch in ['cnn','lstm']:
 for mode in ['input','output','combined']:
    x=V['core'][arch+'_'+mode]
    rows.append([arch.upper(),{'input':'Prompt','output':'Response','combined':'Both'}[mode],f"{100*x['accuracy_mean']:.2f} +/- {100*x['accuracy_sd']:.2f}",f"{100*x['macro_f1_mean']:.2f} +/- {100*x['macro_f1_sd']:.2f}",interval(x['accuracy_ci95'])])
table(['Model','Input','Accuracy (%)','Macro-F1 (%)','95% accuracy CI'],rows,[.12,.14,.25,.25,.24],small=True)
fig('optimization_results','Figure 7. Initial versus optimized recipes on the same test split. Means over three seeds; intervals resample matched-prompt clusters.',maxheight=198)
for arch in ['cnn','lstm']:
    gain=V['contrasts'][arch+'_output_gain'];prompt=V['contrasts'][arch+'_prompt_gain']
    p(f"<b>{arch.upper()}:</b> response-only improvement is {pp(gain['accuracy_difference'])} percentage points (paired 95% CI {interval(gain['ci95'])}). Within the optimized recipe, adding the prompt changes accuracy by {pp(prompt['accuracy_difference'])} pp (CI {interval(prompt['ci95'])}).")
p('For both optimized architectures, the combined-minus-response intervals include zero. The lower point estimates therefore do not establish a clear benefit or harm from adding prompts under the optimized recipes.',small=True)
p(f"Optimization used {V['fit_count']} fits and {V['training_seconds']:.1f} seconds of recorded training/validation time; process startup and analysis are excluded. Only the six selected conditions across three seeds were evaluated on test. The full candidate/seed ledger is available as CSV.",small=True)
p('These are improvements within a bounded search, not a global optimum or a state-of-the-art claim. All optimized choices used validation metrics, but the original test results motivated this follow-up, so this is not an untouched confirmatory test. The original source-shift and signal-removal findings concern the initial networks and are not automatically transferred to optimized models. A new independent dataset would strengthen the conclusions.',small=True)

page('Appendix A. Initial neural run ledger')
p('All initial primary and auxiliary runs are listed; no unfavorable seed is omitted. The 32 optimization fits are listed separately in results/v2/run_ledger.csv. Values are test accuracy and macro-F1. Epoch is chosen on validation data. Timing pilots have no test score and are excluded.',small=True)
ledger=[]
for model in ['cnn','lstm']:
 for mode in ['input','output','combined']:
  for seed in [448,449,450]:
   r=result(f'{model}_{mode}_s{seed}');ledger.append([model.upper(),mode,str(seed),r['best_epoch'],f'{100*r["test"]["accuracy"]:.2f}',f'{100*r["test"]["macro_f1"]:.2f}'])
for model in ['cnn','lstm']:
 for suffix,label in [('_normalized','normalized'),('_source_holdout','source held out'),('_source_control','source control'),('_shuffled','shuffled labels')]:
  r=result(f'{model}_output_s448{suffix}');ledger.append([model.upper(),label,'448',r['best_epoch'],f'{100*r["test"]["accuracy"]:.2f}',f'{100*r["test"]["macro_f1"]:.2f}'])
table(['Model','Condition','Seed','Epoch','Acc. (%)','F1 (%)'],ledger,[.13,.27,.13,.12,.175,.175],small=True)
p('Detailed class-wise metrics, losses, and confidence intervals are available in results/*.json and results/summary.json. The package includes aggregate metrics, non-text row-level predictions, and seed-448 response-only CNN/LSTM weights. Other checkpoints stay local and can be regenerated.',small=True)

page('Appendix B. References and submission contents')
refs=[
('[1] OpenBMB. UltraFeedback dataset and dataset card.','https://huggingface.co/datasets/openbmb/UltraFeedback'),
('[2] Cui, G., Yuan, L., Ding, N., et al. UltraFeedback: Boosting Language Models with Scaled AI Feedback. ICML 2024.','https://arxiv.org/abs/2310.01377'),
('[3] CMPSC 448. Chapter 4: Deep Learning Architectures. Course slides supplied by the student.','CNN / pooling: pp. 16-41; 1D convolution: p. 36; RNN / LSTM: pp. 44-79.'),
('[4] CMPSC 448. Chapter 9: ML Training--Gradient Descent. Course slides supplied by the student.','Mini-batch optimization: pp. 41-48; Adam: pp. 62-65.'),
('[5] PyTorch documentation. Conv1d, LSTM, and Adam.','https://docs.pytorch.org/docs/stable/'),
('[6] CMPSC 448. Who Wrote It? Identifying LLMs from Their Responses. Midterm project instructions, modified September 17, 2026.','Local assignment supplied by the student; deadline October 11, 2026, 11:59 pm.')]
for title,detail in refs:
 p('<b>'+html.escape(title)+'</b>')
 p(html.escape(detail),small=True)
sub('Immutable data provenance')
p('Original dataset revision: '+P['original_revision'],small=True)
p('Converted Parquet revision: '+P['parquet_revision'],small=True)
p('Artifact hashes and download URLs are recorded in docs/data_provenance.json. The conversion revision is pinned separately from the dataset-card revision. Access and experiment date: October 10, 2026.',small=True)
sub('Repository contents')
table(['Path','Purpose'],[['src/','Data acquisition, curation, models, experiments, statistics, and report builder'],['tests/','Checks that guard against leakage and padding errors'],['models/','Initial and optimized CNN/LSTM demonstration weights (seed 448)'],['results/','Initial and optimized metrics, non-text predictions, logs, and figures'],['docs/','Provenance, experiment plan, and supplementary protocol'],['report/','This PDF and editable Markdown source'],['README.md; requirements-lock.txt','Run instructions and exact dependency versions']],[.36,.64],small=True)
p('The repository intentionally excludes the source dataset, processed response text, lecture PDFs, assignment PDF, local environment, and additional model checkpoints. This keeps the submission compact and avoids republishing third-party course materials. Reproduction scripts recover the data from its original host.',small=True)

class NumberedCanvas(canvas.Canvas):
    def __init__(self,*a,**kw):super().__init__(*a,**kw);self.states=[]
    def showPage(self):self.states.append(dict(self.__dict__));self._startPage()
    def save(self):
        total=len(self.states)
        for state in self.states:
            self.__dict__.update(state);self.setStrokeColor(colors.HexColor('#D6E2E4'));self.line(48,752,564,752)
            self.setFont('Bold',7.5);self.setFillColor(TEAL);self.drawString(48,763,'CMPSC 448 / MIDTERM PROJECT')
            self.setFont('Body',7.5);self.setFillColor(GRAY);self.drawRightString(564,763,'HAODONG HUANG')
            self.drawString(48,27,'Whose Slop Is It? | Matched-prompt evaluation');self.drawRightString(564,27,f'{self._pageNumber} / {total}')
            super().showPage()
        super().save()

out=ROOT/'report';out.mkdir(exist_ok=True)
doc=SimpleDocTemplate(str(out/'Whose_Slop_Is_It_Haodong_Huang.pdf'),pagesize=(612,792),leftMargin=48,rightMargin=48,topMargin=55,bottomMargin=48,
                      title='Whose Slop Is It? Identifying LLM Families',author='Haodong Huang')
doc.build(story,canvasmaker=NumberedCanvas)
(out/'report.md').write_text('\n'.join(md),encoding='utf-8')
print(out/'Whose_Slop_Is_It_Haodong_Huang.pdf')
