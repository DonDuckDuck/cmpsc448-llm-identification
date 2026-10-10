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
S=json.loads((ROOT/'results/arena/summary.json').read_text());A=json.loads((ROOT/'results/arena/data_audit.json').read_text())
P=json.loads((ROOT/'docs/arena_data_provenance.json').read_text())
V=json.loads((ROOT/'results/arena/verification.json').read_text())
def result(t):return json.loads((ROOT/'results/arena'/f'{t}.json').read_text())
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
from arena_train import LABELS,CANDIDATES,tag
core=S['core']
def f1(arch,mode='output'):return pct(core[arch+'_'+mode]['macro_f1_mean'])
def change(c):return f"{pp(c['macro_f1_difference'])} pp; 95% CI {interval(c['ci95'])}"
def yesno(c):return 'The interval excludes zero.' if c['ci95'][0]>0 or c['ci95'][1]<0 else 'The interval includes zero; the direction is not resolved.'

story.append(Spacer(1,25));story.append(Paragraph('WHO WROTE IT?',styles['TitleCustom']))
p('<b>Identifying LLMs from Their Responses</b>')
p('Haodong Huang | Individual project<br/>CMPSC 448 | October 10, 2026')
md.insert(0,'# Who Wrote It? Identifying LLMs from Their Responses\n')
story.append(Spacer(1,16));sub('Abstract')
p(f'This project identifies four LLM sources using a convolutional neural network (CNN) and an LSTM recurrent neural network (RNN). A pinned Arena Human Preference 140k release provides {A["retained_rows"]:,} curated English, single-turn responses from Claude Sonnet 4, GPT-4.1, Grok 3, and DeepSeek V3. Sessions, exact and near-duplicate prompts, and exact repeated answers remain within one split. Model identity tokens are masked. A four-candidate validation screen selects one CNN and one LSTM recipe, each evaluated on three random seeds and all three input conditions. Response-only macro-F1 is {f1("cnn")} for CNN and {f1("lstm")} for LSTM. Classical baselines, normalization ablations, shuffled labels, and a matched-size domain-transfer experiment probe what the models learn. These results concern four specific releases on a curated Arena distribution, not universal authorship attribution.')
sub('Research questions and evidence')
table(['Question','Evidence'],[
['RQ1: Identify the author?',f'CNN accuracy {pct(core["cnn_output"]["accuracy_mean"])}; LSTM accuracy {pct(core["lstm_output"]["accuracy_mean"])}. Macro-F1 and class recall accompany accuracy.'],
['RQ2: Does the prompt help?',f'Combined-minus-output macro-F1: CNN {pp(S["contrasts"]["cnn_prompt_gain"]["macro_f1_difference"])} pp; LSTM {pp(S["contrasts"]["lstm_prompt_gain"]["macro_f1_difference"])} pp.'],
['RQ3: Transfer to coding?', 'Train on non-code and evaluate coding; compare with a mixed-domain control matched by training class counts.'],
['RQ4: Which signals matter?', 'Word-only normalization and explicit structural-feature classifiers probe lexical and formatting signals.']],[.31,.69])
p(f'Scope: {A["retained_rows"]:,} responses; four classes; {S["neural_fits"]} neural fits and {len(S["baselines"])} classical fits; {S["test_runs"]} test prediction files; 2,000 group-bootstrap resamples. The earlier UltraFeedback study is preserved separately, not pooled into these scores.',small=True)

page('1. Dataset and model provenance')
p('The official Arena release contains 135,634 battles collected in April-July 2025 [1,2]. Each side provides a model ID and a conversation. We convert a retained side into (LLM_name, LLM_Input, LLM_output). Human preference votes are not author labels and never become input features. The source revision and seven file hashes are recorded in docs/arena_data_provenance.json.')
table(['Class','Exact model ID','Before filters','Retained'],[[f,next(k for k,v in A['models'].items() if v==f),A['selected_before_filter'][f],A['family_counts'][f]] for f in LABELS],[.13,.43,.21,.23])
p('One explicit, non-thinking model variant per family controls a major source of confounding. Grok 4 has only 1,561 sides in the complete release, whereas Grok 3 has 6,040; the study selects Grok 3 for sample support. Therefore class names are convenient abbreviations for these four fixed IDs. Later releases and alternative reasoning settings remain outside the measured scope.')
sub('Curation sequence')
table(['Rule','Reason'],[['Exact four IDs; language=en','Keep the authorship target and language explicit.'],['evaluation_order=1; exactly user then assistant','Avoid attributing inherited or multi-model history to the current author.'],['Text-only content; prompt >=3 words, answer >=10','Exclude multimodal records, empty outputs, and uninformative fragments.'],['Exclude email/API-key patterns and extreme lengths','Basic content screening; not a guarantee of complete de-identification.'],['Deduplicate identical prompt/model/answer triples','Avoid repeatedly counting an identical observation.'],['Mask provider and model-name strings in both fields','Reduce direct identity shortcuts.']],[.48,.52],small=True)
p(f'The four selected IDs contribute {sum(A["selected_before_filter"].values()):,} sides before filtering; {A["retained_rows"]:,} remain. All eligible responses are retained rather than selecting easy examples. Identity masking affects {A["mask_affected_outputs"]:,} retained answers. Public artifacts contain no raw conversation text.',small=True)

page('2. Splits, leakage controls, and scope')
table(['Split','Responses','Groups','Claude / GPT / Grok / DeepSeek'],[[s.title(),A['splits'][s]['rows'],A['splits'][s]['groups'],' / '.join(str(A['splits'][s]['family_counts'][f]) for f in LABELS)] for s in ['train','val','test']],[.15,.16,.16,.53])
p(f'Rows are linked by shared evaluation session, normalized prompt, or normalized exact answer. Character TF-IDF prompt similarity of at least 0.90 also links rows. Transitive connected components form {A["groups"]:,} groups over {A["unique_prompts"]:,} exact prompt IDs; the largest group contains {A["largest_group"]} responses. GroupShuffleSplit uses seed 1448 for the 15% test-group split and 1449 for validation groups from the remainder. Ratios are approximate by response count because groups are indivisible.')
p('Duplicate detection uses the complete corpus without labels or performance feedback. This is a disclosed transductive cleaning step, not a predictive feature transformation. Neural vocabularies, TF-IDF features used by classifiers, scalers, and class weights are fitted on training data only. The verification checks session, prompt, and duplicate-group disjointness independently.')
sub('What enters each classifier?')
table(['Included','Excluded'],[['Masked question and/or answer tokens','Model ID, provider name metadata, timestamps'],['A training-only vocabulary','Human winner votes, category metadata'],['Observed punctuation and newline tokens','Dataset quality scores or hidden reasoning traces']],[.50,.50])
p('Arena pairs two models on a question; it does not supply all four selected authors for every question. Consequently prompts are not balanced across classes by construction. Prompt-only performance is an empirical confounding check, not a guaranteed 25%. Shared prompts and sessions still cannot cross splits.')
p(f'Class balance is reasonably close but not exact. The test majority-class accuracy is {pct(S["majority_accuracy"])}; uniform random accuracy is 25%, and an always-one-class predictor has 25% balanced accuracy. Macro-F1 is the primary model-selection metric.',small=True)

page('3. CNN and LSTM implementation')
p('Both models use randomly initialized trainable token embeddings. Tokenization lowercases text, preserves punctuation, and represents newlines explicitly. Training-only vocabularies contain at most 20,000 tokens, with PAD, UNK, and a separator. No pretrained LLM is used as a classifier. Chapter 4 supplies the convolution/pooling and recurrent-network concepts [3].')
rows=[]
for arch in ['cnn','lstm']:
 c=S['selection'][arch];cfg=c['config'];r=result(tag(c['candidate'],'output',1448))
 rows.append([arch.upper(),f'{cfg["emb"]}-D',str(cfg['hidden']),str(cfg['max_tokens']),f'{r["parameters"]:,}'])
table(['Selected model','Embedding','Width','Answer tokens','Parameters'],rows,[.24,.17,.15,.21,.23])
sub('CNN: local patterns plus global pooling')
p('Three 1-D convolutions have widths 3, 4, and 5. ReLU activates each feature map; global max pooling retains a strong occurrence anywhere in the valid sequence. A mask excludes padded windows. The pooled features are concatenated, passed through dropout, and projected into four logits. This structure can capture repeated phrases, punctuation patterns, and formatting cues without reading a response recurrently.')
sub('LSTM: contextual states plus sequence pooling')
cfg=S['selection']['lstm']['config']
p(f'The selected LSTM is {"bidirectional" if cfg["bidirectional"] else "unidirectional"}. It generates hidden states for each real token, then concatenates a masked mean and masked maximum over states. A dropout layer and linear layer output four logits. The bidirectional candidate uses separate forward and reversed valid-prefix LSTMs; padding is excluded when reversing and pooling. Its behavior is checked against packed PyTorch recurrence.')
sub('Input conditions')
table(['Condition','Token budget'],[['Input only','First 96 question tokens'],['Output only','First 384 answer tokens for CNN; 192 for LSTM'],['Input + output','96 question tokens + separator + the full architecture-specific answer budget']],[.31,.69])
p('The different answer budgets are disclosed design choices. Within each architecture, input-condition comparisons preserve its answer budget; the CNN-versus-LSTM result is a recipe comparison, not a controlled claim that architecture alone caused the difference.',small=True)

page('4. Training and frozen model selection')
p('Training minimizes class-weighted cross-entropy using Adam (learning rate 0.001, weight decay 0.0001), gradient norm clipping at 1.0, batches of 128, embedding dropout 0.10, and model dropout 0.30 or 0.40. The class weight is N/(4*n_class), calculated only on the training subset. These choices build on Chapter 9 optimization concepts [4].')
p('Each fit runs at most 20 epochs, with early stopping after four epochs without improvement. Checkpoints maximize validation macro-F1, then minimize validation loss to break a tie. The test set is not loaded during training or configuration selection. A pre-evaluation masking repair removed three release suffixes; all reported fits use the corrected text and regenerated split. Four response-only candidates are screened at seed 1448:')
table(['ID','Architecture / change','Validation F1','Selected'],[[cid,('CNN' if cfg['architecture']=='cnn' else 'LSTM')+f'; emb {cfg["emb"]}, width {cfg["hidden"]}'+('; bidirectional' if cfg['bidirectional'] else ''),pct(result(tag(cid,'output',1448))['best_val_macro_f1']),'Yes' if cid in [x['candidate'] for x in S['selection'].values()] else 'No'] for cid,cfg in CANDIDATES.items()],[.09,.51,.23,.17])
p('The winning recipe for each architecture is repeated at seeds 1448, 1449, and 1450 for question-only, answer-only, and combined inputs. Selected recipes also receive seed-1448 diagnostic fits. All neural and classical checkpoints are hashed before any test evaluation; rejected candidates receive no test scores.')
fig('arena_learning','Validation curves for the selected response-only seed-1448 runs. Checkpoint selection uses these curves, not test scores.',maxheight=195)
p('A single-seed screen is a bounded resource choice and can select a noisy winner. Three-seed final summaries expose some optimization variability, but this is not an exhaustive hyperparameter search.',small=True)

page('5. Main results: RQ1 and RQ2')
table(['Model / input','Accuracy','Macro-F1','Balanced acc.','F1 seed SD'],[[arch.upper()+' / '+mode,pct(core[arch+'_'+mode]['accuracy_mean']),pct(core[arch+'_'+mode]['macro_f1_mean']),pct(core[arch+'_'+mode]['balanced_accuracy_mean']),f'{100*core[arch+"_"+mode]["macro_f1_sd"]:.2f} pp'] for arch in ['cnn','lstm'] for mode in ['input','output','combined']],[.30,.18,.18,.19,.15])
fig('arena_results','Bars: means over three seeds. Error bars: 95% percentile intervals from 2,000 resamples of held-out session/duplicate groups.',maxheight=210)
p(f'For response-only attribution, the CNN-minus-LSTM macro-F1 difference is {change(S["contrasts"]["cnn_minus_lstm"])} {yesno(S["contrasts"]["cnn_minus_lstm"])} Both architectures are evaluated against the same rows and grouped resampling draws.')
p('Intervals resample test groups while holding the three fitted models fixed. They describe uncertainty from sampled test groups conditional on those fits; they do not fully capture model-selection uncertainty, future releases, platform changes, or a new prompt distribution.',small=True)

page('6. Errors and per-class performance')
fig('arena_confusion','Response-only confusion matrices, pooled over three seeds and normalized by true class. Every row sums to 100%.',maxheight=230)
table(['True class','CNN recall','LSTM recall','Test support'],[[f,pct(np.array(core['cnn_output']['confusion_matrix_sum'])[i,i]/np.array(core['cnn_output']['confusion_matrix_sum'])[i].sum()),pct(np.array(core['lstm_output']['confusion_matrix_sum'])[i,i]/np.array(core['lstm_output']['confusion_matrix_sum'])[i].sum()),A['splits']['test']['family_counts'][f]] for i,f in enumerate(LABELS)],[.27,.25,.25,.23])
p('A high aggregate accuracy can hide one poorly identified source. Class recall shows which author is missed; off-diagonal cells show the competing source. These are source-attribution errors, not judgments about response truthfulness or usefulness.')
sub('Descriptive task slices')
table(['Model','Coding F1','Non-coding F1'],[[a.upper(),pct(S['subgroups'][a+'_code']['macro_f1_mean']),pct(S['subgroups'][a+'_noncode']['macro_f1_mean'])] for a in ['cnn','lstm']],[.34,.33,.33])
p('These slices use models trained on mixed tasks. They are descriptive performance breakdowns, not evidence of generalization to an unseen domain. The next domain-transfer experiment specifically removes coding tasks from training.',small=True)

page('7. Does the question help?')
for arch in ['cnn','lstm']:
 c=S['contrasts'][arch+'_prompt_gain'];sub(arch.upper())
 p(f'Question-only macro-F1 is {f1(arch,"input")}, response-only is {f1(arch)}, and combined is {f1(arch,"combined")}. The paired combined-minus-response difference is {change(c)} {yesno(c)}')
sub('How to interpret input-only prediction')
p('A user prompt precedes the model answer, so prompt-only discrimination cannot be an answer-style fingerprint. It may reflect sampling differences, task mix, time-dependent model availability, remaining template similarities, or chance. Arena randomizes pairs, but curation to four releases and available first-turn English records can still change the prompt distributions across classes.')
p('Adding a prompt can provide helpful context, but can also encourage prompt-topic shortcuts or dilute the answer representation. The result should therefore be judged through the measured paired difference and its uncertainty, rather than assuming that more text always helps.')
sub('Classical prompt check')
p(f'The fixed word TF-IDF question-only baseline achieves {pct(S["baselines"]["word_input"]["accuracy"])} accuracy and {pct(S["baselines"]["word_input"]["macro_f1"])} macro-F1. It offers an independent check for prompt information with a different representation.')
p('The answer prefix is unchanged when the question is added: combined input concatenates the 96-token question, a separator, and the same 384-token CNN or 192-token LSTM answer prefix. Thus a change in score is not caused by truncating away the answer to make room for the question.')

page('8. Cross-task generalization: RQ3')
p('A non-coding training subset excludes all rows with Arena is_code=true. Both domain-transfer models validate on non-code rows only. A mixed-domain control uses exactly the same number of training examples in each class, sampled from the original training split. Both are evaluated on the identical coding subset of the untouched test split. This separates the loss of coding exposure from a simple reduction in training size.')
table(['Model / training','Train rows','Coding accuracy','Coding F1'],[[arch.upper()+' / '+kind,S['domain_transfer'][arch][kind]['training_rows'],pct(S['domain_transfer'][arch][kind]['accuracy']),pct(S['domain_transfer'][arch][kind]['macro_f1'])] for arch in ['cnn','lstm'] for kind in ['holdout','control']],[.37,.17,.23,.23])
for arch in ['cnn','lstm']:
 d=S['domain_transfer'][arch];p(f'{arch.upper()}: non-code-minus-control coding macro-F1 is {pp(d["holdout_minus_control_macro_f1"])} pp, with paired 95% group-bootstrap interval {interval(d["difference_ci95"])}. Coding test support is {d["holdout"]["test_rows"]} responses.')
p('Above-chance performance after removing coding training suggests some transferable signal. A negative holdout-versus-control difference suggests that task exposure still matters. Neither result proves a universal fingerprint: Arena code labels are imperfect proxies for domain, and non-code includes diverse tasks rather than one homogeneous task.')
p('This diagnostic uses one seed per condition, with checkpoint choice on non-code validation only. Architecture recipes were inherited from the main mixed-domain validation screen, so this is a training-domain holdout, not a completely target-label-blind protocol. Its intervals describe held-out groups conditional on those fits. Multiple domain definitions and repeated training seeds would be needed for a stronger transfer claim.',small=True)

page('9. Fingerprint signals: RQ4')
table(['Baseline','Test accuracy','Macro-F1'],[[name.replace('_',' '),pct(m['accuracy']),pct(m['macro_f1'])] for name,m in S['baselines'].items()],[.50,.25,.25])
p(f'The character baseline reaches {pct(S["baselines"]["char_output"]["accuracy"])} accuracy and {pct(S["baselines"]["char_output"]["macro_f1"])} macro-F1, a slightly higher point estimate than the CNN mean. This highlights the strength of local lexical patterns; the small ranking difference alone does not establish reliable superiority.')
p('Word TF-IDF uses 1-2 grams; character TF-IDF uses 3-5 grams. Each is capped at 40,000 features, min_df=3, with sublinear term frequency and balanced logistic regression (C=2). Text baselines use the first 96 question and/or 384 answer tokens. Their longer answer budget than LSTM is disclosed. All transforms are fitted on training data.')
p('Structural baselines use full-answer statistics: log character/word counts, average word length, lexical diversity, newline/punctuation/digit/uppercase rates, list-line rate, and code-fence count. The no-length variant removes the two explicit log length features; other features can still correlate with length. These diagnostics are not controlled equal-input-budget competitors.')
sub('Remove punctuation and newline information')
table(['Model','Raw answer F1, seed 1448','Word-only F1','Difference'],[[a.upper(),pct(result(tag(S['selection'][a]['candidate'],'output',1448))['test']['macro_f1']),pct(S['diagnostics'][a+'_normalized']['macro_f1']),pp(S['contrasts'][a+'_normalization']['macro_f1_difference'])+' pp'] for a in ['cnn','lstm']],[.17,.32,.25,.26])
for a in ['cnn','lstm']:p(f'{a.upper()} normalization contrast: {change(S["contrasts"][a+"_normalization"])}',small=True)
p('Word-only tokenization removes punctuation/newline tokens and changes the number of words visible under a fixed token cap. This is a broad representation ablation, not a causal estimate for one formatting feature. Successful structural or lexical baselines show available predictive signals, not necessarily the exact features used internally by a neural model.',small=True)

page('10. Structural profiles and interpretation')
p('The following summaries use training responses only. They describe the curated distribution, rather than selecting examples after seeing test errors. Medians are robust summaries but conceal substantial within-class variation.')
table(['Source','Median words','Lexical diversity','Newline / 1k chars','Punctuation / 1k chars'],[[f,f'{np.expm1(S["training_style_profiles"][f]["log_words"]):.0f}',f'{S["training_style_profiles"][f]["lexical_diversity"]:.3f}',f'{1000*S["training_style_profiles"][f]["newline_rate"]:.1f}',f'{1000*S["training_style_profiles"][f]["punctuation_rate"]:.1f}'] for f in LABELS],[.16,.18,.22,.22,.22])
p('Here a word is a regex word token, lexical diversity is unique lowercased words divided by all words, and character rates are measured on the full masked answer. These word counts differ from whitespace counts used in the raw curation audit and from the neural tokenizer.')
p('Response length is itself a source signal in this snapshot. Long model answers can create apparent stylistic separation even when their substance is similar. Prefix truncation bounds the neural input, but shorter answers still expose length through how many real tokens remain. The structural classifier without explicit length features is a useful partial check, not complete length control.')
sub('What the analyses establish')
p('Lexical baselines test whether word and character patterns are sufficient for attribution. Structural baselines test whether a small set of formatting and length summaries is sufficient. The normalization ablation tests the broader effect of removing non-word tokens while retraining. Together these offer complementary evidence; none identifies a unique causal fingerprint or reconstructs the internal decision rules of a neural model.')
sub('What would strengthen a later study?')
p('A future independent experiment could collect the same prompts from all four models under matched generation settings, compare exact equal-length text windows, and hold out newer model versions. That would separate provider, prompt distribution, generation settings, and length more directly. Such follow-up work needs a new evaluation set, because this test set has now been used for the reported study.')

page('11. Checks, limitations, and lessons')
p(f'The shuffled-training-label CNN has test accuracy {pct(S["diagnostics"]["shuffled_cnn"]["accuracy"])} and macro-F1 {pct(S["diagnostics"]["shuffled_cnn"]["macro_f1"])}. Labels are permuted only in training; validation and test labels remain unchanged. This is a pipeline sanity check, not a proof that all confounding is eliminated.')
table(['Verified check','Evidence'],[['Session, prompt, duplicate-group separation','Independent assertions across all splits'],['Validation-only model choice',f'{V["validation_files_recomputed"]} validation prediction files recomputed; selected IDs reproduced'],['Reported test scores',f'{V["test_prediction_files_recomputed"]} prediction files independently rescored'],['Saved-model inference','CNN/LSTM checkpoint replay against stored probabilities'],['Implementation invariants','13 unit tests: masking, attribution, vocabulary, padding, recurrent states, class count']],[.40,.60])
sub('Material limitations')
p('The study uses one Arena snapshot, four releases, English first turns, and filtered nontrivial answers. It excludes later model generations, reasoning variants, multilingual text, multi-turn conversations, and multimodal inputs. Automatic identity masking cannot remove all semantic self-identification. Model-side prompts may differ by author; prompt-only controls measure part of this concern but cannot erase it.')
p('Question or answer truncation may discard useful evidence. Softmax values are uncalibrated and should not be presented as trustworthy probabilities for arbitrary text. Future reuse of this test set for tuning would turn subsequent results into follow-up exploration, requiring another independent test for confirmation.')
sub('Lessons supported by the experiments')
p('Curation changes the research question: a 135k-battle corpus becomes a 9.6k-response controlled subset. A recurrent network must handle sequence padding and pooling correctly. Model selection needs a separate validation set. A simple lexical baseline can be an informative competitor, and test accuracy alone does not explain which cues drive attribution.')
p('AI assistance: ChatGPT/Codex assisted with implementation, debugging, experiment orchestration, analysis, and report drafting. The named student is responsible for understanding, verifying, and explaining the submitted work. No claim is made that an automated result establishes student understanding.',small=True)

page('12. Project evolution and dataset choice')
p('The project developed in three stages. Earlier code, measured results, demonstration weights, and PDF reports remain available so that the final study can be traced back to its initial implementation. The table reports response-only accuracy, averaged over three seeds within each stage.')
table(['Stage','Data / author classes','Responses','CNN','LSTM'],[
['Initial implementation','UltraFeedback: Falcon, GPT, Llama','4,254','77.52%','64.01%'],
['Same-data optimization','Same UltraFeedback data and split','4,254','80.91%','77.00%'],
['Current study','Arena: Claude, GPT, Grok, DeepSeek',f'{A["retained_rows"]:,}',f'{100*core["cnn_output"]["accuracy_mean"]:.2f}%',f'{100*core["lstm_output"]["accuracy_mean"]:.2f}%']],[.22,.33,.15,.15,.15])
sub('Why change the dataset?')
p('<b>1. More recognizable model families.</b> The project owner wanted to study Claude, GPT, Grok, and DeepSeek. Arena contains identifiable releases from all four, making the target authors more relevant to that interest than the original Falcon/GPT/Llama subset. This is a relevance motivation, not a measured ranking of dataset popularity or model market share. The selected records are from 2025, not the latest 2026 releases.')
p(f'<b>2. More usable examples.</b> After curation, the sample increases from 4,254 to {A["retained_rows"]:,} responses ({A["retained_rows"]/4254:.2f} times as many). The motivation was to give the models more training evidence and test whether performance could improve. The dataset name "140k" describes the source release, not the number of responses used for training.')
sub('What improved, and what can we conclude?')
p('On the same UltraFeedback task and split, bounded validation search, a longer CNN answer budget, and pooled LSTM states produced accuracy gains of 3.39 and 12.99 percentage points respectively. This was exploratory follow-up after the original test results had been seen, rather than a new independent confirmation.')
p('The Arena study adds a four-class target, explicit release selection, session/near-duplicate grouping, a frozen validation screen before test evaluation, and matched-size domain controls. Its higher accuracy is an observed result on a different task. Changes in author classes, text distribution, and split design prevent attributing the difference to sample size. A fixed-Arena-split learning curve would be needed to isolate the benefit of more training examples; that experiment has not been run.')
p('Archive: report/archive/Who_Wrote_It_Haodong_Huang_v1.pdf and _v2.pdf; historical methods: docs/ITERATION_NOTES_ZH.md; metrics: results/ and results/v2/; current metrics: results/arena/. See docs/PROJECT_EVOLUTION_ZH.md for a Chinese explanation.',small=True)

page('13. Reproduction and submission')
p('Install Python 3.12 and the exact dependencies in requirements-lock.txt. The recorded runs use PyTorch '+result(tag(S['selection']['cnn']['candidate'],'output',1448))['torch_version']+' with Apple MPS. CPU inference replay verifies the two distributed demonstration weights; training on other devices can yield numerical differences.')
table(['Step','Command'],[['Download pinned data','python src/arena_data.py download'],['Create the fixed split once','python src/arena_data.py prepare'],['Screen, select, and fit core models','python src/arena_train.py train'],['Fit planned domain-transfer checks','python src/arena_train.py domain'],['Evaluate frozen models','python src/arena_train.py evaluate'],['Verify metrics and build figures','python src/arena_analyze.py'],['Run implementation checks','python -m unittest discover -s tests -v'],['Build this report','python src/build_report.py']],[.40,.60],small=True)
p('The prepare command refuses to overwrite existing split text and verifies published assignments in a fresh clone. Reproduce from a fresh clone and preserve the archived results. Training caches require matching configuration and split hashes. The domain phase must precede first test evaluation. Two selected response-only weights are included for a local demonstration; other checkpoints remain local and can be regenerated.')
p(f'Retained final-study fit/validation time totals {S["fit_seconds"]/60:.1f} minutes for this experiment collection. This excludes the roughly 1.61 GB download, curation, analysis, report generation, and manual review, and superseded development fits; it is not an end-to-end runtime promise.')
p('Submission: GitHub repository URL containing code, this PDF, author name, manifests, non-text predictions, and measured results. Raw Arena conversations and course handouts are not published. This report does not indicate that a Canvas submission has been made.',small=True)

page('Appendix. Sources and audit trail')
refs=[('[1] Arena Team. Arena Human Preference 140k.','https://huggingface.co/datasets/lmarena-ai/arena-human-preference-140k'),('[2] Arena Team. A Deep Dive into Recent Arena Data.','https://arena.ai/blog/opendata-july2025'),('[3] CMPSC 448. Chapter 4: Deep Learning Architectures.','Supplied lecture: convolution, pooling, recurrent networks, and LSTM.'),('[4] CMPSC 448. Chapter 9: ML Training--Gradient descent.','Supplied lecture: loss functions, mini-batches, and Adam.'),('[5] CMPSC 448. Who Wrote It? Identifying LLMs from Their Responses.','Assignment modified September 17, 2026; deadline October 11, 2026, 11:59 pm.'),('[6] PyTorch and scikit-learn API documentation.','https://pytorch.org/docs/stable/ ; https://scikit-learn.org/stable/')]
for title,url in refs:p('<b>'+html.escape(title)+'</b>');p(html.escape(url),small=True)
sub('Immutable source revision')
p(P['revision'],small=True)
p('The official dataset card licenses user prompts under CC-BY-4.0 and states that model outputs are governed by the respective providers\' terms. The repository references the source and reproduces acquisition without redistributing raw conversation text.',small=True)
table(['Artifact','Contents'],[['docs/arena_data_provenance.json','Seven pinned URLs, file sizes, and SHA-256 hashes'],['data/arena/split_manifest.json','Non-text IDs, model labels, and fixed group assignments'],['docs/arena_experiment_plan.json; arena_selection.json','Candidates and the validation-selected configuration freeze'],['docs/arena_domain_plan.json; arena_training_complete.json','Domain protocol and checkpoint hashes before test access'],['results/arena/run_ledger.csv','Every fit, validation score, test score if evaluated, and time'],['results/arena/predictions/; verification.json','Non-text predictions, full score audit, and checkpoint replay'],['report/archive/','Earlier UltraFeedback reports and editable source']],[.47,.53],small=True)
p('Repository: https://github.com/DonDuckDuck/cmpsc448-llm-identification',small=True)

class NumberedCanvas(canvas.Canvas):
    def __init__(self,*a,**kw):super().__init__(*a,**kw);self.states=[]
    def showPage(self):self.states.append(dict(self.__dict__));self._startPage()
    def save(self):
        total=len(self.states)
        for state in self.states:
            self.__dict__.update(state);self.setStrokeColor(colors.HexColor('#D6E2E4'));self.line(48,752,564,752)
            self.setFont('Bold',7.5);self.setFillColor(TEAL);self.drawString(48,763,'CMPSC 448 / MIDTERM PROJECT')
            self.setFont('Body',7.5);self.setFillColor(GRAY);self.drawRightString(564,763,'HAODONG HUANG')
            self.drawString(48,27,'Who Wrote It? | Arena four-model study');self.drawRightString(564,27,f'{self._pageNumber} / {total}');super().showPage()
        super().save()
out=ROOT/'report';out.mkdir(exist_ok=True)
doc=SimpleDocTemplate(str(out/'Who_Wrote_It_Haodong_Huang.pdf'),pagesize=(612,792),leftMargin=48,rightMargin=48,topMargin=55,bottomMargin=48,title='Who Wrote It? Identifying LLMs from Their Responses',author='Haodong Huang')
doc.build(story,canvasmaker=NumberedCanvas);(out/'report.md').write_text('\n'.join(md));print(out/'Who_Wrote_It_Haodong_Huang.pdf')
