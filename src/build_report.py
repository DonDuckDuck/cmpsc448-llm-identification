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
F=json.loads((ROOT/'results/arena_followup/posthoc.json').read_text());C=json.loads((ROOT/'results/arena_followup/curve.json').read_text());DM=json.loads((ROOT/'results/arena_followup/domain.json').read_text())['domains']
FOLLOW_NEURAL=sum(len(v) for k,e in C['fractions'].items() if k!='1' for a,v in e.items() if a!='char')+sum(r['source']=='follow-up' for d in DM.values() for a in ['cnn','lstm'] for r in d[a]['runs']);FOLLOW_CLASSICAL=sum(len(e['char']) for k,e in C['fractions'].items() if k!='1')
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

story.append(Spacer(1,4));story.append(Paragraph('WHO WROTE IT?',styles['TitleCustom']))
p('<b>Identifying LLMs from Their Responses</b>')
p('Haodong Huang | Individual project<br/>CMPSC 448 | October 10, 2026')
md.insert(0,'# Who Wrote It? Identifying LLMs from Their Responses\n')
story.append(Spacer(1,2));sub('Abstract')
p(f'This project asks whether a neural classifier can tell which LLM wrote a response. From a pinned Arena Human Preference 140k release I curate {A["retained_rows"]:,} English, single-turn responses from four fixed releases (Claude Sonnet 4, GPT-4.1, Grok 3, DeepSeek V3), with model names masked and no prompt or session shared across splits. A validation screen selects one CNN and one LSTM recipe, each trained with three seeds on prompt-only, response-only, and combined input. Response-only macro-F1 is {f1("cnn")} for the CNN and {f1("lstm")} for the LSTM; prompts alone are near chance, and adding them does not help. Follow-up experiments show that accuracy keeps rising with training data, survives when length is hidden and when two models answer the identical prompt, comes mainly from how responses open, and largely transfers to task types removed from training. An equal-weight ensemble with character TF-IDF reaches {pct(F["ensemble"]["equal"]["macro_f1"])} ({pct(F["ensemble"]["weighted"]["macro_f1"])} with validation-chosen weights). The results describe four specific releases, not universal authorship attribution.')
sub('Where to find each grading criterion')
table(['Rubric criterion','Sections'],[['Problem definition and dataset curation','1 (data and provenance), 2 (splits and leakage control)'],['CNN/RNN implementation and training','3 (architectures), 4 (training and model selection)'],['Results and presentation','5 (main results), 6 (per-class errors), 7 (prompt effect)'],['In-depth analyses and experiments','8 (RQ3 domain transfer), 9-10 (RQ4 signals), 11 (follow-up experiments), 12 (checks)'],['Lessons and experience learned','14 (also 13: how the project evolved)']],[.40,.60],small=True)
sub('Research questions and evidence')
table(['Question','Evidence'],[
['RQ1: Identify the author?',f'CNN accuracy {pct(core["cnn_output"]["accuracy_mean"])}; LSTM accuracy {pct(core["lstm_output"]["accuracy_mean"])}. Macro-F1 and class recall accompany accuracy.'],
['RQ2: Does the prompt help?',f'Combined-minus-output macro-F1: CNN {pp(S["contrasts"]["cnn_prompt_gain"]["macro_f1_difference"])} pp; LSTM {pp(S["contrasts"]["lstm_prompt_gain"]["macro_f1_difference"])} pp. Follow-up: on {F["matched_pairs"]["pairs"]} identical-prompt pairs the CNN assigns both authors correctly {pct(F["matched_pairs"]["cnn"]["pairwise_assignment"])} of the time.'],
['RQ3: Transfer to new tasks?', f'Without coding in training, CNN coding F1 falls from {pct(DM["code"]["cnn"]["control_macro_f1_mean"])} to {pct(DM["code"]["cnn"]["holdout_macro_f1_mean"])}; without creative writing (the assignment\'s example), from {pct(DM["writing"]["cnn"]["control_macro_f1_mean"])} to {pct(DM["writing"]["cnn"]["holdout_macro_f1_mean"])}. Both far above 25% chance.'],
['RQ4: Which signals matter?', f'Word-only normalization, structural classifiers, equal-length inputs (CNN {pct(F["equal_length"]["cnn"]["macro_f1"])} F1 with length hidden), window position, and the n-grams the CNN relies on.']],[.31,.69])

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
p('Duplicate detection runs on the whole corpus but never sees labels or results. Everything a classifier learns from (vocabularies, TF-IDF features, scalers, class weights) is fitted on training data only, and an independent check confirms that no session, prompt, or duplicate group crosses splits.')
sub('What enters each classifier?')
table(['Included','Excluded'],[['Masked question and/or answer tokens','Model ID, provider name metadata, timestamps'],['A training-only vocabulary','Human winner votes, category metadata'],['Observed punctuation and newline tokens','Dataset quality scores or hidden reasoning traces']],[.50,.50])
p('Arena shows each question to only two models, so the four authors do not answer the same set of prompts. Prompt-only accuracy is therefore not fixed at 25%; it measures how much the prompts alone reveal about the author.')
p(f'Class balance is reasonably close but not exact. The test majority-class accuracy is {pct(S["majority_accuracy"])}; uniform random accuracy is 25%, and an always-one-class predictor has 25% balanced accuracy. Macro-F1 is the primary model-selection metric.',small=True)
p(f'Main study: {A["retained_rows"]:,} responses; four classes; {S["neural_fits"]} neural and {len(S["baselines"])} classical fits; {S["test_runs"]} test prediction files; 2,000 group-bootstrap resamples. Section 11 adds {FOLLOW_NEURAL} neural and {FOLLOW_CLASSICAL} classical follow-up fits, reported separately from the main-study numbers. The earlier UltraFeedback study is kept separate and is not pooled into these scores.',small=True)

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
p('The CNN reads more answer tokens than the LSTM, so CNN versus LSTM compares complete recipes, not architectures alone. Within each architecture, all three input conditions use the same answer budget.',small=True)

page('4. Training and frozen model selection')
p('Training minimizes class-weighted cross-entropy using Adam (learning rate 0.001, weight decay 0.0001), gradient norm clipping at 1.0, batches of 128, embedding dropout 0.10, and model dropout 0.30 or 0.40. The class weight is N/(4*n_class), calculated only on the training subset. These choices build on Chapter 9 optimization concepts [4].')
p('Each fit runs at most 20 epochs, with early stopping after four epochs without improvement. Checkpoints maximize validation macro-F1, then minimize validation loss to break a tie. The test set is not loaded during training or configuration selection. A pre-evaluation masking repair removed three release suffixes; all reported fits use the corrected text and regenerated split. Four response-only candidates are screened at seed 1448:')
table(['ID','Architecture / change','Validation F1','Selected'],[[cid,('CNN' if cfg['architecture']=='cnn' else 'LSTM')+f'; emb {cfg["emb"]}, width {cfg["hidden"]}'+('; bidirectional' if cfg['bidirectional'] else ''),pct(result(tag(cid,'output',1448))['best_val_macro_f1']),'Yes' if cid in [x['candidate'] for x in S['selection'].values()] else 'No'] for cid,cfg in CANDIDATES.items()],[.09,.51,.23,.17])
p('The winning recipe for each architecture is repeated at seeds 1448, 1449, and 1450 for question-only, answer-only, and combined inputs. Selected recipes also receive seed-1448 diagnostic fits. All neural and classical checkpoints are hashed before any test evaluation; rejected candidates receive no test scores.')
fig('arena_learning','Validation curves for the selected response-only seed-1448 runs. Checkpoint selection uses these curves, not test scores.',maxheight=195)
p('The screen uses one seed and four candidates, so it is a small search; the three-seed final runs show how much results vary between trainings.',small=True)

page('5. Main results: RQ1 and RQ2')
table(['Model / input','Accuracy','Macro-F1','Balanced acc.','F1 seed SD'],[[arch.upper()+' / '+mode,pct(core[arch+'_'+mode]['accuracy_mean']),pct(core[arch+'_'+mode]['macro_f1_mean']),pct(core[arch+'_'+mode]['balanced_accuracy_mean']),f'{100*core[arch+"_"+mode]["macro_f1_sd"]:.2f} pp'] for arch in ['cnn','lstm'] for mode in ['input','output','combined']],[.30,.18,.18,.19,.15])
fig('arena_results','Bars: means over three seeds. Error bars: 95% percentile intervals from 2,000 resamples of held-out session/duplicate groups.',maxheight=210)
p(f'For response-only attribution, the CNN-minus-LSTM macro-F1 difference is {change(S["contrasts"]["cnn_minus_lstm"])} {yesno(S["contrasts"]["cnn_minus_lstm"])} Both architectures are evaluated against the same rows and grouped resampling draws.')
p('The intervals capture uncertainty from which test questions happened to be sampled; they do not cover retraining, model selection, or future model releases.',small=True)

page('6. Errors and per-class performance')
fig('arena_confusion','Response-only confusion matrices, pooled over three seeds and normalized by true class. Every row sums to 100%.',maxheight=230)
table(['True class','CNN recall','LSTM recall','Test support'],[[f,pct(np.array(core['cnn_output']['confusion_matrix_sum'])[i,i]/np.array(core['cnn_output']['confusion_matrix_sum'])[i].sum()),pct(np.array(core['lstm_output']['confusion_matrix_sum'])[i,i]/np.array(core['lstm_output']['confusion_matrix_sum'])[i].sum()),A['splits']['test']['family_counts'][f]] for i,f in enumerate(LABELS)],[.27,.25,.25,.23])
CM={a:np.array(core[a+'_output']['confusion_matrix_sum'])/np.array(core[a+'_output']['confusion_matrix_sum']).sum(1,keepdims=True) for a in ['cnn','lstm']}
p(f'Claude is the easiest author to recognize (CNN recall {pct(CM["cnn"][0,0])}). DeepSeek is the hardest: the LSTM recognizes {pct(CM["lstm"][3,3])} of its answers and labels {pct(CM["lstm"][3,1])} of them GPT. GPT and DeepSeek are the most confused pair in both directions for both models, so their styles are the most alike in this data.')
sub('Descriptive task slices')
table(['Model','Coding F1','Non-coding F1'],[[a.upper(),pct(S['subgroups'][a+'_code']['macro_f1_mean']),pct(S['subgroups'][a+'_noncode']['macro_f1_mean'])] for a in ['cnn','lstm']],[.34,.33,.33])
p('These slices use models trained on mixed tasks. They are descriptive performance breakdowns, not evidence of generalization to an unseen domain. The next domain-transfer experiment specifically removes coding tasks from training.',small=True)

page('7. Does the question help?')
for arch in ['cnn','lstm']:
 c=S['contrasts'][arch+'_prompt_gain'];sub(arch.upper())
 p(f'Question-only macro-F1 is {f1(arch,"input")}, response-only is {f1(arch)}, and combined is {f1(arch,"combined")}. The paired combined-minus-response difference is {change(c)} {yesno(c)}')
sub('How to interpret input-only prediction')
p('Prompt-only accuracy is near chance. The prompt is written before the model answers, so whatever it reveals cannot be the model\'s fingerprint; any residual signal can only come from which prompts each model happened to receive. For example, Claude Sonnet 4 first appears in this data on May 22, 2025, so it never answered April prompts.')
p('Adding the prompt to the answer does not help: for the CNN the change is within noise, and for the LSTM it slightly hurts, plausibly because the extra tokens dilute the answer signal. Section 11C tests the prompt question directly with pairs of answers to the identical prompt.')
sub('Classical prompt check')
p(f'The fixed word TF-IDF question-only baseline achieves {pct(S["baselines"]["word_input"]["accuracy"])} accuracy and {pct(S["baselines"]["word_input"]["macro_f1"])} macro-F1. It offers an independent check for prompt information with a different representation.')
p('The answer prefix is unchanged when the question is added: combined input concatenates the 96-token question, a separator, and the same 384-token CNN or 192-token LSTM answer prefix. Thus a change in score is not caused by truncating away the answer to make room for the question.')

page('8. Cross-task generalization: RQ3')
p('A non-coding training subset excludes all rows with Arena is_code=true. Both domain-transfer models validate on non-code rows only. A mixed-domain control uses exactly the same number of training examples in each class, sampled from the original training split. Both are evaluated on the identical coding subset of the untouched test split. This separates the loss of coding exposure from a simple reduction in training size.')
table(['Model / training','Train rows','Coding accuracy','Coding F1'],[[arch.upper()+' / '+kind,S['domain_transfer'][arch][kind]['training_rows'],pct(S['domain_transfer'][arch][kind]['accuracy']),pct(S['domain_transfer'][arch][kind]['macro_f1'])] for arch in ['cnn','lstm'] for kind in ['holdout','control']],[.37,.17,.23,.23])
for arch in ['cnn','lstm']:
 d=S['domain_transfer'][arch];p(f'{arch.upper()}: non-code-minus-control coding macro-F1 is {pp(d["holdout_minus_control_macro_f1"])} pp, with paired 95% group-bootstrap interval {interval(d["difference_ci95"])}. Coding test support is {d["holdout"]["test_rows"]} responses.')
p('Without any coding examples, both models still identify authors on coding prompts far above chance, so part of the fingerprint transfers. The drop against the control shows that seeing the task type in training still helps. Arena\'s code label is only a proxy for domain, so this is evidence of partial transfer, not of a universal fingerprint.')
p('This main-study version uses one seed, and the architecture recipes come from the mixed-domain screen, so coding was not completely unseen during design. Section 11F repeats it with three seeds and adds a second domain, the assignment\'s own example: training without creative writing and testing on it.',small=True)

page('9. Fingerprint signals: RQ4')
table(['Baseline','Test accuracy','Macro-F1'],[[name.replace('_',' '),pct(m['accuracy']),pct(m['macro_f1'])] for name,m in S['baselines'].items()],[.50,.25,.25])
p(f'The character baseline reaches {pct(S["baselines"]["char_output"]["accuracy"])} accuracy and {pct(S["baselines"]["char_output"]["macro_f1"])} macro-F1, a slightly higher point estimate than the CNN mean. This highlights the strength of local lexical patterns; the small ranking difference alone does not establish reliable superiority.')
p('Word TF-IDF uses 1-2 grams; character TF-IDF uses 3-5 grams. Each is capped at 40,000 features, min_df=3, with sublinear term frequency and balanced logistic regression (C=2). Text baselines use the first 96 question and/or 384 answer tokens. Their longer answer budget than LSTM is disclosed. All transforms are fitted on training data.')
p('Structural baselines use full-answer statistics: log character/word counts, average word length, lexical diversity, newline/punctuation/digit/uppercase rates, list-line rate, and code-fence count. The no-length variant removes the two explicit log length features; other features can still correlate with length. These diagnostics are not controlled equal-input-budget competitors.')
sub('Remove punctuation and newline information')
table(['Model','Raw answer F1, seed 1448','Word-only F1','Difference'],[[a.upper(),pct(result(tag(S['selection'][a]['candidate'],'output',1448))['test']['macro_f1']),pct(S['diagnostics'][a+'_normalized']['macro_f1']),pp(S['contrasts'][a+'_normalization']['macro_f1_difference'])+' pp'] for a in ['cnn','lstm']],[.17,.32,.25,.26])
for a in ['cnn','lstm']:p(f'{a.upper()} normalization contrast: {change(S["contrasts"][a+"_normalization"])}',small=True)
p(f'Removing punctuation and newlines costs the CNN {-100*S["contrasts"]["cnn_normalization"]["macro_f1_difference"]:.1f} points and the LSTM {-100*S["contrasts"]["lstm_normalization"]["macro_f1_difference"]:.1f}. Because the word-only version also fits more words into the same token budget, this measures the value of formatting tokens as a group, not of any single feature.',small=True)

page('10. Structural profiles and interpretation')
p('The following summaries use training responses only. They describe the curated distribution, rather than selecting examples after seeing test errors. Medians are robust summaries but conceal substantial within-class variation.')
table(['Source','Median words','Lexical diversity','Newline / 1k chars','Punctuation / 1k chars'],[[f,f'{np.expm1(S["training_style_profiles"][f]["log_words"]):.0f}',f'{S["training_style_profiles"][f]["lexical_diversity"]:.3f}',f'{1000*S["training_style_profiles"][f]["newline_rate"]:.1f}',f'{1000*S["training_style_profiles"][f]["punctuation_rate"]:.1f}'] for f in LABELS],[.16,.18,.22,.22,.22])
p('Here a word is a regex word token, lexical diversity is unique lowercased words divided by all words, and character rates are measured on the full masked answer. These word counts differ from whitespace counts used in the raw curation audit and from the neural tokenizer.')
p('Response length is itself a source signal: Grok\'s median answer is about four times as long as Claude\'s. Truncation limits how much the networks read, but a short answer still reveals its length through how many real tokens it has. Section 11B tests this directly by giving the models inputs of identical length.')
sub('What the analyses establish')
p(f'Word and character patterns alone are enough to identify authors well (char TF-IDF {pct(S["baselines"]["char_output"]["accuracy"])}). A handful of formatting and length summaries reaches {pct(S["baselines"]["style_all"]["accuracy"])}. Removing punctuation and newlines hurts both networks. None of these shows the network\'s internal decision rule directly; Section 11D traces the CNN\'s decisions to specific phrases.')
sub('What would strengthen a later study?')
p('A future independent experiment could collect the same prompts from all four models under matched generation settings and hold out newer model versions. Section 11 takes first steps with equal-length inputs and the identical-prompt pairs Arena already contains. That would separate provider, prompt distribution, generation settings, and length more directly. Such follow-up work needs a new evaluation set, because this test set has now been used for the reported study.')

page('11. Follow-up experiments after the main evaluation')
p('The main study above was frozen and evaluated once. This section asks four questions the main study left open: does more training data help, is the classifier only reading answer length, does the fingerprint survive when the prompt is identical, and where in an answer does it live? Part F then extends RQ3. The fractions, seeds, and ensemble rule were fixed in code and recorded in docs/arena_followup_plan.json before the runs, and no configuration was chosen using test labels. Because the test split had already been seen, these are follow-up analyses, not a second confirmation. One disclosure: the length strata and matched-pair counts were previewed from saved predictions before the plan was written; nothing was changed as a result.')
sub('A. Learning curve: does more training data help?')
keys=sorted(C['summary'],key=float)
table(['Training responses','CNN macro-F1','LSTM macro-F1','Char TF-IDF macro-F1'],[[f"{C['rows'][k]:,} ({100*float(k):g}%)"]+[pct(C['summary'][k][m]['macro_f1_mean'])+(f" ± {100*C['summary'][k][m]['macro_f1_sd']:.1f}" if C['summary'][k][m]['macro_f1_sd'] else '') for m in ['cnn','lstm','char']] for k in keys],[.28,.24,.24,.24],small=True)
fig('followup_learning_curve','Selected CNN/LSTM recipes and the char TF-IDF baseline retrained on class-stratified subsets of the same training split; validation and test are unchanged. Points: mean ± SD over seeds 1448-1450 (the full-data char baseline is a single fit).',maxheight=200)
gain={m:C['summary']['1'][m]['macro_f1_mean']-C['summary']['0.125'][m]['macro_f1_mean'] for m in ['cnn','lstm','char']}
p(f'Going from one eighth of the training data to all of it adds {100*gain["cnn"]:.1f} points of macro-F1 for the CNN, {100*gain["lstm"]:.1f} for the LSTM, and {100*gain["char"]:.1f} for char TF-IDF. No curve has flattened at the right edge, so more data would likely help further. The CNN tracks char TF-IDF closely at every size, while the LSTM, which learns its sequence representation from scratch, gains the most from data and is still well behind at full size. This is the experiment Section 13 needed: within one fixed task, more examples did improve attribution.')
sub('B. Is the classifier just measuring length?')
L=F['length_strata'];E=F['equal_length']
p(f'Grok answers are long (median about 1,480 neural tokens in test) and Claude answers short (about 300), so a model could in principle count padding. Two checks argue against length as the main cue. First, on answers of at least 384 tokens, where the CNN window is completely full and contains no padding, CNN macro-F1 is {pct(L[">=384"]["cnn"]["macro_f1"])} ({L[">=384"]["rows"]} answers). Second, the frozen models were re-run with every test answer cut to exactly its first 128 tokens, dropping the {A["splits"]["test"]["rows"]-E["rows"]} shorter answers, so all inputs have identical length: CNN {pct(E["cnn"]["macro_f1"])}, LSTM {pct(E["lstm"]["macro_f1"])}, char TF-IDF {pct(E["char"]["macro_f1"])} macro-F1.')
fig('followup_length','Response-only macro-F1 by answer length (CNN and LSTM: three-seed means). Short answers are hardest: they carry fewer distinctive tokens.',maxheight=190)
p('About 82% macro-F1 with length hidden is lower than the full-input score, so length and later tokens carry some signal, but most of the attribution survives without them.',small=True)
sub('C. Same prompt, two authors (RQ2)')
M=F['matched_pairs']
p(f'For {M["pairs"]} test battles both responses survived curation, giving identical prompts answered by two different models. A prompt-only classifier cannot separate them: the CNN prompt-only model scores exactly {pct(M["cnn_prompt_only"]["pairwise_assignment"])} on the pairwise task. Given the two answers and the two candidate authors, the response-only CNN assigns them correctly {pct(M["cnn"]["pairwise_assignment"])} of the time (LSTM {pct(M["lstm"]["pairwise_assignment"])}, char TF-IDF {pct(M["char"]["pairwise_assignment"])}); the CNN labels both sides correctly among all four classes {pct(M["cnn"]["both_sides_correct"])} of the time. With the topic held fixed, the answer still identifies its author. The sample is small, so treat these as a direction rather than a precise rate.')
sub('D. Where is the fingerprint? (RQ4)')
Wp=F['window_position']
table(['192-token window','CNN macro-F1','LSTM macro-F1'],[[w.title(),pct(Wp['cnn'][w]['macro_f1']),pct(Wp['lstm'][w]['macro_f1'])] for w in ['first','middle','last']],[.4,.3,.3],small=True)
p(f'On the {Wp["rows"]} test answers of at least 576 tokens, frozen models read a first, middle, or last 192-token window. Accuracy is highest on the opening, collapses in the middle, and partly recovers at the end. Both models were trained on answer prefixes, which favors the first window. Neither was trained specifically on endings; they saw sign-offs only in answers short enough to fit their window. That the last window still recovers suggests sign-offs such as "Would you like…" are distinctive. Claude has only {Wp["family_counts"].get("Claude",0)} answers this long.')
def show(g):
    for c in '#*-':g=re.sub(rf'(?<=\{c}) (?=\{c})','',g)
    g=re.sub(r"(?<=\w) ([’']) (?=\w)",r'\1',g);g=re.sub(r" ([’']) ",r'\1',g);g=re.sub(r' ([.,!?:])',r'\1',g);return g
NG=F['cnn_ngrams']
table(['Author','Most decisive CNN n-grams (training answers, share of correctly classified answers)'],[[f,'; '.join(f'{show(x["ngram"])} ({100*x["share_of_docs"]:.0f}%)' for x in NG[f]['top'][:6])] for f in LABELS],[.15,.85],small=True)
p('For each correctly classified training answer, the five convolution filters that push hardest toward the true class were traced back to the n-gram at their max-pooled position. Shown are n-grams appearing for at least 15 answers and at least 60% of the time for one class (↵ is a newline). The pattern is readable: GPT opens with "Certainly! Here\'s", "Absolutely! Here\'s", or "Great question!"; Grok with "Below is a detailed…"; Claude says "I\'d recommend" and closes by asking a question ("Is there…? That would help"); DeepSeek uses bolded numbered headers and closes with "Would you like…". The fingerprint is largely conversational framing and Markdown habits, which explains why the opening window is most informative and why removing punctuation and newlines (Section 9) costs so much.')
sub('E. Two more checks')
K=F['ensemble']
p(f'<b>Identity placeholder.</b> Removing the {F["mask_free"]["excluded"]} test answers that contain a [MODEL] mask leaves CNN macro-F1 at {pct(F["mask_free"]["cnn"]["macro_f1"])} and LSTM at {pct(F["mask_free"]["lstm"]["macro_f1"])}, essentially unchanged, so masked self-references are not a shortcut.')
p(f'<b>Ensemble.</b> Averaging the probabilities of the CNN (three seeds), the LSTM (three seeds), and char TF-IDF combines different views of the text. Weights chosen on validation (CNN {K["weights"]["cnn"]:.2f}, LSTM {K["weights"]["lstm"]:.2f}, char {K["weights"]["char"]:.2f}) reach {pct(K["weighted"]["macro_f1"])} test macro-F1, {pp(K["weighted"]["minus_char"])} pp over char TF-IDF, 95% CI {interval(K["weighted"]["minus_char_ci95"])}. The pre-specified equal-weight average reaches {pct(K["equal"]["macro_f1"])}, {pp(K["equal"]["minus_char"])} pp, CI {interval(K["equal"]["minus_char_ci95"])}. Validation-tuned weights did not beat the simple average here, a reminder that 1,442 validation answers make weight tuning noisy. The equal-weight result suggests the CNN and LSTM capture information that char TF-IDF misses; the validation-weighted interval includes zero, so the size of that gain is uncertain.')
sub("F. RQ3 extended: three seeds and the assignment's writing example")
p("Section 8 ran one domain split with one seed. Here the coding holdout is repeated with seeds 1449 and 1450 (seed 1448 is the frozen main-study fit), and a second split follows the assignment's own example: train on everything except creative writing, so coding, math, factual, and other prompts remain, and test on creative writing. Each holdout is compared with a mixed-domain control of identical per-class training size, sampled from the full training split. This extension was planned in docs/arena_followup_domain_plan.json and committed to Git before any of its models were trained.")
TASK={'code':'Coding','writing':'Creative writing'}
table(['Test task (rows)','Model','Control F1','Holdout F1','Holdout minus control [95% CI]'],[[f'{TASK[d]} ({DM[d]["test_rows"]})',a.upper(),f'{pct(DM[d][a]["control_macro_f1_mean"])} ± {100*DM[d][a]["control_macro_f1_sd"]:.1f}',f'{pct(DM[d][a]["holdout_macro_f1_mean"])} ± {100*DM[d][a]["holdout_macro_f1_sd"]:.1f}',f'{pp(DM[d][a]["difference"])} pp {interval(DM[d][a]["difference_ci95"])}'] for d in ['code','writing'] for a in ['cnn','lstm']],[.22,.11,.17,.17,.33],small=True)
fig('followup_domain','Target-task test macro-F1, mean ± SD over seeds 1448-1450. Control: mixed-domain training of the same per-class size. Holdout: the target task type removed from training and validation.',maxheight=200)
p(f'Removing a task type from training costs about 4 to 9 points of macro-F1 on that task. The drop is clear for both models on coding and for the LSTM on writing; for the CNN on writing the interval reaches zero, with only {DM["writing"]["test_rows"]} test answers. Yet every holdout model stays far above the 25% chance level. Most of the fingerprint is therefore general and carries over to an unseen task type, while a smaller part is task-specific. The three-seed coding results confirm the single-seed finding of Section 8, and the LSTM drop ({pp(DM["code"]["lstm"]["difference"])} pp) is larger than its seed-1448 estimate ({pp(S["domain_transfer"]["lstm"]["holdout_minus_control_macro_f1"])} pp), which shows why one seed is not enough.')
R={d:DM[d]['cnn']['recall'] for d in DM}
table(['Author','Coding: control','Coding: holdout','Writing: control','Writing: holdout'],[[f]+[pct(R[d][k][f]) for d in ['code','writing'] for k in ['control','holdout']] for f in LABELS],[.2,.2,.2,.2,.2],small=True)
p(f'The table shows CNN recall by author (mean of three seeds); the LSTM shows the same pattern. This answers the assignment\'s question of whether some models are more consistently identifiable across domains: Claude stays recognizable on both unseen task types ({pct(R["code"]["holdout"]["Claude"])} on coding, {pct(R["writing"]["holdout"]["Claude"])} on writing, close to its control). DeepSeek loses the most, about 20 points on each task, so what separates it from the other authors depends more on having seen that kind of task. Per-author writing numbers rest on {min(DM["writing"]["test_family_counts"].values())} to {max(DM["writing"]["test_family_counts"].values())} test answers each.')
page('12. Checks and limitations')
p(f'The shuffled-training-label CNN has test accuracy {pct(S["diagnostics"]["shuffled_cnn"]["accuracy"])} and macro-F1 {pct(S["diagnostics"]["shuffled_cnn"]["macro_f1"])}. Labels are permuted only in training; validation and test labels remain unchanged. This is a pipeline sanity check, not a proof that all confounding is eliminated.')
table(['Verified check','Evidence'],[['Session, prompt, duplicate-group separation','Independent assertions across all splits'],['Validation-only model choice',f'{V["validation_files_recomputed"]} validation prediction files recomputed; selected IDs reproduced'],['Reported test scores',f'{V["test_prediction_files_recomputed"]} prediction files independently rescored'],['Saved-model inference','CNN/LSTM checkpoint replay against stored probabilities'],['Implementation invariants','16 unit tests: masking, attribution, vocabulary, padding, recurrent states, class count, follow-up subsetting and domain switching']],[.40,.60])
sub('Material limitations')
p('The study uses one Arena snapshot, four releases, English first turns, and filtered nontrivial answers. It excludes later model generations, reasoning variants, multilingual text, multi-turn conversations, and multimodal inputs. Automatic identity masking cannot remove all semantic self-identification. Model-side prompts may differ by author; prompt-only controls measure part of this concern but cannot erase it.')
p('Question or answer truncation may discard useful evidence. Softmax values are uncalibrated and should not be presented as trustworthy probabilities for arbitrary text. Future reuse of this test set for tuning would turn subsequent results into follow-up exploration, requiring another independent test for confirmation.')

page('13. Project evolution and dataset choice')
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
p(f'The Arena study adds a four-class target, explicit release selection, session/near-duplicate grouping, a frozen validation screen before test evaluation, and matched-size domain controls. Its higher accuracy is an observed result on a different task. Changes in author classes, text distribution, and split design prevent attributing the difference to sample size. The follow-up learning curve in Section 11 isolates training size on the fixed Arena split: with the same recipes, validation set, and test set, CNN macro-F1 rises from {pct(C['summary']['0.125']['cnn']['macro_f1_mean'])} with {C['rows']['0.125']:,} training responses to {pct(C['summary']['1']['cnn']['macro_f1_mean'])} with {C['rows']['1']:,}. More examples did help within Arena; the curve still says nothing about the cross-dataset jump.')

page('14. Lessons and experience learned')
cs=C['summary'];V1,V2=json.loads((ROOT/'results/summary.json').read_text()),json.loads((ROOT/'results/v2/summary.json').read_text());UF_TRAIN=sum(V1['source_sizes']['train'].values())
def acc_of(s,a):return s['core'][a+'_output']['accuracy_mean']
p('This section is written from my own notes on the project. The earlier sections report what was measured; this one reports what I learned while measuring it, what surprised me, what went wrong and how it was fixed, and what I would do differently. Every number below comes from the saved result files used in the rest of the report.')
sub('Lessons at a glance')
table(['#','Lesson','Evidence in this project'],[
['1','The dataset sets the ceiling.',f'Same-data tuning took the LSTM from {pct(acc_of(V1,"lstm"))} to {pct(acc_of(V2,"lstm"))}; only a new dataset moved it past 80%.'],
['2','More data beat more difficulty.',f'Four classes instead of three, yet the LSTM reached {pct(core["lstm_output"]["accuracy_mean"])}. At the old training size it sits near the old 77% plateau.'],
['3','Always run a cheap baseline.',f'Char TF-IDF ({pct(S["baselines"]["char_output"]["macro_f1"])} F1) edged out the CNN ({pct(core["cnn_output"]["macro_f1_mean"])}); an equal-weight combination gave {pct(F["ensemble"]["equal"]["macro_f1"])}.'],
['4','The fingerprint is in the framing.','Openers such as "Certainly! Here\'s" and "Below is a detailed" drive decisions; the first window is easiest.'],
['5','Choose the data before the model.','Arena covers mainstream, actively updated models; Falcon-era data could not grow.'],
['6','Test for shortcuts before trusting a score.','Length, identity-mask, and same-prompt controls in Section 11 all held up.']],[.05,.33,.62],small=True)

sub('1. The data limited the first version more than the model did')
p(f'The hardest part of the project was the first version. On UltraFeedback (Falcon, GPT, Llama; 4,254 usable responses) the LSTM reached only {pct(acc_of(V1,"lstm"))} accuracy. I spent the second stage improving the model on the same data: a bounded validation search over twelve candidates, a longer CNN input, and LSTM pooling. That lifted the LSTM to {pct(acc_of(V2,"lstm"))} and the CNN to {pct(acc_of(V2,"cnn"))}, but I could not find a way past that level. I also could not simply add examples, because public datasets that contain those same model releases barely exist, especially for Falcon. To make a real breakthrough I had to change the dataset and essentially start again on Arena.')
fig('lessons_journey','Response-only test accuracy at each stage (three-seed means). Stages 1-2 share one dataset and three classes; stage 3 is a different, four-class task, so the jump is not a controlled comparison.',maxheight=200)
p('The lesson is that the dataset sets the ceiling. It decides how many examples exist, which authors can be studied, and therefore how far any later tuning can go. Model work in stage 2 was still valuable, since it is where I learned how much pooling matters for an LSTM, but it could not compensate for a small, fixed dataset.')

sub('2. More classes, yet higher accuracy')
p(f'I expected four authors to be harder to separate than three, since chance accuracy drops from 33% to 25%. Instead, accuracy went up: the LSTM reached {pct(core["lstm_output"]["accuracy_mean"])} and the CNN {pct(core["cnn_output"]["accuracy_mean"])}, close to 90%. I believe a large part of the reason is simply more data. The follow-up learning curve tests this within Arena, holding the task, recipe, validation set, and test set fixed and changing only the number of training answers.')
fig('lessons_data','Arena learning curve in test accuracy (three-seed means). Dotted lines mark where the UltraFeedback LSTM stopped and the dashed line marks the UltraFeedback training size. UltraFeedback is a different three-class task, so these marks are a visual reference, not a controlled comparison.',maxheight=200)
p(f'The first version trained on {UF_TRAIN:,} answers. At a similar size, {C["rows"]["0.5"]:,} Arena answers, the Arena LSTM reaches {pct(cs["0.5"]["lstm"]["accuracy_mean"])} accuracy, about where the tuned UltraFeedback LSTM stopped ({pct(acc_of(V2,"lstm"))}), even though the Arena task has four classes instead of three. With all {C["rows"]["1"]:,} answers it reaches {pct(cs["1"]["lstm"]["accuracy_mean"])}. This splits the story in two. The very first version\'s {pct(acc_of(V1,"lstm"))} came mainly from the LSTM design, which read only the final hidden state; stage 2 fixed that. What then held the LSTM near 77% was the amount of training data. Had I stayed with a dataset of the old size, I would very likely still be stuck there. The two datasets also differ in authors and prompts, so data size is a strong explanation but not the only one.')

sub('3. Assumptions versus what happened')
table(['A natural assumption','What actually happened','Takeaway'],[
['Four classes would be harder than three.',f'Accuracy rose to {pct(core["cnn_output"]["accuracy_mean"])} (CNN).','Sample size can matter more than task difficulty.'],
['A deep model would beat a classical one.',f'Char TF-IDF scored {pct(S["baselines"]["char_output"]["macro_f1"])} F1 vs. CNN {pct(core["cnn_output"]["macro_f1_mean"])}.','Always measure against a cheap baseline.'],
['Adding the prompt would help.',f'Combined input changed F1 by {pp(S["contrasts"]["cnn_prompt_gain"]["macro_f1_difference"])} pp (CNN) and {pp(S["contrasts"]["lstm_prompt_gain"]["macro_f1_difference"])} pp (LSTM).','The answer alone carries the fingerprint.'],
['Models would be told apart by content.','Openers, sign-offs, and Markdown habits drive decisions.','Style of presentation is the main signal.'],
['Long Grok answers might make length a shortcut.',f'Equal 128-token inputs still give {pct(F["equal_length"]["cnn"]["macro_f1"])} CNN F1.','Length helps a little, but is not the main cue.'],
['Fingerprints would not survive a new task type.',f'Trained without coding or writing, CNN F1 on those tasks is {pct(DM["code"]["cnn"]["holdout_macro_f1_mean"])} and {pct(DM["writing"]["cnn"]["holdout_macro_f1_mean"])}.','Most of the fingerprint is general; a smaller part is task-specific.'],
['The LSTM would read text better than the CNN.',f'The CNN led by {pp(S["contrasts"]["cnn_minus_lstm"]["macro_f1_difference"])} pp F1.','Local n-gram detectors suit this task well.']],[.30,.36,.34],small=True)
p(f'The baseline surprise was the most useful one. Character TF-IDF with logistic regression trains in seconds, yet it slightly outscored my CNN. A deep model is not automatically better than a well-chosen classical one. The two are complementary, though: averaging the CNN, LSTM, and TF-IDF probabilities with equal weights reached {pct(F["ensemble"]["equal"]["macro_f1"])} macro-F1, the best result in the project ({pct(F["ensemble"]["weighted"]["macro_f1"])} with validation-chosen weights).')

sub('4. The fingerprint is in how a response starts')
p('The third surprise was where the evidence lives. When I traced the CNN\'s decisions back to the phrases that triggered them, they were mostly about how the model frames a response rather than what it says.')
fig('lessons_signatures','Phrases at the max-pooled positions of the CNN filters that push hardest toward the correct author, on correctly classified training answers (Section 11D method; phrases with at least two words).',maxheight=215)
p('GPT opens with enthusiasm ("Certainly! Here\'s", "Great question!"), Grok announces structure ("Below is a detailed…"), Claude gives a personal recommendation and asks a follow-up ("I\'d recommend", "Is there…"), and DeepSeek offers more help ("Would you like…"). This matches the window experiment: identification is easiest on the first 192 tokens and hardest in the middle of a long answer. What I learned is that an LLM "fingerprint" in this data is mostly a habit of presentation, which also suggests it could change quickly when providers update their models.')

sub('5. Was switching to Arena worth it?')
table(['','UltraFeedback (stages 1-2)','Arena (stage 3)'],[['Authors','Falcon, GPT, Llama','Claude, GPT, Grok, DeepSeek'],['Relevance','Older or less widely used models','Mainstream models that keep releasing new versions'],['Usable responses','4,254',f'{A["retained_rows"]:,}'],['Room to grow','Few datasets contain the same releases','Large public release with exact model IDs']],[.22,.39,.39],small=True)
p('Looking back, switching was definitely worth it. ChatGPT, Claude, and Grok are used far more than Falcon, so studying them answers a question about text people actually read today. A reminder for my future self: before starting, search for the best and most current dataset for the question and understand what it contains, instead of picking a dataset more or less at random and starting work right away.')

sub('6. Mistakes, risks, and how they were fixed')
table(['Problem or risk','How it was noticed','Fix','Result'],[
['LSTM summarized the answer with only its final hidden state.','Weak LSTM scores in stage 1.','Masked mean and max pooling over all real tokens.',f'LSTM {pct(acc_of(V1,"lstm"))} to {pct(acc_of(V2,"lstm"))}.'],
['Stage-2 tuning happened after seeing stage-1 test scores.','Reviewing the protocol.','Stage 2 labelled exploratory; Arena selection frozen on validation before any test access.','Clean main study.'],
['Three answers still contained model-version suffixes after masking.','Audit before test evaluation.','Masking repaired; every fit rerun from scratch.','No identity strings left.'],
['Near-duplicate prompts could leak across splits.','Popular prompts recur across Arena battles.','Group by session, prompt, answer, and TF-IDF similarity of at least 0.90.','No group crosses splits.'],
['The classifier might just count answer length.','Grok answers are about five times longer than Claude\'s.','Equal 128-token inputs; length strata.',f'CNN still {pct(F["equal_length"]["cnn"]["macro_f1"])} F1.'],
['[MODEL] placeholders might act as a shortcut.','Placeholder rates differ slightly by class.','Score answers without any placeholder.',f'CNN {pct(F["mask_free"]["cnn"]["macro_f1"])} F1, unchanged.']],[.27,.22,.30,.21],small=True)
p('Most of these were not visible in the accuracy number. A model can score well for the wrong reason, so each risk needed its own targeted check. Writing the plan and freezing the model choice before looking at test results also turned out to matter: once a test set has been seen, every later change becomes exploratory.')

sub('7. What I would change if I started again')
table(['Step','What I would do','Why'],[['1','Search for the best, most current dataset and read its documentation first.','The dataset sets the ceiling (Lessons 1, 2, 5).'],['2','Write down the split, metric, and test policy before training.','Avoids test reuse and leakage (Lesson 6).'],['3','Start with the simplest approach: a TF-IDF baseline.','Shows how strong simple patterns are (Lesson 3).'],['4','Work up to the CNN and then the LSTM, comparing each with the step before.','Every gain from complexity becomes visible.'],['5','Run shortcut checks (length, identity, same prompt) early.','Confirms the model learns style, not artifacts.']],[.08,.52,.40],small=True)

page('15. Reproduction and submission')
p('Install Python 3.12 and the exact dependencies in requirements-lock.txt. The recorded runs use PyTorch '+result(tag(S['selection']['cnn']['candidate'],'output',1448))['torch_version']+' with Apple MPS. CPU inference replay verifies the two distributed demonstration weights; training on other devices can yield numerical differences.')
table(['Step','Command'],[['Download pinned data','python src/arena_data.py download'],['Create the fixed split once','python src/arena_data.py prepare'],['Screen, select, and fit core models','python src/arena_train.py train'],['Fit planned domain-transfer checks','python src/arena_train.py domain'],['Evaluate frozen models','python src/arena_train.py evaluate'],['Verify metrics and build figures','python src/arena_analyze.py'],['Run implementation checks','python -m unittest discover -s tests -v'],['Follow-up plan, diagnostics, learning curve, RQ3 extension','python src/arena_followup.py plan | posthoc | curve | domain'],['Lessons figures','python src/build_lessons_figures.py'],['Build this report','python src/build_report.py']],[.40,.60],small=True)
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
