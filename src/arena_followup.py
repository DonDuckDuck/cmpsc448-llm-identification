"""Follow-up Arena study, added after the main test evaluation.

Nothing here changes the frozen main study. New fits reuse arena_train.fit with
redirected output folders, so the training recipe is identical. The test split was
already used once, so every number produced here is follow-up analysis: no
configuration, threshold, or weight is chosen with test labels.

Phases:
  plan     write docs/arena_followup_plan.json before any follow-up result exists
  curve    learning curve: selected CNN/LSTM and char TF-IDF on class-stratified training fractions
  posthoc  no training: length strata, matched-prompt pairs, mask-free subset,
           equal-length windows, validation-weighted ensemble, CNN n-gram evidence
"""
import argparse,contextlib,itertools,json,re,time
from collections import Counter,defaultdict
import joblib,numpy as np,torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import arena_train
from arena_data import ROOT,DEST,LABELS,write
from arena_train import RESULT,CHECK,SEEDS,Data,read,digest,load,metrics,tag,device,baseline_features
from arena_analyze import scores
from train import tokenize,collate,predict
from models_v2 import make_model

OUT=ROOT/'results/arena_followup';FOLLOW_CHECK=ROOT/'results/checkpoints/arena_followup';FIGURES=ROOT/'results/figures'
PLAN_PATH=ROOT/'docs/arena_followup_plan.json'
FRACTIONS=[.125,.25,.5]
COLORS=['#0B7774','#DC8242','#536FA8','#915E91']
PLAN={'status':'Follow-up after the main frozen test evaluation; the test split has been seen, so results are exploratory and are reported separately from the main study.',
      'frozen_inputs':'docs/arena_selection.json chooses the CNN and LSTM recipes; no new candidates, epochs, or hyperparameters are searched.',
      'learning_curve':{'fractions':FRACTIONS,'seeds':SEEDS,'models':['selected CNN','selected LSTM','char TF-IDF 3-5 grams'],
                        'sampling':'Per-class random subset of the training split, drawn with numpy default_rng([seed, fraction*1000]); validation and test are the unchanged full splits.',
                        'full_data_point':'Reuses the existing main-study fits (seeds 1448-1450; char TF-IDF single fit) at fraction 1.0.'},
      'posthoc':{'length_strata':'Neural-tokenizer answer length <192, 192-383, >=384 tokens.',
                 'matched_pairs':'Test battles whose two sides were both retained: identical prompt, two different authors.',
                 'mask_free':'Test rows whose answer contains no [MODEL] placeholder.',
                 'equal_length':'Frozen models read exactly the first 128 answer tokens of every test answer with at least 128 tokens.',
                 'window_position':'Frozen models read 192-token first, middle, and last windows of test answers with at least 576 tokens.',
                 'ensemble':'Average class probabilities of CNN (3-seed mean), LSTM (3-seed mean), and char TF-IDF. Weights on a 0.05 simplex grid maximize validation macro-F1; the equal-weight average is reported too.',
                 'cnn_ngrams':'Seed-1448 CNN on training answers only: n-grams at the max-pooled position of the filters that contribute most to the correct class.'},
      'uncertainty':'2,000 paired resamples of test duplicate groups, seed 1448, matching the main study.'}

def plan():
    if PLAN_PATH.exists():assert read(PLAN_PATH)['plan']==PLAN,'Follow-up plan changed after it was frozen';return
    write(PLAN_PATH,{'plan':PLAN,'frozen_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
                     'selection_sha256':digest(ROOT/'docs/arena_selection.json'),'split_manifest_sha256':digest(DEST/'split_manifest.json')})

def subsample(rows,fraction,seed):
    """Class-stratified subset; keeps the original row order."""
    rng=np.random.default_rng([seed,int(round(fraction*1000))]);keep=[]
    for family in LABELS:
        idx=[i for i,r in enumerate(rows) if r['family']==family]
        keep.extend(rng.choice(idx,int(round(fraction*len(idx))),replace=False).tolist())
    return [rows[i] for i in sorted(keep)]

@contextlib.contextmanager
def redirected(folder,train_rows):
    """Run arena_train.fit unchanged, but write to a follow-up folder and train on a subset."""
    saved=arena_train.RESULT,arena_train.CHECK,arena_train.load
    check=FOLLOW_CHECK/folder.name;folder.mkdir(parents=True,exist_ok=True);check.mkdir(parents=True,exist_ok=True)
    arena_train.RESULT,arena_train.CHECK=folder,check
    arena_train.load=lambda split:train_rows if split=='train' else saved[2](split)
    try:yield check
    finally:arena_train.RESULT,arena_train.CHECK,arena_train.load=saved

def run_checkpoint(path,rows,windows=None):
    """Class probabilities from a saved neural checkpoint; `windows` optionally replaces each answer's tokens."""
    saved=torch.load(path,map_location='cpu',weights_only=True);dev=device()
    model=make_model(saved['config'],len(saved['vocab'])).to(dev);model.load_state_dict(saved['state_dict'])
    data=Data(rows,saved['vocab'],saved['mode'],saved['config'])
    if windows is not None:data.items=[([saved['vocab'].get(t,1) for t in w] or [1],y) for w,(_,y) in zip(windows,data.items)]
    y,p,_=predict(model,DataLoader(data,batch_size=128,collate_fn=collate),dev);return y,p

def short(m):return {k:m[k] for k in ['accuracy','macro_f1','balanced_accuracy']}

def curve():
    plan();selection=read(ROOT/'docs/arena_selection.json')['selection'];train,test=load('train'),load('test')
    testy=np.array([LABELS.index(r['family']) for r in test]);out={'fractions':{},'rows':{}}
    for fraction,seed in itertools.product(FRACTIONS,SEEDS):
        rows=subsample(train,fraction,seed);key=f'{fraction:g}';folder=OUT/'curve'/f'frac{int(fraction*1000):04d}_s{seed}'
        out['rows'][key]=len(rows);entry=out['fractions'].setdefault(key,{})
        with redirected(folder,rows) as check:
            for arch,c in selection.items():
                r=arena_train.fit(c['candidate'],'output',seed);y,p=run_checkpoint(check/f'{tag(c["candidate"],"output",seed)}.pt',test)
                entry.setdefault(arch,[]).append({'seed':seed,'val_macro_f1':r['best_val_macro_f1'],'best_epoch':r['best_epoch'],'test':short(metrics(y,p))})
        clf=make_pipeline(TfidfVectorizer(analyzer='char',ngram_range=(3,5),min_df=3,max_features=40000,sublinear_tf=True),
                          LogisticRegression(C=2,max_iter=1000,class_weight='balanced',random_state=1448))
        clf.fit(baseline_features(rows,'char_output'),[LABELS.index(r['family']) for r in rows])
        entry.setdefault('char',[]).append({'seed':seed,'test':short(metrics(testy,clf.predict_proba(baseline_features(test,'char_output'))))})
        print('CURVE',key,seed,{k:round(v[-1]['test']['macro_f1'],4) for k,v in entry.items()},flush=True)
    full={arch:[{'seed':s,'test':short(read(RESULT/f'{tag(c["candidate"],"output",s)}.json')['test'])} for s in SEEDS] for arch,c in selection.items()}
    full['char']=[{'seed':1448,'test':short(read(RESULT/'char_output.json')['test'])}]
    out['fractions']['1']=full;out['rows']['1']=len(train)
    out['summary']={k:{m:{'macro_f1_mean':float(np.mean([x['test']['macro_f1'] for x in v])),'macro_f1_sd':float(np.std([x['test']['macro_f1'] for x in v],ddof=1)) if len(v)>1 else 0.0,
                          'accuracy_mean':float(np.mean([x['test']['accuracy'] for x in v]))} for m,v in e.items()} for k,e in out['fractions'].items()}
    write(OUT/'curve.json',out)
    keys=sorted(out['summary'],key=float);n=[out['rows'][k] for k in keys]
    fig,ax=plt.subplots(figsize=(7,3.6))
    for i,(m,label) in enumerate([('cnn','CNN'),('lstm','LSTM'),('char','Char TF-IDF')]):
        mean=np.array([out['summary'][k][m]['macro_f1_mean'] for k in keys]);sd=np.array([out['summary'][k][m]['macro_f1_sd'] for k in keys])
        ax.errorbar(n,100*mean,yerr=100*sd,marker='o',capsize=3,color=COLORS[i],label=label)
    ax.set_xscale('log',base=2);ax.set_xticks(n,[f'{x:,}' for x in n]);ax.minorticks_off();ax.set_xlabel('Training responses (log scale)');ax.set_ylabel('Test macro-F1 (%)')
    ax.set_title('Learning curve on the fixed Arena split (mean ± SD over seeds)');ax.grid(alpha=.2);ax.legend(frameon=False);fig.tight_layout();fig.savefig(FIGURES/'followup_learning_curve.png',dpi=180);plt.close(fig)
    print(json.dumps(out['summary'],indent=1))

def preds(name):return [json.loads(l) for l in (RESULT/'predictions'/f'{name}.jsonl').read_text().splitlines()]
def macro_f1(y,pred):
    cm=np.zeros((4,4));np.add.at(cm,(y,pred),1);return float(scores(cm)[1])
def group_bootstrap_diff(groups,y,pred_a,pred_b):
    """Paired macro-F1 difference a-b over resampled duplicate groups."""
    names=sorted(set(groups));idx={g:i for i,g in enumerate(names)};gi=np.array([idx[g] for g in groups])
    cms=np.zeros((2,len(names),4,4));np.add.at(cms[0],(gi,y,pred_a),1);np.add.at(cms[1],(gi,y,pred_b),1)
    rng=np.random.default_rng(1448);out=[]
    for _ in range(2000):
        ii=rng.integers(len(names),size=len(names));out.append(np.subtract(*scores(cms[:,ii].sum(1))[1]))
    return np.percentile(out,[2.5,97.5]).tolist()

def posthoc():
    plan();selection=read(ROOT/'docs/arena_selection.json')['selection'];test=load('test');val=load('val')
    y=np.array([LABELS.index(r['family']) for r in test]);groups=[r['duplicate_group'] for r in test]
    names={arch:[tag(c['candidate'],'output',s) for s in SEEDS] for arch,c in selection.items()}
    probs={arch:[np.array([r['probs'] for r in preds(n)]) for n in ns] for arch,ns in names.items()}
    probs['char']=[np.array([r['probs'] for r in preds('char_output')])]
    out={'note':PLAN['status']}
    def mean_f1(arch,mask):return float(np.mean([metrics(y[mask],p[mask])['macro_f1'] for p in probs[arch]]))
    def mean_acc(arch,mask):return float(np.mean([(p[mask].argmax(1)==y[mask]).mean() for p in probs[arch]]))
    # 1. Answer length strata on the neural tokenizer.
    length=np.array([len(tokenize(r['LLM_output'])) for r in test]);out['length_strata']={}
    for label,mask in [('<192',length<192),('192-383',(length>=192)&(length<384)),('>=384',length>=384)]:
        out['length_strata'][label]={'rows':int(mask.sum()),'family_counts':dict(Counter(r['family'] for r,m in zip(test,mask) if m)),
                                     **{arch:{'accuracy':mean_acc(arch,mask),'macro_f1':mean_f1(arch,mask)} for arch in probs}}
    # 2. Matched prompts: both sides of one battle survive curation and land in test.
    by=defaultdict(list)
    for i,r in enumerate(test):by[r['row_id'][:-2]].append(i)
    pairs=[v for v in by.values() if len(v)==2];assert all(test[a]['LLM_Input']==test[b]['LLM_Input'] for a,b in pairs)
    def pairwise(p):
        ok=[]
        for a,b in pairs:
            right,swap=p[a,y[a]]*p[b,y[b]],p[a,y[b]]*p[b,y[a]];ok.append(1.0 if right>swap else .5 if right==swap else 0.0)
        return float(np.mean(ok))
    out['matched_pairs']={'pairs':len(pairs),'family_pairs':{'/'.join(k):v for k,v in Counter(tuple(sorted((test[a]['family'],test[b]['family']))) for a,b in pairs).items()}}
    for arch in probs:
        out['matched_pairs'][arch]={'pairwise_assignment':float(np.mean([pairwise(p) for p in probs[arch]])),
                                    'both_sides_correct':float(np.mean([np.mean([p[a].argmax()==y[a] and p[b].argmax()==y[b] for a,b in pairs]) for p in probs[arch]]))}
    pin=[np.array([r['probs'] for r in preds(tag(selection['cnn']['candidate'],'input',s))]) for s in SEEDS]
    out['matched_pairs']['cnn_prompt_only']={'pairwise_assignment':float(np.mean([pairwise(p) for p in pin]))}
    # 3. Answers without any identity placeholder.
    mask=np.array(['[MODEL]' not in r['LLM_output'] for r in test])
    out['mask_free']={'rows':int(mask.sum()),'excluded':int((~mask).sum()),**{arch:{'accuracy':mean_acc(arch,mask),'macro_f1':mean_f1(arch,mask)} for arch in probs}}
    # 4. Equal-length inputs and window position, using frozen checkpoints.
    toks=[tokenize(r['LLM_output']) for r in test];out['equal_length']={};out['window_position']={}
    long128=np.array([len(t)>=128 for t in toks]);sub128=[r for r,m in zip(test,long128) if m]
    long576=np.array([len(t)>=576 for t in toks]);sub576=[r for r,m in zip(test,long576) if m];t576=[t for t,m in zip(toks,long576) if m]
    windows={'first':[t[:192] for t in t576],'middle':[t[len(t)//2-96:len(t)//2+96] for t in t576],'last':[t[-192:] for t in t576]}
    for arch,ns in names.items():
        eq=[];pos=defaultdict(list)
        for n in ns:
            yy,p=run_checkpoint(CHECK/f'{n}.pt',sub128,[tokenize(r['LLM_output'])[:128] for r in sub128]);eq.append(short(metrics(yy,p)))
            for w,ws in windows.items():
                yy,p=run_checkpoint(CHECK/f'{n}.pt',sub576,ws);pos[w].append(short(metrics(yy,p)))
        out['equal_length'][arch]={k:float(np.mean([m[k] for m in eq])) for k in eq[0]}
        out['window_position'][arch]={w:{k:float(np.mean([m[k] for m in v])) for k in v[0]} for w,v in pos.items()}
    eqchar=joblib.load(CHECK/'char_output.joblib').predict_proba([' '.join(tokenize(r['LLM_output'])[:128]) for r in sub128])
    out['equal_length']['char']=short(metrics(np.array([LABELS.index(r['family']) for r in sub128]),eqchar))
    out['equal_length']['rows']=len(sub128);out['equal_length']['family_counts']=dict(Counter(r['family'] for r in sub128))
    out['window_position']['rows']=len(sub576);out['window_position']['family_counts']=dict(Counter(r['family'] for r in sub576))
    # 5. Ensemble: weights chosen on validation only.
    valy=np.array([LABELS.index(r['family']) for r in val])
    vprob={arch:np.mean([np.load(RESULT/f'{n}_val.npz')['probs'] for n in ns],0) for arch,ns in names.items()};vprob['char']=np.load(RESULT/'char_output_val.npz')['probs']
    for k,v in vprob.items():assert len(v)==len(valy)
    tprob={k:np.mean(v,0) for k,v in probs.items()};parts=['cnn','lstm','char'];grid=[]
    for a in range(21):
        for b in range(21-a):
            w=np.array([a,b,20-a-b])/20;grid.append((macro_f1(valy,sum(wi*vprob[k] for wi,k in zip(w,parts)).argmax(1)),tuple(w)))
    best_val,best_w=max(grid,key=lambda x:(round(x[0],10),-max(x[1])))
    ens={'weights':dict(zip(parts,best_w)),'val_macro_f1':best_val}
    for label,w in [('weighted',best_w),('equal',(1/3,1/3,1/3))]:
        p=sum(wi*tprob[k] for wi,k in zip(w,parts));ens[label]=short(metrics(y,p));pred=p.argmax(1)
        ens[label]['minus_char_ci95']=group_bootstrap_diff(groups,y,pred,tprob['char'].argmax(1))
        ens[label]['minus_char']=ens[label]['macro_f1']-metrics(y,tprob['char'])['macro_f1']
        cm=np.array(metrics(y,p)['confusion_matrix']);ens[label]['recall']=dict(zip(LABELS,(np.diag(cm)/cm.sum(1)).tolist()))
    out['ensemble']=ens
    # 6. CNN n-grams at the max-pooled positions that push hardest toward the true class (training answers only).
    out['cnn_ngrams']=cnn_ngrams(selection['cnn']['candidate'])
    write(OUT/'posthoc.json',out);figures(out);print(json.dumps({k:v for k,v in out.items() if k!='cnn_ngrams'},indent=1))

def cnn_ngrams(cid,top_filters=5,top_n=10):
    saved=torch.load(CHECK/f'{tag(cid,"output",1448)}.pt',map_location='cpu',weights_only=True)
    model=make_model(saved['config'],len(saved['vocab']));model.load_state_dict(saved['state_dict']);model.eval()
    inverse={i:t for t,i in saved['vocab'].items()};train=load('train');data=Data(train,saved['vocab'],'output',saved['config'])
    W=model.classifier.weight.detach();counts={f:Counter() for f in LABELS};docs={f:0 for f in LABELS}
    with torch.inference_mode():
        for batch in DataLoader(data,batch_size=256,collate_fn=collate):
            tokens,lengths,labels=batch;x=model.embedding(tokens).transpose(1,2);acts=[];where=[];widths=[]
            for conv in model.convs:
                k=conv.kernel_size[0];h=torch.relu(conv(x));valid=(lengths-k+1).clamp(min=1)
                h=h.masked_fill(torch.arange(h.size(2))[None,None,:]>=valid[:,None,None],-1e4)
                a,pos=h.max(2);acts.append(a);where.append(pos);widths+= [k]*h.size(1)
            acts=torch.cat(acts,1);where=torch.cat(where,1);logits=model.classifier(acts)
            for i in range(len(labels)):
                c=int(labels[i])
                if int(logits[i].argmax())!=c:continue
                docs[LABELS[c]]+=1;seen=set()
                for f in (W[c]*acts[i]).topk(top_filters).indices.tolist():
                    s=int(where[i,f]);gram=' '.join(inverse.get(int(t),'?') for t in tokens[i,s:s+widths[f]].tolist())
                    if gram not in seen:seen.add(gram);counts[LABELS[c]][gram]+=1
    total=Counter();[total.update(c) for c in counts.values()];out={}
    for f in LABELS:
        ranked=[(g,n) for g,n in counts[f].most_common() if n>=15 and n/total[g]>=.6][:top_n]
        out[f]={'correct_training_docs':docs[f],'top':[{'ngram':g.replace('[nl]','↵'),'docs':n,'share_of_docs':n/docs[f],'class_purity':n/total[g]} for g,n in ranked]}
    return out

def figures(out):
    fig,ax=plt.subplots(figsize=(7.6,3.4));bins=list(out['length_strata']);x=np.arange(len(bins))
    for i,(m,label) in enumerate([('cnn','CNN'),('lstm','LSTM'),('char','Char TF-IDF')]):
        ax.bar(x+(i-1)*.26,[100*out['length_strata'][b][m]['macro_f1'] for b in bins],.26,color=COLORS[i],label=label)
    ax.set_xticks(x,[f'{b} tokens\n(n={out["length_strata"][b]["rows"]})' for b in bins]);ax.set_ylabel('Test macro-F1 (%)');ax.set_ylim(0,112);ax.set_yticks(range(0,101,20))
    ax.set_title('Response-only macro-F1 by answer length');ax.legend(frameon=False,ncol=3,loc='upper center');ax.grid(axis='y',alpha=.2)
    fig.tight_layout();fig.savefig(FIGURES/'followup_length.png',dpi=180);plt.close(fig)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['plan','curve','posthoc']);args=p.parse_args()
    {'plan':plan,'curve':curve,'posthoc':posthoc}[args.phase]()
