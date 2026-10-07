"""Arena training: validation-only selection, then one frozen test evaluation."""
import argparse,hashlib,json,random,time
from collections import Counter
import joblib,numpy as np,torch
from torch import nn
from torch.utils.data import Dataset,DataLoader
from sklearn.metrics import accuracy_score,balanced_accuracy_score,f1_score,confusion_matrix,classification_report
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from arena_data import ROOT,DEST,LABELS,write
from train import tokenize,collate,predict
from models_v2 import make_model
from baselines import style,FEATURES
RESULT=ROOT/'results/arena';CHECK=ROOT/'results/checkpoints/arena';SEEDS=[1448,1449,1450]
BASE={'emb':64,'hidden':64,'dropout':.3,'lr':.001,'max_tokens':384,'normalized':False,'bidirectional':False,'pooling':'meanmax','epochs':20,'patience':4,'batch':128,'num_classes':4}
CANDIDATES={'c1':dict(BASE,architecture='cnn'),'c2':dict(BASE,architecture='cnn',emb=128,hidden=96,dropout=.4),
            'r1':dict(BASE,architecture='lstm',max_tokens=192),'r2':dict(BASE,architecture='lstm',max_tokens=192,bidirectional=True)}
def read(p):return json.loads(p.read_text())
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load(split):
    # JSON permits literal Unicode line separators inside strings; split only at file newlines.
    path=DEST/f'{split}.jsonl'
    expected=ROOT/'docs/arena_processed_hashes.json'
    if expected.exists():assert digest(path)==read(expected)[path.name],'Processed text hash mismatch'
    with path.open() as f:return [json.loads(l) for l in f]
def parts(r,mode,cfg):
    p=tokenize(r['LLM_Input'],cfg['normalized'])[:96];a=tokenize(r['LLM_output'],cfg['normalized'])[:cfg['max_tokens']]
    return p if mode=='input' else a if mode=='output' else p+['[sep]']+a

def vocab_for(rows,mode,cfg):
    counts=Counter(t for r in rows for t in parts(r,mode,cfg));v={'[pad]':0,'[unk]':1,'[sep]':2}
    for t,n in sorted(counts.items(),key=lambda x:(-x[1],x[0])):
        if n>=2 and t not in v and len(v)<20000:v[t]=len(v)
    return v
class Data(Dataset):
    def __init__(self,rows,vocab,mode,cfg):self.items=[([vocab.get(t,1) for t in parts(r,mode,cfg)] or [1],LABELS.index(r['family'])) for r in rows]
    def __len__(self):return len(self.items)
    def __getitem__(self,i):return self.items[i]
def metrics(y,p):
    pred=p.argmax(1);labels=list(range(4))
    return {'accuracy':float(accuracy_score(y,pred)),'balanced_accuracy':float(balanced_accuracy_score(y,pred)),
            'macro_f1':float(f1_score(y,pred,labels=labels,average='macro',zero_division=0)),
            'confusion_matrix':confusion_matrix(y,pred,labels=labels).tolist(),
            'classification_report':classification_report(y,pred,labels=labels,target_names=LABELS,output_dict=True,zero_division=0)}
def device():return 'mps' if torch.backends.mps.is_available() else 'cpu'
def tag(cid,mode,seed,suffix=''):return f'{cid}_{mode}_s{seed}{suffix}'
def fit(cid,mode,seed,suffix=''):
    cfg=dict(CANDIDATES[cid]);cfg['normalized']=suffix=='_normalized';name=tag(cid,mode,seed,suffix)
    rp=RESULT/f'{name}.json';cp=CHECK/f'{name}.pt';vp=RESULT/f'{name}_val.npz'
    if rp.exists() and cp.exists() and vp.exists():
        old=read(rp);assert old['config']==cfg and old['split_manifest_sha256']==digest(DEST/'split_manifest.json');return old
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.set_num_threads(4);dev=device()
    if dev=='mps':torch.mps.manual_seed(seed)
    train,val=load('train'),load('val')
    if suffix in ['_domain','_domain_control']:
        noncode=[r for r in train if not r['is_code']]
        if suffix=='_domain':train=noncode
        else:
            counts=Counter(r['family'] for r in noncode);rng=np.random.default_rng(seed)
            train=[rows[i] for family in LABELS for rows in [[r for r in train if r['family']==family]] for i in sorted(rng.choice(len(rows),counts[family],replace=False))]
        val=[r for r in val if not r['is_code']]
    vocab=vocab_for(train,mode,cfg)
    td=Data(train,vocab,mode,cfg);vd=Data(val,vocab,mode,cfg)
    if suffix=='_shuffled':
        labels=np.random.default_rng(seed).permutation([y for _,y in td.items]);td.items=[(x,int(y)) for (x,_),y in zip(td.items,labels)]
    weights=len(td)/(4*np.bincount([y for _,y in td.items],minlength=4));weight=torch.tensor(weights,dtype=torch.float32,device=dev)
    dl=DataLoader(td,batch_size=cfg['batch'],shuffle=True,collate_fn=collate,generator=torch.Generator().manual_seed(seed));vl=DataLoader(vd,batch_size=128,collate_fn=collate)
    model=make_model(cfg,len(vocab)).to(dev);opt=torch.optim.Adam(model.parameters(),lr=cfg['lr'],weight_decay=1e-4)
    start=time.perf_counter();history=[];best=-1;bestloss=float('inf');bad=0;state=None
    print('START',name,'rows',len(train),len(val),'device',dev,flush=True)
    for epoch in range(1,cfg['epochs']+1):
        t=time.perf_counter();model.train();total=0;loss_sum=0
        for tokens,lengths,y in dl:
            tokens,lengths,y=tokens.to(dev),lengths.to(dev),y.to(dev)
            opt.zero_grad(set_to_none=True);logits=model(tokens,lengths)
            loss=nn.functional.cross_entropy(logits,y,weight=weight);loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.0);opt.step()
            total+=len(y);loss_sum+=loss.item()*len(y)
        y,p,vloss=predict(model,vl,dev);m=metrics(y,p);f1=m['macro_f1']
        history.append({'epoch':epoch,'train_weighted_loss':loss_sum/total,'val_loss':vloss,'val_accuracy':m['accuracy'],'val_macro_f1':f1,'seconds':time.perf_counter()-t})
        print(name,epoch,'valF1',round(f1,4),'seconds',round(time.perf_counter()-t,1),flush=True)
        if f1>best+1e-8 or (abs(f1-best)<=1e-8 and vloss<bestloss):
            best=f1;bestloss=vloss;bad=0;epoch_best=epoch;valprobs=p
            state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        else:bad+=1
        if bad>=cfg['patience']:break
    out={'tag':name,'candidate':cid,'mode':mode,'seed':seed,'suffix':suffix,'config':cfg,'best_epoch':epoch_best,'best_val_macro_f1':best,'best_val_loss':bestloss,
         'history':history,'training_seconds':time.perf_counter()-start,'device':dev,'torch_version':torch.__version__,
         'vocab_size':len(vocab),'parameters':sum(x.numel() for x in model.parameters()),'class_weights':weights.tolist(),
         'training_family_counts':dict(Counter(r['family'] for r in train)),'training_rows':len(train),'validation_rows':len(val),
         'split_manifest_sha256':digest(DEST/'split_manifest.json'),'test_accessed':False}
    torch.save({'state_dict':state,'vocab':vocab,'config':cfg,'mode':mode,'labels':LABELS,'dataset':'arena'},cp)
    write(rp,out);np.savez_compressed(vp,probs=valprobs,labels=y)
    print('FIT',name,'valF1',round(best,4),'epoch',epoch_best,flush=True);return out

def baseline_features(rows,name):
    if name.startswith('style'):
        a=np.array([style(r['LLM_output']) for r in rows]);return a[:,2:] if name=='style_no_length' else a
    mode=name.split('_',1)[1];cfg=dict(BASE)
    return [' '.join(parts(r,mode,cfg)) for r in rows]
BASELINES=['word_input','word_output','word_combined','char_output','style_all','style_no_length']
def fit_baselines():
    train,val=load('train'),load('val');y=np.array([LABELS.index(r['family']) for r in train]);vy=np.array([LABELS.index(r['family']) for r in val])
    for name in BASELINES:
        rp=RESULT/f'{name}.json';cp=CHECK/f'{name}.joblib'
        if rp.exists() and cp.exists():continue
        start=time.perf_counter();lr=LogisticRegression(C=2,max_iter=1000,class_weight='balanced',random_state=1448)
        if name.startswith('style'):clf=make_pipeline(StandardScaler(),lr)
        else:
            opts={'analyzer':'char','ngram_range':(3,5)} if name.startswith('char') else {'ngram_range':(1,2)}
            clf=make_pipeline(TfidfVectorizer(min_df=3,max_features=40000,sublinear_tf=True,**opts),lr)
        clf.fit(baseline_features(train,name),y);p=clf.predict_proba(baseline_features(val,name));joblib.dump(clf,cp)
        out={'tag':name,'kind':'baseline','best_val_macro_f1':metrics(vy,p)['macro_f1'],'training_seconds':time.perf_counter()-start,'test_accessed':False,'split_manifest_sha256':digest(DEST/'split_manifest.json')}
        if name.startswith('style'):out['features']=FEATURES[2:] if name=='style_no_length' else FEATURES
        write(rp,out);np.savez_compressed(RESULT/f'{name}_val.npz',labels=vy,probs=p);print('BASELINE',name,out['best_val_macro_f1'],flush=True)

def train_all():
    plan={'dataset_revision':'6322995ab34d7c2693e3f47dd13fa5caa0789a74','split_manifest_sha256':digest(DEST/'split_manifest.json'),
          'candidates':CANDIDATES,'seeds':SEEDS,'selection':'Output-only screen of two CNN and two LSTM recipes at seed 1448. Select each architecture by validation macro-F1, then loss, then ID. Repeat selected configurations on all three input modes and three seeds.',
          'test_policy':'No test text loaded by training, candidate selection, or baseline fitting. Freeze all planned checkpoints before test evaluation.',
          'diagnostics':'Selected output CNN and LSTM word-only normalization at seed 1448; shuffled-training-label CNN control; six fixed classical baselines.',
          'primary_metric':'macro-F1; also accuracy, balanced accuracy, per-class recall; cluster bootstrap over duplicate/session groups',
          'scope':'Four fixed non-reasoning model IDs; no claim about entire families or later releases.'}
    path=ROOT/'docs/arena_experiment_plan.json'
    if path.exists():assert read(path)==plan
    else:write(path,plan)
    first={cid:fit(cid,'output',1448) for cid in CANDIDATES};selection={}
    for arch in ['cnn','lstm']:
        ids=[c for c in CANDIDATES if CANDIDATES[c]['architecture']==arch]
        chosen=sorted(ids,key=lambda c:(-first[c]['best_val_macro_f1'],first[c]['best_val_loss'],c))[0]
        selection[arch]={'candidate':chosen,'config':CANDIDATES[chosen],'screen_val_macro_f1':first[chosen]['best_val_macro_f1']}
    freeze=ROOT/'docs/arena_selection.json'
    previous=read(freeze)['selection'] if freeze.exists() else None
    if previous is not None:
        assert all(previous[a]['candidate']==selection[a]['candidate'] for a in selection),'Reproduction chose different candidates; preserve the published study and use a separate experiment directory.'
    if previous!=selection:write(freeze,{'selection':selection,'frozen_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'plan_sha256':digest(path)})
    for arch,choice in selection.items():
        for mode in ['input','output','combined']:
            for seed in SEEDS:fit(choice['candidate'],mode,seed)
        fit(choice['candidate'],'output',1448,'_normalized')
    fit(selection['cnn']['candidate'],'output',1448,'_shuffled');fit_baselines()
    # Hash every checkpoint before test access, including fixed baselines.
    write(ROOT/'docs/arena_training_complete.json',{'selection_sha256':digest(freeze),'checkpoints':{p.name:digest(p) for p in sorted(CHECK.iterdir())},'completed_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())})

def save_predictions(name,rows,y,p):
    folder=RESULT/'predictions';folder.mkdir(exist_ok=True)
    (folder/f'{name}.jsonl').write_text(''.join(json.dumps({'row_id':r['row_id'],'prompt_id':r['prompt_id'],'duplicate_group':r['duplicate_group'],'family':r['family'],'is_code':r['is_code'],'true':int(t),'pred':int(v.argmax()),'probs':v.tolist()})+'\n' for r,t,v in zip(rows,y,p)))
def domain():
    assert not any(read(p).get('test_accessed',False) for p in RESULT.glob('*.json')),'Freeze domain experiment before evaluating the current fitted runs'
    write(ROOT/'docs/arena_domain_plan.json',{'seed':1448,'training':'Non-code only, compared with mixed-domain control matched exactly by per-class training count. Both validate on non-code only. Existing selected architecture recipes are reused.','evaluation':'Coding subset of the untouched test split. Single-seed exploratory domain-transfer study; code labels come from Arena metadata.'})
    frozen=read(ROOT/'docs/arena_selection.json')
    for c in frozen['selection'].values():
        for suffix in ['_domain','_domain_control']:fit(c['candidate'],'output',1448,suffix)
    write(ROOT/'docs/arena_training_complete.json',{'selection_sha256':digest(ROOT/'docs/arena_selection.json'),'domain_plan_sha256':digest(ROOT/'docs/arena_domain_plan.json'),'checkpoints':{p.name:digest(p) for p in sorted(CHECK.iterdir())},'completed_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())})
def evaluate():
    marker=ROOT/'docs/arena_training_complete.json';assert marker.exists(),'Finish and freeze all training first'
    for name,h in read(marker)['checkpoints'].items():assert digest(CHECK/name)==h
    rows=load('test');dev=device();torch.set_num_threads(4);frozen=read(ROOT/'docs/arena_selection.json');names=[]
    for arch,c in frozen['selection'].items():
        names.extend(tag(c['candidate'],m,s) for m in ['input','output','combined'] for s in SEEDS)
        names.append(tag(c['candidate'],'output',1448,'_normalized'))
        if (ROOT/'docs/arena_domain_plan.json').exists():
            names.extend(tag(c['candidate'],'output',1448,s) for s in ['_domain','_domain_control'])
    names.append(tag(frozen['selection']['cnn']['candidate'],'output',1448,'_shuffled'))
    for name in names+BASELINES:
        r=read(RESULT/f'{name}.json')
        if name in BASELINES:
            clf=joblib.load(CHECK/f'{name}.joblib');y=np.array([LABELS.index(r['family']) for r in rows]);p=clf.predict_proba(baseline_features(rows,name))
        else:
            saved=torch.load(CHECK/f'{name}.pt',map_location='cpu',weights_only=True);model=make_model(saved['config'],len(saved['vocab'])).to(dev);model.load_state_dict(saved['state_dict'])
            dl=DataLoader(Data(rows,saved['vocab'],saved['mode'],saved['config']),batch_size=128,collate_fn=collate);y,p,loss=predict(model,dl,dev)
        r['test']=metrics(y,p)
        if r.get('suffix') in ['_domain','_domain_control']:
            coded=np.array([x['is_code'] for x in rows],dtype=bool);r['coding_test']=metrics(y[coded],p[coded])
        r['test_accessed']=True;r['training_complete_sha256']=digest(marker);write(RESULT/f'{name}.json',r);save_predictions(name,rows,y,p)
        print('TEST',name,'accuracy',round(r['test']['accuracy'],4),'F1',round(r['test']['macro_f1'],4),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['train','domain','evaluate']);args=p.parse_args()
    RESULT.mkdir(parents=True,exist_ok=True);CHECK.mkdir(parents=True,exist_ok=True)
    {'train':train_all,'domain':domain,'evaluate':evaluate}[args.phase]()
