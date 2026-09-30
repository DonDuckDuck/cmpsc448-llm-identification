"""V2: bounded validation-only search, repeated-seed selection, frozen evaluation."""
import argparse,hashlib,json,platform,random,time
from pathlib import Path
from collections import Counter
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader,Dataset
from train import ROOT,LABELS,load_rows,tokenize,collate,predict,metrics
from models_v2 import make_model
DEST=ROOT/'results/v2';CHECK=ROOT/'results/checkpoints/v2'
BASE={'emb':64,'hidden':64,'dropout':.3,'lr':.001,'max_tokens':192,'normalized':False,'bidirectional':False,'pooling':'meanmax','epochs':24,'patience':5,'batch':64}
CANDIDATES={
 'c01':dict(BASE,architecture='cnn'),
 'c02':dict(BASE,architecture='cnn',max_tokens=384),
 'c03':dict(BASE,architecture='cnn',emb=128,hidden=96,dropout=.4,lr=.0007),
 'c04':dict(BASE,architecture='cnn',emb=128,hidden=96,dropout=.4,lr=.0007,max_tokens=384),
 'r01':dict(BASE,architecture='lstm',pooling='last'),
 'r02':dict(BASE,architecture='lstm'),
 'r03':dict(BASE,architecture='lstm',bidirectional=True),
 'r04':dict(BASE,architecture='lstm',normalized=True),
 'r05':dict(BASE,architecture='lstm',bidirectional=True,normalized=True),
 'r06':dict(BASE,architecture='lstm',bidirectional=True,max_tokens=384),
 'r07':dict(BASE,architecture='lstm',emb=128,hidden=96,dropout=.4,lr=.0007,bidirectional=True,max_tokens=384),
 'r08':dict(BASE,architecture='lstm',lr=.0005),
}

def components(row,mode,cfg):
    prompt=tokenize(row['LLM_Input'],cfg['normalized'])[:96]
    answer=tokenize(row['LLM_output'],cfg['normalized'])[:cfg['max_tokens']]
    return prompt if mode=='input' else answer if mode=='output' else prompt+['[sep]']+answer

def make_vocab(rows,mode,cfg):
    counts=Counter(t for r in rows for t in components(r,mode,cfg));vocab={'[pad]':0,'[unk]':1,'[sep]':2}
    for t,n in sorted(counts.items(),key=lambda x:(-x[1],x[0])):
        if n>=2 and t not in vocab and len(vocab)<16000:vocab[t]=len(vocab)
    return vocab
class Data(Dataset):
    def __init__(self,rows,vocab,mode,cfg):
        self.items=[([vocab.get(t,1) for t in components(r,mode,cfg)] or [1],LABELS.index(r['family'])) for r in rows]
    def __getitem__(self,i):return self.items[i]
    def __len__(self):return len(self.items)
def write(path,data):path.write_text(json.dumps(data,indent=2)+'\n')
def tag(cid,mode,seed):return f'{cid}_{mode}_s{seed}'
def device():return 'mps' if torch.backends.mps.is_available() else 'cpu'

def fit(cid,mode,seed):
    cfg=CANDIDATES[cid];name=tag(cid,mode,seed);rp=DEST/f'{name}.json';cp=CHECK/f'{name}.pt'
    if rp.exists() and cp.exists() and (DEST/f'{name}_val.npz').exists():
        old=json.loads(rp.read_text());assert old['config']==cfg
        return old
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.set_num_threads(4)
    dev=device()
    if dev=='mps':torch.mps.manual_seed(seed)
    # Intentionally load only training/validation. Test loading exists solely in evaluate().
    train,val=load_rows('train'),load_rows('val');vocab=make_vocab(train,mode,cfg)
    dl=DataLoader(Data(train,vocab,mode,cfg),batch_size=cfg['batch'],shuffle=True,collate_fn=collate,generator=torch.Generator().manual_seed(seed))
    vl=DataLoader(Data(val,vocab,mode,cfg),batch_size=cfg['batch'],collate_fn=collate)
    model=make_model(cfg,len(vocab)).to(dev);opt=torch.optim.Adam(model.parameters(),lr=cfg['lr'],weight_decay=1e-4)
    start=time.perf_counter();history=[];best=-1;bestloss=float('inf');bad=0;state=None;epoch_best=0;valprobs=None
    for epoch in range(1,cfg['epochs']+1):
        t=time.perf_counter();model.train();total=0;loss_sum=0
        for tokens,lengths,y in dl:
            tokens,lengths,y=tokens.to(dev),lengths.to(dev),y.to(dev)
            opt.zero_grad(set_to_none=True);logits=model(tokens,lengths)
            loss=nn.functional.cross_entropy(logits,y);loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.0);opt.step()
            total+=len(y);loss_sum+=loss.item()*len(y)
        y,p,vloss=predict(model,vl,dev);m=metrics(y,p);f1=m['macro_f1']
        history.append({'epoch':epoch,'train_loss':loss_sum/total,'val_loss':vloss,'val_accuracy':m['accuracy'],'val_macro_f1':f1,'seconds':time.perf_counter()-t})
        if f1>best+1e-8 or (abs(f1-best)<=1e-8 and vloss<bestloss):
            best=f1;bestloss=vloss;bad=0;epoch_best=epoch;valprobs=p
            state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        else:bad+=1
        if bad>=cfg['patience']:break
    out={'tag':name,'candidate':cid,'mode':mode,'seed':seed,'config':cfg,'best_epoch':epoch_best,'best_val_macro_f1':best,
         'best_val_loss':bestloss,'history':history,'training_seconds':time.perf_counter()-start,'device':dev,
         'torch_version':torch.__version__,'vocab_size':len(vocab),'parameters':sum(x.numel() for x in model.parameters()),
         'split_manifest_sha256':hashlib.sha256((ROOT/'data/processed/split_manifest.json').read_bytes()).hexdigest(),
         'test_accessed':False}
    torch.save({'state_dict':state,'vocab':vocab,'config':cfg,'mode':mode,'labels':LABELS},cp)
    write(rp,out)
    np.savez_compressed(DEST/f'{name}_val.npz',probs=valprobs,labels=y)
    print('FIT',name,'val F1',round(best,4),'epoch',epoch_best,'seconds',round(out['training_seconds'],1),flush=True)
    return out

def search():
    plan=ROOT/'docs/iteration_plan.json'
    if not plan.exists():write(plan,{'created_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'candidates':CANDIDATES,
        'selection':'Seed 448 output-only screen; top two per architecture repeat seeds 449 and 450; choose highest mean validation macro-F1 over 3 seeds. Tie breaks: validation loss, then ID.',
        'final_conditions':'Selected CNN and selected LSTM, all 3 input modes, seeds 448/449/450. Same split as v1.',
        'test_policy':'No test text or metrics accessed in candidate training/selection. V1 test results have previously been seen, so v2 is explicitly follow-up evaluation, not a fresh blind test.',
        'scope':'bounded search, no claim of global optimum; all candidates disclosed'})
    else:assert json.loads(plan.read_text())['candidates']==CANDIDATES
    first={cid:fit(cid,'output',448) for cid in CANDIDATES}
    finalists={};selection={}
    for arch in ['cnn','lstm']:
        ids=[c for c in CANDIDATES if CANDIDATES[c]['architecture']==arch]
        ids=sorted(ids,key=lambda c:(-first[c]['best_val_macro_f1'],first[c]['best_val_loss'],c))[:2];finalists[arch]=ids
        scores=[]
        for cid in ids:
            runs=[fit(cid,'output',s) for s in [448,449,450]]
            scores.append((float(np.mean([r['best_val_macro_f1'] for r in runs])),float(np.mean([r['best_val_loss'] for r in runs])),cid))
        best=sorted(scores,key=lambda x:(-x[0],x[1],x[2]))[0]
        selection[arch]={'candidate':best[2],'validation_macro_f1_mean':best[0],'validation_loss_mean':best[1],
                         'config':CANDIDATES[best[2]],'finalist_scores':[{'candidate':c,'mean_val_macro_f1':f,'mean_val_loss':l} for f,l,c in scores]}
    frozen={'frozen_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'selection':selection,'finalists':finalists,
            'plan_sha256':hashlib.sha256(plan.read_bytes()).hexdigest()}
    freeze_path=ROOT/'docs/iteration_selection.json'
    if freeze_path.exists():
        previous=json.loads(freeze_path.read_text())
        assert all(previous['selection'][a]['candidate']==selection[a]['candidate'] for a in selection), 'Reproduction selected different candidates; preserve the published study and create a separate experiment directory.'
        if previous['selection']!=selection:write(freeze_path,frozen)
    else:write(freeze_path,frozen)
    print('FROZEN',json.dumps(selection),flush=True)

def confirm():
    frozen=json.loads((ROOT/'docs/iteration_selection.json').read_text())
    for arch,chosen in frozen['selection'].items():
        for mode in ['input','output','combined']:
            for seed in [448,449,450]:fit(chosen['candidate'],mode,seed)

def evaluate():
    frozen=json.loads((ROOT/'docs/iteration_selection.json').read_text())
    expected=[tag(chosen['candidate'],mode,s) for chosen in frozen['selection'].values() for mode in ['input','output','combined'] for s in [448,449,450]]
    assert all((DEST/f'{t}.json').exists() and (CHECK/f'{t}.pt').exists() for t in expected),'Complete all training before test evaluation'
    rows=load_rows('test');torch.set_num_threads(4);dev=device();predroot=DEST/'predictions';predroot.mkdir(exist_ok=True)
    for name in expected:
        r=json.loads((DEST/f'{name}.json').read_text());saved=torch.load(CHECK/f'{name}.pt',map_location='cpu',weights_only=True)
        model=make_model(saved['config'],len(saved['vocab'])).to(dev);model.load_state_dict(saved['state_dict'])
        dl=DataLoader(Data(rows,saved['vocab'],saved['mode'],saved['config']),batch_size=64,collate_fn=collate)
        y,p,loss=predict(model,dl,dev);r['test']=metrics(y,p);r['test']['loss']=loss;r['test_accessed']=True
        r['selection_sha256']=hashlib.sha256((ROOT/'docs/iteration_selection.json').read_bytes()).hexdigest();write(DEST/f'{name}.json',r)
        (predroot/f'{name}.jsonl').write_text(''.join(json.dumps({'prompt_id':a['prompt_id'],'duplicate_group':a['duplicate_group'],'source':a['source'],
                 'family':a['family'],'true':int(b),'pred':int(c.argmax()),'probs':c.tolist()})+'\n' for a,b,c in zip(rows,y,p)))
        print('TEST',name,'accuracy',round(r['test']['accuracy'],4),'F1',round(r['test']['macro_f1'],4),flush=True)
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['search','confirm','evaluate']);args=parser.parse_args()
    DEST.mkdir(parents=True,exist_ok=True);CHECK.mkdir(parents=True,exist_ok=True)
    {'search':search,'confirm':confirm,'evaluate':evaluate}[args.phase]()
