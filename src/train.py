"""Deterministic data setup, validation-only checkpoint choice, test evaluation."""
from pathlib import Path
from collections import Counter
import argparse, copy, hashlib, json, os, platform, random, re, time
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from models import TextCNN, TextLSTM
ROOT=Path(__file__).resolve().parents[1]
LABELS=['Falcon','GPT','Llama']
TOKEN=re.compile(r'\[model\]|\[nl\]|\w+|[^\w\s]',re.UNICODE)

def tokenize(text,normalized=False):
    if normalized: return re.findall(r'\w+',text.lower())
    return TOKEN.findall(text.lower().replace('\n',' [nl] '))
def load_rows(split):
    return [json.loads(line) for line in (ROOT/f'data/processed/{split}.jsonl').read_text().splitlines()]
def components(row,mode,normalized=False):
    a=tokenize(row['LLM_Input'],normalized); b=tokenize(row['LLM_output'],normalized)
    if mode=='input': return a[:96]
    if mode=='output': return b[:192]
    return a[:96]+['[sep]']+b[:192]
def make_vocab(rows,mode,normalized=False):
    counts=Counter(t for r in rows for t in components(r,mode,normalized))
    vocab={'[pad]':0,'[unk]':1,'[sep]':2}
    for word,count in sorted(counts.items(),key=lambda x:(-x[1],x[0])):
        if count>=2 and word not in vocab and len(vocab)<12000: vocab[word]=len(vocab)
    return vocab
class TextData(Dataset):
    def __init__(self,rows,vocab,mode,normalized=False):
        self.items=[]
        for row in rows:
            ids=[vocab.get(t,1) for t in components(row,mode,normalized)] or [1]
            self.items.append((ids,LABELS.index(row['family'])))
    def __len__(self):return len(self.items)
    def __getitem__(self,i):return self.items[i]
def collate(batch):
    lengths=torch.tensor([len(x[0]) for x in batch],dtype=torch.long)
    tokens=torch.zeros(len(batch),max(5,int(lengths.max())),dtype=torch.long)
    for i,(ids,y) in enumerate(batch):tokens[i,:len(ids)]=torch.tensor(ids)
    labels=torch.tensor([x[1] for x in batch],dtype=torch.long)
    return tokens,lengths,labels

def predict(model,loader,device):
    model.eval(); probs=[]; ys=[];losses=0;count=0
    with torch.inference_mode():
        for tokens,lengths,labels in loader:
            tokens,lengths,labels=tokens.to(device),lengths.to(device),labels.to(device)
            logits=model(tokens,lengths)
            losses+=nn.functional.cross_entropy(logits,labels,reduction='sum').item();count+=len(labels)
            probs.extend(logits.softmax(dim=1).cpu().numpy().tolist());ys.extend(labels.cpu().tolist())
    probs=np.asarray(probs);ys=np.asarray(ys)
    return ys,probs,losses/count

def metrics(y,p):
    pred=np.argmax(p,axis=1)
    return {'accuracy':float(accuracy_score(y,pred)), 'macro_f1':float(f1_score(y,pred,labels=[0,1,2],average='macro',zero_division=0)),
            'confusion_matrix':confusion_matrix(y,pred,labels=[0,1,2]).tolist(),
            'classification_report':classification_report(y,pred,labels=[0,1,2],target_names=LABELS,zero_division=0,output_dict=True)}

def run(args):
    if args.domain and args.source_control:raise ValueError('Choose holdout or control, not both')
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed);torch.set_num_threads(4)
    device=args.device
    if device=='auto':device='mps' if torch.backends.mps.is_available() else 'cpu'
    if device=='mps':torch.mps.manual_seed(args.seed)
    rows={s:load_rows(s) for s in ['train','val','test']}
    if args.domain or args.source_control:
        # Original splits retained. Domain definition is source identity, not manual task annotation.
        if args.domain:
            rows['train']=[r for r in rows['train'] if r['source']!='evol_instruct']
        else:
            target=len({r['prompt_id'] for r in rows['train'] if r['source']!='evol_instruct'})
            all_ids=sorted({r['prompt_id'] for r in rows['train']})
            selected=set(np.random.default_rng(448).choice(all_ids,target,replace=False))
            rows['train']=[r for r in rows['train'] if r['prompt_id'] in selected]
        rows['val']=[r for r in rows['val'] if r['source']!='evol_instruct']
        rows['test']=[r for r in rows['test'] if r['source']=='evol_instruct']
    vocab=make_vocab(rows['train'],args.mode,args.normalized)
    datasets={s:TextData(rows[s],vocab,args.mode,args.normalized) for s in rows}
    if args.shuffle_labels:
        perm=np.random.default_rng(args.seed).permutation([y for _,y in datasets['train'].items])
        datasets['train'].items=[(ids,int(y)) for (ids,_),y in zip(datasets['train'].items,perm)]
    generator=torch.Generator().manual_seed(args.seed)
    loaders={s:DataLoader(datasets[s],batch_size=args.batch,shuffle=s=='train',collate_fn=collate,num_workers=0,
                          generator=generator if s=='train' else None) for s in rows}
    cls=TextCNN if args.model=='cnn' else TextLSTM
    model=cls(len(vocab),emb=64,hidden=64,dropout=.3).to(device)
    optimizer=torch.optim.Adam(model.parameters(),lr=.001,weight_decay=1e-4)
    tag=f'{args.model}_{args.mode}_s{args.seed}'+('_normalized' if args.normalized else '')+('_source_holdout' if args.domain else '')+('_source_control' if args.source_control else '')+('_shuffled' if args.shuffle_labels else '')+('_pilot' if args.pilot else '')
    print(f'START {tag} device={device} rows={ {s:len(r) for s,r in rows.items()} } vocab={len(vocab)}',flush=True)
    start=time.perf_counter();history=[];best=-1;best_loss=float('inf');bad=0;state=None;best_epoch=None
    for epoch in range(1,args.epochs+1):
        epoch_start=time.perf_counter();model.train();total=0;loss_sum=0
        for tokens,lengths,labels in loaders['train']:
            tokens,lengths,labels=tokens.to(device),lengths.to(device),labels.to(device)
            optimizer.zero_grad(set_to_none=True);logits=model(tokens,lengths)
            loss=nn.functional.cross_entropy(logits,labels);loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),1.0);optimizer.step()
            loss_sum+=loss.item()*len(labels);total+=len(labels)
        y,p,valloss=predict(model,loaders['val'],device);v=metrics(y,p)
        record={'epoch':epoch,'train_loss':loss_sum/total,'val_loss':valloss,'val_accuracy':v['accuracy'],
                'val_macro_f1':v['macro_f1'],'seconds':time.perf_counter()-epoch_start};history.append(record)
        print(json.dumps(record),flush=True)
        # Ties resolved by validation loss. No test metric is observed until training ends.
        if v['macro_f1']>best+1e-8 or (abs(v['macro_f1']-best)<=1e-8 and valloss<best_loss):
            best=v['macro_f1'];best_loss=valloss;best_epoch=epoch;bad=0
            state={k:x.detach().cpu().clone() for k,x in model.state_dict().items()}
        else:bad+=1
        if bad>=3:break
    model.load_state_dict(state);model.to(device)
    training_seconds=time.perf_counter()-start
    out={'tag':tag,'config':vars(args),'device':device,'torch_version':torch.__version__,
         'platform':platform.platform(),'labels':LABELS,'vocab_size':len(vocab),'parameters':sum(p.numel() for p in model.parameters()),
         'row_counts':{s:len(r) for s,r in rows.items()},'best_epoch':best_epoch,'best_val_macro_f1':best,
         'history':history,'training_seconds':training_seconds,
         'split_manifest_sha256':hashlib.sha256((ROOT/'data/processed/split_manifest.json').read_bytes()).hexdigest(),
         'token_stats':{s:{'unk_fraction':sum(x==1 for ids,_ in datasets[s].items for x in ids)/sum(len(ids) for ids,_ in datasets[s].items),
                           'mean_input_tokens':float(np.mean([len(ids) for ids,_ in datasets[s].items]))} for s in rows}}
    if not args.pilot:
        y,p,loss=predict(model,loaders['test'],device);out['test']=metrics(y,p);out['test']['loss']=loss
        pred_dir=ROOT/'results/predictions';pred_dir.mkdir(exist_ok=True)
        (pred_dir/f'{tag}.jsonl').write_text(''.join(json.dumps({'prompt_id':r['prompt_id'],'duplicate_group':r['duplicate_group'],
                     'source':r['source'],'family':r['family'],'true':int(yi),'pred':int(pi.argmax()),'probs':pi.tolist()})+'\n'
                      for r,yi,pi in zip(rows['test'],y,p)))
    checkpoint_dir=ROOT/'results/checkpoints';checkpoint_dir.mkdir(exist_ok=True)
    torch.save({'state_dict':state,'vocab':vocab,'config':vars(args),'labels':LABELS},checkpoint_dir/f'{tag}.pt')
    (ROOT/'results'/f'{tag}.json').write_text(json.dumps(out,indent=2)+'\n')
    print('DONE',tag,json.dumps(out.get('test',{})),flush=True)
    return out
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--model',choices=['cnn','lstm'],required=True)
    p.add_argument('--mode',choices=['input','output','combined'],required=True);p.add_argument('--seed',type=int,default=448)
    p.add_argument('--epochs',type=int,default=8);p.add_argument('--batch',type=int,default=64)
    p.add_argument('--device',choices=['auto','cpu','mps'],default='auto')
    p.add_argument('--shuffle-labels',action='store_true');p.add_argument('--source-control',action='store_true');p.add_argument('--normalized',action='store_true');p.add_argument('--domain',action='store_true');p.add_argument('--pilot',action='store_true')
    run(p.parse_args())
