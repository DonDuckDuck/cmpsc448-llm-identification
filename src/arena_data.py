"""Pinned Arena acquisition and prompt/session-disjoint four-model curation."""
import argparse,hashlib,json,re,time,unicodedata
from collections import Counter,defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
from sklearn.model_selection import GroupShuffleSplit
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from download_data import download
from train import ROOT
REV='6322995ab34d7c2693e3f47dd13fa5caa0789a74'
LABELS=['Claude','GPT','Grok','DeepSeek']
MODELS={'claude-sonnet-4-20250514':'Claude','gpt-4.1-2025-04-14':'GPT','grok-3-preview-02-24':'Grok','deepseek-v3-0324':'DeepSeek'}
DEST=ROOT/'data/arena';SEED=1448
IDENTITY=re.compile(r'\b(?:chatgpt|gpt(?:[- ]?(?:4(?:\.1|o)?|3\.5|5)(?:[- ]?(?:mini|turbo))?)?|openai|claude(?:[- ]?(?:sonnet|opus|haiku))?(?:[- ]?[34](?:\.[57])?)?|anthropic|sonnet|opus|haiku|grok(?:[- ]?[34])?|xai|x\.ai|deepseek(?:[- ]?[rv]?[134](?:\.\d+)?)?)\b',re.I)
SENSITIVE=re.compile(r'\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b|\bsk-[A-Za-z0-9_-]{20,}\b')
def norm(s):return ' '.join(unicodedata.normalize('NFKC',s).lower().split())
def uid(s):return hashlib.sha256(s.encode()).hexdigest()[:24]
def mask(s):
    s=unicodedata.normalize('NFKC',s)
    # Mask complete selected model IDs before shorter family names, including release suffixes.
    aliases=[re.escape(m).replace(r'\-',r'[- ]') for m in sorted(MODELS,key=len,reverse=True)]
    s=re.sub('|'.join(aliases),'[MODEL]',s,flags=re.I)
    return IDENTITY.sub('[MODEL]',s).strip()
def write(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')
def text(message):
    parts=message.get('content',[])
    if not parts or any(p.get('type')!='text' or p.get('image') is not None for p in parts):return None
    return '\n'.join(p.get('text') or '' for p in parts).strip()
def extract(row,side):
    """Accept only unambiguous first-evaluation, single-turn text responses."""
    model=row['model_'+side]
    if model not in MODELS:return None,'other_model'
    if row['language']!='en':return None,'not_english'
    if row['evaluation_order']!=1:return None,'later_evaluation'
    conv=row['conversation_'+side]
    if len(conv)!=2 or [m['role'] for m in conv]!=['user','assistant']:return None,'not_single_turn'
    prompt,answer=map(text,conv)
    if prompt is None or answer is None:return None,'non_text'
    if len(prompt.split())<3 or len(answer.split())<10:return None,'short'
    if SENSITIVE.search(prompt) or SENSITIVE.search(answer):return None,'email_or_key_pattern'
    p,a=mask(prompt),mask(answer)
    if len(p)>50000 or len(a)>100000:return None,'extreme_length'
    return {'row_id':row['id']+'_'+side,'session_id':row['evaluation_session_id'],
            'prompt_id':uid(norm(p)),'family':MODELS[model],'LLM_name':model,
            'LLM_Input':p,'LLM_output':a,'source':'Arena','is_code':bool(row['is_code']),
            'date':str(row['timestamp'])[:10],
            'identity_prompt':len(IDENTITY.findall(prompt)),'identity_output':len(IDENTITY.findall(answer))},None

def acquire():
    raw=ROOT/'data/raw/arena';raw.mkdir(parents=True,exist_ok=True)
    def one(i):
        name=f'train-{i:05d}-of-00007.parquet'
        return download(f'https://huggingface.co/datasets/lmarena-ai/arena-human-preference-140k/resolve/{REV}/data/{name}',raw/name)
    with ThreadPoolExecutor(max_workers=3) as pool:files=list(pool.map(one,range(7)))
    path=ROOT/'docs/arena_data_provenance.json'
    if path.exists():
        old=json.loads(path.read_text());assert [x['sha256'] for x in old['files']]==[x['sha256'] for x in files]
    write(path,{'dataset':'lmarena-ai/arena-human-preference-140k','revision':REV,'files':files})

def prepare():
    published=(DEST/'split_manifest.json').read_bytes() if (DEST/'split_manifest.json').exists() else None
    if any((DEST/f'{s}.jsonl').exists() for s in ['train','val','test']):
        raise RuntimeError('Arena split text already exists; reproduce in a fresh clone instead of overwriting a study.')
    filters=Counter();rows=[];seen=set();selected_before=Counter();raw=sorted((ROOT/'data/raw/arena').glob('*.parquet'))
    assert len(raw)==7,'Finish the pinned download first'
    for path in raw:
        for b in pq.ParquetFile(path).iter_batches(batch_size=256,columns=['id','model_a','model_b','evaluation_session_id','evaluation_order','conversation_a','conversation_b','language','is_code','timestamp']):
            for r in b.to_pylist():
                filters['raw_battles']+=1
                for side in ['a','b']:
                    if r['model_'+side] in MODELS:selected_before[MODELS[r['model_'+side]]]+=1
                    x,reason=extract(r,side)
                    if reason:filters[reason]+=1;continue
                    key=(x['prompt_id'],x['LLM_name'],uid(norm(x['LLM_output'])))
                    if key in seen:filters['duplicate_prompt_model_answer']+=1;continue
                    seen.add(key);rows.append(x)
        print('CURATE',path.name,'retained',len(rows),flush=True)
    rows.sort(key=lambda r:r['row_id']);n=len(rows);parent=list(range(n))
    def find(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return i
    def union(i,j):
        a,b=find(i),find(j)
        if a!=b:parent[max(a,b)]=min(a,b)
    maps=[{}, {}, {}];links=Counter()
    for i,r in enumerate(rows):
        for name,key,owners in zip(['session','prompt','answer'],[r['session_id'],r['prompt_id'],uid(norm(r['LLM_output']))],maps):
            if key in owners:union(i,owners[key]);links[name]+=1
            else:owners[key]=i
    # Full-corpus text is used only to detect duplicates, never to fit predictive features.
    pids=sorted(maps[1]);owners=[maps[1][p] for p in pids]
    vec=TfidfVectorizer(analyzer='char_wb',ngram_range=(3,5),max_features=30000,dtype=np.float32)
    X=vec.fit_transform([rows[i]['LLM_Input'] for i in owners]);nn=NearestNeighbors(metric='cosine',radius=.1,n_jobs=2).fit(X)
    for start in range(0,len(owners),256):
        nearby=nn.radius_neighbors(X[start:start+256],return_distance=False)
        for offset,ids in enumerate(nearby):
            i=start+offset
            for j in ids:
                if j>i:union(owners[i],owners[j]);links['near_prompt']+=1
    # Refusal/templates with exactly matching output stay together even across prompts.
    cluster=np.array([find(i) for i in range(n)]);indices=np.arange(n)
    trainval,test=next(GroupShuffleSplit(n_splits=1,test_size=.15,random_state=SEED).split(indices,groups=cluster))
    a,b=next(GroupShuffleSplit(n_splits=1,test_size=.15/.85,random_state=SEED+1).split(trainval,groups=cluster[trainval]))
    splits={'train':trainval[a],'val':trainval[b],'test':test};manifest=[];splitstats={}
    DEST.mkdir(parents=True,exist_ok=True)
    for split,ids in splits.items():
        selected=[]
        for i in sorted(ids):
            r=dict(rows[i],split=split,duplicate_group=rows[find(i)]['row_id']);selected.append(r)
            manifest.append({k:r[k] for k in ['row_id','session_id','prompt_id','family','LLM_name','split','duplicate_group','is_code','date']})
        (DEST/f'{split}.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in selected))
        splitstats[split]={'rows':len(selected),'groups':len({r['duplicate_group'] for r in selected}),
                           'family_counts':dict(Counter(r['family'] for r in selected))}
        assert all(splitstats[split]['family_counts'].get(f,0)>=100 for f in LABELS),'Insufficient class support'
    if published is not None:
        assert json.loads(published)==manifest,'Recreated split differs from the published manifest; preserve the original and investigate.'
    write(DEST/'split_manifest.json',manifest)
    hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(DEST.glob('*.jsonl'))}
    hash_path=ROOT/'docs/arena_processed_hashes.json'
    if hash_path.exists():assert json.loads(hash_path.read_text())==hashes,'Processed text differs from the published cleaning hashes'
    else:write(hash_path,hashes)
    audit={'revision':REV,'seed':SEED,'models':MODELS,'filters':dict(filters),'selected_before_filter':dict(selected_before),
           'retained_rows':n,'family_counts':dict(Counter(r['family'] for r in rows)),
           'unique_prompts':len(pids),'groups':len(set(cluster)),'largest_group':int(max(Counter(cluster).values())),
           'duplicate_links':dict(links),'splits':splitstats,'mask_affected_outputs':sum(r['identity_output']>0 for r in rows),
           'response_words_by_family':{f:{'median':float(np.median([len(r['LLM_output'].split()) for r in rows if r['family']==f]))} for f in LABELS},
           'code_by_family':{f:float(np.mean([r['is_code'] for r in rows if r['family']==f])) for f in LABELS},
           'split_manifest_sha256':hashlib.sha256((DEST/'split_manifest.json').read_bytes()).hexdigest(),
           'note':'Natural class counts retained; training uses inverse-frequency class weights. Full corpus duplicate detection is unsupervised. Test predictions are not accessed during curation.'}
    write(ROOT/'results/arena/data_audit.json',audit);print(json.dumps(audit,indent=2),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['download','prepare']);args=p.parse_args()
    (acquire if args.phase=='download' else prepare)()
