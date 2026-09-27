"""Build a paired, prompt-disjoint benchmark with auditable duplicate controls."""
from pathlib import Path
from collections import Counter, defaultdict
import hashlib, json, re, unicodedata
import numpy as np
import pyarrow.parquet as pq
from sklearn.model_selection import GroupShuffleSplit
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
ROOT=Path(__file__).resolve().parents[1]
FAMILIES={'Falcon':['falcon-40b-instruct'], 'GPT':['gpt-3.5-turbo','gpt-4'],
          'Llama':['llama-2-7b-chat','llama-2-13b-chat','llama-2-70b-chat']}
SEED=448
IDENTITY=re.compile(r'\b(?:chatgpt|gpt(?:[- ]?(?:3\.5|4)(?:[- ]?turbo)?)?|openai|llama(?:[- ]?2)?|meta|falcon(?:[- ]?40b)?|tii|technology innovation institute)\b',re.I)
SENSITIVE=re.compile(r'\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b|\bsk-[A-Za-z0-9]{20,}\b')
def norm(s): return ' '.join(unicodedata.normalize('NFKC',s).lower().split())
def uid(s): return hashlib.sha256(norm(s).encode()).hexdigest()[:20]
def mask(s): return IDENTITY.sub('[MODEL]',unicodedata.normalize('NFKC',s)).strip()
def write_json(p,x): p.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')

def main():
    audit=Counter(); model_counts=Counter(); groups={}
    for path in sorted((ROOT/'data/raw').glob('*.parquet')):
        for batch in pq.ParquetFile(path).iter_batches(batch_size=256,columns=['source','instruction','completions']):
            for row in batch.to_pylist():
                audit['original_prompt_rows']+=1
                # Avoid redistributing real user chat; no raw data are included in the repo.
                if row['source']=='sharegpt': audit['excluded_sharegpt']+=1; continue
                prompt=row['instruction'].strip(); pid=uid(prompt)
                if pid in groups: audit['duplicate_matched_prompt_rows']+=1; continue
                options={k:[] for k in FAMILIES}
                for c in row['completions']:
                    model_counts[c['model']]+=1
                    for family,models in FAMILIES.items():
                        if c['model'] in models and isinstance(c.get('response'),str) and len(c['response'].split())>=5:
                            options[family].append(c)
                if not all(options.values()): audit['not_complete_three_family_prompt']+=1; continue
                rng=np.random.default_rng(int(pid[:8],16)+SEED)
                chosen=[(f,options[f][int(rng.integers(len(options[f])))]) for f in sorted(FAMILIES)]
                if SENSITIVE.search(prompt) or any(SENSITIVE.search(c['response']) for _,c in chosen):
                    audit['excluded_email_or_key_pattern']+=1; continue
                if len(prompt.split())<3: audit['short_prompt']+=1; continue
                rows=[]
                for f,c in chosen:
                    rows.append({'prompt_id':pid,'family':f,'LLM_name':c['model'],'source':row['source'],
                                 'LLM_Input':mask(prompt),'LLM_output':mask(c['response']),
                                 'principle':c.get('principle','unknown'),
                                 'identity_mentions_prompt':len(IDENTITY.findall(prompt)),
                                 'identity_mentions_output':len(IDENTITY.findall(c['response']))})
                groups[pid]=rows
    # Sampling decision made before seeing performance. Keep at most 1,500 matched prompts.
    keys=sorted(groups); rng=np.random.default_rng(SEED)
    if len(keys)>1500: keys=sorted(rng.choice(keys,1500,replace=False).tolist())
    data=[groups[k] for k in keys]; prompts=[g[0]['LLM_Input'] for g in data]
    if len(data)<100: raise ValueError('Too few matched prompts for the planned study')
    parent=list(range(len(data)))
    def find(x):
        while parent[x]!=x: parent[x]=parent[parent[x]]; x=parent[x]
        return x
    def union(a,b):
        a,b=find(a),find(b)
        if a!=b: parent[max(a,b)]=min(a,b)
    # Unsupervised full-corpus duplicate detection only; predictive vocab uses training data only.
    vec=TfidfVectorizer(analyzer='char_wb',ngram_range=(3,5),max_features=30000,dtype=np.float32)
    X=vec.fit_transform(prompts)
    nn=NearestNeighbors(metric='cosine',radius=.10,n_jobs=2).fit(X)
    distances,neighbors=nn.radius_neighbors(X)
    near_pairs=0
    for i,indices in enumerate(neighbors):
        for j in indices:
            if j>i: union(i,int(j)); near_pairs+=1
    answer_owner={}; repeated_answer_links=0
    for i,g in enumerate(data):
        for r in g:
            # Long exact shared outputs cannot leak across train and test.
            answer=norm(r['LLM_output'])
            if len(answer.split())>=10:
                if answer in answer_owner:
                    union(i,answer_owner[answer]); repeated_answer_links+=1
                else: answer_owner[answer]=i
    cluster=np.array([find(i) for i in range(len(data))]); indices=np.arange(len(data))
    trainval,test=next(GroupShuffleSplit(n_splits=1,test_size=.15,random_state=SEED).split(indices,groups=cluster))
    a,b=next(GroupShuffleSplit(n_splits=1,test_size=.15/.85,random_state=SEED+1).split(trainval,groups=cluster[trainval]))
    splits={'train':trainval[a],'val':trainval[b],'test':test}
    processed=ROOT/'data/processed';processed.mkdir(exist_ok=True)
    manifest=[]; counts={}; rows=[]
    for split,idx in splits.items():
        subset=[]
        for i in sorted(idx):
            for r in data[i]:
                r=dict(r,split=split,duplicate_group=str(cluster[i]));subset.append(r);rows.append(r)
                manifest.append({k:r[k] for k in ['prompt_id','family','LLM_name','source','split','duplicate_group']})
        (processed/f'{split}.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in subset))
        counts[split]={'rows':len(subset),'prompts':len(idx),'family_counts':dict(Counter(r['family'] for r in subset)),
                       'sources':dict(Counter(r['source'] for r in subset)),
                       'models':dict(Counter(r['LLM_name'] for r in subset))}
    for a in splits:
        for b in splits:
            if a!=b: assert not set(cluster[splits[a]])&set(cluster[splits[b]])
    for split in splits:
        assert len(set(counts[split]['family_counts'].values()))==1
    write_json(processed/'split_manifest.json',manifest)
    summary={'seed':SEED,'family_mapping':FAMILIES,'filters':dict(audit),'raw_model_counts_excluding_sharegpt':dict(model_counts),
             'eligible_matched_prompts':len(groups),'selected_prompts':len(data),'selected_rows':len(rows),
             'duplicate_clusters':len(set(cluster)),'near_duplicate_prompt_pairs':near_pairs,
             'repeated_long_answer_links':repeated_answer_links,'split_statistics':counts,
             'identity_mentions_output':sum(r['identity_mentions_output'] for r in rows),
             'identity_affected_outputs':sum(r['identity_mentions_output']>0 for r in rows),
             'principles_by_family':{f:dict(Counter(r['principle'] for r in rows if r['family']==f)) for f in FAMILIES},
             'response_words_by_family':{f:{'median':float(np.median([len(r['LLM_output'].split()) for r in rows if r['family']==f])),
                                          'mean':float(np.mean([len(r['LLM_output'].split()) for r in rows if r['family']==f]))} for f in FAMILIES}}
    write_json(ROOT/'results/data_audit.json',summary)
    print(json.dumps(summary,indent=2))
if __name__=='__main__': main()
