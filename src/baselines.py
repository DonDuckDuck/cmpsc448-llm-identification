"""Training-only TF-IDF and structural baselines; complements required neural models."""
from pathlib import Path
import json,re,time
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from train import ROOT,LABELS,load_rows,metrics,components
FEATURES=['log_chars','log_words','mean_word_length','lexical_diversity','newline_rate','punctuation_rate','digit_rate','uppercase_rate','list_line_rate','code_fence_count']
def style(text):
    words=re.findall(r'\w+',text);n=max(len(text),1);w=max(len(words),1);lines=text.splitlines()
    return [np.log1p(len(text)),np.log1p(len(words)),sum(len(x) for x in words)/w,
            len(set(x.lower() for x in words))/w,text.count('\n')/n,
            sum(not c.isalnum() and not c.isspace() for c in text)/n,
            sum(c.isdigit() for c in text)/n,sum(c.isupper() for c in text)/n,
            sum(bool(re.match(r'^\s*(?:[-*]|\d+[.)])\s',line)) for line in lines)/max(len(lines),1),text.count('```')]
def save(tag,clf,Xtest,test,start,extra=None):
    y=np.array([LABELS.index(r['family']) for r in test]);p=clf.predict_proba(Xtest)
    record={'tag':tag,'test':metrics(y,p),'seconds':time.perf_counter()-start,'extra':extra or {}}
    (ROOT/'results'/f'{tag}.json').write_text(json.dumps(record,indent=2)+'\n')
    dest=ROOT/'results/predictions';dest.mkdir(exist_ok=True)
    (dest/f'{tag}.jsonl').write_text(''.join(json.dumps({'prompt_id':r['prompt_id'],'duplicate_group':r['duplicate_group'],'source':r['source'],
        'family':r['family'],'true':int(yi),'pred':int(pi.argmax()),'probs':pi.tolist()})+'\n' for r,yi,pi in zip(test,y,p)))
    print(tag,record['test']['accuracy'],record['test']['macro_f1'],flush=True)
def main():
    train,val,test=[load_rows(s) for s in ['train','val','test']];y=[LABELS.index(r['family']) for r in train]
    for mode in ['input','output','combined']:
        start=time.perf_counter()
        # Same truncated tokens as the neural models, not unrestricted full text.
        texts=lambda rows:[' '.join(components(r,mode)) for r in rows]
        clf=make_pipeline(TfidfVectorizer(ngram_range=(1,2),min_df=2,max_features=30000,sublinear_tf=True),
                          LogisticRegression(C=1,max_iter=1000,random_state=448))
        clf.fit(texts(train),y);save('tfidf_'+mode,clf,texts(test),test,start)
    for mode,indices in [('style_all',list(range(10))),('style_no_length',list(range(2,10)))]:
        start=time.perf_counter();features=lambda rows:np.array([style(r['LLM_output']) for r in rows])[:,indices]
        clf=make_pipeline(StandardScaler(),LogisticRegression(C=1,max_iter=1000,random_state=448))
        clf.fit(features(train),y)
        coefs=clf[-1].coef_.tolist()
        save(mode,clf,features(test),test,start,{'features':[FEATURES[i] for i in indices],'standardized_coefficients':coefs})
if __name__=='__main__':main()
