"""Aggregate fixed experiments with cluster bootstrap uncertainty and reproducible plots."""
from pathlib import Path
from collections import defaultdict,Counter
import csv,json,re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from train import ROOT,LABELS,load_rows,tokenize
from baselines import style,FEATURES
SEEDS=[448,449,450];BOOT=2000
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,
                     'figure.facecolor':'white','axes.facecolor':'white','savefig.dpi':180})
COLORS=['#0B7774','#D18635','#5366AB']
def result(tag):return json.loads((ROOT/'results'/f'{tag}.json').read_text())
def preds(tag):return [json.loads(l) for l in (ROOT/'results/predictions'/f'{tag}.jsonl').read_text().splitlines()]
def score_cm(cm):
    diag=np.diagonal(cm,axis1=-2,axis2=-1);total=cm.sum(axis=(-2,-1))
    acc=diag.sum(axis=-1)/np.maximum(total,1)
    denom=cm.sum(axis=-2)+cm.sum(axis=-1)
    f1=np.divide(2*diag,denom,out=np.zeros_like(diag,dtype=float),where=denom!=0).mean(axis=-1)
    return acc,f1

def bootstrap(tags):
    allrows=[preds(t) for t in tags];clusters=sorted({r['duplicate_group'] for r in allrows[0]})
    lookup={k:i for i,k in enumerate(clusters)};cms=np.zeros((len(tags),len(clusters),3,3))
    for j,rows in enumerate(allrows):
        for r in rows:cms[j,lookup[r['duplicate_group']],r['true'],r['pred']]+=1
    rng=np.random.default_rng(448);acc=[];f1=[]
    for _ in range(BOOT):
        ids=rng.integers(0,len(clusters),len(clusters));a,f=score_cm(cms[:,ids].sum(axis=1))
        acc.append(a.mean());f1.append(f.mean())
    return {'accuracy_ci95':np.percentile(acc,[2.5,97.5]).tolist(),'macro_f1_ci95':np.percentile(f1,[2.5,97.5]).tolist(),
            'bootstrap_clusters':len(clusters)},np.array(acc)

def paired_difference(tags_a,tags_b):
    # The same seed and bootstrap cluster indices are used for both sets.
    a,aa=bootstrap(tags_a);b,bb=bootstrap(tags_b)
    return {'accuracy_difference':float(np.mean([result(t)['test']['accuracy'] for t in tags_a])-np.mean([result(t)['test']['accuracy'] for t in tags_b])),
            'ci95':np.percentile(aa-bb,[2.5,97.5]).tolist()}

def savefig(fig,name):
    fig.tight_layout();fig.savefig(ROOT/'results/figures'/f'{name}.png',bbox_inches='tight');plt.close(fig)

def main():
    (ROOT/'results/figures').mkdir(exist_ok=True)
    summary={'bootstrap_replicates':BOOT,'bootstrap_unit':'near-duplicate cluster; all three answers kept together; same resamples across contrasts',
             'core':{},'contrasts':{},'auxiliary':{},'runtime_seconds':0}
    flat=[]
    for model in ['cnn','lstm']:
        for mode in ['input','output','combined']:
            tags=[f'{model}_{mode}_s{s}' for s in SEEDS];runs=[result(t) for t in tags]
            ci,_=bootstrap(tags)
            x={'model':model,'mode':mode,'tags':tags,**ci}
            for metric in ['accuracy','macro_f1']:
                values=[r['test'][metric] for r in runs];x[metric+'_mean']=float(np.mean(values));x[metric+'_sd']=float(np.std(values,ddof=1))
            x['best_epochs']=[r['best_epoch'] for r in runs]
            x['training_seconds']=sum(r['training_seconds'] for r in runs);summary['runtime_seconds']+=x['training_seconds']
            summary['core'][model+'_'+mode]=x;flat.append({k:v for k,v in x.items() if isinstance(v,(str,int,float))})
        summary['contrasts'][model+'_prompt_gain']=paired_difference([f'{model}_combined_s{s}' for s in SEEDS],[f'{model}_output_s{s}' for s in SEEDS])
    for tag in ['tfidf_input','tfidf_output','tfidf_combined','style_all','style_no_length']+[f'{m}_output_s448{suffix}' for m in ['cnn','lstm'] for suffix in ['_normalized','_source_holdout','_source_control','_shuffled']]:
        r=result(tag);ci,_=bootstrap([tag]);summary['auxiliary'][tag]={**r['test'],**ci}
        summary['runtime_seconds']+=r.get('training_seconds',r.get('seconds',0))
    for model in ['cnn','lstm']:
        summary['contrasts'][model+'_normalization_effect']=paired_difference([f'{model}_output_s448_normalized'],[f'{model}_output_s448'])
        summary['contrasts'][model+'_source_shift']=paired_difference([f'{model}_output_s448_source_holdout'],[f'{model}_output_s448_source_control'])
    summary['contrasts']['style_length_removal']=paired_difference(['style_no_length'],['style_all'])
    # Descriptive metadata/truncation audit does not influence model fitting.
    test=load_rows('test');allrows=sum([load_rows(s) for s in ['train','val','test']],[])
    summary['truncation']={s:{'prompt_over96':float(np.mean([len(tokenize(r['LLM_Input']))>96 for r in load_rows(s)])),
                              'response_over192':float(np.mean([len(tokenize(r['LLM_output']))>192 for r in load_rows(s)]))} for s in ['train','val','test']}
    main_predictions=preds('cnn_output_s448')
    version_stats={}
    for model in sorted({r['LLM_name'] for r in test}):
        ids=[i for i,r in enumerate(test) if r['LLM_name']==model]
        version_stats[model]={'n':len(ids),'accuracy':float(np.mean([main_predictions[i]['true']==main_predictions[i]['pred'] for i in ids]))}
    summary['cnn_seed448_version_recall']=version_stats
    summary['source_sizes']={s:dict(Counter(r['source'] for r in load_rows(s))) for s in ['train','val','test']}
    (ROOT/'results/summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    pd.DataFrame(flat).to_csv(ROOT/'results/core_metrics.csv',index=False)
    # Main six conditions with cluster-bootstrap confidence intervals.
    fig,ax=plt.subplots(figsize=(8.5,4.0));positions=np.arange(3);w=.30
    for j,model in enumerate(['cnn','lstm']):
        xs=[summary['core'][model+'_'+mode] for mode in ['input','output','combined']]
        means=np.array([x['accuracy_mean'] for x in xs]);intervals=np.array([x['accuracy_ci95'] for x in xs])
        ax.bar(positions+(j-.5)*w,means*100,w,color=COLORS[j],label=model.upper(),zorder=3)
        ax.errorbar(positions+(j-.5)*w,means*100,yerr=np.maximum(0,np.stack([means-intervals[:,0],intervals[:,1]-means]))*100,fmt='none',color='#243240',capsize=4,zorder=4)
    ax.axhline(100/3,color='#7E8992',ls='--',lw=1,label='Chance (33.3%)');ax.set_xticks(positions,['Prompt only','Response only','Prompt + response'])
    ax.set_ylim(0,100);ax.set_ylabel('Test accuracy (%)');ax.grid(axis='y',alpha=.15,zorder=0);ax.legend(ncol=3,loc='upper left',frameon=False)
    savefig(fig,'main_comparison')
    # Confusion matrices use averaged counts across 3 seeds, row-normalized.
    fig,axes=plt.subplots(1,2,figsize=(8.5,3.5))
    for ax,model in zip(axes,['cnn','lstm']):
        cm=np.mean([result(f'{model}_output_s{s}')['test']['confusion_matrix'] for s in SEEDS],axis=0)
        cm=cm/cm.sum(axis=1,keepdims=True)
        ax.imshow(cm,cmap='Blues',vmin=0,vmax=1)
        for i in range(3):
            for j in range(3):ax.text(j,i,f'{100*cm[i,j]:.1f}%',ha='center',va='center',color='white' if cm[i,j]>.55 else '#142838')
        ax.set_xticks(range(3),LABELS);ax.set_yticks(range(3),LABELS);ax.set_xlabel('Predicted family');ax.set_ylabel('True family');ax.set_title(model.upper()+' / response only')
    savefig(fig,'confusion_matrices')
    fig,axes=plt.subplots(1,2,figsize=(8.5,3.1))
    for ax,model in zip(axes,['cnn','lstm']):
        h=result(f'{model}_output_s448')['history'];epochs=[r['epoch'] for r in h]
        ax.plot(epochs,[r['train_loss'] for r in h],marker='o',color=COLORS[0],label='Training')
        ax.plot(epochs,[r['val_loss'] for r in h],marker='o',color=COLORS[1],label='Validation')
        ax.set_xlabel('Epoch');ax.set_ylabel('Cross-entropy loss');ax.set_title(model.upper()+' / seed 448');ax.legend(frameon=False)
    savefig(fig,'learning_curves')
    fig,axes=plt.subplots(1,2,figsize=(8.5,3.3))
    for ax,feature,label in zip(axes,['words','punctuation'],['Response length (words)','Punctuation / characters']):
        values=[]
        for fam in LABELS:
            subset=[r for r in allrows if r['family']==fam]
            values.append([len(r['LLM_output'].split()) if feature=='words' else style(r['LLM_output'])[5] for r in subset])
        box=ax.boxplot(values,tick_labels=LABELS,showfliers=False,patch_artist=True)
        for patch,color in zip(box['boxes'],COLORS):patch.set_facecolor(color);patch.set_alpha(.7)
        ax.set_ylabel(label);ax.grid(axis='y',alpha=.15)
    savefig(fig,'style_distributions')
    fig,ax=plt.subplots(figsize=(8.5,3.6));items=['style_all','style_no_length','tfidf_output','cnn_output','lstm_output']
    vals=[summary['auxiliary'][t]['accuracy'] if t in summary['auxiliary'] else summary['core'][t]['accuracy_mean'] for t in items]
    ax.barh(['Structural features','Structural: no length','TF-IDF + logistic regression','CNN (3-seed mean)','LSTM (3-seed mean)'],np.array(vals)*100,color=[COLORS[2],COLORS[2],'#537E95',COLORS[0],COLORS[1]])
    ax.axvline(100/3,color='gray',ls='--');ax.set_xlim(0,100);ax.set_xlabel('Response-only test accuracy (%)');ax.invert_yaxis()
    savefig(fig,'baselines')
    fig,axes=plt.subplots(1,2,figsize=(8.5,3.4))
    for ax,kind,title in zip(axes,['normalized','source'],['Formatting intervention','Unseen source: Evol-Instruct']):
        if kind=='normalized':
            before=[result(f'{m}_output_s448')['test']['accuracy'] for m in ['cnn','lstm']]
            after=[summary['auxiliary'][f'{m}_output_s448_normalized']['accuracy'] for m in ['cnn','lstm']];labs=['Original','Normalized + retrained']
        else:
            before=[summary['auxiliary'][f'{m}_output_s448_source_control']['accuracy'] for m in ['cnn','lstm']]
            after=[summary['auxiliary'][f'{m}_output_s448_source_holdout']['accuracy'] for m in ['cnn','lstm']];labs=['Mixed-source training','Source held out']
        pos=np.arange(2);ax.bar(pos-.17,np.array(before)*100,.34,color=COLORS[0],label=labs[0]);ax.bar(pos+.17,np.array(after)*100,.34,color=COLORS[1],label=labs[1])
        ax.set_xticks(pos,['CNN','LSTM']);ax.set_ylim(0,100);ax.set_ylabel('Test accuracy (%)');ax.set_title(title);ax.legend(fontsize=8,frameon=False,loc='upper left')
    savefig(fig,'interventions')
    print(json.dumps(summary['core'],indent=2));print('CONTRASTS',json.dumps(summary['contrasts'],indent=2))
if __name__=='__main__':main()
