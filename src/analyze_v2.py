"""Report validation search and paired v1/v2 differences without cherry-picking seeds."""
import json,csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from train import ROOT,load_rows,tokenize
from analyze import score_cm,COLORS
from iterate import CANDIDATES,tag
DEST=ROOT/'results/v2'
def read(path):return json.loads(path.read_text())
def predictions(version,name):
    root=ROOT/'results' if version==1 else DEST
    return [json.loads(l) for l in (root/'predictions'/f'{name}.jsonl').read_text().splitlines()]
def bootstrap(version,tags):
    rows=[predictions(version,t) for t in tags];clusters=sorted({x['duplicate_group'] for x in rows[0]});lookup={c:i for i,c in enumerate(clusters)}
    cms=np.zeros((len(tags),len(clusters),3,3))
    keys0={(x['prompt_id'],x['family']) for x in rows[0]}
    for s,rr in enumerate(rows):
        assert {(x['prompt_id'],x['family']) for x in rr}==keys0
        for r in rr:cms[s,lookup[r['duplicate_group']],r['true'],r['pred']]+=1
    rng=np.random.default_rng(448);acc=[];f1=[]
    for _ in range(2000):
        ids=rng.integers(len(clusters),size=len(clusters));a,f=score_cm(cms[:,ids].sum(1));acc.append(a.mean());f1.append(f.mean())
    return np.array(acc),np.array(f1)
def main():
    frozen=read(ROOT/'docs/iteration_selection.json');old=read(ROOT/'results/summary.json')
    summary={'title':'Who Wrote It?','selection':frozen['selection'],'core':{},'contrasts':{},'search_ledger':[],
             'evaluation_note':'V2 is follow-up on the previously reported split. Configurations selected only with validation metrics. Test is not a new blind sample.'}
    for cid,cfg in CANDIDATES.items():
        rr=[read(p) for p in sorted(DEST.glob(f'{cid}_output_s*.json'))]
        first=next(r for r in rr if r['seed']==448)
        summary['search_ledger'].append({'candidate':cid,**cfg,'seed448_val_f1':first['best_val_macro_f1'],
             'evaluated_seeds':[r['seed'] for r in rr], 'mean_val_f1':float(np.mean([r['best_val_macro_f1'] for r in rr])),
             'selected':cid==frozen['selection'][cfg['architecture']]['candidate']})
    for arch,chosen in frozen['selection'].items():
        for mode in ['input','output','combined']:
            names=[tag(chosen['candidate'],mode,s) for s in [448,449,450]];rr=[read(DEST/f'{n}.json') for n in names]
            aa,ff=bootstrap(2,names)
            x={'tags':names,'candidate':chosen['candidate'],'accuracy_ci95':np.percentile(aa,[2.5,97.5]).tolist(),
               'macro_f1_ci95':np.percentile(ff,[2.5,97.5]).tolist(),'best_epochs':[r['best_epoch'] for r in rr]}
            for metric in ['accuracy','macro_f1']:
                vals=[r['test'][metric] for r in rr];x[metric+'_mean']=float(np.mean(vals));x[metric+'_sd']=float(np.std(vals,ddof=1))
            summary['core'][arch+'_'+mode]=x
            a_old,f_old=bootstrap(1,[f'{arch}_{mode}_s{s}' for s in [448,449,450]])
            summary['contrasts'][arch+'_'+mode+'_gain']={'accuracy_difference':x['accuracy_mean']-old['core'][arch+'_'+mode]['accuracy_mean'],
                      'ci95':np.percentile(aa-a_old,[2.5,97.5]).tolist()}
        a_both,_=bootstrap(2,summary['core'][arch+'_combined']['tags']);a_out,_=bootstrap(2,summary['core'][arch+'_output']['tags'])
        summary['contrasts'][arch+'_prompt_gain']={'accuracy_difference':summary['core'][arch+'_combined']['accuracy_mean']-summary['core'][arch+'_output']['accuracy_mean'],
                                                  'ci95':np.percentile(a_both-a_out,[2.5,97.5]).tolist()}
    summary['test_response_truncation']={arch:float(np.mean([len(tokenize(r['LLM_output'],choice['config']['normalized']))>choice['config']['max_tokens'] for r in load_rows('test')])) for arch,choice in frozen['selection'].items()}
    allruns=[read(p) for p in DEST.glob('*.json') if p.name!='summary.json']
    allruns=[r for r in allruns if 'training_seconds' in r]
    summary['fit_count']=len(allruns);summary['test_run_count']=sum('test' in r for r in allruns)
    summary['training_seconds']=sum(r['training_seconds'] for r in allruns)
    (DEST/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    fields=['candidate','mode','seed','best_epoch','best_val_macro_f1','test_accuracy','test_macro_f1','training_seconds']
    with (DEST/'run_ledger.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=fields,lineterminator="\n");w.writeheader()
        for r in sorted(allruns,key=lambda r:r['tag']):
            w.writerow({k:r.get(k,r.get('test',{}).get(k.replace('test_',''))) for k in fields})
    fig,axes=plt.subplots(1,2,figsize=(9,3.7))
    for ax,arch in zip(axes,['cnn','lstm']):
        pos=np.arange(3);modes=['input','output','combined']
        for i,(version,source) in enumerate([('Initial',old),('Optimized',summary)]):
            mean=np.array([source['core'][arch+'_'+m]['accuracy_mean'] for m in modes]);ci=np.array([source['core'][arch+'_'+m]['accuracy_ci95'] for m in modes])
            x=pos+(i-.5)*.34;ax.bar(x,mean*100,.34,color=COLORS[i],label=version,zorder=2)
            ax.errorbar(x,mean*100,yerr=np.maximum(0,np.stack([mean-ci[:,0],ci[:,1]-mean]))*100,fmt='none',color='#253A47',capsize=3,zorder=3)
        ax.set_xticks(pos,['Prompt','Response','Both']);ax.set_ylim(0,100);ax.set_ylabel('Test accuracy (%)');ax.set_title(arch.upper());ax.legend(frameon=False);ax.axhline(100/3,color='gray',ls='--',lw=.8)
    fig.tight_layout();fig.savefig(ROOT/'results/figures/optimization_results.png',dpi=180,bbox_inches='tight');plt.close(fig)
    print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
