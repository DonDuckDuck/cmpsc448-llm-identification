"""Audit measured Arena results and create figures and grouped uncertainty estimates."""
import csv,json,shutil
from collections import Counter
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from arena_data import ROOT,DEST,LABELS,write
from arena_train import RESULT,CHECK,SEEDS,CANDIDATES,BASELINES,Data,read,digest,load,metrics,tag
from train import collate,predict
from models_v2 import make_model
from baselines import style,FEATURES
COLORS=['#0B7774','#DC8242','#536FA8','#915E91']
def scores(c):
    tp=np.diagonal(c,axis1=-2,axis2=-1);true=c.sum(-1);pred=c.sum(-2)
    accuracy=tp.sum(-1)/np.maximum(c.sum((-1,-2)),1)
    f1=np.divide(2*tp,true+pred,out=np.zeros_like(tp),where=(true+pred)>0).mean(-1)
    bal=np.divide(tp,true,out=np.zeros_like(tp),where=true>0).mean(-1)
    return accuracy,f1,bal

def prediction(name):return [json.loads(l) for l in (RESULT/'predictions'/f'{name}.jsonl').read_text().splitlines()]
def bootstrap(names,code_only=False):
    rr=[[r for r in prediction(t) if not code_only or r['is_code']] for t in names];groups=sorted({r['duplicate_group'] for r in rr[0]});idx={g:i for i,g in enumerate(groups)}
    cms=np.zeros((len(names),len(groups),4,4))
    for s,rows in enumerate(rr):
        for r in rows:cms[s,idx[r['duplicate_group']],r['true'],r['pred']]+=1
    rng=np.random.default_rng(1448);out=[]
    for _ in range(2000):
        ii=rng.integers(len(groups),size=len(groups));out.append([a.mean() for a in scores(cms[:,ii].sum(1))])
    return np.array(out)

def main():
    manifest=read(DEST/'split_manifest.json');audit=read(RESULT/'data_audit.json');selection=read(ROOT/'docs/arena_selection.json')['selection']
    # Audit all three grouping keys independently, not just the assigned split IDs.
    for key in ['session_id','prompt_id','duplicate_group']:
        by={}
        for r in manifest:
            if r[key] in by:assert by[r[key]]==r['split'],f'{key} leakage'
            by[r[key]]=r['split']
    assert digest(DEST/'split_manifest.json')==audit['split_manifest_sha256']
    val=load('val');valy=np.array([LABELS.index(r['family']) for r in val]);verified_val=0
    for p in sorted(RESULT.glob('*_val.npz')):
        d=np.load(p);r=read(RESULT/(p.stem.replace('_val','')+'.json'))
        expected=np.array([LABELS.index(x['family']) for x in val if not x['is_code']]) if r.get('suffix') in ['_domain','_domain_control'] else valy
        np.testing.assert_array_equal(d['labels'],expected)
        assert abs(metrics(expected,d['probs'])['macro_f1']-r['best_val_macro_f1'])<1e-10;verified_val+=1
    for arch,c in selection.items():
        ids=[i for i,x in CANDIDATES.items() if x['architecture']==arch]
        best=min(ids,key=lambda i:(-read(RESULT/f'{tag(i,"output",1448)}.json')['best_val_macro_f1'],read(RESULT/f'{tag(i,"output",1448)}.json')['best_val_loss'],i))
        assert c['candidate']==best
    test=load('test');test_ids=[r['row_id'] for r in test];testy=np.array([LABELS.index(r['family']) for r in test]);verified=0
    for p in sorted((RESULT/'predictions').glob('*.jsonl')):
        rows=prediction(p.stem);assert [r['row_id'] for r in rows]==test_ids
        y=np.array([r['true'] for r in rows]);probs=np.array([r['probs'] for r in rows]);np.testing.assert_array_equal(y,testy)
        np.testing.assert_array_equal([r['pred'] for r in rows],probs.argmax(1))
        np.testing.assert_allclose(probs.sum(1),1,atol=1e-6)
        assert np.isfinite(probs).all() and np.all(probs>=0)
        actual=metrics(y,probs);stored=read(RESULT/f'{p.stem}.json')['test']
        for k in ['accuracy','macro_f1','balanced_accuracy']:assert abs(actual[k]-stored[k])<1e-12
        assert actual['confusion_matrix']==stored['confusion_matrix'];verified+=1
    summary={'dataset':'Arena Human Preference 140k','labels':LABELS,'core':{},'contrasts':{},'baselines':{},'diagnostics':{},'selection':selection,
             'test_note':'First held-out evaluation on a new dataset. Selection used only training/validation. This fixed-version four-class task differs from the historical three-class task.'}
    dist={}
    for arch,c in selection.items():
        for mode in ['input','output','combined']:
            names=[tag(c['candidate'],mode,s) for s in SEEDS];rr=[read(RESULT/f'{n}.json') for n in names];bs=bootstrap(names);dist[arch+'_'+mode]=bs
            x={'tags':names,'best_epochs':[r['best_epoch'] for r in rr],'confusion_matrix_sum':np.array([r['test']['confusion_matrix'] for r in rr]).sum(0).tolist()}
            for i,k in enumerate(['accuracy','macro_f1','balanced_accuracy']):
                values=[r['test'][k] for r in rr];x[k+'_mean']=float(np.mean(values));x[k+'_sd']=float(np.std(values,ddof=1));x[k+'_ci95']=np.percentile(bs[:,i],[2.5,97.5]).tolist()
            summary['core'][arch+'_'+mode]=x
        delta=dist[arch+'_combined']-dist[arch+'_output'];summary['contrasts'][arch+'_prompt_gain']={
            'macro_f1_difference':summary['core'][arch+'_combined']['macro_f1_mean']-summary['core'][arch+'_output']['macro_f1_mean'],
            'ci95':np.percentile(delta[:,1],[2.5,97.5]).tolist()}
        normname=tag(c['candidate'],'output',1448,'_normalized');summary['diagnostics'][arch+'_normalized']=read(RESULT/f'{normname}.json')['test']
        base=tag(c['candidate'],'output',1448);d=bootstrap([normname])-bootstrap([base])
        summary['contrasts'][arch+'_normalization']={'macro_f1_difference':summary['diagnostics'][arch+'_normalized']['macro_f1']-read(RESULT/f'{base}.json')['test']['macro_f1'],'ci95':np.percentile(d[:,1],[2.5,97.5]).tolist()}
    d=dist['cnn_output']-dist['lstm_output'];summary['contrasts']['cnn_minus_lstm']={'macro_f1_difference':summary['core']['cnn_output']['macro_f1_mean']-summary['core']['lstm_output']['macro_f1_mean'],'ci95':np.percentile(d[:,1],[2.5,97.5]).tolist()}
    for name in BASELINES:summary['baselines'][name]=read(RESULT/f'{name}.json')['test']
    summary['diagnostics']['shuffled_cnn']=read(RESULT/f'{tag(selection["cnn"]["candidate"],"output",1448,"_shuffled")}.json')['test']
    summary['majority_accuracy']=max(Counter(testy).values())/len(testy);summary['uniform_chance']=.25
    trainrows=load('train');summary['training_style_profiles']={}
    for family in LABELS:
        values=np.array([style(r['LLM_output']) for r in trainrows if r['family']==family])
        summary['training_style_profiles'][family]=dict(zip(FEATURES,np.median(values,axis=0).tolist()))
    summary['domain_transfer']={}
    for arch,c in selection.items():
        names={kind:tag(c['candidate'],'output',1448,suffix) for kind,suffix in [('holdout','_domain'),('control','_domain_control')]}
        domain={}
        for kind,name in names.items():
            rr=[r for r in prediction(name) if r['is_code']]
            domain[kind]=metrics(np.array([r['true'] for r in rr]),np.array([r['probs'] for r in rr]))
            run=read(RESULT/f'{name}.json')
            for metric in ['accuracy','macro_f1','balanced_accuracy']:assert abs(domain[kind][metric]-run['coding_test'][metric])<1e-12
            domain[kind]['training_rows']=run['training_rows'];domain[kind]['training_family_counts']=run['training_family_counts'];domain[kind]['test_rows']=len(rr)
        assert domain['holdout']['training_family_counts']==domain['control']['training_family_counts']
        bs=bootstrap([names['holdout']],True)-bootstrap([names['control']],True)
        domain['holdout_minus_control_macro_f1']=domain['holdout']['macro_f1']-domain['control']['macro_f1']
        domain['difference_ci95']=np.percentile(bs[:,1],[2.5,97.5]).tolist();summary['domain_transfer'][arch]=domain
    # Descriptive subgroup performance; neither subgroup is a held-out domain experiment.
    summary['subgroups']={}
    for arch,c in selection.items():
        rr=[prediction(tag(c['candidate'],'output',s)) for s in SEEDS]
        for iscode in [False,True]:
            mm=[]
            for rows in rr:
                subset=[r for r in rows if r['is_code']==iscode]
                mm.append(metrics(np.array([r['true'] for r in subset]),np.array([r['probs'] for r in subset])))
            summary['subgroups'][arch+('_code' if iscode else '_noncode')]={'rows':len(subset),'accuracy_mean':float(np.mean([m['accuracy'] for m in mm])),'macro_f1_mean':float(np.mean([m['macro_f1'] for m in mm]))}
    runs=[read(p) for p in RESULT.glob('*.json') if 'training_seconds' in read(p)]
    summary['fits']=len(runs);summary['neural_fits']=sum('config' in r for r in runs);summary['test_runs']=verified;summary['fit_seconds']=sum(r['training_seconds'] for r in runs)
    write(RESULT/'summary.json',summary)
    fields=['tag','best_val_macro_f1','test_accuracy','test_macro_f1','test_balanced_accuracy','coding_test_accuracy','coding_test_macro_f1','training_seconds']
    with (RESULT/'run_ledger.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=fields,lineterminator='\n');w.writeheader()
        for r in sorted(runs,key=lambda x:x['tag']):
            w.writerow({k:(r.get('coding_test',{}).get(k.removeprefix('coding_test_')) if k.startswith('coding_test_') else r.get(k,r.get('test',{}).get(k.replace('test_','')))) for k in fields})
    # Replay the two published seed-1448 output demonstrations against saved predictions.
    replay={};torch.set_num_threads(4)
    for arch,c in selection.items():
        name=tag(c['candidate'],'output',1448);saved=torch.load(CHECK/f'{name}.pt',map_location='cpu',weights_only=True)
        model=make_model(saved['config'],len(saved['vocab']));model.load_state_dict(saved['state_dict'])
        dl=DataLoader(Data(test,saved['vocab'],'output',saved['config']),batch_size=128,collate_fn=collate)
        y,p,_=predict(model,dl,'cpu');old=np.array([r['probs'] for r in prediction(name)])
        err=float(np.max(np.abs(p-old)));assert err<2e-4;replay[arch]=err
        shutil.copy2(CHECK/f'{name}.pt',ROOT/f'models/arena_{arch}_output_s1448.pt')
    write(RESULT/'verification.json',{'validation_files_recomputed':verified_val,'test_prediction_files_recomputed':verified,'selection_recomputed':True,
        'session_prompt_duplicate_groups_disjoint':True,'checkpoint_cpu_replay_max_error':replay,'training_complete_sha256':digest(ROOT/'docs/arena_training_complete.json')})
    figures=ROOT/'results/figures';figures.mkdir(exist_ok=True)
    fig,ax=plt.subplots(figsize=(8,3.5));x=np.arange(3)
    for i,arch in enumerate(['cnn','lstm']):
        a=[summary['core'][arch+'_'+m] for m in ['input','output','combined']];means=np.array([z['macro_f1_mean'] for z in a]);ci=np.array([z['macro_f1_ci95'] for z in a])
        ax.bar(x+(i-.5)*.32,100*means,.32,label=arch.upper(),color=COLORS[i]);ax.errorbar(x+(i-.5)*.32,100*means,yerr=100*np.maximum(0,np.array([means-ci[:,0],ci[:,1]-means])),fmt='none',capsize=3,color='#253A47')
    ax.set_xticks(x,['Prompt only','Response only','Prompt + response']);ax.set_ylabel('Test macro-F1 (%)');ax.set_ylim(0,100);ax.legend(frameon=False);ax.set_title('Arena: three seeds, 95% group-bootstrap intervals');fig.tight_layout();fig.savefig(figures/'arena_results.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(9,4))
    for ax,arch in zip(axes,['cnn','lstm']):
        cm=np.array(summary['core'][arch+'_output']['confusion_matrix_sum']);a=cm/cm.sum(1,keepdims=True);ax.imshow(a,vmin=0,vmax=1,cmap='Blues')
        for i in range(4):
            for j in range(4):ax.text(j,i,f'{a[i,j]*100:.1f}',ha='center',va='center',color='white' if a[i,j]>.55 else 'black')
        ax.set_xticks(range(4),LABELS,rotation=25);ax.set_yticks(range(4),LABELS);ax.set_xlabel('Predicted');ax.set_ylabel('True');ax.set_title(arch.upper()+' response-only recall (%)')
    fig.tight_layout();fig.savefig(figures/'arena_confusion.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(9,3.4))
    for ax,arch in zip(axes,['cnn','lstm']):
        c=selection[arch]['candidate'];h=read(RESULT/f'{tag(c,"output",1448)}.json')['history'];ax.plot([x['epoch'] for x in h],[100*x['val_macro_f1'] for x in h],color=COLORS[0],label='Validation macro-F1');ax.set_xlabel('Epoch');ax.set_ylabel('Validation macro-F1 (%)');ax.set_title(arch.upper()+' seed 1448');ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(figures/'arena_learning.png',dpi=180);plt.close(fig)
    print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
