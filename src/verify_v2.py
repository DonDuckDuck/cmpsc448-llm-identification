"""Audit every v2 prediction and replay the two selected demonstration models."""
import hashlib,json
import numpy as np
import torch
from torch.utils.data import DataLoader
from train import ROOT,LABELS,metrics,predict,collate,load_rows
from iterate import Data
from models_v2 import make_model

def main():
    dest=ROOT/'results/v2';selected=json.loads((ROOT/'docs/iteration_selection.json').read_text())
    selection_hash=hashlib.sha256((ROOT/'docs/iteration_selection.json').read_bytes()).hexdigest()
    validation_checked=0
    for path in sorted(dest.glob('*_s*.json')):
        record=json.loads(path.read_text());arrays=np.load(dest/f'{path.stem}_val.npz')
        recalculated=metrics(arrays['labels'],arrays['probs'])
        assert abs(recalculated['macro_f1']-record['best_val_macro_f1'])<1e-12
        validation_checked+=1
    for arch,choice in selected['selection'].items():
        candidates=choice['finalist_scores']
        ranked=sorted(candidates,key=lambda c:(-c['mean_val_macro_f1'],c['mean_val_loss'],c['candidate']))
        assert choice['candidate']==ranked[0]['candidate']
        for candidate in candidates:
            rr=[json.loads((dest/f'{candidate["candidate"]}_output_s{seed}.json').read_text()) for seed in [448,449,450]]
            assert abs(np.mean([r['best_val_macro_f1'] for r in rr])-candidate['mean_val_macro_f1'])<1e-12
    checked=0
    for p in sorted((dest/'predictions').glob('*.jsonl')):
        rows=[json.loads(x) for x in p.read_text().splitlines()];r=json.loads((dest/f'{p.stem}.json').read_text())
        assert r['selection_sha256']==selection_hash
        probs=np.array([x['probs'] for x in rows]);y=np.array([x['true'] for x in rows]);m=metrics(y,probs)
        np.testing.assert_allclose(probs.sum(1),1,atol=1e-5)
        for key in ['accuracy','macro_f1']:assert abs(m[key]-r['test'][key])<1e-12
        assert m['confusion_matrix']==r['test']['confusion_matrix']
        assert all(x['family']==LABELS[x['true']] and x['pred']==int(np.argmax(x['probs'])) for x in rows)
        if '_input_' in p.stem:assert abs(m['accuracy']-1/3)<1e-12
        checked+=1
    assert checked==18
    torch.set_num_threads(4);dev='mps' if torch.backends.mps.is_available() else 'cpu';replayed={}
    for arch,choice in selected['selection'].items():
        name=f'{choice["candidate"]}_output_s448';path=ROOT/f'results/checkpoints/v2/{name}.pt'
        if not path.exists():path=ROOT/f'models/v2_{arch}_output_s448.pt'
        ck=torch.load(path,map_location='cpu',weights_only=True);model=make_model(ck['config'],len(ck['vocab'])).to(dev);model.load_state_dict(ck['state_dict'])
        dl=DataLoader(Data(load_rows('test'),ck['vocab'],'output',ck['config']),batch_size=64,collate_fn=collate)
        y,p,_=predict(model,dl,dev);original=np.array([json.loads(l)['probs'] for l in (dest/'predictions'/f'{name}.jsonl').read_text().splitlines()])
        np.testing.assert_allclose(p,original,atol=2e-4,rtol=2e-4)
        assert np.array_equal(p.argmax(1),original.argmax(1));replayed[arch]=float(np.max(np.abs(p-original)))
    out={'validation_files_verified':validation_checked,'selection_recomputed':True,'prediction_files_verified':checked,'metrics_recomputed':True,'frozen_selection_hash':selection_hash,'checkpoint_replay_max_error':replayed}
    (dest/'verification.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
if __name__=='__main__':main()
