"""Recompute every published metric and replay a saved neural checkpoint."""
import json,hashlib
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from train import ROOT,LABELS,load_rows,metrics,TextData,collate,predict
from models import TextCNN

def main():
    checked=[]
    manifest_hash=hashlib.sha256((ROOT/'data/processed/split_manifest.json').read_bytes()).hexdigest()
    for path in sorted((ROOT/'results/predictions').glob('*.jsonl')):
        rows=[json.loads(l) for l in path.read_text().splitlines()]
        y=np.array([r['true'] for r in rows]);probs=np.array([r['probs'] for r in rows])
        assert np.isfinite(probs).all() and (probs>=0).all()
        np.testing.assert_allclose(probs.sum(axis=1),1,atol=1e-5)
        assert all(LABELS[r['true']]==r['family'] and r['pred']==int(np.argmax(r['probs'])) for r in rows)
        r=json.loads((ROOT/'results'/f'{path.stem}.json').read_text());calculated=metrics(y,probs)
        for key in ['accuracy','macro_f1']:assert abs(calculated[key]-r['test'][key])<1e-12
        assert calculated['confusion_matrix']==r['test']['confusion_matrix']
        if 'split_manifest_sha256' in r:assert r['split_manifest_sha256']==manifest_hash
        if '_input' in path.stem:assert abs(calculated['accuracy']-1/3)<1e-12
        checked.append(path.stem)
    assert len(checked)==31,checked
    torch.set_num_threads(4)
    checkpoint=ROOT/'results/checkpoints/cnn_output_s448.pt'
    if not checkpoint.exists():checkpoint=ROOT/'models/cnn_output_s448.pt'
    saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
    device='mps' if torch.backends.mps.is_available() else 'cpu'
    model=TextCNN(len(saved['vocab'])).to(device);model.load_state_dict(saved['state_dict'])
    test=load_rows('test');loader=DataLoader(TextData(test,saved['vocab'],'output'),batch_size=64,collate_fn=collate)
    y,p,_=predict(model,loader,device)
    original=[json.loads(l) for l in (ROOT/'results/predictions/cnn_output_s448.jsonl').read_text().splitlines()]
    expected=np.array([r['probs'] for r in original])
    np.testing.assert_allclose(p,expected,atol=2e-4,rtol=2e-4)
    assert np.array_equal(p.argmax(axis=1),expected.argmax(axis=1))
    summary={'prediction_files_verified':len(checked),'all_metrics_match_saved_predictions':True,
             'prompt_only_accuracy_exactly_one_third':True,'split_manifest_hash':manifest_hash,
             'checkpoint_replay':'cnn_output_s448','checkpoint_replay_device':device,
             'max_probability_difference':float(np.max(np.abs(p-expected)))}
    (ROOT/'results/verification.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
