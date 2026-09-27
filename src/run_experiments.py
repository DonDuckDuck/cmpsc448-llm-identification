"""Fixed experiment matrix. Restart safely by skipping complete result files."""
from pathlib import Path
import json,subprocess,sys,time
ROOT=Path(__file__).resolve().parents[1]
def main():
    commands=[]
    for seed in [448,449,450]:
        for model in ['cnn','lstm']:
            for mode in ['input','output','combined']:
                commands.append((f'{model}_{mode}_s{seed}',['--model',model,'--mode',mode,'--seed',str(seed)]))
    for model in ['cnn','lstm']:
        commands.append((f'{model}_output_s448_normalized',['--model',model,'--mode','output','--normalized']))
        commands.append((f'{model}_output_s448_source_holdout',['--model',model,'--mode','output','--domain']))
        commands.append((f'{model}_output_s448_source_control',['--model',model,'--mode','output','--source-control']))
    for model in ['cnn','lstm']:
        commands.append((f'{model}_output_s448_shuffled',['--model',model,'--mode','output','--shuffle-labels']))
    plan={'created_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'primary_seeds':[448,449,450],
          'max_epochs':8,'patience':3,'selection':'validation macro-F1; tie-break validation loss',
          'commands':commands,'notes':'No test-based hyperparameter selection; auxiliary experiments use seed 448. Pilot runs evaluate validation only.'}
    planpath=ROOT/'docs/experiment_plan.json'
    if not planpath.exists():planpath.write_text(json.dumps(plan,indent=2)+'\n')
    for tag,args in commands:
        if all(p.exists() for p in [ROOT/'results'/f'{tag}.json', ROOT/'results/predictions'/f'{tag}.jsonl', ROOT/'results/checkpoints'/f'{tag}.pt']):continue
        print('RUN',tag,flush=True)
        with (ROOT/'results'/f'{tag}.log').open('w') as log:
            subprocess.run([sys.executable,str(ROOT/'src/train.py'),*args],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        r=json.loads((ROOT/'results'/f'{tag}.json').read_text())
        print('COMPLETE',tag,'acc',round(r['test']['accuracy'],4),'F1',round(r['test']['macro_f1'],4),flush=True)
if __name__=='__main__':main()
