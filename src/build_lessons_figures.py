"""Figures for the report's Lessons section, drawn only from saved result files."""
import json,re
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
ROOT=Path(__file__).resolve().parents[1];FIGURES=ROOT/'results/figures'
COLORS=['#0B7774','#DC8242','#536FA8','#915E91'];LABELS=['Claude','GPT','Grok','DeepSeek']
def read(p):return json.loads((ROOT/p).read_text())
v1,v2,arena=read('results/summary.json'),read('results/v2/summary.json'),read('results/arena/summary.json')
curve,post=read('results/arena_followup/curve.json'),read('results/arena_followup/posthoc.json')
def acc(s,arch):return 100*s['core'][arch+'_output']['accuracy_mean']
def show(g):
    for c in '#*-':g=re.sub(rf'(?<=\{c}) (?=\{c})','',g)
    g=re.sub(r"(?<=\w) ([’']) (?=\w)",r'\1',g);return re.sub(r' ([.,!?:])',r'\1',g)

# 1. Three project stages.
fig,ax=plt.subplots(figsize=(7.6,3.5));stages=['1. Initial\nUltraFeedback, 3 classes','2. Same-data tuning\nUltraFeedback, 3 classes','3. Arena\n4 classes, 2.27x data'];x=np.arange(3)
for i,arch in enumerate(['cnn','lstm']):
    vals=[acc(s,arch) for s in [v1,v2,arena]];bars=ax.bar(x+(i-.5)*.34,vals,.34,color=COLORS[i],label=arch.upper())
    for b,v in zip(bars,vals):ax.text(b.get_x()+b.get_width()/2,v+1,f'{v:.1f}',ha='center',fontsize=8)
ax.axvline(1.5,color='#888',ls='--',lw=1);ax.text(1.53,96,'dataset changed',fontsize=8,color='#555')
ax.set_xticks(x,stages,fontsize=8.5);ax.set_ylim(0,100);ax.set_ylabel('Response-only test accuracy (%)');ax.set_title('Three stages of the project (three-seed means)')
ax.legend(frameon=False,loc='upper left');ax.grid(axis='y',alpha=.2);fig.tight_layout();fig.savefig(FIGURES/'lessons_journey.png',dpi=180);plt.close(fig)

# 2. Arena learning curve against the accuracy where the first version stopped.
keys=sorted(curve['summary'],key=float);n=[curve['rows'][k] for k in keys]
fig,ax=plt.subplots(figsize=(7.6,3.5))
for i,arch in enumerate(['cnn','lstm']):ax.plot(n,[100*curve['summary'][k][arch]['accuracy_mean'] for k in keys],marker='o',color=COLORS[i],label=f'{arch.upper()} on Arena subsets')
for y,label in [(acc(v1,'lstm'),'first-version LSTM (UltraFeedback)'),(acc(v2,'lstm'),'tuned LSTM (UltraFeedback)')]:
    ax.axhline(y,color='#888',ls=':',lw=1.2);ax.text(n[0],y+.6,f'{label}: {y:.1f}%',ha='left',fontsize=8,color='#555')
ax.set_xscale('log',base=2);ax.set_xticks(n,[f'{x:,}' for x in n]);ax.minorticks_off();ax.set_xlabel('Training responses (log scale)');ax.set_ylabel('Test accuracy (%)')
ax.set_title('More training data moves the LSTM past where the first version stopped');ax.set_ylim(62,90);ax.legend(frameon=False,loc='upper left');ax.grid(alpha=.2)
fig.tight_layout();fig.savefig(FIGURES/'lessons_data.png',dpi=180);plt.close(fig)

# 3. Signature phrases the CNN relies on.
rows=[(f,show(x['ngram']),100*x['share_of_docs']) for f in LABELS for x in [t for t in post['cnn_ngrams'][f]['top'] if len(re.findall(r"[a-z]+(?:'[a-z]+)?",show(t['ngram'])))>=2][:3]]
fig,ax=plt.subplots(figsize=(7.6,3.9));y=np.arange(len(rows))[::-1]
ax.barh(y,[r[2] for r in rows],color=[COLORS[LABELS.index(r[0])] for r in rows])
ax.set_yticks(y,[f'{r[0]}:  "{r[1]}"' for r in rows],fontsize=8.5)
for yy,r in zip(y,rows):ax.text(r[2]+.3,yy,f'{r[2]:.0f}%',va='center',fontsize=8)
ax.set_xlabel('Share of correctly classified training answers where the CNN relied on the phrase (%)',fontsize=8.5)
ax.set_title('Signature phrases behind CNN decisions (↵ = newline)',fontsize=11);ax.grid(axis='x',alpha=.2)
fig.tight_layout();fig.savefig(FIGURES/'lessons_signatures.png',dpi=180);plt.close(fig)
print('lessons figures written')
