"""Plot RL-only saved results without selecting or tuning policies."""
import json
from pathlib import Path
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cf_pilot.benchmark import file_hash, write_json


if __name__ == "__main__":
    config = json.loads((ROOT / "config/deep-rl.json").read_text())
    report = ROOT / "reports" / config["run_id"]
    result = json.loads((report / "results.json").read_text())
    legacy = json.loads((report / "legacy-rl-fresh-test.json").read_text())
    curves = json.loads((report / "learning-curves.json").read_text())
    plt.rcParams.update({"font.size":11,"axes.spines.top":False,"axes.spines.right":False})
    names = []
    def save(fig,name):
        for ext in (".png",".svg"): fig.savefig(report/(name+ext),dpi=180,bbox_inches="tight")
        plt.close(fig); names.append(name)
    fig,ax=plt.subplots(figsize=(11,5),layout="constrained")
    for candidate in config['candidates']:
        points=curves[candidate['name']+'_42']
        ax.plot([p['updates']/1000 for p in points],[p['validation']['f1']*100 for p in points],marker='o',label=candidate['name'].replace('_',' '))
    ax.set(xlabel='Training updates (thousands)',ylabel='Validation F1 (%)',title='RL-only model search: continuous behavioural features')
    ax.grid(alpha=.2);ax.legend(fontsize=9,ncol=2)
    save(fig,'rl-only-validation-curves')
    labels=['Q-learning\n5 seeds','SARSA\n5 seeds','DQN\n1 seed','Dueling Double DQN\nselected, 1 seed']
    families=[legacy['seed_aggregate']['q_learning'],legacy['seed_aggregate']['sarsa']]
    plain=result['fresh_test']['dqn64_p025_42']['all']; selected=result['fresh_test'][result['selected']]['all']
    fig,ax=plt.subplots(figsize=(11,5.2),layout='constrained');x=np.arange(4)
    for shift,field,color in [(-.24,'precision','#1565c0'),(0,'recall','#c26513'),(.24,'f1','#31714b')]:
        v=[family[field]['mean'] for family in families]+[plain[field],selected[field]]
        err=[family[field]['std'] for family in families]+[0,0]
        ax.bar(x+shift,np.array(v)*100,.22,label=field.capitalize(),color=color,yerr=np.array(err)*100,capsize=3)
    ax.set(xticks=x,xticklabels=labels,ylabel='Percent',title='Fresh contest 2274: RL algorithms only')
    ax.set_ylim(0,85);ax.legend(loc='upper left');ax.grid(axis='y',alpha=.2)
    save(fig,'rl-only-fresh-comparison')
    fig,ax=plt.subplots(figsize=(10,4.8),layout='constrained')
    amounts=[f['average_inspections']['mean'] for f in families]+[plain['average_inspections'],selected['average_inspections']]
    ax.bar(x,amounts,color=['#88949e','#88949e','#1565c0','#31714b'])
    ax.set(xticks=x,xticklabels=labels,ylabel='Mean additional groups inspected',title='Policy evidence inspection; costs are simulated')
    ax.grid(axis='y',alpha=.2)
    save(fig,'rl-only-inspections')
    c=selected['confusion'];counts=np.array([[c['tn'],c['fp']],[c['fn'],c['tp']]])
    normalized=counts/counts.sum(axis=1,keepdims=True)
    fig,ax=plt.subplots(figsize=(7.5,5.1),layout='constrained');ax.imshow(normalized,cmap='Blues',vmin=0,vmax=1)
    for i in range(2):
        for j in range(2): ax.text(j,i,f'{counts[i,j]:,}\n{normalized[i,j]:.1%} of row',ha='center',va='center',fontsize=14,color='white' if normalized[i,j]>.55 else 'black')
    ax.set(xticks=[0,1],xticklabels=['Low priority','Review'],yticks=[0,1],yticklabels=['No-skipped\ncomparison','All-skipped\nmultiple problems'],xlabel='Policy recommendation',ylabel='Recorded outcome',title='Validation-selected dueling Double DQN: fresh test')
    ax.spines[:].set_visible(False)
    save(fig,'rl-only-selected-confusion')
    write_json(report/'plot-manifest.json',{'results_sha256':file_hash(report/'results.json'),'plot_code_sha256':file_hash(Path(__file__)),'plots':{n+e:file_hash(report/(n+e)) for n in names for e in ['.png','.svg']}})
