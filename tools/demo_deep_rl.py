"""Restore the selected RL network and replay a held-out example without its target."""
import argparse
import csv
from itertools import islice
import json
from pathlib import Path
import sys
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from cf_pilot.benchmark import FEATURE_GROUPS,file_hash
from cf_pilot.deep_rl import QNetwork,infer,rewards
from cf_pilot.rl_data import load_data


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--case',choices=['true_positive','true_negative','false_positive','false_negative'],default='true_positive');args=parser.parse_args()
    torch.set_num_threads(1)
    config=json.loads((ROOT/'config/deep-rl.json').read_text())
    models=ROOT/'data'/config['run_id'];report=ROOT/'reports'/config['run_id']
    selection=json.loads((models/'selection.json').read_text());examples=json.loads((report/'demo-cases.json').read_text())
    case=examples['cases'][args.case];idx=int(case['example_id'].split('_')[-1])
    folder=ROOT/'data'/config['benchmark_id'];summary=json.loads((folder/'summary.json').read_text());pre=json.loads((folder/'preprocessor.json').read_text())
    data=load_data(folder,'test',summary,pre)
    path=models/(selection['selected']+'.pt')
    if file_hash(path)!=selection['model_sha256'][selection['selected']]:raise ValueError('Selected weights changed')
    candidate=selection['selected_candidate'];network=QNetwork(candidate['width'],candidate['dueling'])
    network.load_state_dict(torch.load(path,map_location='cpu',weights_only=True))
    pred,count,masks,actions=infer(network,data['x'][idx:idx+1],traces=True)
    names=['inspect_pacing','inspect_trajectory','inspect_language','low_priority','review']
    observed_actions=[names[a] for a in actions[0]]
    if observed_actions!=case['actions']:raise ValueError('Restored deep policy differs from saved example')
    with (folder/'test/features.csv').open(encoding='utf-8',newline='') as stream:
        raw=next(islice(csv.DictReader(stream),idx,idx+1))
    def values(group):return {name:round(float(raw[name]),4) if raw[name] else 'missing' for name in FEATURE_GROUPS[group]}
    print('RL-only post-contest review demonstration')
    print('Policy:',selection['selected'],'| Example:',case['example_id'])
    print('Initial state summaries:',values('basic'))
    for step,action in enumerate(actions[0],1):
        print('Action',step,':',names[action])
        if action<3:print('Revealed:',values(('pacing','trajectory','language')[action]))
    print('Recommendation:','REVIEW' if pred[0] else 'LOW_PRIORITY')
    print('Additional groups inspected:',int(count[0]))
    target=int(data['y'][idx]); print('Recorded outcome revealed after inference:',target)
    total=float(rewards([4 if pred[0] else 3],[target],candidate['false_positive_penalty'])[0])-float(count[0])*config['settings']['inspection_cost']
    print('Evaluation reward including simulated costs:',round(total,4))
    print('Target is the all-skipped multi-problem outcome, not verified cheating or AI use.')
