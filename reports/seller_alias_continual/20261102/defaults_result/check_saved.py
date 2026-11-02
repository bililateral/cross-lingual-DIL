"""Linux independent arithmetic check of saved outputs; never opens labels."""
from pathlib import Path
import argparse
import json
import math
import time

import numpy as np


def main(root, old_root):
    started = time.monotonic()
    result = json.loads((root/'result.json').read_text())
    cols = result['columns']; roles = ('raw', 'stage-cal', 'first-cal')
    names = ('S_default', 'H_default', 'S_tuned', 'H_pos_tuned')
    orders = ('ABC', 'BCA', 'CAB'); seeds = (4,5,6)
    terms = {'O':[(3,1,.5),(3,2,.5)],'N':[(2,2,.5),(3,3,.5)],'Z':[(3,3,1)],
        'A_acq':[(1,1,.5),(2,2,.5)],'F':[(1,1,.5),(2,2,.5),(3,1,-.5),(3,2,-.5)],
        'F_first':[(1,1,1),(3,1,-1)],'D2':[(1,1,1),(2,1,-1)],
        'D3':[(2,1,.5),(2,2,.5),(3,1,-.5),(3,2,-.5)],
        'final_mean':[(3,1,1/3),(3,2,1/3),(3,3,1/3)]}
    # Frequencies reproduce exactly the fixed draws, using matrix products
    # rather than production's per-draw index averaging.
    rng = np.random.Generator(np.random.PCG64(2026110202))
    indices = rng.integers(20, size=(5000,3,20))
    frequencies = np.zeros((5000,3,20))
    for b in range(5000):
        for d in range(3): frequencies[b,d] = np.bincount(indices[b,d], minlength=20)/20
    vectors = {}; old_compared = 0; matrix_count = 0; max_error = 0.
    def equal(a,b):
        nonlocal max_error
        a,b=np.asarray(a,float),np.asarray(b,float)
        np.testing.assert_allclose(a,b,rtol=1e-11,atol=1e-12,equal_nan=True)
        ok=np.isfinite(a)&np.isfinite(b)
        if ok.any():max_error=max(max_error,float(np.max(abs(a[ok]-b[ok]))))
    for name in names:
        vectors[name] = {}
        for role in roles:
            tracks = {field:[] for field in terms}
            for order in orders:
                for seed in seeds:
                    rid=f'{order}_s{seed}_{name}'; saved={}
                    for stage in (1,2,3):
                        stem=f'{rid}_{stage}_{role}'
                        payload=json.loads((root/'metrics'/(stem+'.json')).read_text())
                        assert payload['columns']==cols
                        values=np.array(payload['values'],float);counts=payload['counts']
                        queries=np.load(root/'metrics'/(stem+'_queries.npy'),allow_pickle=False)
                        scores=np.load(root/'scores'/(stem+'.npy'),allow_pickle=False)
                        assert values.shape==(60,22) and queries.shape==(60,28,10) and scores.shape==(60,378)
                        equal(values[:,12:],queries.mean(1))
                        for i,c in enumerate(counts):
                            tp,fp,fn,tn=[c[k] for k in ('tp','fp','fn','tn')]
                            assert tp+fn==20 and tn+fp==358
                            assert int((scores[i]>=0).sum())==tp+fp
                            den=math.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
                            expected=[tp/(tp+fp) if tp+fp else np.nan,tp/20,2*tp/(2*tp+fp+fn),tn/358,(tp/20+tn/358)/2,(tp*tn-fp*fn)/den if den else np.nan]
                            equal(values[i,6:12],expected)
                        if name in ('S_tuned','H_pos_tuned'):
                            prior=json.loads((old_root/(stem+'.json')).read_text())
                            equal(values,prior['values']);assert counts==prior['counts'];old_compared+=1
                        for d,domain in enumerate('ABC'):saved[stage,domain]=values[d*20:(d+1)*20]
                        matrix_count+=1
                    for field,components in terms.items():
                        v=np.zeros((3,20,22))
                        for stage,pos,weight in components:
                            domain=order[pos-1];v['ABC'.index(domain)]+=weight*saved[stage,domain]
                        equal(v.mean(1).sum(0),result['per_run'][rid][role][field]);tracks[field].append(v)
            vectors[name][role]={field:np.mean(vals,axis=0) for field,vals in tracks.items()}
    checked_intervals=0
    def verify_intervals(v, entries):
        nonlocal checked_intervals
        point=v.mean(1).sum(0)
        for j,col in enumerate(cols):
            equal(point[j],entries[col]['mean'])
            if np.isnan(v[:,:,j]).any():
                assert entries[col]['conditional_95pct_interval']==[None,None]
            else:
                samples=sum(frequencies[:,d,:]@v[d,:,j] for d in range(3))
                equal(np.quantile(samples,[.025,.975]),entries[col]['conditional_95pct_interval'])
            checked_intervals+=1
    for name in names:
        for role in roles:
            for field,v in vectors[name][role].items():verify_intervals(v,result['absolute'][name][role][field])
    for key,roles_saved in result['comparisons'].items():
        a,b=key.split('-')
        for role in roles:
            for field in terms:verify_intervals(vectors[a][role][field]-vectors[b][role][field],roles_saved[role][field])
    return dict(status='PASS_SAVED_RESULTS_NO_LABEL_REREAD',matrices=matrix_count,old_reference_matrices_identical=old_compared,
                interval_entries=checked_intervals,maximum_absolute_difference=max_error,seconds=time.monotonic()-started,
                scope='Saved counts/query means, old tuned development records, endpoint arithmetic, fixed-seed paired intervals; no new bootstrap design or label/model access')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('old_root',type=Path);a=p.parse_args()
    print(json.dumps(main(a.root,a.old_root),indent=2))
