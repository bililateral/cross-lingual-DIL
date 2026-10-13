#!/usr/bin/env python3
"""Small artificial-table checks for a static review; NOT a project implementation.

Only reads inputs.json beside this file. No project imports, torch, data, models,
Memory, network, training, or optimizer. Exact rational enumerations plus a small
NumPy finite-difference calculation. Run: python check_math.py
"""
from __future__ import annotations
import itertools
import json
import math
from fractions import Fraction as F
from pathlib import Path
from typing import Iterator
import numpy as np

ROOT = Path(__file__).resolve().parent
CFG = json.loads((ROOT / 'inputs.json').read_text(encoding='utf-8'))
RESULTS: list[dict] = []
ENUMERATIONS: list[dict] = []


def note(name: str, **details) -> None:
    RESULTS.append({'name': name, 'status': 'PASS', **details})
    print('PASS', name, json.dumps(details, ensure_ascii=False, sort_keys=True))


def partitions(items: tuple[int, ...], sizes: tuple[int, ...]) -> Iterator[tuple[tuple[int, ...], ...]]:
    """All equiprobable labelled-slot partitions; order within a slot irrelevant."""
    if len(sizes) == 1:
        assert len(items) == sizes[0]
        yield (items,)
        return
    for first in itertools.combinations(items, sizes[0]):
        used = set(first)
        rest = tuple(i for i in items if i not in used)
        for suffix in partitions(rest, sizes[1:]):
            yield (first,) + suffix


def favg(values) -> F:
    values = list(values)
    return sum(values, F(0)) / len(values)


def finite_cases() -> None:
    for case in CFG['finite_population_cases']:
        counts = tuple(case['counts']); e = [F(x) for x in case['e']]
        M = len(e); assert M == sum(counts) and sum(e) == 0
        origins = [a for a,n in enumerate(counts) for _ in range(n)]
        for a in range(len(counts)):
            assert sum(e[r] for r in range(M) if origins[r] == a) == 0
        d = [F(case['d'][a]) for a in origins]
        rows = []
        for cells in partitions(tuple(range(M)), counts):
            means = [favg(e[r] for r in slot) for slot in cells]
            comp = [favg(d[r] for r in slot) for slot in cells]
            rows.append((cells, means, comp))
        sse = sum(x*x for x in e)
        expected = [F(M-n, n*M*(M-1))*sse for n in counts]
        observed = [favg(m[j]**2 for _,m,_ in rows) for j in range(len(counts))]
        assert observed == expected
        covariances = []
        for j in range(len(counts)):
            assert favg(m[j] for _,m,_ in rows) == 0
            assert favg(m[j]*c[j] for _,m,c in rows) == 0
            assert favg((m[j]+c[j])**2 for _,m,c in rows) == expected[j] + favg(c[j]**2 for _,m,c in rows)
            dbar = favg(d)
            comp_formula = dbar*dbar + F(M-counts[j], counts[j]*M*(M-1))*sum((x-dbar)**2 for x in d)
            assert comp_formula == favg(c[j]**2 for _,m,c in rows)
            for k in range(j+1, len(counts)):
                cov = favg(m[j]*m[k] for _,m,_ in rows)
                assert cov == -sse / (M*(M-1))
                covariances.append({'j':j,'k':k,'covariance':str(cov)})
        b = sum((F(M-n, n*M*(M-1)) for n in counts), F(0))
        assert sum(observed) == b*sse
        ENUMERATIONS.append({'counts':counts, 'e':case['e'], 'd':case['d'], 'partitions':[
            {'slots':cells, 'residual_means':[str(x) for x in means], 'composition_means':[str(x) for x in comp]}
            for cells,means,comp in rows]})
        note('finite_population_'+ '_'.join(map(str,counts)), partitions=len(rows),
             slot_second_moments=[str(x) for x in observed], sum_second_moments=str(sum(observed)),
             b=str(b), covariances=covariances, composition_cross_terms='exactly zero')


def means_for_table(E: np.ndarray, counts: list[int], components: list[list[int]], norm: int):
    cuts = np.cumsum([0]+counts)
    accounts = [np.arange(cuts[a],cuts[a+1]) for a in range(len(counts))]
    comp_of = {a:g for g,grp in enumerate(components) for a in grp}
    e = np.zeros((len(counts),len(E))); mu = np.zeros((len(counts),len(counts)))
    b = np.zeros((len(counts),len(components)))
    for q in range(len(counts)):
        for a in range(len(counts)):
            if q == a: continue
            u = E[np.ix_(accounts[q],accounts[a])].mean(axis=0)
            mu[q,a] = u.mean(); e[q,accounts[a]] = u-u.mean()
        for g,grp in enumerate(components):
            H = [a for a in grp if a!=q]
            M = sum(counts[a] for a in H)
            assert M >= 2
            b[q,g] = sum((M-counts[a])/(counts[a]*M*(M-1)) for a in H)
    R = sum(b[q,comp_of[a]] * float(e[q,accounts[a]] @ e[q,accounts[a]])
            for q in range(len(counts)) for a in range(len(counts)) if a!=q) / norm
    D0 = sum(mu[q,a]**2 for q in range(len(counts)) for a in range(q+1,len(counts))) / (norm/2)
    return R,D0,e,b,accounts,comp_of


def gradient_check() -> None:
    c = CFG['table_gradient_toy']; counts=c['counts']; comps=c['components']; norm=c['normalizer']
    N=sum(counts); E=np.zeros((N,N),dtype=np.float64)
    for i in range(N):
        for j in range(i+1,N):
            E[i,j]=E[j,i]=(((i+2)*(j+3)+3*i-2*j)%19-9)/7
    R,D0,e,b,accounts,comp_of=means_for_table(E,counts,comps,norm)
    owner={int(r):a for a,rs in enumerate(accounts) for r in rs}
    eps=c['finite_difference_epsilon']; diffs=[]; ignored=[]; derivatives=[]
    for i in range(N):
        for j in range(i+1,N):
            q,a=owner[i],owner[j]
            analytic=0.0 if q==a else (2/norm)*(b[q,comp_of[a]]*e[q,j]/counts[q] + b[a,comp_of[q]]*e[a,i]/counts[a])
            plus=E.copy(); minus=E.copy()
            plus[i,j]+=eps; plus[j,i]+=eps; minus[i,j]-=eps; minus[j,i]-=eps
            numerical=(means_for_table(plus,counts,comps,norm)[0]-means_for_table(minus,counts,comps,norm)[0])/(2*eps)
            diffs.append(abs(analytic-numerical))
            if q==a: ignored.append(abs(numerical))
            derivatives.append({'i':i,'j':j,'analytic':analytic,'numerical':numerical})
    assert max(diffs)<c['absolute_tolerance'] and max(ignored)==0
    (ROOT/'gradient_details.json').write_text(json.dumps({'error_table':E.tolist(),'derivatives':derivatives},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    note('symmetric_table_gradient_both_query_roles', records=N, unique_pairs=len(diffs), reduced_toy_normalizer=norm,
         R=R,D0=D0,max_absolute_error=max(diffs),same_origin_numeric_gradient_max=max(ignored),
         final_objective_R_multiplier=0.25)


def sp(x: float) -> float:
    return max(x,0.0)+math.log1p(math.exp(-abs(x)))


def lse(xs: list[float]) -> float:
    m=max(xs); return m+math.log(sum(math.exp(x-m) for x in xs))


def row_terms(Q: np.ndarray, comps: list[list[int]]) -> tuple[float,float,list[list[int]]]:
    n=len(Q); owner={q:g for g,grp in enumerate(comps) for q in grp}
    ranks=[]; hards=[]; sets=[]
    for q in range(n):
        pos=[j for j in range(n) if j!=q and owner[j]==owner[q]]
        neg=[j for j in range(n) if owner[j]!=owner[q]]
        hard=sorted(neg,key=lambda j:(-Q[q,j],j))[:5]
        ranks.append(lse([float(Q[q,j]) for j in range(n) if j!=q])-sum(Q[q,p] for p in pos)/len(pos))
        hards.append(sum(sp(float(Q[q,k]-Q[q,p])) for p in pos for k in hard)/(len(pos)*len(hard)))
        sets.append(hard)
    return sum(ranks)/n,sum(hards)/n,sets


def brow(Q: np.ndarray, comps: list[list[int]]) -> tuple[float,dict]:
    n=len(Q); owner={q:g for g,grp in enumerate(comps) for q in grp}
    ce=sum(sp(float(Q[q,j]))-int(owner[q]==owner[j])*Q[q,j]
           for q in range(n) for j in range(n) if q!=j)/(n*(n-1))
    rank,hard,sets=row_terms(Q,comps)
    return float(ce+rank+0.5*hard),{'bce':float(ce),'rank':float(rank),'hard_mean':float(hard),'top5':sets}


def btri(Q: np.ndarray, comps: list[list[int]]) -> float:
    # Deliberately uses only the upper triangle, as the ORIGINAL symmetric interface does.
    n=len(Q); owner={q:g for g,grp in enumerate(comps) for q in grp}
    ce=sum(sp(float(Q[q,j]))-int(owner[q]==owner[j])*Q[q,j]
           for q in range(n) for j in range(q+1,n))/(n*(n-1)/2)
    reconstructed=np.triu(Q,1); reconstructed=reconstructed+reconstructed.T
    rank,hard,_=row_terms(reconstructed,comps)
    return float(ce+rank+0.5*hard)


def supervision_check() -> None:
    c=CFG['supervised_toy']; n=c['accounts']; comps=c['components']; Q=np.zeros((n,n))
    for i in range(n):
        for j in range(i+1,n): Q[i,j]=Q[j,i]=((7*i+11*j)%23-11)/5
    row,parts=brow(Q,comps); tri=btri(Q,comps)
    assert abs(row-tri)<c['absolute_tolerance']
    asym=Q.copy(); p=c['directed_perturbation']; asym[p['row'],p['column']]+=p['increment']
    ar,ap=brow(asym,comps); at=btri(asym,comps)
    assert abs(ar-at)>1e-5 and at==tri
    _,ties=brow(np.zeros((n,n)),comps)
    owner={q:g for g,grp in enumerate(comps) for q in grp}
    assert all(hard==[j for j in range(n) if owner[q]!=owner[j]][:5] for q,hard in enumerate(ties['top5']))
    positives=sum(len(grp)*(len(grp)-1) for grp in comps)
    note('natural_Brow_Btri_and_directed_non_equivalence', directed_cells=n*(n-1), positive_cells=positives,
         negative_cells=n*(n-1)-positives, symmetric_Brow=row,symmetric_Btri=tri, symmetric_error=abs(row-tri),
         directed_Brow=ar,incorrect_triangular_loss=at, top5_all_ties_query0=ties['top5'][0],
         K2_zero_R_positive_cells=16,K3_potentially_active_R_positive_cells=24,
         natural_components={k:v for k,v in parts.items() if k!='top5'})


def average_block(E, left, right) -> float:
    return float(E[np.ix_(left,right)].mean())


def counterexamples() -> None:
    # Same-source channel in S mixed bags but not a natural query.
    E=np.zeros((4,4)); E[0,1]=E[1,0]=1; E[2,3]=E[3,2]=1
    nat=average_block(E,[0,1],[2,3]); mixed=average_block(E,[0,2],[1,3])
    assert nat==0 and mixed==0.5
    note('same_source_channel', natural=nat,S_mixed=mixed,query_exclusive=nat)
    # Legal independent rows in a K3 produce non-symmetric Q.
    E=np.zeros((6,6)); block=np.array([[2.,0.],[2.,0.]])
    E[:2,2:4]=block; E[2:4,:2]=block.T
    qa=average_block(E,[0,1],[2,4]); qb=average_block(E,[2,3],[0,4])
    assert qa==1 and qb==0.5
    note('legal_directed_K3_rows',Q_01=qa,Q_10=qb)
    # A cross-origin block in K3: b>0 in both query roles.
    comps=[[0,1,2]]; counts=[2,2,2]
    for name,block in [('mean_blind',np.array(CFG['block_toys']['mean_blind'],float)),
                       ('zero_marginal',np.array(CFG['block_toys']['zero_marginal'],float))]:
        E=np.zeros((6,6)); E[:2,2:4]=block; E[2:4,:2]=block.T
        R,D0,_,_,_,_=means_for_table(E,counts,comps,6)
        assert D0==0
        if name=='mean_blind': assert R>0
        else:
            assert R==0
            u=np.vstack([E[2*q:2*q+2,:].mean(axis=0) for q in range(3)])
            assert np.count_nonzero(u)==0
            assert average_block(E,[0,2],[1,3]) == -0.5
        note('block_'+name,D0=D0,R=R,normalizer=6,
             S_mixed_block_score=average_block(E,[0,2],[1,3]))
    # K2 mates have b=0, including nonzero row/column effects.
    E=np.zeros((4,4)); block=np.array(CFG['block_toys']['mean_blind'],float)
    E[:2,2:]=block; E[2:,:2]=block.T
    R,D0,_,b,_,_=means_for_table(E,[2,2],[[0,1]],2)
    assert R==0 and D0==0 and np.count_nonzero(b)==0
    note('K2_positive_residual_blind',R=R,D0=D0,normalizer=2)
    # Query-independent record offsets survive per-original-account centering.
    t=np.array([1.,-1.,2.,-2.,3.,-3.]); E=t[:,None]+t[None,:]; np.fill_diagonal(E,0)
    R,D0,_,_,_,_=means_for_table(E,[2,2,2],[[0,1,2]],6)
    assert D0==0 and R>0
    note('query_independent_nuisance_is_penalized',R=R,D0=D0,normalizer=6,offsets=t.tolist())
    # Ordinary with-replacement bootstrap's own second moment differs from R_q's coefficient.
    e=[F(1),F(-1)]; draws=list(itertools.product(range(2),repeat=2))
    second=favg(favg(e[r] for r in draw)**2 for draw in draws)
    assert second==F(1,2)==sum(x*x for x in e)/4
    note('bootstrap_not_Rq_expectation',within_account_bootstrap_second_moment=str(second),
         K2_positive_query_exclusive_second_moment='0')


def main() -> None:
    finite_cases(); gradient_check(); supervision_check(); counterexamples()
    (ROOT/'enumerations.json').write_text(json.dumps(ENUMERATIONS,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    out={'scope':CFG['scope'],'all_passed':True,'check_groups':len(RESULTS),'results':RESULTS}
    (ROOT/'results.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('COMPLETE',len(RESULTS),'check groups; artificial mathematics only.')


if __name__=='__main__':
    main()
