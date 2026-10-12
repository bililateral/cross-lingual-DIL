"""Independent finite-score reference for a static scientific review.

No project imports, model, optimizer, formal data, training, or project-server access.
Float64 finite differences check real-valued calculus only. FP32 arithmetic is
checked separately. The permutation examples enumerate allowed toy outcomes;
they do not claim to reproduce the project's future fixed-seed outcome.
"""
from __future__ import annotations
import argparse
import hashlib
import itertools
import json
import math
import platform
import sys
from collections import defaultdict
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path
import numpy as np

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def graph(components):
    n = sum(map(len, components))
    assert n == 28 and sorted(itertools.chain.from_iterable(components)) == list(range(n))
    y = np.zeros((n, n), dtype=bool)
    for comp in components:
        for i, j in itertools.combinations(comp, 2):
            y[i,j] = y[j,i] = True
    pairs = list(itertools.combinations(range(n), 2))
    pair_index = {pair:k for k,pair in enumerate(pairs)}
    triples, weights = [], []
    for q in range(n):
        pos = np.flatnonzero(y[q]).tolist()
        neg = [i for i in range(n) if i != q and not y[q,i]]
        for p in pos:
            for r in neg:
                triples.append((q,p,r))
                weights.append(1/(n*len(pos)*len(neg)))
    return y, pairs, pair_index, triples, np.asarray(weights, dtype=np.float64)

def scores_from_labels(y, positive, negative):
    x = np.where(y, positive, negative).astype(np.float64)
    np.fill_diagonal(x, 0)
    return x

def margins32(scores, triples):
    s = np.asarray(scores, dtype=np.float32)
    with np.errstate(over="ignore", invalid="ignore"):
        m = np.asarray([np.subtract(s[q,p],s[q,n],dtype=np.float32)
                        for q,p,n in triples],dtype=np.float32)
    if not np.isfinite(m).all():
        raise ValueError("Nonfinite defined FP32 margins")
    return m

def floors(f, o, triples):
    mf, mo = margins32(f,triples), margins32(o,triples)
    b = np.where((mf>0)&(mo>0), np.minimum(mf,mo), np.float32(0)).astype(np.float32)
    assert np.isfinite(b).all()
    return mf,mo,b

def loss_gradient(scores, b, triples, weights, pairs, pair_index):
    # This is a real-valued calculus reference on finite numerical inputs.
    s = np.asarray(scores,dtype=np.float64)
    m = np.asarray([s[q,p]-s[q,n] for q,p,n in triples])
    r = np.where(b>0,np.maximum(np.asarray(b,dtype=np.float64)-m,0),0)
    value = float(np.dot(weights,r*r))
    grad = np.zeros(len(pairs),dtype=np.float64)
    for (q,p,n),w,deficit in zip(triples,weights,r,strict=True):
        grad[pair_index[tuple(sorted((q,p)))]] -= 2*w*deficit
        grad[pair_index[tuple(sorted((q,n)))]] += 2*w*deficit
    return value,grad,int(np.sum((b>0)&(m<np.asarray(b,dtype=np.float64))))

def exact_map(scores,y):
    values=[]
    for q in range(len(y)):
        ordered=sorted((i for i in range(len(y)) if i!=q),key=lambda i:(-float(scores[q,i]),i))
        hits=0
        total=Fraction(0)
        for rank,i in enumerate(ordered,1):
            if y[q,i]:
                hits+=1
                total+=Fraction(hits,rank)
        values.append(total/int(y[q].sum()))
    return sum(values,Fraction(0))/len(y),values

def enumerate_allocations(capacities, original):
    c=tuple(float(np.float32(x)) for x in capacities)
    orig=tuple(float(np.float32(x)) for x in original)
    assert all(a>=0 and (a==0 or a<=v) for a,v in zip(orig,c,strict=True))
    values=sorted([(a,i) for i,a in enumerate(orig) if a>0],key=lambda x:(-x[0],x[1]))
    outcomes=defaultdict(Fraction)
    traces=0
    def walk(k,used,target,probability):
        nonlocal traces
        if k==len(values):
            outcomes[tuple(target)]+=probability
            traces+=1
            return
        a,_=values[k]
        eligible=[j for j,v in enumerate(c) if j not in used and v>=a]
        assert eligible, "Descending feasible assignment unexpectedly failed"
        for j in eligible:
            out=list(target);out[j]=a
            walk(k+1,used|{j},out,probability/Fraction(len(eligible)))
    walk(0,set(),[0.0]*len(c),Fraction(1))
    assert sum(outcomes.values(),Fraction(0))==1
    for out in outcomes:
        assert sorted(x for x in out if x>0)==sorted(x for x in orig if x>0)
        assert all(a==0 or a<=v for a,v in zip(out,c,strict=True))
    assert len(set(outcomes.values()))==1
    return {"labelled_choice_traces":traces,"distinct_outputs":len(outcomes),
            "uniform_over_distinct_feasible_outputs_in_this_example":True,
            "outputs":[{"b":list(k),"probability_exact":str(v)} for k,v in sorted(outcomes.items())]}

def check_equal(a,b,atol=1e-12):
    assert math.isclose(float(a),float(b),rel_tol=0,abs_tol=atol),(a,b)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--input",required=True)
    parser.add_argument("--output",required=True)
    args=parser.parse_args()
    inp=json.loads(Path(args.input).read_text(encoding="utf-8"))
    y,pairs,pair_index,triples,w=graph(inp["components"])
    triple_index={e:i for i,e in enumerate(triples)}
    out={"scope":inp["scope"],"source_sha256":sha(__file__),"input_sha256":sha(args.input),
         "environment":{"python":sys.version,"numpy":np.__version__,"platform":platform.platform(),
                        "project_python_or_native_qualification":False},
         "checks":{}}
    checks=out["checks"]
    degrees=y.sum(1)
    assert int(y[np.triu_indices(28,1)].sum())==20
    assert int((degrees==1).sum())==16 and int((degrees==2).sum())==12
    assert len(triples)==1016
    check_equal(w.sum(),1)
    for q in range(28):
        check_equal(w[[i for i,e in enumerate(triples) if e[0]==q]].sum(),1/28)
    checks["topology_and_weights"]={"queries":28,"undirected_pairs":len(pairs),"positive_pairs":20,
          "directed_comparisons":len(triples),"weight_sum":float(w.sum()),
          "k2_comparison_weight":"1/728","k3_comparison_weight":"1/1400"}
    print("PASS topology_and_query_macro_weights")

    scalars=[]
    for item in inp["scalar_cases"]:
        mf,mo,m=float(item["mF"]),float(item["mO"]),float(item["current"])
        b=min(mf,mo) if mf>0 and mo>0 else 0.
        r=max(b-m,0) if b>0 else 0.
        scalars.append({**item,"b":b,"unweighted_loss":r*r,
                        "derivative_positive":-2*r,"derivative_negative":2*r})
    check_equal(scalars[0]["unweighted_loss"],.09)
    check_equal(scalars[0]["derivative_positive"],-.6)
    assert all(row["unweighted_loss"]==0 for row in scalars[1:])
    checks["scalar_floor_cases"]=scalars
    print("PASS scalar_min_gate_zero_sentinel_one_sided_loss")

    checks["permutation_examples"]={case["name"]:enumerate_allocations(case["capacities"],case["original"])
                                    for case in inp["permutation_cases"]}
    # Minimal counterexample to using ascending order with arbitrary choices.
    checks["ascending_order_can_fail"]={"capacities":[1,2],"thresholds_ascending":[1,2],
        "first_assignment":{"threshold":1,"chosen_capacity":2},"remaining_capacity":1,
        "second_threshold":2,"failure":True,"scope":"Not the proposed descending algorithm."}
    print("PASS descending_feasibility_and_toy_conditional_uniformity")

    sc=inp["scale_counterexample"]
    f=np.full((28,28),sc["default_offdiagonal"],dtype=np.float64);np.fill_diagonal(f,0)
    for a,b,val in sc["overrides"]:f[a,b]=f[b,a]=val
    o=f*sc["o_scale"]
    mf,mo,b=floors(f,o,triples)
    perm=b.copy()
    i,j=[triple_index[tuple(e)] for e in sc["swap_triples"]]
    perm[i],perm[j]=b[j],b[i]
    for qp in sorted({e[:2] for e in triples}):
        ids=[k for k,e in enumerate(triples) if e[:2]==qp]
        assert sorted(b[ids].tolist())==sorted(perm[ids].tolist())
    assert np.all(perm[perm>0]<=mf[perm>0])
    lcf,_,_=loss_gradient(o,b,triples,w,pairs,pair_index)
    lp,_,vp=loss_gradient(o,perm,triples,w,pairs,pair_index)
    check_equal(lcf,0);check_equal(lp,Fraction(1,2912))
    assert vp==1 and np.sum(b>0)==4
    checks["permutation_attribution_counterexample"]={
      "source_relation":"O=0.5*F; all source query ranks and ties identical",
      "enabled":[{"triple":list(e),"mF":float(mf[k]),"mO":float(mo[k]),"b":float(b[k]),"b_perm":float(perm[k])}
                 for k,e in enumerate(triples) if b[k]>0 or perm[k]>0],
      "F_feasible_both":True,"O_feasible_cf":True,"O_feasible_perm":False,
      "cf_loss_at_O":lcf,"perm_loss_at_O":lp,"perm_loss_exact":"1/2912",
      "outer_0_5_perm_loss_exact":"1/5824",
      "fixed_project_seed_outcome_claimed":False}
    print("PASS symmetric_28_account_permutation_attribution_counterexample")

    tr=inp["translation"]
    f=scores_from_labels(y,tr["positive"],tr["negative"])
    _,_,b=floors(f,f,triples)
    g=f.copy();q=tr["query"]
    for n in range(28):
        if n!=q:g[q,n]=g[n,q]=f[q,n]+tr["shift"]
    d,_,v=loss_gradient(g,b,triples,w,pairs,pair_index)
    expected=Fraction(331,9100)
    check_equal(d,expected)
    own=[k for k,e in enumerate(triples) if e[0]==q]
    assert np.array_equal(margins32(f,triples)[own],margins32(g,triples)[own])
    allshift=f+tr["shift"];np.fill_diagonal(allshift,0)
    dall,_,_=loss_gradient(allshift,b,triples,w,pairs,pair_index)
    check_equal(dall,0)
    checks["symmetric_query_shift"]={"query_own_margins_unchanged":True,
       "whole_group_loss":d,"whole_group_loss_exact":str(expected),
       "violated_comparisons":v,"all_edges_common_shift_loss":dall}
    print("PASS single_query_shift_is_not_global_symmetric_invariance")

    zc=inp["zero_loss_map"]
    f=scores_from_labels(y,zc["f_positive"],zc["f_negative"])
    o=f.copy();a,n=zc["disabled_negative"]
    o[a,n]=o[n,a]=zc["o_and_current_negative"]
    _,_,b=floors(f,o,triples)
    dg,_,vg=loss_gradient(o,b,triples,w,pairs,pair_index)
    mfmap,_=exact_map(f,y);mgmap,aps=exact_map(o,y)
    check_equal(dg,0)
    assert mfmap==1 and mgmap==Fraction(27,28) and vg==0
    assert np.sum(b>0)==1014
    checks["zero_history_loss_does_not_protect_map"]={"enabled":int(np.sum(b>0)),
       "disabled_triples":[list(e) for k,e in enumerate(triples) if b[k]==0],
       "loss_at_current":dg,"MAP_F_exact":str(mfmap),"MAP_current_exact":str(mgmap),
       "changed_query_AP_exact":{str(a):str(aps[a]),str(n):str(aps[n])},
       "scope":"Handwritten cached-score example; not a trained model or formal result."}
    print("PASS zero_new_term_and_lower_cached_MAP")

    f=scores_from_labels(y,1,0);_,_,b=floors(f,f,triples)
    g=f.copy();a,n=inp["positive_edge_drop"]["edge"];eps=inp["positive_edge_drop"]["amount"]
    g[a,n]=g[n,a]=f[a,n]-eps
    dc,_,_=loss_gradient(g,b,triples,w,pairs,pair_index)
    d0=float(np.mean([(g[i,j]-f[i,j])**2 for i,j in pairs]))
    check_equal(dc,eps*eps/14);check_equal(d0,eps*eps/378);check_equal(dc/d0,27)
    checks["same_outer_coefficient_different_strength"]={"epsilon":eps,
          "D_cf":dc,"D0":d0,"ratio":dc/d0,"exact_ratio":"27"}
    print("PASS penalty_strength_ratio_27_for_one_K2_positive_edge")

    count=inp["records_per_account"];slots=np.repeat(np.arange(28),count)
    rpairs=list(itertools.combinations(range(len(slots)),2))
    pool=np.zeros((len(pairs),len(rpairs)),dtype=np.float64)
    for k,(a,n) in enumerate(rpairs):
        if slots[a]!=slots[n]:
            pool[pair_index[tuple(sorted((int(slots[a]),int(slots[n]))))],k]=1/(count*count)
    g=scores_from_labels(y,inp["vjp"]["current_positive"],inp["vjp"]["current_negative"])
    table=np.asarray([0 if slots[a]==slots[n] else g[slots[a],slots[n]] for a,n in rpairs])
    vector=pool@table
    pooled=np.zeros((28,28),dtype=np.float64)
    for (a,n),val in zip(pairs,vector,strict=True):pooled[a,n]=pooled[n,a]=val
    dc,gscore,_=loss_gradient(pooled,b,triples,w,pairs,pair_index)
    gtable=pool.T@gscore*.5
    h=inp["vjp"]["finite_difference_h"]
    def objective(k):
        s=np.zeros((28,28),dtype=np.float64)
        for (a,n),v in zip(pairs,pool@k,strict=True):s[a,n]=s[n,a]=v
        return .5*loss_gradient(s,b,triples,w,pairs,pair_index)[0]
    directions=[]
    for pair in [(0,1),(0,2),(0,4)]:
        v=np.zeros(len(rpairs));v[rpairs.index(pair)]=1
        directions.append(("record_edge_"+str(pair),v))
    v=np.sin(np.arange(len(rpairs),dtype=np.float64)+.37);v/=np.linalg.norm(v)
    directions.append(("dense_deterministic_direction",v))
    checks_dir=[]
    for name,v in directions:
        fd=(objective(table+h*v)-objective(table-h*v))/(2*h)
        an=float(np.dot(gtable,v))
        assert abs(fd-an)<2e-9,(name,fd,an)
        checks_dir.append({"name":name,"finite_difference":float(fd),"analytic":an,"absolute_error":abs(fd-an)})
    same_account=[k for k,(a,n) in enumerate(rpairs) if slots[a]==slots[n]]
    assert np.all(gtable[same_account]==0)
    checks["account_pooling_VJP_reference"]={"record_count":len(slots),"table_values":len(rpairs),
       "loss_before_outer_coefficient":dc,"cross_account_derivative_factor":"1/(r_i*r_j)=1/4",
       "within_account_A0_only_gradient_zero":True,"directional_checks":checks_dir,
       "precision":"float64 calculus reference with fixed FP32 target; no head/BGE/Adam executed",
       "proves_native_encoder_connection":False}
    print("PASS account_pooling_chain_rule_only_no_native_BGE_claim")

    fp=inp["fp32"]
    p=np.float32(fp["f_positive"]);neg=np.float32(2.**fp["f_negative_power2"])
    m32=np.subtract(p,neg,dtype=np.float32);m64=float(p)-float(neg)
    bf=np.minimum(m32,np.float32(fp["o_margin"]))
    assert m32==np.float32(1) and m64<float(bf)
    shift=np.float32(fp["large_common_shift"])
    shifted=np.subtract(np.add(np.float32(1),shift,dtype=np.float32),
                        np.add(np.float32(0),shift,dtype=np.float32),dtype=np.float32)
    assert shifted==0
    tiny=np.float32(fp["tiny_residual"])
    tiny_square=np.multiply(tiny,tiny,dtype=np.float32)
    tiny_derivative=np.multiply(np.float32(2),tiny,dtype=np.float32)
    assert tiny_square==0 and tiny_derivative>0
    maximum=np.finfo(np.float32).max
    with np.errstate(over="ignore"):
        overflow=np.subtract(maximum,-maximum,dtype=np.float32)
    assert np.isfinite(maximum) and not np.isfinite(overflow)
    checks["FP32_semantics"]={"F_positive":float(p),"F_negative":float(neg),"margin32":float(m32),
        "margin64_after_promoting_scores":m64,"b":float(bf),
        "born_feasible_using_defined_margin32":bool(m32>=bf),
        "infeasible_after_changing_to_margin64":bool(m64<float(bf)),
        "large_common_shift":float(shift),"margin_after_FP32_shift":float(shifted),
        "tiny_residual":float(tiny),"FP32_squared_residual":float(tiny_square),
        "formal_square_derivative_in_FP32":float(tiny_derivative),
        "finite_source_subtraction_can_overflow":True,
        "note":"Loss rounding to zero is not a native-autograd test; zero/positive gates must retain specified semantics."}
    print("PASS FP32_rounding_precision_change_translation_and_finiteness")

    c=inp["work_counts"];main=c["main_stage_updates"];aux=c["aux_source_updates"]*c["aux_sources"]
    updates=main+aux+c["b_branches"]*main
    presentations=main+aux+c["b_branches"]*main*2
    oldsecs=c["previous_hours"]*3600+c["previous_minutes"]*60+c["previous_seconds"]
    assert updates==1440 and presentations==2304
    checks["work_and_payload_arithmetic"]={"physical_updates":updates,"gradient_group_presentations":presentations,
        "main_endpoints":1+c["b_branches"],"main_epoch_diagnostics":(1+c["b_branches"])*c["epochs"],
        "independent_S_updates":2*main,"independent_Ccf_updates":2*main+aux,
        "Ccf_extra_update_fraction_vs_S":aux/(2*main),
        "b_bytes_one_group":len(triples)*c["fp32_bytes"],
        "b_bytes_six_groups":len(triples)*c["fp32_bytes"]*c["retained_groups"],
        "minimum_S_table_values_one_group":math.comb(56,2),
        "minimum_S_table_bytes_one_group":math.comb(56,2)*4,
        "linear_training_volume_reference_hours_only":oldsecs*presentations/c["previous_group_presentations"]/3600,
        "runtime_or_output_budget_qualification":False}
    print("PASS work_counts_and_raw_target_bytes")
    out["status"]="PASS_FINITE_REFERENCE_ONLY"
    out["executed_utc"]=datetime.now(timezone.utc).isoformat()
    Path(args.output).write_text(json.dumps(out,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print("COMPLETED",len(checks),"reference sections; project training/qualification: not performed")

if __name__=="__main__":
    main()
