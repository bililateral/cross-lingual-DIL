"""Design arithmetic and analytic counterexamples only.
No project imports, scientific input, labels, models, fitting, or training.
Source clauses: docs/SELLER_ALIAS_BGE_CONTINUAL.zh.md, stated line ranges.
"""
from fractions import Fraction as F
from collections import Counter
from pathlib import Path
import json, math, platform, sys


def schedule(total, warm):
    peak=F(1,100000)
    return [peak*F(i,warm) if i<=warm else peak*F(total-i,total-warm)
            for i in range(1,total+1)]


def bernoulli_loss(q,p):
    return -q*math.log(p)-(1-q)*math.log1p(-p)


def logit(p):
    return math.log(p)-math.log1p(-p)


def loss_at_bins(true_probs, predicted_probs):
    return {
      'expected_brier':sum(q*(1-p)**2+(1-q)*p**2 for q,p in zip(true_probs,predicted_probs))/2,
      'expected_nll':sum(bernoulli_loss(q,p) for q,p in zip(true_probs,predicted_probs))/2}

out={'scope':'Arithmetic/analytic illustrations; NOT project experiments or an implementation validation.',
     'python':platform.python_version(),'executable':sys.executable}
orders=('ABC','BCA','CAB')
transitions=Counter(a+b for o in orders for a,b in zip(o,o[1:]))
counts={'groups_fit_per_domain':48,'epochs':6,'stages':3,'orders':3,'updated_arms':3,
        'updates_per_stage':288,'shared_updates':3*288,'later_updates':3*3*2*288,
        'physical_optimizer_calls':3*288+3*3*2*288,'history_presentations':2*3*2*288,
        'all_gradient_group_presentations':3*288+3*3*2*288+2*3*2*288,
        'unique_stage_endpoints_and_maps':3+3*3*2,
        'current_calibration_pairs':12*378,'current_calibration_positives':12*20,
        'reservoir_accounts':6*28,'reservoir_pairs':6*378,'reservoir_positives':6*20,
        'reference_scores_float32_bytes':6*378*4,'first_map_float64_scalar_bytes':2*8}
out['counts']={'source':'draft L56-L68,L94,L102,L193-L204','values':counts}
short=schedule(288,29); long=schedule(864,87)
assert sum(short)*3==sum(long)
out['schedule']={'source':'draft L56-L58; RANKING L21',
    'phase_sum':float(sum(short)),'three_phase_sum':float(sum(short)*3),
    'old_864_sum':float(sum(long)), 'exact_sums_equal':True,
    'steps_1_29_30_287_288':[float(short[i-1]) for i in [1,29,30,287,288]],
    'positive_lr_steps_per_path':3*sum(x>0 for x in short),
    'old_positive_lr_steps':sum(x>0 for x in long),
    'physical_positive_lr_calls':21*sum(x>0 for x in short),
    'interpretation':'Equal sum does not imply same optimization trajectory or parameter displacement.'}
p0=F(math.comb(48,6),math.comb(96,6))
out['reservoir_analytic']={'source':'draft L62-L68; uniform six-of-96 assumption for Algorithm R',
    'historical_fit_fraction_stage2':6/48,'historical_fit_fraction_stage3':6/96,
    'expected_history_presentations_per_retained_group_in_one_stage':288/6,
    'current_presentations_per_group':6,'expected_repeat_ratio':8,
    'probability_named_domain_absent':float(p0),
    'probability_one_of_two_domains_absent':float(2*p0),
    'expected_old_domain_count':3,
    'fraction_of_1MiB_reference_numerics':9072/(1024**2),
    'not_claimed':'These are distributional calculations, not actual chosen groups or memory-byte measurements.'}
out['orders']={'source':'draft L35','transitions':dict(transitions),
    'missing_directed_transitions':['AC','CB','BA'],
    'position_balanced':True,'directed_transition_balanced':False}
# Exact prior 20/378 shared by both analytic two-bin distributions.
prior=20/378
qold=[0.01,2*prior-0.01]; qnew=[0.04,2*prior-0.04]
oldmap=[(logit(qold[1])-logit(qold[0]))/2,(logit(qold[0])+logit(qold[1]))/2]
newmap=[(logit(qnew[1])-logit(qnew[0]))/2,(logit(qnew[0])+logit(qnew[1]))/2]
lold=loss_at_bins(qold,qold); lnewonold=loss_at_bins(qold,qnew)
out['calibration_counterexample']={
    'source':'reviewer analytic example addressing draft L102-L108; two equally likely score bins -1,+1',
    'not_data':'Not the project generator, not sampled labels, not actual metrics.',
    'same_positive_prior':prior,'old_true_probabilities':qold,'new_true_probabilities':qnew,
    'old_optimal_positive_affine_map':oldmap,'new_optimal_positive_affine_map':newmap,
    'old_loss_using_old_map':lold,'old_loss_using_new_map':lnewonold,
    'old_brier_increase':lnewonold['expected_brier']-lold['expected_brier'],
    'old_nll_increase':lnewonold['expected_nll']-lold['expected_nll'],
    'score_change':0,'rank_change':0,
    'claim':'Current-domain-perfect calibration may worsen old probability loss without model forgetting, despite equal class priors.'}
# Purely specified scalar performance tables, not generated project samples.
r={'comparator':{'O':.5,'R22':.4,'R33':.6},'candidate':{'O':.55,'R22':.75,'R33':.3}}
for v in r.values():
    v['N']=(v['R22']+v['R33'])/2; v['final_all']=(2*v['O']+v['R33'])/3
out['endpoint_scope_counterexample']={'source':'draft L118-L123,L147-L150',
    'not_data':'Illustrative performance entries only; not observations.',
    'values':r,'claim':'O and N can increase while final newest-domain and final overall MAP decrease.'}
out['path_not_transition_counterexample']={'source':'draft L137',
    'not_data':'Illustrative scalar entries, no generated input or labels.',
    'first_domain_MAP_stages':[.6,.5,.5],'new_domain_gains_stages_2_3':[-.1,.3],
    'first_loss':.1,'average_new_gain':.1,
    'claim':'Same-order positive final forgetting and average new learning need not co-occur in a single transition.'}
seconds=14*3600+25*60+21
out['resource_arithmetic']={'source':'draft L193-L208; old elapsed is background report, not remeasured',
    'linear_group_presentation_extrapolation_seconds':seconds*9504/7776,
    'linear_extrapolation_hours':seconds*9504/7776/3600,
    'twenty_one_model_bytes_if_old_average_file_size_applies':23516221872*21//18,
    'is_formal_runtime_or_storage_bound':False,
    'later_4_arms_3_seeds_3_orders_calls_if_864_per_arm_and_shared_first':9*(1+4*2)*288,
    'net_new_calls_if_first_6048_all_reusable':9*(1+4*2)*288-6048,
    'baseline_BGE_4_arms_3_seeds_ABC_calls_if_same_288_per_stage':3*(1+4*2)*288,
    'not_included':'Hyperparameter development, additional budgets, ablations, holdout generation or evaluation; candidate schedule not yet agreed.'}
root=Path(__file__).resolve().parents[1]
(root/'outputs'/'derivations.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(out,ensure_ascii=False,indent=2))
