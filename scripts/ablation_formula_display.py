"""Calculate a declared hypothetical score-distribution example from saved metrics."""
import hashlib
import json
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path

import scipy
from scipy.integrate import quad
from scipy.special import expit, gammaln

ROOT = Path('/home/yongpeng/cross-lingual')
SOURCE = ROOT/'reports/documentation/20261107/h_ablation_prepare/workspace/formal/confirmation/summary.json'
OUTPUT = ROOT/'reports/documentation/20261116/h_ablation_result/presentation_formula.json'


def ranked(positives, negatives, rate):
    """Expected finite-ranking metrics for independent exponential edge clocks."""
    r, n = positives, negatives
    dp = [[0.0]*(n+1) for _ in range(r+1)]
    dp[0][0] = 1.0
    out = dict(average_precision=0.0, trapezoidal_pr_auc=0.0,
               recall_at_fpr_1pct=0.0, mrr=0.0)
    cuts = (1, 3, 5, 10)
    for cut in cuts:
        out[f'recall_at_{cut}'] = out[f'ndcg_at_{cut}'] = 0.0
    idcg = {k: sum(1/math.log2(j+1) for j in range(1, min(k, r)+1)) for k in cuts}
    for i in range(r+1):
        for j in range(n+1):
            prob = dp[i][j]
            denominator = (r-i)*rate+n-j
            if not prob or not denominator:
                continue
            position = i+j+1
            positive = (r-i)*rate/denominator
            if i < r:
                mass = prob*positive
                dp[i+1][j] += mass
                after = (i+1)/position
                before = i/(position-1) if position > 1 else 1.0
                out['average_precision'] += mass*after/r
                out['trapezoidal_pr_auc'] += mass*(before+after)/(2*r)
                if j <= math.floor(.01*n):
                    out['recall_at_fpr_1pct'] += mass/r
                if i == 0:
                    out['mrr'] += mass/position
                for cut in cuts:
                    if position <= cut:
                        out[f'recall_at_{cut}'] += mass/r
                        out[f'ndcg_at_{cut}'] += mass/math.log2(position+1)/idcg[cut]
            if j < n:
                dp[i][j+1] += prob*(1-positive)
    return out


def query_metrics(rate):
    one, two = ranked(1, 26, rate), ranked(2, 25, rate)
    return {key: (16*one[key]+12*two[key])/28 for key in one}


def expected_loss(rate, slope, offset, positive, logarithmic):
    def integrand(u):
        logit = slope*(offset-u/rate)
        if logarithmic:
            z = -logit if positive else logit
            value = max(z, 0)+math.log1p(math.exp(-abs(z)))
        else:
            probability = float(expit(logit))
            value = ((1-probability) if positive else probability)**2
        return math.exp(-u)*value
    return quad(integrand, 0, math.inf, epsabs=1e-12, epsrel=1e-11)[0]


def binomial(n, probability):
    return [math.exp(float(gammaln(n+1)-gammaln(k+1)-gammaln(n-k+1))
                     +k*math.log(probability)+(n-k)*math.log1p(-probability))
            for k in range(n+1)]


def main():
    saved = json.loads(SOURCE.read_text())
    columns = saved['metric_columns']
    index = {name: i for i, name in enumerate(columns)}
    full = saved['absolute']['Full']
    original = full['raw']['final_mean'][index['map']]
    target = original+.02
    low, high = 1e-8, 1e8
    for _ in range(85):
        middle = math.sqrt(low*high)
        if query_metrics(middle)['average_precision'] < target:
            low = middle
        else:
            high = middle
    rate = math.sqrt(low*high)
    query, pair = query_metrics(rate), ranked(20, 358, rate)
    fpr = 1-full['stage-cal']['final_mean'][index['specificity']]
    slope, offset = 10.0, -math.log1p(-fpr)
    tpr = -math.expm1(-rate*offset)
    tp, fp = binomial(20, tpr), binomial(358, fpr)
    f1 = sum(p*q*2*i/(20+i+j) for i, p in enumerate(tp) for j, q in enumerate(fp))
    values = dict(average_precision=pair['average_precision'],
                  trapezoidal_pr_auc=pair['trapezoidal_pr_auc'], roc_auc=rate/(1+rate),
                  recall_at_fpr_1pct=pair['recall_at_fpr_1pct'], precision=None,
                  recall=tpr, f1=f1, specificity=1-fpr,
                  balanced_accuracy=(tpr+1-fpr)/2, mcc=None,
                  map=query['average_precision'], mrr=query['mrr'])
    for name, logarithmic in [('brier', False), ('log_loss', True)]:
        values[name] = (20*expected_loss(rate, slope, offset, True, logarithmic)
                        +358*expected_loss(1, slope, offset, False, logarithmic))/378
    for cut in (1, 3, 5, 10):
        for name in ('recall_at_', 'ndcg_at_'):
            values[name+str(cut)] = query[name+str(cut)]
    harmonic = sum(1/k for k in range(1, 28))
    random_ap = (16*harmonic/27+12*(harmonic/27+(27-harmonic)/(27*26)))/28
    assert abs(query_metrics(1)['average_precision']-random_ap) < 1e-12
    assert abs(values['map']-target) < 1e-12
    assert set(values) == set(columns)
    assert values['recall_at_1'] <= 11/14+1e-12
    assert all(x is None or math.isfinite(x) and 0 <= x <= 1 for x in values.values())
    arms = ['Full', 'ER', 'KD', 'SEQ', 'NoRegroup', 'MeanPool', 'StandardRank', 'AccountPool']
    probability_metrics = {'brier', 'log_loss', 'precision', 'recall', 'f1',
                           'specificity', 'balanced_accuracy', 'mcc'}
    actual = {arm: {key: saved['absolute'][arm][
        'stage-cal' if key in probability_metrics else 'raw']['final_mean'][index[key]]
        for key in columns} for arm in arms}
    fields = ('A_acq', 'O', 'F', 'N', 'Z', 'F_first', 'final_mean')
    components = {arm: {key: saved['absolute'][arm]['raw'][key][index['map']]
                        for key in fields} for arm in arms}
    early = components['Full']
    r22 = 2*early['N']-early['Z']
    r11 = 2*early['A_acq']-r22
    example = dict(A_acq=early['A_acq'], O=target, F=early['A_acq']-target,
                   N=(r22+target)/2, Z=target, F_first=r11-target, final_mean=target)
    assert abs(example['A_acq']-example['O']-example['F']) < 1e-14
    assert abs((2*example['O']+example['Z'])/3-example['final_mean']) < 1e-14
    result = dict(created_at=datetime.now(timezone(timedelta(hours=8))).isoformat(),
        purpose='Declared formula example, not new experiment results or a revised scientific gate',
        source=dict(path=str(SOURCE), bytes=SOURCE.stat().st_size,
                    sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest()),
        environment=dict(python='/home/yongpeng/miniconda3/envs/py310/bin/python', scipy=scipy.__version__),
        metric_columns=columns, arms=arms, saved_final_values=actual,
        full_formula_values=values, saved_map_components=components,
        full_formula_map_components=example,
        parameters=dict(original_map=original, target_map=target, delta=.02,
                        positive_rate=rate, negative_rate=1, score_slope=slope,
                        score_offset=offset, preserved_false_positive_rate=fpr),
        assumptions=dict(group='28 accounts, 20 positive/358 negative unordered pairs; 16 queries with one positive and 12 with two',
            scores='Independent edge clocks T_positive~Exp(w), T_negative~Exp(1); p=sigmoid(10*(b-T)); larger p ranks first',
            ranking='Exact expectation over finite possible rankings using remaining positive/negative exponential rates',
            classification='p>=0.5; TP~Binom(20,1-exp(-w*b)), FP~Binom(358,1-exp(-b)); F1 averaged over the joint count distribution',
            losses='Expected pair-level Brier/log-loss integrated over the same score distributions with 20:358 class weights',
            undefined='Precision/MCC remain N/A: zero-denominator groups have positive probability',
            temporal='Earlier acquisition endpoints fixed; each final domain has the formula score distribution; derive F/N/Z/F_first from endpoint identities',
            scope='Hypothetical distribution expectations, not a finite sample, learned model, confidence interval, or values uniquely implied by MAP; original experimental verdict unchanged'),
        checks=dict(map_target_error=values['map']-target,
                    random_rank_ap_error=query_metrics(1)['average_precision']-random_ap,
                    model_loading=False, label_reading=False, rescoring=False, resampling=False))
    with OUTPUT.open('x') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps(dict(path=str(OUTPUT), bytes=OUTPUT.stat().st_size,
                          sha256=hashlib.sha256(OUTPUT.read_bytes()).hexdigest(),
                          parameters=result['parameters'], values=values), ensure_ascii=False))


if __name__ == '__main__':
    main()
