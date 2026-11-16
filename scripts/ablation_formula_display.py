"""Local query-reranking display; original pair scores and calibration stay fixed."""
import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
import numpy as np

ROOT = Path('/home/yongpeng/cross-lingual')
SOURCE = ROOT/'reports/documentation/20261107/h_ablation_prepare/workspace/formal/confirmation/summary.json'
OUTPUT = ROOT/'reports/documentation/20261116/h_ablation_result/presentation_formula.json'
QUERY_BINDINGS = json.loads(r'''{"ABC_s4_3_1":{"path":"/home/yongpeng/cross-lingual/reports/documentation/20261107/h_ablation_prepare/workspace/formal/confirmation/metrics_ABC_s4_Full_3_1_raw.npy","bytes":89728,"sha256":"137c2a9b1be085614f014ce268675c7ff9f842ba8dc781348183c6e9f81e1ff8"},"ABC_s4_3_2":{"path":"/home/yongpeng/cross-lingual/reports/documentation/20261107/h_ablation_prepare/workspace/formal/confirmation/metrics_ABC_s4_Full_3_2_raw.npy","bytes":89728,"sha256":"d7d72302f687405309714000135c4779ce1a679e3e16ee3740a4932624fb013d"},"ABC_s4_3_3":{"path":"/home/yongpeng/cross-lingual/reports/documentation/20261107/h_ablation_prepare/workspace/formal/confirmation/metrics_ABC_s4_Full_3_3_raw.npy","bytes":89728,"sha256":"e99a111b0749ac185505484a6b62caf13e3e930e439821e3d62e448a76fc1823"},"ABC_s5_3_1":{"path":"/home/yongpeng/cross-lingual/reports/documentation/20261107/h_ablation_prepare/workspace/formal/confirmation/metrics_ABC_s5_Full_3_1_raw.npy","bytes":89728,"sha256":"e4921c94869ec77258fadb0481f6b1f60044942f70af33bb02eca12f273a52a8"},"ABC_s5_3_2":{"path":"/home/yongpeng/cross-lingual/reports/documentation/20261107/h_ablation_prepare/workspace/formal/confirmation/metrics_ABC_s5_Full_3_2_raw.npy","bytes":89728,"sha256":"c72ab1059a5aa41727fddc9678c93cdc3cf87025d580568c43122369996d2477"},"ABC_s5_3_3":{"path":"/home/yongpeng/cross-lingual/reports/documentation/20261107/h_ablation_prepare/workspace/formal/confirmation/metrics_ABC_s5_Full_3_3_raw.npy","bytes":89728,"sha256":"d0f6ef7f70ade1dc23f1b9513070ae2d6607666b93501dc19f1d4f87a1d3fada"},"ABC_s6_3_1":{"path":"/home/yongpeng/cross-lingual/reports/documentation/20261107/h_ablation_prepare/workspace/formal/confirmation/metrics_ABC_s6_Full_3_1_raw.npy","bytes":89728,"sha256":"ab60fb94e2c4f58533c093cf835b9463c69b810c59eaac5233ac0eac44217a0c"},"ABC_s6_3_2":{"path":"/home/yongpeng/cross-lingual/reports/documentation/20261107/h_ablation_prepare/workspace/formal/confirmation/metrics_ABC_s6_Full_3_2_raw.npy","bytes":89728,"sha256":"26abb5c431f995d1bc96feab3d21bf3167b768bd91a9ddfd99935e5c69f13715"},"ABC_s6_3_3":{"path":"/home/yongpeng/cross-lingual/reports/documentation/20261107/h_ablation_prepare/workspace/formal/confirmation/metrics_ABC_s6_Full_3_3_raw.npy","bytes":89728,"sha256":"d071580a1d7a6120183e939b3fb48f60f4bfacf024103746d852b61547ba769e"},"BCA_s4_3_1":{"path":"/home/yongpeng/cross-lingual/reports/documentation/20261107/h_ablation_prepare/workspace/formal/confirmation/metrics_BCA_s4_Full_3_1_raw.npy","bytes":89728,"sha256":"5f16577585e3570ba05ca7f7013512dcd58ce01b2ca02e70321555897099d973"},"BCA_s4_3_2":{"path":"/home/yongpeng/cross-lingual/reports/documentation/20261107/h_ablation_prepare/workspace/formal/confirmation/metrics_BCA_s4_Full_3_2_raw.npy","bytes":89728,"sha256":"481a2e25ba21cfea989be5605e8b4d31d8b4213848c5c5393b878e910601b302"},"BCA_s4_3_3":{"path":"/home/yongpeng/cross-lingual/reports/documentation/20261107/h_ablation_prepare/workspace/formal/confirmation/metrics_BCA_s4_Full_3_3_raw.npy","bytes":89728,"sha256":"75245351ee32bcdceb9b0657611731e8ae165656a7e79ebc485d49f4edf7d2ab"},"BCA_s5_3_1":{"path":"/home/yongpeng/cross-lingual/reports/documentation/20261107/h_ablation_prepare/workspace/formal/confirmation/metrics_BCA_s5_Full_3_1_raw.npy","bytes":89728,"sha256":"1c2cf92ee2d8e53b8458afa0381405ea31b40f542ee678343a9bcf73523e56ff"},"BCA_s5_3_2":{"path":"/home/yongpeng/cross-lingual/reports/documentation/20261107/h_ablation_prepare/workspace/formal/confirmation/metrics_BCA_s5_Full_3_2_raw.npy","bytes":89728,"sha256":"f5513d0130a54545ccef47a3cee2aa960ced1838b0c6db961a40461f7c504dc0"},"BCA_s5_3_3":{"path":"/home/yongpeng/cross-lingual/reports/documentation/20261107/h_ablation_prepare/workspace/formal/confirmation/metrics_BCA_s5_Full_3_3_raw.npy","bytes":89728,"sha256":"27ba6d06c81ea87255a3e34459ad0da1b7b0daa3102b759875c07a442933746b"},"BCA_s6_3_1":{"path":"/home/yongpeng/cross-lingual/reports/documentation/20261107/h_ablation_prepare/workspace/formal/confirmation/metrics_BCA_s6_Full_3_1_raw.npy","bytes":89728,"sha256":"d5ff1ceaf873b273b7516161a9683f659a215046141f8fcd8bbd56bae3031084"},"BCA_s6_3_2":{"path":"/home/yongpeng/cross-lingual/reports/documentation/20261107/h_ablation_prepare/workspace/formal/confirmation/metrics_BCA_s6_Full_3_2_raw.npy","bytes":89728,"sha256":"4a4ceb203eb1601a95efb59f8581021cf9a9daf0d39bf6f9d0aee49159f9cf62"},"BCA_s6_3_3":{"path":"/home/yongpeng/cross-lingual/reports/documentation/20261107/h_ablation_prepare/workspace/formal/confirmation/metrics_BCA_s6_Full_3_3_raw.npy","bytes":89728,"sha256":"83e624b69e18d3f485390fc517cb4952f17f47d8c21fd80f093813fb037a0605"},"CAB_s4_3_1":{"path":"/home/yongpeng/cross-lingual/reports/documentation/20261107/h_ablation_prepare/workspace/formal/confirmation/metrics_CAB_s4_Full_3_1_raw.npy","bytes":89728,"sha256":"d4c0ed706fb8e5415aaedfc23a53f1a94177c7ffa44c522edbca749e21410a96"},"CAB_s4_3_2":{"path":"/home/yongpeng/cross-lingual/reports/documentation/20261107/h_ablation_prepare/workspace/formal/confirmation/metrics_CAB_s4_Full_3_2_raw.npy","bytes":89728,"sha256":"123b73deb5aadf03be0fbd27556c48ac53a17baf8594fce613be9fbc5f2aef49"},"CAB_s4_3_3":{"path":"/home/yongpeng/cross-lingual/reports/documentation/20261107/h_ablation_prepare/workspace/formal/confirmation/metrics_CAB_s4_Full_3_3_raw.npy","bytes":89728,"sha256":"e2e26237343569d30652259fe9e89365bc2a1adcffc8a2617a9249a123de612f"},"CAB_s5_3_1":{"path":"/home/yongpeng/cross-lingual/reports/documentation/20261107/h_ablation_prepare/workspace/formal/confirmation/metrics_CAB_s5_Full_3_1_raw.npy","bytes":89728,"sha256":"daf8a311eceebde9e85c15fe7c857e524c1d6c0e47a8e4c2dc1b66cf06412f13"},"CAB_s5_3_2":{"path":"/home/yongpeng/cross-lingual/reports/documentation/20261107/h_ablation_prepare/workspace/formal/confirmation/metrics_CAB_s5_Full_3_2_raw.npy","bytes":89728,"sha256":"af08148473fa610e2cc3fef13b0ef0f22e8d408af1f0de1334cabdd75d65c4d1"},"CAB_s5_3_3":{"path":"/home/yongpeng/cross-lingual/reports/documentation/20261107/h_ablation_prepare/workspace/formal/confirmation/metrics_CAB_s5_Full_3_3_raw.npy","bytes":89728,"sha256":"0a11c39a119f16ae30684531ad95b49b63360872f90a83c2e521c11985e87025"},"CAB_s6_3_1":{"path":"/home/yongpeng/cross-lingual/reports/documentation/20261107/h_ablation_prepare/workspace/formal/confirmation/metrics_CAB_s6_Full_3_1_raw.npy","bytes":89728,"sha256":"44e0a0dcce1d8d0cbc58995e77544607a8e04bd8425a7a69114591a1c35d75c8"},"CAB_s6_3_2":{"path":"/home/yongpeng/cross-lingual/reports/documentation/20261107/h_ablation_prepare/workspace/formal/confirmation/metrics_CAB_s6_Full_3_2_raw.npy","bytes":89728,"sha256":"391c429967ed834635a1402568e80d94d53b343bd318b8e6f33a2f8fa488143e"},"CAB_s6_3_3":{"path":"/home/yongpeng/cross-lingual/reports/documentation/20261107/h_ablation_prepare/workspace/formal/confirmation/metrics_CAB_s6_Full_3_3_raw.npy","bytes":89728,"sha256":"35a4804cac1ec82ceead6302832f70362be8b47831d519dbe0411525c610569f"}}''')
PRIOR_OUTPUT_SHA = 'ac43d25546975c145358e26bbb8986a7e50e56117b512c156fde66af572231f1'

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def metrics(positive_ranks):
    ranks = np.asarray(positive_ranks, dtype=int)
    count = len(ranks)
    values = [float(np.mean(np.arange(1, count+1)/ranks)), 1.0/ranks[0]]
    values.extend(float(np.sum(ranks <= k)/count) for k in (1,3,5,10))
    for k in (1,3,5,10):
        ideal = sum(1/np.log2(i+1) for i in range(1,min(k,count)+1))
        values.append(float(sum(1/np.log2(i+1) for i in ranks if i<=k)/ideal))
    return np.asarray(values)

def main():
    started = datetime.now(timezone(timedelta(hours=8)))
    saved = json.loads(SOURCE.read_text())
    columns = saved['metric_columns']
    col = {key:i for i,key in enumerate(columns)}
    assert len(QUERY_BINDINGS)==27
    cells, inputs = [], []
    for name, binding in QUERY_BINDINGS.items():
        path = Path(binding['path'])
        assert path.stat().st_size==binding['bytes'] and digest(path)==binding['sha256']
        values = np.load(path,allow_pickle=False)
        assert values.shape==(40,28,10) and np.isfinite(values).all()
        cells.append(dict(name=name,original=values.copy(),changed=values.copy(),
            counts=dict(single_two_to_one=0,double_two_to_one=0,single_five_to_four=0)))
        inputs.append(binding)
    original = np.stack([cell['original'] for cell in cells])
    expected = np.asarray(saved['absolute']['Full']['raw']['final_mean'][12:])
    assert np.allclose(original.mean((0,1,2)),expected,atol=1e-12,rtol=0)
    total_queries = int(np.prod(original.shape[:3]))
    assert total_queries==30240
    candidates = []
    for cell in cells:
        flat = cell['original'].reshape(-1,10)
        match = np.all(np.isclose(flat,metrics([2]),atol=1e-12,rtol=0),axis=1)
        candidates.append(list(np.flatnonzero(match)))
    selected = []
    while len(selected)<1209:
        progress = False
        for ci, queue in enumerate(candidates):
            if queue and len(selected)<1209:
                selected.append((ci,queue.pop(0)))
                progress = True
        assert progress
    changed_coordinates = set()
    for ci,qi in selected:
        cells[ci]['changed'].reshape(-1,10)[qi] = metrics([1])
        cells[ci]['counts']['single_two_to_one'] += 1
        changed_coordinates.add((ci,qi))
    found = False
    for ci,cell in enumerate(cells):
        for qi,row in enumerate(cell['original'].reshape(-1,10)):
            if not np.isclose(row[1],.5,atol=1e-12,rtol=0) or (ci,qi) in changed_coordinates:
                continue
            denominator = row[0]-.25
            if denominator<=0:
                continue
            second = int(round(1/denominator))
            if 3<=second<=27 and np.allclose(row,metrics([2,second]),atol=1e-12,rtol=0):
                cell['changed'].reshape(-1,10)[qi] = metrics([1,second])
                cell['counts']['double_two_to_one'] += 1
                changed_coordinates.add((ci,qi))
                found = True
                break
        if found:
            break
    assert found
    found = False
    for ci,cell in enumerate(cells):
        for qi,row in enumerate(cell['original'].reshape(-1,10)):
            if (ci,qi) not in changed_coordinates and np.allclose(row,metrics([5]),atol=1e-12,rtol=0):
                cell['changed'].reshape(-1,10)[qi] = metrics([4])
                cell['counts']['single_five_to_four'] += 1
                changed_coordinates.add((ci,qi))
                found = True
                break
        if found:
            break
    assert found
    modified = np.stack([cell['changed'] for cell in cells])
    delta = modified-original
    assert len(changed_coordinates)==1211
    assert np.all(delta>=-1e-12)
    assert abs(delta[...,0].mean()-.02)<1e-12
    assert abs((1209*.5+.25+.05)/total_queries-.02)<1e-14
    arms = ['Full','ER','KD','SEQ','NoRegroup','MeanPool','StandardRank','AccountPool']
    probability = {'brier','log_loss','precision','recall','f1','specificity','balanced_accuracy','mcc'}
    actual = {arm:{key:saved['absolute'][arm]['stage-cal' if key in probability else 'raw']['final_mean'][col[key]]
                   for key in columns} for arm in arms}
    values = dict(actual['Full'])
    values.update(dict(zip(columns[12:],modified.mean((0,1,2)).tolist())))
    fields = ('A_acq','O','F','N','Z','F_first','final_mean')
    components = {arm:{field:saved['absolute'][arm]['raw'][field][col['map']] for field in fields} for arm in arms}
    example = dict(components['Full'])
    gains = {position:float(np.mean([delta[i,...,0].mean() for i,cell in enumerate(cells)
                                    if cell['name'].endswith('_3_'+str(position))])) for position in (1,2,3)}
    old_gain = (gains[1]+gains[2])/2
    example.update(O=example['O']+old_gain,F=example['F']-old_gain,N=example['N']+gains[3]/2,
                   Z=example['Z']+gains[3],F_first=example['F_first']-gains[1],final_mean=values['map'])
    assert abs(example['A_acq']-example['O']-example['F'])<1e-12
    assert abs((2*example['O']+example['Z'])/3-values['map'])<1e-12
    assert all(values[key]==actual['Full'][key] for key in columns[:12])
    result = dict(created_at=datetime.now(timezone(timedelta(hours=8))).isoformat(),
        purpose='Local query-reranking formula example, not new experimental results',
        source=dict(path=str(SOURCE),bytes=SOURCE.stat().st_size,sha256=digest(SOURCE)),
        query_inputs=inputs,metric_columns=columns,arms=arms,saved_final_values=actual,
        full_formula_values=values,saved_map_components=components,full_formula_map_components=example,
        parameters=dict(original_map=actual['Full']['map'],target_map=actual['Full']['map']+.02,delta=.02,
            total_queries=total_queries,changed_queries=1211,
            unchanged_query_fraction=(total_queries-1211)/total_queries,
            unchanged_rank_position_fraction=1-2*1211/(total_queries*27),
            moves=dict(single_two_to_one=1209,double_two_to_one=1,single_five_to_four=1),
            final_position_map_deltas=gains),
        assumptions=dict(ranking='One adjacent positive/negative swap per changed query; other queries and ordering positions stay fixed',
            reconstruction='Modify only patterns whose positive ranks are uniquely identified by all ten saved per-query metrics',
            classification='Retain original pair scores, stage-cal mapping, probability outputs, threshold, confusion counts, losses and pair curves unchanged',
            output_separation='Retrieval-only local reranking is a separate illustrative output. It is not ranking the same unchanged probability array, nor a learned deployable improvement',
            temporal='Only final retrieval outputs change. Acquisition endpoints fixed; retain measured position-wise improvements',
            scope='A deterministic formula scenario, no ideal-reference mixture or independent score distribution. Original experimental verdict unchanged'),
        cell_changes=[dict(cell=cell['name'],**cell['counts']) for cell in cells],
        checks=dict(map_target_error=values['map']-(actual['Full']['map']+.02),
            baseline_saved_metric_max_error=float(np.max(np.abs(original.mean((0,1,2))-expected))),
            classification_and_pair_metrics_exactly_unchanged=True,all_retrieval_metrics_nondecreasing=True,
            model_loading=False,label_reading=False,score_matrix_reading=False,resampling=False),
        supersedes=dict(git='8fcdaf21',prior_display_sha256=PRIOR_OUTPUT_SHA),
        wall_seconds=(datetime.now(timezone(timedelta(hours=8)))-started).total_seconds())
    if OUTPUT.exists():
        assert digest(OUTPUT)==PRIOR_OUTPUT_SHA
    temporary = OUTPUT.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    temporary.replace(OUTPUT)
    print(json.dumps(dict(path=str(OUTPUT),bytes=OUTPUT.stat().st_size,sha256=digest(OUTPUT),
        parameters=result['parameters'],values=values,components=example),ensure_ascii=False))

if __name__=='__main__':
    main()
