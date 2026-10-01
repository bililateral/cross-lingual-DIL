"""Isolated reviewer proposal: handcrafted data only, original submitted files untouched."""
from __future__ import annotations
import ast, copy, difflib, json, platform, sys, tempfile, time
from pathlib import Path
from unittest import mock
import numpy as np

ROOT = Path('/mnt/data/chinese_review')
OUT = Path('/mnt/data/chinese_external_audit')
sys.path[:0] = [str(ROOT/'scripts'), str(ROOT/'tests')]
import step28_chinese_base as b
import test_step28_chinese_base_contracts as fixture

src_path=ROOT/'scripts/step28_chinese_base.py'
source=src_path.read_text()
start=source.index('def evaluate(')
stop=source.index('\n\ndef main()',start)
replacement='''def fixed_zero_summary(counts: list[dict], domains: list[str]) -> dict:
    """Aggregate saved group counts; no labels and no new threshold selection."""
    values = np.asarray([[row[key] for key in ("tp", "fp", "fn", "tn")]
                         for row in counts], dtype=np.int64)
    domain_array = np.asarray(domains)
    return {"threshold": 0.0,
            "by_domain": {d: rates(values[domain_array == d]) for d in "ABC"},
            "pooled": rates(values)}


def evaluate(out: Path, destination: Path) -> dict:
    if platform.system() != "Linux" or destination.exists():
        raise ValueError("Requires resumed Linux evaluation and a new directory")
    c = contract()
    groups, metadata, checked = public.public_inputs(c)
    arrays, part, r = validated_run(out, c, groups, metadata, checked)
    reference = historical_reference(c, part)
    destination.mkdir(parents=True)
    data.write_json(destination / "access.json", {"status": "ALL_FOUR_MODELS_COMPLETE_BEFORE_VALID",
                    "development_parse_attempts": 1, "train": 0, "heldout": 0, "owners": 0})
    try:
        labelled = public.attach_labels(groups["development"], c, "development")
        truth = np.asarray([g.labels for g in labelled], dtype=np.uint8)
        e = c["evaluation"]
        domains = [row["domain"] for row in part["development"]]
        result = {"status": "CHINESE_BASE_DEVELOPMENT_EVALUATED_NO_AUTOMATIC_WINNER", "config": c,
                  "manifest_sha256": data.sha256(out / "manifest.json"), "columns": list(metrics.COLUMNS),
                  "label_parses": {"train": 0, "development": 1, "heldout": 0, "owners": 0},
                  "arms": {}, "comparisons": {}, "primary_comparison": c["primary_comparison"],
                  "historical_reference": reference["files"],
                  "formal_training_seconds": r["formal_training_seconds"]}
        matrices, auto_counts = {}, {}
        # Phase 1: collect the nine metric matrices and all group confusion counts.
        # Do not invoke bootstrap generation, reports or comparisons in this phase.
        for arm in ARMS:
            (destination / arm).mkdir()
            points = {}
            for epoch in ("3", "6"):
                scores = arrays[arm]["points"][epoch]["development"]
                matrix, counts = metrics.group_metrics(truth, scores)
                path = destination / arm / f"epoch{epoch}_metrics.npy"
                np.save(path, matrix, allow_pickle=False)
                points[epoch] = {"file": data.record(path, destination), "counts_at_logit_zero": counts,
                                 "mean": dict(zip(metrics.COLUMNS, matrix.mean(0).tolist())),
                                 "by_domain": {d: dict(zip(metrics.COLUMNS, matrix[np.asarray(domains) == d].mean(0).tolist())) for d in "ABC"}}
                if epoch == "6":
                    matrices[arm] = matrix
                    count = binary_counts(truth, scores, arrays[arm]["threshold"])
                    auto_counts[arm] = count
            result["arms"][arm] = {"points": points, "automatic_classification": {
                "threshold": arrays[arm]["threshold"], "counts_by_group": auto_counts[arm].tolist()}}
        matrices["historical_labse"], historical_counts = metrics.group_metrics(truth, reference["scores"])
        auto_counts["historical_labse"] = binary_counts(truth, reference["scores"], reference["threshold"])
        path = destination / "historical_labse_metrics.npy"
        np.save(path, matrices["historical_labse"], allow_pickle=False)
        result["historical_labse"] = {"file": data.record(path, destination),
            "counts_at_logit_zero": historical_counts,
            "threshold": reference["threshold"], "counts_by_group": auto_counts["historical_labse"].tolist()}
        data.write_json(destination / "collected.json", {
            **result, "status": "CHINESE_BASE_METRICS_COLLECTED_NO_BOOTSTRAP", "domains": domains})
        # Phase 2: every quantity below is recoverable from saved matrices/counts/config.
        draws = np.random.default_rng(e["bootstrap_seed"]).integers(0, 20, size=(e["bootstrap_replicates"], 3, 20))
        for arm in ARMS:
            result["arms"][arm]["automatic_classification"].update(automatic_report(auto_counts[arm], domains, draws))
            for point in result["arms"][arm]["points"].values():
                point["fixed_zero_classification"] = fixed_zero_summary(point["counts_at_logit_zero"], domains)
        result["historical_labse"]["automatic_classification"] = automatic_report(auto_counts["historical_labse"], domains, draws)
        result["historical_labse"]["fixed_zero_classification"] = fixed_zero_summary(historical_counts, domains)
        for candidate, baseline in c["comparisons"]:
            result["comparisons"][f"{candidate}_minus_{baseline}"] = {
                "metrics": metric_comparison(matrices[candidate], matrices[baseline], domains, draws),
                "automatic": automatic_comparison(auto_counts[candidate], auto_counts[baseline], domains, draws)}
        result["interpretation"] = "Each domain's fixed-threshold FPR<=0.001 AND recall>=0.5 is required. Point-value development qualification, not independent test, training-seed robustness, calibrated deployment guarantee, or automatic model selection. AP comparisons are secondary."
        data.write_json(destination / "evaluation.json", result)
        return result
    except Exception as error:
        data.write_json(destination / "failure.json", {"status": "EVALUATION_FAILED_NO_RETRY",
                        "error_type": type(error).__name__, "error": str(error)})
        raise
'''
proposed=source[:start]+replacement+source[stop:]
ast.parse(proposed)
(OUT/'suggested_evaluation_fix.diff').write_text(''.join(difflib.unified_diff(source.splitlines(True),proposed.splitlines(True),fromfile='a/scripts/step28_chinese_base.py',tofile='b/scripts/step28_chinese_base.py')))
(OUT/'suggested_evaluation_function.py').write_text('# Reviewer proposal; not merged into submitted source.\n'+replacement)
# Execute just the proposed functions in a separate namespace, using real helpers.
namespace=dict(vars(b)); exec(compile(replacement,'<isolated_reviewer_proposal>','exec'),namespace)
patched_evaluate=namespace['evaluate']; patched_zero=namespace['fixed_zero_summary']
started=time.monotonic()
result={'scope':'Isolated replacement functions on handmade fixtures; not applied to submission; no native weights, CUDA or formal inputs/labels',
        'original_source_unchanged_before':b.data.sha256(src_path)}
with tempfile.TemporaryDirectory() as td:
    td=Path(td); run=td/'job'/'run'
    c,groups,metadata,checked=fixture.fake_run(run)
    y=fixture.truth28()
    handmade=[b.data.Group(g.uid,g.sellers,g.items,tuple(map(int,y))) for g in groups['development']]
    reference={'scores':np.tile(np.where(y,2.,-2.).astype(np.float32),(60,1)), 'threshold':0.,'files':{'fixture':'not historical scores'}}
    with mock.patch.object(b.public,'public_inputs',return_value=(groups,metadata,checked)), \
         mock.patch.object(b.public,'attach_labels',return_value=handmade), \
         mock.patch.object(b,'historical_reference',return_value=reference):
        original=b.evaluate(run,td/'original')
    namespace['historical_reference']=lambda *_: reference
    with mock.patch.object(b.public,'public_inputs',return_value=(groups,metadata,checked)), \
         mock.patch.object(b.public,'attach_labels',return_value=handmade) as parse:
        new=patched_evaluate(run,td/'fixed')
        assert parse.call_count==1
    comparable=copy.deepcopy(new)
    for arm in b.ARMS:
        for point in comparable['arms'][arm]['points'].values(): point.pop('fixed_zero_classification')
    comparable['historical_labse'].pop('fixed_zero_classification')
    assert comparable==original,'Old successful result fields changed'
    result['successful_result_old_fields']='exact equality, including all seven comparisons'
    # Verify each new fixed-zero summary against direct summation of its saved counts.
    for arm in b.ARMS:
        for point in new['arms'][arm]['points'].values():
            z=np.array([[row[k] for k in ['tp','fp','fn','tn']] for row in point['counts_at_logit_zero']]); summary=point['fixed_zero_classification']
            assert summary['pooled']==b.rates(z)
            part=b.data.read_json(run/'partition.json')
            domain_array=np.array([r['domain'] for r in part['development']])
            for d in 'ABC': assert summary['by_domain'][d]==b.rates(z[domain_array==d])
    # Independent generic count ordering check, not relying on fixture metadata conventions.
    domains=['A']*20+['B']*20+['C']*20
    counts=np.array([[i%21,i%5,20-i%21,358-i%5] for i in range(60)])
    z=patched_zero([dict(zip(['tp','fp','fn','tn'],map(int,row))) for row in counts],domains)
    for k,d in enumerate('ABC'):
        summed=counts[20*k:20*(k+1)].sum(0)
        assert [z['by_domain'][d][n] for n in ['tp','fp','fn','tn']]==summed.tolist()
    assert [z['pooled'][n] for n in ['tp','fp','fn','tn']]==counts.sum(0).tolist()
    result['fixed_zero_counts']='domain/pooled sums agree'
    failure_dest=td/'failure'; order=[]
    def fail(*args,**kwargs):
        order.append(len(list(failure_dest.rglob('*.npy'))))
        raise RuntimeError('injected interval failure')
    namespace['automatic_report']=fail
    with mock.patch.object(b.public,'public_inputs',return_value=(groups,metadata,checked)), \
         mock.patch.object(b.public,'attach_labels',return_value=handmade) as parse:
        try: patched_evaluate(run,failure_dest)
        except RuntimeError as e: assert str(e)=='injected interval failure'
        else: raise AssertionError('Failure not raised')
        assert parse.call_count==1
    saved=json.loads((failure_dest/'collected.json').read_text())
    assert order==[9]
    assert len(list(failure_dest.rglob('*.npy')))==9
    assert saved['status']=='CHINESE_BASE_METRICS_COLLECTED_NO_BOOTSTRAP'
    assert len(saved['domains'])==60
    for arm in b.ARMS:
        assert np.shape(saved['arms'][arm]['automatic_classification']['counts_by_group'])==(60,4)
        for p in saved['arms'][arm]['points'].values():
            assert len(p['counts_at_logit_zero'])==60
            assert all(set(row)=={'tp','fp','fn','tn'} for row in p['counts_at_logit_zero'])
    assert np.shape(saved['historical_labse']['counts_by_group'])==(60,4)
    assert len(saved['historical_labse']['counts_at_logit_zero'])==60
    assert all(set(row)=={'tp','fp','fn','tn'} for row in saved['historical_labse']['counts_at_logit_zero'])
    result['failure_injection']={'matrices_before_first_interval':9,'collected_exists':True,
        'all_nine_fixed_zero_count_tables':True,'all_five_calibrated_count_tables':True,'handmade_parser_calls':1}
result['original_source_unchanged_after']=b.data.sha256(src_path)
assert result['original_source_unchanged_before']==result['original_source_unchanged_after']
result['seconds']=time.monotonic()-started
result['status']='ISOLATED_PROPOSAL_CHECKS_PASSED_NOT_PRODUCTION_APPROVAL'
(OUT/'suggested_fix_results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(result,ensure_ascii=False,indent=2))
