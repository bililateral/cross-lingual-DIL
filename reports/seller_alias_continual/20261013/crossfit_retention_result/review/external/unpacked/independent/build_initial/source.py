#!/usr/bin/env python3
"""Assemble the report indexes and figures from already verified saved results."""
from pathlib import Path
import collections, csv, datetime, hashlib, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE=Path(__file__).resolve().parents[1]
INPUT=BASE/'input'
RESULT=BASE/'independent/attempts/run03/output'

def dump(path,value):
    (BASE/path).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def read(path):return json.loads((BASE/path).read_text())
def sha(data):return hashlib.sha256(data).hexdigest()

def main():
    summary=read('independent/attempts/run03/output/result_summary.json')
    diagnostics=read('independent/attempts/run03/output/diagnostics_recomputed.json')
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    colors={'S':'#374f72','C_cf':'#c65136','C_perm':'#197f78'}
    for i,d in enumerate('AB'):
        ax=axes[0,i]
        for j,contrast in enumerate(['C_cf-S','C_cf-C_perm']):
            r=next(r for r in summary['primary'] if r['kind']=='contrast' and r['point_or_contrast']==contrast and r['domain']==d)
            ax.errorbar(r['mean'],1-j,xerr=[[r['mean']-r['lower_95']],[r['upper_95']-r['mean']]],
                        color=colors['C_cf' if j==0 else 'C_perm'],fmt='o',markersize=7,capsize=5,elinewidth=2)
            ax.annotate(f"{r['mean']:+.6f}",(r['mean'],1-j),xytext=(0,14),textcoords='offset points',ha='center',fontsize=10)
        ax.axvline(0,color='#6b7280',linestyle='--',linewidth=1)
        ax.set(yticks=[1,0],yticklabels=['C_cf - S','C_cf - C_perm'],xlim=(-.021,.011),ylim=(-.55,1.55),
               xlabel='Final paired MAP difference; conditional 95% interval',title=('Old domain A' if d=='A' else 'New domain B'))
        ax.grid(axis='x',alpha=.16)
        ax=axes[1,i]
        for arm in ['S','C_cf','C_perm']:
            values=np.asarray(summary['valid_curves']['B_'+arm][d])[:,0]
            ax.plot(range(1,7),values,marker='o',color=colors[arm],linewidth=2,label=arm)
        shared=summary['absolute']['A_shared']['raw'][d]['map']
        ax.axhline(shared,color='#7a8189',linestyle='--',linewidth=1.2,label='Shared A endpoint')
        ax.set(xticks=range(1,7),xlabel='B-stage epoch (all six saved points)',ylabel='Unseen-account MAP',
               title=('A validation groups' if d=='A' else 'B validation groups'),ylim=(.451,.512))
        ax.grid(alpha=.16)
        ax.legend(loc='lower right',frameon=False,fontsize=9)
    for ax in axes.flat:
        ax.spines[['top','right']].set_visible(False)
    fig.suptitle('C_cf fixed A-to-B result review | s0 | complete S comparator',fontsize=16,fontweight='bold')
    fig.get_layout_engine().set(rect=(0,.045,1,.965))
    fig.text(.5,.012,'Saved results only. Same registered group bootstrap. No epoch selection, new seeds, or training uncertainty coverage.',ha='center',fontsize=9,color='#4b5563')
    fig.savefig(BASE/'05_MAP_REVIEW.png',dpi=180)
    plt.close(fig)

    findings=[
      {'id':'N01','category':'合理非阻断建议','status':'VERIFIED_EXISTING_CONCLUSION','evidence':['E4','E8'],
       'finding':'四个主要MAP差均负点估计，条件区间跨零；未建立改进，不能称等效或家族无效。','minimum_action':'保持本固定配置／开发条件的限定；无需改正式数值。'},
      {'id':'N02','category':'合理非阻断建议','status':'ALREADY_ADEQUATE','evidence':['E1','E4','E8'],
       'finding':'raw概率误差改善与stage-cal不占优、first-cal误报代价共存。','minimum_action':'三角色和合并计数继续并列，不能替代主要排序目标。'},
      {'id':'N03','category':'合理非阻断建议','status':'OPTIONAL_CLARIFICATION','evidence':['E6','E8'],
       'finding':'原判据21项均未触发，但C_cf训练—验证差距确实扩大；这既不确诊也不排除过拟合。','minimum_action':'可增加已有gap-widening数值及未触发原因，不需新实验。'},
      {'id':'N04','category':'合理非阻断建议','status':'ALREADY_ADEQUATE','evidence':['E1','E4','E7'],
       'finding':'C_perm保留O参与产生的下限多重集合和F可行性，非无OOF或总体随机置换对照。','minimum_action':'保留一次固定赋值比较的归因边界，不造p值。'},
      {'id':'N05','category':'合理非阻断建议','status':'ALREADY_ADEQUATE','evidence':['E7','E8'],
       'finding':'候选项被消费不等于强度最优或信息有效；新损失与S的MSE不能按数值大小比较强度。','minimum_action':'维持描述性解释，不从总梯度范数宣布纯保持项导数或根因。'},
      {'id':'N06','category':'合理非阻断建议','status':'ALREADY_ADEQUATE','evidence':['E1','E3','E7','E8'],
       'finding':'独立候选路径更新多50%、群呈现多1/3，末态Memory少30.3377402%。','minimum_action':'保留成本分母与本批六群限制，不冒称墙钟／FLOPs比例。'},
      {'id':'N07','category':'合理非阻断建议','status':'SUPPLEMENTED_IN_THIS_REVIEW','evidence':['E5','E6','E8'],
       'finding':'Linux参考summary转录interpretation；网页已按冻结规则重算21项并全部一致。','minimum_action':'接纳本次补充即可；无需修改冻结实现或重跑正式实验。'},
      {'id':'O01','category':'OUT_OF_SCOPE_OVERDESIGN','status':'NOT_PERFORMED','evidence':['E1'],
       'finding':'新增臂／seed／折／错配抽样／选轮／旧C独立诊断。','minimum_action':'不设为当前结果的自动阻断。'},
      {'id':'O02','category':'OUT_OF_SCOPE_OVERDESIGN','status':'NOT_PERFORMED','evidence':['E1','E7'],
       'finding':'要求上传或重开正式标签、模型、Memory、F/O、封存资产。','minimum_action':'明确保存证据边界；不扩大读取。'},
      {'id':'O03','category':'OUT_OF_SCOPE_OVERDESIGN','status':'NOT_PERFORMED','evidence':['E1','E2'],
       'finding':'重开创新和全历史实现审查、通用审计／权限／加固框架。','minimum_action':'复用已关闭结论，维持针对性结果审查。'}]
    dump('02_FINDINGS.json',{'scientific_blockers':0,'open_reproducibility_defects':0,'mandatory_corrections':[],
       'verdict':'Accept valid completion and improvement-not-established for this fixed development configuration; close original contract without automatic expansion.', 'findings':findings})

    failures=[
      {'id':'P01','scope':'project native qualification','status':'FAILED_THEN_EXPLICITLY_AUTHORIZED_SUPPLEMENT',
       'paths':['input/qualification/evidence/gpu/attempt01','input/qualification/native_disposition.zh.txt','input/qualification/native_retry_authorization.json'],
       'cause':'torch.isfinite on None gradient at paired_off; five prior updates inferred from stage, not five saved per-step receipts.',
       'disposition':'Original failure retained; CPU04 checker correction and one approved seven-step GPU02; cumulative 12 steps, 341 conservative seconds.'},
      {'id':'P02','scope':'project saved-result reader','status':'FAILED_THEN_READER_ONLY_FIXED',
       'paths':['input/analysis_attempt01','input/step28_crossfit_result_audit.py','independent/supplementary/linux_reader_fix.diff'],
       'cause':'execution.json sources field read as source_files; KeyError before statistics.',
       'disposition':'One reader-only diff hunk; no formal source or result mutation.'},
      {'id':'W00','scope':'web exploratory display','status':'FAILED_ORIGINAL_OUTPUT_TRUNCATED_EXPLICIT_REPLAY_RETAINED',
       'paths':['independent/exploration_failure_replay.py','independent/exploration_failure_replay'],
       'cause':'Called .items() on list-root analysis_manifest.json.',
       'disposition':'Original interactive full stdout/stderr not retained; exact command transcription and labelled replay preserve complete reproducer logs. Not passed off as original logs.'},
      {'id':'W01','scope':'web independent numerical reference','status':'FAILED_OWN_ASSERTION_FIXED',
       'paths':['independent/attempts/run01'],
       'cause':'Compared 288 A-stage row-level history_uid=None entries to empty history_ids draw list.',
       'disposition':'Assert each correct representation separately; scientific formulas and inputs unchanged.'},
      {'id':'W02','scope':'web independent numerical reference','status':'FAILED_OWN_RECURSION_FIXED',
       'paths':['independent/attempts/run02'],
       'cause':'Generic numeric comparison failed to recurse into list of dictionary residual records.',
       'disposition':'Container handling fixed only; run03 fully passes, all partial outputs and snapshots retained.'},
      {'id':'W03','scope':'web supplementary receipt check','status':'FAILED_OWN_TEMPORAL_ASSUMPTION_FIXED',
       'paths':['independent/supplementary_attempt01','independent/supplementary'],
       'cause':'Assumed access_at_start meant pre-parse train=0; observed receipt is after first real update and correctly train=1.',
       'disposition':'Check startup_observation time/status/first event and train=1; no project access ledger mutation.'},
      {'id':'H01','scope':'earlier implementation review','status':'HISTORICAL_NARRATIVE_ONLY_IN_CURRENT_INPUT',
       'paths':['input/review/support/reports/documentation/20261012/crossfit_retention/review/external/unpacked/01_REVIEW.zh.txt'],
       'cause':'Earlier reviewer tar checker treated directory members as ordinary files.',
       'disposition':'Current 303-member package provides narrative, not original prior stderr or full 19-member return ZIP. No claim of independent reread of unavailable raw log.'}]
    dump('04_FAILURE_LEDGER.json',{'formal_experiment_failure_observed_in_supplied_evidence':False,
      'original_formal_exit_code':0,'main_web_success':'independent/attempts/run03','failures':failures,
      'not_failures':'CPU attempts 01-04 PASS; expected negative fixtures not formal failures; output truncation alone not a script failure.'})

    events=[json.loads(s) for s in (BASE/'independent/reading_events.jsonl').read_text().splitlines()]
    # Reads performed in earlier direct tool commands, recorded retrospectively and labelled.
    manual=[{'path':'report.zh.txt','read_start':1,'read_end':len((INPUT/'report.zh.txt').read_text().splitlines()),'mode':'human_full_read_direct_tool'},
      {'path':'analysis_attempt01/source.py','mode':'one_hunk_diff_against_fully_read_success_script'},
      {'path':'review/support/docs/SELLER_ALIAS_RECORD_ATTRIBUTION.zh.md','mode':'targeted_keyword_excerpts_only','read_start':45,'read_end':62},
      {'path':'review/support/docs/SELLER_ALIAS_RECORD_REPLAY_PILOT.zh.md','mode':'targeted_keyword_excerpts_only','read_start':35,'read_end':35},
      {'path':'qualification/evidence/formal/startup_observation.json','mode':'human_structured_full_read_except_no_sources_field'},
      {'path':'qualification/evidence/formal/access_at_start.json','mode':'human_full_read_direct_tool'},
      {'path':'job/run/maps/A_shared.json','mode':'human_structured_selected_fields'},
      {'path':'job/run/maps/B_C_cf.json','mode':'human_structured_full_read'},
      {'path':'job/run/sources/birth.json','mode':'human_structured_full_read'},
      {'path':'job/run/points/B_C_cf.json','mode':'human_structured_full_read'},
      {'path':'job/run/diagnostics/B_C_cf_epoch1.json','mode':'human_structured_full_read'},
      {'path':'qualification/evidence/gpu/attempt02/result/result.json','mode':'human_native_fields_full_read_plus_programmatic_source_check'}]
    dump('independent/manual_reading_events.json',{'note':'Retrospective accurate descriptions of earlier visible tool reads; not newly timestamped original read logs. Keyword excerpts are not complete contiguous section reads.','events':manual})
    bypath=collections.defaultdict(list)
    for ev in events+manual:bypath[ev['path']].append(ev)
    checked={r['path']:r['modes'] for r in read('independent/attempts/run03/output/input_reads.json')}
    rows=[]
    for p in sorted(INPUT.rglob('*')):
        if not p.is_file():continue
        name=str(p.relative_to(INPUT));modes=checked.get(name,[])
        if name in bypath:
            human='; '.join((e['mode']+(f" [{e['read_start']}:{e['read_end']}]" if 'read_start' in e else '')) for e in bypath[name])
        else:human='no full human prose/source review claimed'
        if name.endswith('.npy'):
            if '/diagnostics/' in name and name.endswith('_valid.npy'):
                scope='Full bytes/array loaded; shape/finite check; epoch6 additionally bitwise compared with endpoint. Earlier epochs not recomputed from labels.'
            elif name.endswith('_calibration.npy'):
                scope='Full bytes/array loaded; shape/finite check only; no calibration refit or truth validation.'
            elif '/run/scores/' in name:
                scope='Full saved scores loaded; affine, ordering/ties, and predicted-positive totals checked where applicable; no truth reconstruction.'
            else:scope='Full saved metric array loaded; sums/paired statistics/curves checked; original labels not available.'
        elif name.endswith('source.tar.gz'):
            scope='All ordinary member bytes compared with own receipt; no source execution or full semantic rereview.'
        elif name=='analysis/map_comparison.pdf':scope='Byte identity and retention only; not visually rendered.'
        elif name=='analysis/map_comparison.png':scope='Byte identity and retention only; new independent figure uses saved checked results.'
        elif 'json_semantic_check' in modes:scope='Full JSON parsed; selected named fields/identities/statistics checked. Not a claim that every narrative field was reviewed.'
        elif 'csv_numeric_check' in modes:scope='All 990 rows numerically and structurally compared.'
        elif name in bypath:scope='Human scope is explicitly listed; all original bytes also hashed.'
        else:scope='Byte identity verified; no substantive reading or execution claimed.'
        rows.append({'path':name,'bytes':p.stat().st_size,'sha256':sha(p.read_bytes()),
                     'mechanical_modes':'; '.join(modes),'human_scope':human,'limits':scope})
    assert len(rows)==303
    dump('03_READING_SCOPE.json',rows)
    with (BASE/'03_READING_SCOPE.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    # Stable report excerpts for convenient machine extraction; no new statistics.
    with (BASE/'06_PRIMARY_RESULTS.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(summary['primary'][0]));w.writeheader();w.writerows(summary['primary'])
    print(json.dumps({'status':'BUILT_REPORT_INDEXES_AND_SAVED_RESULT_FIGURE','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
      'source_sha256':sha(Path(__file__).read_bytes()),'reading_scope_files':len(rows),'findings':len(findings),
      'blockers':0,'open_reproducibility_defects':0,'new_scientific_comparisons':0,'plot':'05_MAP_REVIEW.png'},ensure_ascii=False))

if __name__=='__main__':main()
