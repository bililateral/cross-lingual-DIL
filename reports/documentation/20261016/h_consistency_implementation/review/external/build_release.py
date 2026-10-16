"""Assemble review artifacts and cross-check their inventory; no research execution."""
from __future__ import annotations
import ast,collections,csv,datetime,hashlib,json,pathlib,shutil,zipfile
ROOT=pathlib.Path(__file__).resolve().parent
IN=ROOT/'input';ART=ROOT/'artifacts'
def js(p):return json.loads(p.read_text())
def dump(name,obj):(ART/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')
def ref(path,a,b=None,field=None):
 d={'path':path,'start_line':a,'end_line':b or a}
 if field:d['field']=field
 return d
findings=[
 {'id':'R01','category':'REPRODUCIBILITY_DEFECT','status':'OPEN','severity':'PRE_GPU_EVIDENCE_FIX_REQUIRED',
  'title':'原生开关C的两组Adam变化记录没有实现',
  'references':[ref('sources/docs/SELLER_ALIAS_H_CONSISTENCY.zh.md',166),ref('sources/scripts/step28_g_half_verify.py',29,31),ref('sources/scripts/step28_g_half_verify.py',44,72),ref('sources/scripts/step28_g_half_verify.py',158,192),ref('sources/scripts/step28_g_half_verify.py',244,250),ref('sources/scripts/step28_h_consistency_run.py',147,151)],
  'trigger':'按当前代码完成原生调用2/3、4/5后，effects仅含参数/梯度差，捕获的off Adam矩未被比较或记录。',
  'established_fact':'component_effects实际仅读取on[model]和on[grads]；gcurrent/gpaired的adam被捕获但在两个比较中未消费，之后删除。',
  'impact':'无法从原生结果审计合同166行约定的两组分组件Adam变化；不证明Adam算法错误、不否定真实参数差。',
  'not_a_substitute':['CPU独立全图Adam正确性','原生调用5/6完整Adam一致性','泛化的both_phases_encoder_head_effect布尔值'],
  'minimal_disposition':'删除off状态前，利用现有model/optimizer与off快照按命名参数映射，记录current/paired两组件的step、exp_avg、exp_avg_sq有限性与实际差异摘要；仅补受影响测试、来源与增量回执，不增更新/精度门/预算。',
  'gate_effect':'建议主执行者关闭前不要为本包签APPROVED_NO_OPEN_BLOCKERS或启动依赖GPU；本报告不代替主审。',
  'scientific_blocker':False},
 {'id':'R02','category':'REPRODUCIBILITY_DEFECT','status':'CLOSED_IN_INPUT','severity':'CORRECTED_HISTORY_RETAINED',
  'title':'attempt01新臂两个证据JSON被旧默认回归覆盖，attempt03已补存',
  'references':[ref('cpu_receipt.json',311,field='reproducibility_correction'),ref('sources/tests/test_step28_h_consistency.py',477,480),ref('sources/tests/test_step28_h_consistency.py',536),ref('cpu/attempt03/result/h_cons_orchestration.json',1),ref('cpu/attempt03/result/h_cons_comparison_and_permissions.json',1)],
  'trigger':'把attempt01中沿用旧名的orchestration/comparison JSON当作当时新臂具体保存证据。',
  'impact':'unittest通过不能恢复已被覆写的细节原件；不属于CPU科学失败。',
  'minimal_disposition':'只使用attempt03的h_cons_前缀原件支持对应细节，保留attempt01、全部收费及按方法继承，不全套重跑。','gate_effect':'已关闭，不另加GPU前提'},
 {'id':'N01','category':'EVIDENCE_LIMITATION_RECOMMENDATION','status':'NONBLOCKING','severity':'NARROW_TEST_SCOPE',
  'title':'项目CPU旧S/H比较夹具共用根与数值，未独立区分两个物理根',
  'references':[ref('sources/tests/test_step28_h_consistency.py',435),ref('sources/tests/test_step28_h_consistency.py',482,508),ref('sources/scripts/step28_g_half_run.py',549,591)],
  'trigger':'将同根mock的PASS宣传为有区分力的双根文件读取实测。',
  'impact':'该夹具不能检出串根；实际finalize按端点origin路由正确，审查侧不同根/同名文件/不同值已通过。',
  'minimal_disposition':'限定原CPU结论；以后相关测试改动时可替换成不同根/值的最小例，不强制新平台或全套重测。','gate_effect':'不阻断；审查侧结果也不替代项目原生资格'},
 {'id':'P01','category':'PENDING_CONTRACT_PREREQUISITE','status':'NOT_EXECUTED','severity':'STAGE_BOUNDARY',
  'title':'新BGE六调用、资源和正式执行仍未发生',
  'references':[ref('sources/docs/SELLER_ALIAS_H_CONSISTENCY.zh.md',153,197),ref('sources/scripts/step28_g_half_verify.py',124,250),ref('sources/scripts/step28_g_half_run.py',616,711)],
  'trigger':'把CPU、旧H资格或本审查写成H_cons原生/正式PASS。',
  'impact':'未知作用及资源被错误确证。',
  'minimal_disposition':'先关闭R01并独立主审，再按原合同唯一六调用，不足即停；实际投影和完整新盲门仍必须履行。','gate_effect':'原生与正式均未取得资格，不是已失败'},
 {'id':'L01','category':'SCIENTIFIC_CLAIM_LIMIT','status':'RETAIN','severity':'INTERPRETATION_BOUNDARY',
  'title':'保存对照回答固定系统性能，不识别B-only因果或H特有协同',
  'references':[ref('sources/docs/SELLER_ALIAS_H_CONSISTENCY.zh.md',30),ref('sources/docs/SELLER_ALIAS_H_CONSISTENCY.zh.md',92,106),ref('sources/scripts/step28_g_half_run.py',593,612)],
  'trigger':'将未来双域胜S/新B胜H推广成根因修复、普遍优势、确认性显著性或已成立论文贡献。',
  'impact':'新A/出生表也变化；单seed、复用valid和条件区间不支持上述推广。',
  'minimal_disposition':'分别报告两个判据、旧A相对H及22列/三角色代价，保留有限主张；不追加S+JS或非劣门。','gate_effect':'不是本轮实现阻断'},
 {'id':'E01','category':'REVIEWER_EXECUTION_ERROR','status':'CLOSED_WITH_RECORDS','severity':'NOT_PROJECT_FAILURE',
  'title':'审查脚本v1误用Group.records字段',
  'references':[{'path':'evidence/reviewer_probes.py','start_line':41,'end_line':41},{'path':'evidence/reviewer_probes_execution/stderr.txt','start_line':1,'end_line':5}],
  'trigger':'仅审查侧脚本创建手写输入记录时AttributeError退出1。',
  'impact':'v1未完成检查；不涉及项目代码、不计项目资格或预算。',
  'minimal_disposition':'保留v1、完整失败记录；v2仅改为items及新输出目录，原输入公式和容差不变，退出0。','gate_effect':'不影响项目准入；不得隐藏这次失败'},
 {'id':'O01','category':'OUT_OF_SCOPE_OVERDESIGN','status':'REJECTED','severity':'DO_NOT_EXPAND',
  'title':'不扩大候选、数据、GPU调用、预算或重审旧项目',
  'references':[ref('request.zh.txt',1,27),ref('sources/docs/SELLER_ALIAS_H_CONSISTENCY.zh.md',9),ref('sources/docs/SELLER_ALIAS_H_CONSISTENCY.zh.md',201,209)],
  'trigger':'以本次意见追加S+JS、grid、seed、正式A1诊断、旧Memory或第七次调用等。',
  'impact':'超出已批准范围且不解决R01这个具体记录缺口。',
  'minimal_disposition':'不实施上述扩展，保留旧结果与费用。','gate_effect':'不新造准入总门'}]
dump('findings.json',{'review_verdict':'CORE_IMPLEMENTATION_ACCEPTABLE_WITH_ONE_OPEN_REPRODUCIBILITY_FIX_BEFORE_GPU','proved_scientific_blockers':0,'open_required_reproducibility_fixes':1,'conditional_next_stage':'Only after independent disposition/closure; unique native six calls remains pending','findings':findings})
trace=[
 ('precision_and_queries','FP32 H→FP64 JS→一次FP32 C；排自环、规范27列、仅28平均','M:42-67; g_half:39-73','CPU/js_precision; shared_K_precision; reviewer probe FP32/FP64','已验证代码和手写例；不是BGE'),
 ('shared_live_graph','两学生/m连通；双向边、两分配在同K汇合','M:47-114; record_replay:159-208','独立解析/FD/完整图与tiny更新','无新增encoder前向；资源仍待测'),
 ('five_term_target','Bbar字段不藏C；current/history/.5 D0分开；正式拒绝去C','M:70-114; R:113-139; S:422-451','CPU total/weight/identity故障检测','目标与新盲门一致'),
 ('training_identity','H评分身份保持，H_cons训练身份贯穿','M:17-39,117-136; S:209-302','CPU capture/restore/missing-training反例','两正式端点未执行'),
 ('one_arm_trajectory','共同预训练→自身A/出生表→自身B；576/864/2/12','R:160-182; S:177-365','attempt03 orchestration; actual tiny另外证明梯度','576为mock控制流，实际训练0'),
 ('memory_and_blind','新臂盲门仅恢复自身Memory，不载旧Memory','S:368-514; record_replay:234-332','CPU actual capture + controlled blind fixture','未实际读取正式Memory'),
 ('saved_baselines','52批准文件，两个旧根，UID/22列/三角色/起点与日程','R:75-110; S:497-612','批准与deployment逐项metadata一致；reviewer不同根例','未在网页读取远端52原件'),
 ('statistics','直接H比較独立于双域S门；低好遗忘及固定群N/A','S:549-612; balanced_pooling_run:648-680','CPU代数例；reviewer root/N/A例','非效果结果；条件区间'),
 ('native_six_calls','W/P固定；1准备，2/3current，4/5paired，6诊断后重复','V:91-250','源代码追踪；CPU geometry固定','原生未执行；R01 Adam记录缺口'),
 ('budget','180/600/28800独立；1446v+36h、1800预留和25%；10GiB峰值','C:176-197; S:151-174; V budget/main','CPU41收费复算；代数投影','所有新原生测量待取得'),
 ('data_permissions','七输入、C技术扫描后筛除、train/valid失败也耗一次、全新门前不开valid','balanced_pooling_run:89-173; previous:76-87; S:506-514','CPU消费失败尝试/盲门拒绝；静态调用','test/owners/private、旧Memory无许可')]
dump('traceability.json',{'reference_key':'M=h_consistency.py; R=h_consistency_run.py; S=g_half_run.py; V=g_half_verify.py; all scripts under sources/scripts; C=approved H_cons contract',
 'requirements':[dict(id=a,requirement=b,source=c,evidence=d,limit=e) for a,b,c,d,e in trace]})
# Requested read logs are preserved separately. A semantic scope is intentionally not all-file full reading.
logs=[json.loads(l) for l in (ART/'reading_log.jsonl').read_text().splitlines()]
by=collections.defaultdict(list)
for x in logs:by[x['path']].append([x['start_line'],x['end_line']])
# Reads performed in short direct display commands rather than read.py.
manual={'evidence/H_result_primary.zh.txt':[[1,19]],'sources/scripts/step28_continual_population_data.py':[[66,102]],
        'sources/scripts/step28_continual_population_evaluate.py':[[1,35]]}
for p,rr in manual.items():by[p]+=rr
metadata_full=set(['manifest.json','source_mappings.json','deployment_receipt.json','cpu_receipt.json',
 'sources/reports/documentation/20261016/h_consistency/baseline_sources.json'])
for p in IN.glob('cpu/attempt*/**/*.json'):metadata_full.add(str(p.relative_to(IN)))
for p in IN.glob('cpu/attempt*/**/*.txt'):metadata_full.add(str(p.relative_to(IN)))
full_core={'request.zh.txt','sources/docs/SELLER_ALIAS_H_CONSISTENCY.zh.md','sources/scripts/step28_h_consistency.py','sources/scripts/step28_h_consistency_run.py','sources/scripts/step28_h_consistency_verify.py','sources/scripts/step28_g_half.py','sources/scripts/step28_g_half_run.py','sources/scripts/step28_g_half_verify.py','sources/scripts/step28_record_replay.py','sources/scripts/step28_replay_diagnostics.py','sources/tests/test_step28_h_consistency.py','sources/schema/step28_h_consistency_policy.json','sources/scripts/run_step28_h_consistency_linux_20261016.sh','sources/reports/documentation/20261016/h_consistency_implementation/authorization.json'}
entries=[]
for p in sorted([IN/r['path'] for r in js(IN/'manifest.json')['files']]+[IN/'manifest.json']):
 rel=str(p.relative_to(IN));b=p.read_bytes();rr=[]
 for a,z in sorted(by.get(rel,[])):
  if rr and a<=rr[-1][1]+1:rr[-1][1]=max(z,rr[-1][1])
  else:rr.append([a,z])
 if rel in full_core:level='COMPLETE_CORE_SOURCE_OR_CONTRACT_READ'
 elif rr:level='TARGETED_TEXT_READ'
 elif rel in metadata_full:level='STRUCTURED_FIELDS_AND_ORIGINAL_LOG_CHECK'
 elif rel in ('cpu/attempt01/sources/tests/test_step28_h_consistency.py','cpu/attempt02/sources/tests/test_step28_h_consistency.py'):level='METHOD_AST_AND_DIFF_CHECK_NOT_FULL_SEMANTIC_REVIEW'
 else:level='IDENTITY_HASH_ONLY_OR_HISTORICAL_METADATA_REUSE_NOT_FULL_REVIEW'
 e={'path':rel,'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest(),'level':level,'text_read_ranges':rr,'all_bytes_hashed':True}
 if rel=='changes.diff':e['note']='全部1806行机械当前侧hunk验证；共享修改1-778行定向阅读；后续新文件diff的科学阅读直接按实际新源文件完成。原始工具日志是请求范围，不以被截断的单次输出证明全文；核心缺口已用短段及实际文件补读。'
 if rel in metadata_full:e['note']='JSON结构/指定字段、字节对应或原始测试/控制台/外层日志核对；不是其中引用的远端载荷内容已被读取。'
 entries.append(e)
dump('reading_scope.json',{'principle':'All bytes verified is not all files scientifically rereviewed. Core source/contract read; unchanged shared/history targeted. Files ZIP text extraction unavailable; mounted archive used.',
 'archive_members':92,'raw_read_log':'artifacts/reading_log.jsonl','scope_entries':entries,'scope_entry_count':len(entries),'bytecode_cache_note':'Python imports may create incidental __pycache__ in the extracted working copy. They are not input ZIP members and are excluded from input reading scope and delivery; original source and ZIP bytes were not changed.',
 'not_performed':['Project Linux connection','Remote 52-file live rehash','Old or new formal Memory loading','BGE or any encoder/head model loading','GPU qualification','Formal training','Formal label reading','Complete rereview of old F or all history','New literature/novelty search'],
 'reviewer_execution':'Pure handwritten tables and saved-statistics fixtures plus byte/AST verification; full failures retained'})
# Cross-check real sources() emitted by reviewer calling production collect/finalize.
actual=js(ROOT/'evidence/probe_outputs_v2/distinct_root_H_cons/collected.json')['source_files']
receipt=js(IN/'cpu_receipt.json')
assert actual==receipt['source_files'],'Production source closure disagrees with CPU receipt'
proof={'current_source_closure_matches_cpu_receipt':True,'source_count':len(actual),'provenance':'Production run.sources() result in actual reviewer-created collected.json compared with input CPU receipt',
 'old_test_method_inheritance':{},'reviewer_original_failure_retained':True}
all_versions={a:IN/'cpu'/a/'sources/tests/test_step28_h_consistency.py' for a in ('attempt01','attempt02')};all_versions['attempt03']=IN/'sources/tests/test_step28_h_consistency.py'
def funcs(p):
 t=ast.parse(p.read_text());out={}
 for n in ast.walk(t):
  if isinstance(n,ast.ClassDef) and n.name=='HConsistencyTests':
   for f in n.body:
    if isinstance(f,ast.FunctionDef):out[f.name]=ast.dump(f,include_attributes=False)
 return out
fm={k:funcs(v) for k,v in all_versions.items()}
for name,a in receipt['qualification_by_test'].items():
 same=fm[a][name]==fm['attempt03'][name]
 # shared-table test from02 unchanged at03; helpers for interface are from03.
 assert same,(name,a)
 proof['old_test_method_inheritance'][name]={'accepted_attempt':a,'test_body_AST_same_as_current':True}
dump('final_cross_checks.json',proof)
# Add context to the mechanically copied manifest's frozen pending text, without changing the input.
v=js(ART/'manifest_verification.json');v['pending_field_interpretation']='Input manifest is a frozen preparation snapshot. This submitted package is now in review; native/formal qualification remains unexecuted. No permission inferred for additional payloads.';dump('manifest_verification.json',v)
(ROOT/'README.zh.md').write_text('''# H_cons 合并反对性审查交付\n\n先读 `artifacts/review.zh.md`；结构化裁决见 `artifacts/findings.json`。核心实现未见已证科学阻断，R01为一项GPU前需主执行者独立关闭的原生Adam变化记录缺口。\n\n`original_input.zip` 为本次实收输入原字节，SHA在input_identity内；它没有被重打包替代。`artifacts/reading_scope.json` 区分全文核心阅读、定向依赖阅读、元数据检查和仅身份核对。`traceability.json` 对应各项合同要求。\n\n`evidence/` 保留本次审查的全部科学隔离检查源码、手写输入、完整数组/统计输出、stdout/stderr、环境及退出。v1字段名错误退出1与v2退出0均保留；机械检查单独留证。这些不是项目Linux资格，不计入其41/180秒，不改变原生0次/正式0次。\n\n复现应在新的空目录中解压原输入到 `input/`，复制evidence中的脚本并保持相对目录。不要在已归档证据目录上覆盖重跑。`launch.py` 是一次性有界执行器，没有自动重试；v1原脚本仍会重现已记录的字段错误，实际成功核对为 `reviewer_probes_v2.py`。无需模型、正式文本/标签或旧Memory。使用包中相同软件环境才能期待同样的数值细节。\n\n导入时生成的临时Python字节码缓存不是原输入成员，不装入交付；阅读清单严格按原ZIP的92成员枚举，原源码和ZIP字节不改。\n\n原始输入92成员仅作为ZIP存一份；为了避免重复，它的解包目录不再同时装入交付。产物 `manifest.json` 列除其自身以外的全部成员大小/SHA；外部members.tsv还包含manifest自身。审查意见不是NATIVE/FORMAL批准门，不改变合同或既有结果。\n''')
shutil.copyfile(pathlib.Path('/mnt/data/review_input(20261016-053330).zip'),ROOT/'original_input.zip')
# Include assembly source itself so final scope/inventory transformations are reproducible.
release_files=[ROOT/'README.zh.md',ROOT/'original_input.zip',ROOT/'build_release.py']
release_files+=sorted(p for p in ART.rglob('*') if p.is_file())
release_files+=sorted(p for p in (ROOT/'evidence').rglob('*') if p.is_file() and '__pycache__' not in str(p))
rows=[]
for p in release_files:
 b=p.read_bytes();rows.append({'path':str(p.relative_to(ROOT)),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()})
manifest={'scope':'H_cons merged adversarial review artifacts, not executable authorization','files':rows,'excludes_self':True}
(ROOT/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
release_files.append(ROOT/'manifest.json')
zip_path=pathlib.Path('/mnt/data/H_cons_merged_adversarial_review_20261016.zip')
with zipfile.ZipFile(zip_path,'w',zipfile.ZIP_DEFLATED,compresslevel=9) as z:
 for p in release_files:z.write(p,str(p.relative_to(ROOT)))
with zipfile.ZipFile(zip_path) as z:
 assert z.testzip() is None
 assert len(z.namelist())==len(set(z.namelist()))==len(rows)+1
 for row in rows:
  b=z.read(row['path']);assert len(b)==row['bytes'] and hashlib.sha256(b).hexdigest()==row['sha256']
 member_rows=[{'path':i.filename,'bytes':i.file_size,'sha256':hashlib.sha256(z.read(i.filename)).hexdigest()} for i in z.infolist()]
public_report=pathlib.Path('/mnt/data/H_cons_merged_review.zh.md');shutil.copyfile(ART/'review.zh.md',public_report)
public_findings=pathlib.Path('/mnt/data/H_cons_review_findings.json');shutil.copyfile(ART/'findings.json',public_findings)
inventory=pathlib.Path('/mnt/data/H_cons_review_members.tsv')
with inventory.open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=['path','bytes','sha256'],delimiter='\t');w.writeheader();w.writerows(member_rows)
final={'zip':str(zip_path),'bytes':zip_path.stat().st_size,'sha256':hashlib.sha256(zip_path.read_bytes()).hexdigest(),'members':len(member_rows),'payloads':len(rows),
       'input_archive_unchanged':hashlib.sha256((ROOT/'original_input.zip').read_bytes()).hexdigest()==hashlib.sha256(pathlib.Path('/mnt/data/review_input(20261016-053330).zip').read_bytes()).hexdigest(),
       'CRC_and_inventory_verified':True,'report':str(public_report),'findings':str(public_findings),'members_inventory':str(inventory)}
pathlib.Path('/mnt/data/H_cons_review_delivery.json').write_text(json.dumps(final,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(final,ensure_ascii=False,indent=2))
