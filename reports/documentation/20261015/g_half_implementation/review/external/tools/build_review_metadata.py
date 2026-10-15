"""Assemble audit metadata from retained evidence; no project execution or scientific recomputation."""
from pathlib import Path
import hashlib, json
R = Path('/mnt/data/g_half_input')
O = Path('/mnt/data/g_half_review')
P = 'reports/documentation/20261015/g_half_implementation/'
Q = P + 'qualification/'
def load(path):
    return json.loads(path.read_text(encoding='utf-8'))
def write(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

full = {
    'docs/SELLER_ALIAS_G_HALF.zh.md': '当前合同；冻结准备版措辞与后续授权按对象和时间区分。',
    P+'authorization.json': '具体完整顺序批准及合同身份。',
    P+'review/request.zh.txt': '完整逐项请求及证据边界。',
    P+'review/attempt_delta.diff': '只改变临时夹具及CPU增量回执接受入口。',
    'schema/step28_g_half_policy.json': '单候选、全部限额、固定比较及身份。',
    'scripts/step28_g_half.py': '公式、两视图目标、top5调用、教师、更新、评分与两个诊断回调。',
    'scripts/step28_g_half_run.py': '单H完整入口、状态、配对、盲门、统计、资源及准入。',
    'scripts/step28_g_half_verify.py': 'CPU与唯一六调用GPU真实路径、保存恢复、资源计量；未执行此入口。',
    'scripts/run_step28_g_half_linux_20261015.sh': '正式deadline、环境、退出留证；未执行。',
    'tests/test_step28_g_half.py': '六项实际测试源码；区分真实tiny计算与mock编排；未在网页重跑测试。',
    'scripts/step28_record_replay.py': '读取全文，语义审查聚焦表/重组/VJP/Memory和H调用；旧C等未重开资格。',
    'scripts/step28_replay_diagnostics.py': '两回调、中立性、残差、曲线方向及AB评价。',
    'reports/documentation/20261015/g_refinement/review/primary_disposition.zh.txt': '已关闭静态方案与有限贡献定位；未重开文献检索。',
    'reports/documentation/20261014/balanced_pooling_implementation/review/primary_disposition.zh.txt': '继承限定实现结论，不替代新H接线。',
    'reports/documentation/20261014/balanced_pooling_implementation/qualification/native_disposition.zh.txt': '旧资格范围、版本与适用边界。',
    'reports/seller_alias_continual/20261015/balanced_pooling_result/review/primary_disposition.zh.txt': '最新闭合结果处置、损失遗忘方向、保管及解释范围。',
    Q+'run_cpu_attempt01.sh': '原始外层时钟、timeout和首轮来源保留。',
    Q+'run_cpu_attempt02.sh': '原始累计5秒、只补两项及外层时钟。',
}
targeted = {
    'docs/RESEARCH_DISCIPLINE.zh.md': ([[32,180]], '授权、技术/开放/选择权限、证据继承、外审分类、资源、保存和交付。'),
    'docs/SELLER_ALIAS_BALANCED_POOLING.zh.md': ([[23,35],[82,176]], '仅继承H明确引用的共同配置、接线、恢复、权限和评价；不重审F求解。'),
    'reports/documentation/20261015/g_refinement/review/external/review.zh.md': ([[123,160],[215,243]], '配对/判据/目标路径与历史隔离核算及最终裁决；文献条目仅作为历史阅读记录，不称本轮查阅原文。'),
    'scripts/step28_balanced_pooling.py': ([[1,100],[150,274]], '必要常量上下文、G/S实际聚合及目标/更新/评分/capture；F求解主体未全审。'),
    'scripts/step28_balanced_pooling_run.py': ([[89,195],[355,373],[595,616],[648,680]], '实际继承的公共输入/AB先过滤、初始化档案、查询指标与N/A/summary；不执行旧三臂。'),
    'scripts/step28_record_replay_run.py': ([[60,104]], 'Budget及输出扫描生命周期；其他依赖按闭包和旧证据复用。'),
    'scripts/step28_record_replay_verify.py': ([[1,44]], 'preflight现场资源前提；不是本次项目现场观测。'),
    'scripts/step28_bge_continual.py': ([[1,88],[242,248]], '配置、数值模式、学习率与Adam时钟。'),
    'scripts/step28_alias_ranking.py': ([[1,35],[89,129]], '导入关系、H实际top5选择与可导值/损失。'),
    'scripts/step28_bge_continual_run.py': ([[20,100]], '数组IO、RNG与parse_once失败消费。'),
    'scripts/step28_continual_population.py': ([[20,56],[157,210]], '必要初始化片段以及真实state digest/保存恢复；不声称其余函数复审。'),
    'scripts/step28_continual_population_run.py': ([[63,74]], 'checkpoint reserve/临时文件删除规则。'),
    'scripts/step28_chinese_base.py': ([[90,191]], '实际模型/文本表示上下文和BCE/queryrank定义；未加载模型。'),
    'tests/test_step28_record_replay.py': ([[1,85]], '手写群和tiny模型夹具定义；没有在网页实例化或执行。'),
}
structured = {
    'source_manifest.json': '77载荷完整集合、字节数与SHA机械逐项核对。',
    'reports/documentation/20261015/g_refinement/baseline_sources.json': '43个许可旧文件的完整身份列表、UID/列/角色和配对元数据；不含对应实际载荷。',
    Q+'cpu_receipt.json': '完整当前来源33项、六项不同测试、两轮继承、历史错误和累计17秒。',
    Q+'baseline_identity.json': '43项Linux身份记录；显式根路径映射后与批准清单比较。',
    Q+'receive_verification.json': '27项全部大小/SHA对应本包。',
    Q+'duplicate_cleanup.json': '64个重复来源均有当前等价身份；保留审查首版类型错误及修正。',
}
entries=[]
for path in sorted(p for p in R.rglob('*') if p.is_file()):
    name=path.relative_to(R).as_posix()
    e={'path':name,'bytes':path.stat().st_size,'sha256':sha(path),'identity_checked':True}
    total=len(path.read_text(encoding='utf-8').splitlines())
    e['total_text_lines']=total
    if name in full:
        e.update(read_level='全文读取_定向语义审查',read_ranges=[[1,total]] if total else [],purpose=full[name])
    elif name in targeted:
        ranges,purpose=targeted[name]
        e.update(read_level='直接相关段落读取',read_ranges=ranges,purpose=purpose)
    elif name in structured:
        e.update(read_level='完整结构化字段读取与机械核对',read_ranges=[[1,total]] if total else [],purpose=structured[name])
    elif name.startswith(Q+'cpu/') and '/sources/' in name:
        e.update(read_level='首轮旧字节_机械差异与四函数AST比较',read_ranges=[],purpose='旧文件全文供哈希/AST/diff，不冒称再次完整逐行科学审查；变更段见保留diff。')
    elif name.startswith(Q+'cpu/') and name.endswith('source_files.json'):
        e.update(read_level='完整来源清单结构化比较',read_ranges=[[1,total]],purpose='两轮33项来源与当前及首轮差异快照逐项核对。')
    elif name.startswith(Q+'cpu/') and '/result/' in name:
        e.update(read_level='原始结果全文或完整字段读取',read_ranges=[[1,total]] if total else [],purpose='读取unittest/真实失败/运行状态或全部小型行为证据字段；不是重新运行。')
    elif name.startswith(Q+'cpu/') and name.endswith(('stderr.txt','stdout.txt','wallclock.json','wrapper.txt')):
        e.update(read_level='原始日志全文读取',read_ranges=[[1,total]] if total else [],purpose='真实stderr/stdout、退出、wallclock及累计计费；空日志如实保留。')
    elif name.startswith('reports/documentation/20261014/') and name.endswith('result.json'):
        e.update(read_level='结构化状态和来源字段定向核对',read_ranges=[],purpose='旧CPU/GPU状态及source_files身份比较；25项可比中23同、2异，不称完整旧资格重新执行或全字段科学复核。')
    elif name.endswith('.py'):
        e.update(read_level='身份及导入闭包机械核对_未全文语义审查',read_ranges=[],purpose='未变间接依赖按已关闭范围复用；AST遍历不等于其全部功能再次通过。')
    elif name.startswith('schema/'):
        e.update(read_level='身份与引用机械核对_旧配置语义继承',read_ranges=[],purpose='不重开未变化旧政策；本轮实际新政策单列全文读取。')
    else:
        e.update(read_level='身份核对_不作为新增行为证明',read_ranges=[],purpose='必要归档/派发/PID/旧日志身份；不凭历史PID声称进程当前存在或已独立全量复审。')
    entries.append(e)
write(O/'reading_scope.json',{'input_archive_sha256':'f0f09f50daddbcf948bf9717edcb9b204fc19ea8f8a6d1ecc74b353f09f403e9','all_input_members':len(entries),'notes':['Full extraction and hashing does not equal full semantic review.','Scope includes source_manifest itself, for 78 members total.','Old baseline payloads 0/43 supplied.','Line ranges refer to original input files; structured/AST reads are explicitly distinguished.','No web, literature search, model, Memory, project tests, GPU, training or delegation.'], 'files':entries})

findings=[
 {'id':'F01','category':'REPRODUCIBILITY_DEFECT','status':'HISTORICAL_FIXED','title':'首轮临时查询夹具重复保存触发正确拒绝覆盖','locations':['tests/test_step28_g_half.py:358',Q+'cpu/attempt01/result/unittest.txt'], 'minimal_counterexample':'临时qpath已经存在后再次save_array同路径；应抛FileExistsError。','impact':'首轮第五项error、第六项failfast未运行；不影响未变四项已通过结果。','minimal_fix':'仅在本测试自己的临时qpath.unlink()后再保存；不改生产拒绝覆盖接口。','necessary_increment':'第二轮仅补保存比较/权限/预算及单H编排，已经完成。','current_blocker':False},
 {'id':'F02','category':'CONTRACT_IMPLEMENTATION_REVIEW','status':'NO_DEMONSTRATED_OPEN_BLOCKER','title':'H公式与实际A0/A1、top5、D0、VJP、诊断和恢复接线一致','locations':['scripts/step28_g_half.py:35-133','scripts/step28_record_replay.py:159-208','tests/test_step28_g_half.py:75-236'], 'support':'项目真实tiny/独立完整图及隔离数表计算共同支持各自有限范围。','limitations':'不是BGE/BF16/最大形状或正式576次资格。','current_blocker':False},
 {'id':'F03','category':'REASONABLE_NONBLOCKING_RECOMMENDATION','status':'OPTIONAL_NOT_REQUIRED_FOR_CURRENT_RECEIPT','title':'CPU增量门只检查qualified_tests长度','locations':['scripts/step28_g_half_run.py:586-588'], 'minimal_counterexample':'[同一个既定测试名]*6能通过该字段的长度条件；纯静态反例，未运行伪批准门。','impact':'当前实际六名恰为完整不同集合，已逐项核对；不能据一般漏检说当前回执失效。','minimal_fix':'后续维护时同时核对长度和既定六名集合。','necessary_increment':'若真的修改，仅针对入口正确/重复/缺项检查并更新来源绑定，保留17秒与未变继承；不要求当前追加。','current_blocker':False},
 {'id':'F04','category':'EVIDENCE_LIMITATION_AND_EXISTING_PREREQUISITE','status':'BASELINE_METADATA_MATCHED_PAYLOADS_NOT_READ','title':'43旧保存文件身份记录吻合，实际载荷不在本包','locations':['reports/documentation/20261015/g_refinement/baseline_sources.json',Q+'baseline_identity.json','scripts/step28_g_half_run.py:87-115,335-469'], 'support':'43项归档/Linux根映射后bytes/SHA全部一致。','limitations':'本网页0/43实际载荷读取；手写旧日志不是实际配对。','required_existing_action':'未来批准路径实际可用且baseline_inputs/实际新轨迹盲门通过；缺失则停，不重训旧对照。','current_blocker':False},
 {'id':'F05','category':'PENDING_NATIVE_AND_FORMAL_PREREQUISITES','status':'NOT_EXECUTED','title':'GPU六调用及实际投影尚未发生，正式准入尚不成立','locations':['scripts/step28_g_half_verify.py:91-222,306-312','scripts/step28_g_half_run.py:573-673'], 'required_existing_action':'独立主审后唯一GPU尝试；真实效果/恢复/最大形状/资源及当前来源门满足后才可正式。','not_authorized_by_review':['自动重试','增加第七次GPU调用','延长600秒或8小时','放宽精度','新候选或正式数据读取'], 'current_blocker':False,'formal_launch_permitted_now':False},
 {'id':'F06','category':'SCIENTIFIC_INTERPRETATION_BOUNDARY','status':'LIMITED_DEVELOPMENT_CANDIDATE','title':'实现可接受不等于性能优越或论文贡献充分','locations':['docs/SELLER_ALIAS_G_HALF.zh.md:15-17,63-75','scripts/step28_g_half_run.py:550-567'], 'support':'H-S双域MAP条件目标、H-G直接新域证据、完整概率/遗忘/N/A/诊断和成本分别解释。','current_blocker':False}
]
write(O/'findings.json',{'document_kind':'EXTERNAL_REVIEW_FINDINGS_NOT_A_RUNTIME_GATE','reviewer_self_description':'GPT-6 Astra Pro','requested_model':'网页GPT 6 Pro','selector_independently_observed':False,'backend_independently_verified':False,'verdict':'CONTRACT_AND_INCREMENT_ACCEPTABLE_FOR_CONDITIONAL_NATIVE_REVIEW_DISPOSITION','open_demonstrated_scientific_blockers':[],'open_necessary_reproducibility_defects':[],'formal_launch_qualified_now':False,'project_code_modified_by_reviewer':False,'findings':findings,'out_of_scope_overdesign':['增加系数/温度/seed/顺序/新教师/新损失/对照臂','增加GPU调用、时限、精度容差或自动重试','全量重审项目/F求解器或通用验证/认证框架','重训S/G、加载旧Memory或封存资产','为结果显著性追加训练']})

trace=[
 ('固定公式与精度',[9,45],['scripts/step28_g_half.py:35-69'],['checks/pure_checks_result.json','checks/raw_numeric_outputs.npz',Q+'cpu/attempt01/result/local_precision.json'],'CPU小例和独立数表一致；GPU未执行'),
 ('两视图当前/历史监督',[23,30],['scripts/step28_g_half.py:72-88','scripts/step28_alias_ranking.py:89-129','scripts/step28_chinese_base.py:165-191'],[Q+'cpu/attempt01/result/objective_witnesses.json'],'遗漏A1可辨别；历史0.1及D0外0.5正确'),
 ('H自己的top5',[23,23],['scripts/step28_alias_ranking.py:89-129','tests/test_step28_g_half.py:103-154'],[Q+'cpu/attempt01/result/objective_witnesses.json','checks/pure_checks_result.json'],'H/G第5候选不同；所选值带梯度'),
 ('教师D0与D1区分',[25,30],['scripts/step28_g_half.py:83-88,126-133','scripts/step28_replay_diagnostics.py:61-129'],[Q+'cpu/attempt01/result/objective_witnesses.json',Q+'cpu/attempt01/result/actual_capture_restore.json'],'先H投影再差平方；D1仅诊断'),
 ('encoder/head真实更新',[21,45],['scripts/step28_g_half.py:91-102','scripts/step28_record_replay.py:159-208','tests/test_step28_g_half.py:155-186'],[Q+'cpu/attempt01/result/independent_full_graph.json'],'真实tiny对完整图；norm<1未主动裁剪缩放；不是576真实更新'),
 ('两诊断回调与中立性',[73,73],['scripts/step28_g_half.py:126-133','scripts/step28_replay_diagnostics.py:31-129','tests/test_step28_g_half.py:187-236'],[Q+'cpu/attempt01/result/actual_capture_restore.json'],'实际48/6/40tiny capture；下一更新一致；GPU仍待执行'),
 ('完整/推理恢复身份',[43,43],['scripts/step28_g_half.py:18-32,105-123','scripts/step28_g_half_run.py:180-270'],[Q+'cpu/attempt01/result/actual_capture_restore.json'],'tiny实际恢复；正式两端点未发生'),
 ('只H和独立A接B',[49,57],['scripts/step28_g_half_run.py:31-58,149-177,273-332'],[Q+'cpu/attempt02/result/orchestration.json'],'576模拟调度/864逻辑群/2端点/12诊断，真实optimizer=0'),
 ('旧结果配对',[51,55],['scripts/step28_g_half_run.py:87-115,335-469'],['evidence/records_v2.json',Q+'baseline_identity.json'],'43身份记录吻合；载荷未在网页读取；未来实际门'),
 ('统一盲门',[108,108],['scripts/step28_g_half_run.py:335-479','tests/test_step28_g_half.py:237-338'],[Q+'cpu/attempt02/result/orchestration.json'],'实际受影响拒绝逻辑在手写fixture中可辨别'),
 ('评价/方向/N/A',[61,75],['scripts/step28_g_half_run.py:482-569','scripts/step28_balanced_pooling_run.py:648-680'],[Q+'cpu/attempt02/result/comparison_and_permissions.json'],'两对手直接共享整群抽样；loss遗忘B-A；未重算旧实际结果'),
 ('技术/开放/选择权限',[94,112],['scripts/step28_balanced_pooling_run.py:89-195','scripts/step28_bge_continual_run.py:76-87','scripts/step28_g_half_run.py:472-479'],[Q+'cpu/attempt02/result/comparison_and_permissions.json'],'正式七文件/AB先过滤/失败消费/valid完整门；未正式读取'),
 ('六次GPU固定构造',[120,133],['scripts/step28_g_half_verify.py:91-222'],[],'纯静态；计数336显式合成；不能称GPU资格已过'),
 ('完整投影/生命周期',[135,166],['scripts/step28_g_half_run.py:123-146,205-270','scripts/step28_g_half_verify.py:185-217'],['checks/pure_checks_result.json'],'1446v+36h与代码数量一致；人工测量只验算式，实际H投影待GPU'),
 ('批准门/停止',[168,174],['scripts/step28_g_half_run.py:573-673','scripts/step28_g_half_verify.py:225-325','scripts/run_step28_g_half_linux_20261015.sh:1-23'],['evidence/source_inheritance.json'],'NATIVE/FORMAL门未创建；不自动重试'),
 ('来源增量/计费',[116,118],['scripts/step28_g_half_run.py:586-588','tests/test_step28_g_half.py:358'],['evidence/source_inheritance.json','evidence/actual_attempt_delta.diff','evidence/records_v2.json'],'31相同+2最小变化；四项函数未变；17/180秒')
]
write(O/'evidence/traceability.json',{'contract':'docs/SELLER_ALIAS_G_HALF.zh.md','rows':[{'claim':a,'contract_lines':b,'code':c,'evidence':d,'assessment':e} for a,b,c,d,e in trace]})

pure=load(O/'checks/pure_checks_result.json')
ledger=[
 {'id':'R01','activity':'原始ZIP和source_manifest机械复核','source':['tools/audit_identity.py'],'outputs':['evidence/input_identity.json','logs/identity.stdout.txt','logs/identity.stderr.txt'],'outcome':'SUCCESS','note':'初始交互式ZIP查看/解包的显示未另存为独立日志；本脚本的重定向输出是随后机械复核的原始输出，不冒称最初调用转录。'},
 {'id':'R02','activity':'当前AST闭包、两轮继承、四函数与旧源比较','source':['tools/audit_inheritance.py'],'outputs':['evidence/source_inheritance.json','evidence/actual_attempt_delta.diff','logs/inheritance.stdout.txt','logs/inheritance.stderr.txt','logs/inheritance.exit_code.txt'],'outcome':'EXIT_0_WITH_BASELINE_PATH_JOIN_LIMITATION','note':'首版完整路径联结使43项未匹配；保留原输出，后续显式根映射修正。'},
 {'id':'R03','activity':'首次元数据摘要读取','source':['tools/inspect_receipts_v1.py'],'outputs':['logs/receipt_inspection_v1.tool_transcript.txt'],'outcome':'REAL_UNEXPECTED_TYPEERROR','exception':'TypeError: object of type int has no len()','note':'对omitted_duplicate_sources整数误用len；初次未独立重定向stdout/stderr，源码和实际失败子段从工具记录转录。外层shell还有后续cat，未独立取得失败Python子进程退出码。未重跑制造失败。'},
 {'id':'R04','activity':'元数据摘要仅修正类型分支','source':['tools/inspect_receipts_v2.py'],'outputs':['logs/receipt_inspection_v2.stdout.txt','logs/receipt_inspection_v2.stderr.txt','logs/receipt_inspection_v2.exit_code.txt'],'outcome':'EXIT_0'},
 {'id':'R05','activity':'43项显式根映射、27回收与64去重身份','source':['tools/audit_records_v2.py'],'outputs':['evidence/records_v2.json','logs/records_v2.stdout.txt','logs/records_v2.stderr.txt','logs/records_v2.exit_code.txt'],'outcome':'EXIT_0','note':'只比较提供的记录，不打开项目Linux上43个载荷。'},
 {'id':'R06','activity':'AST提取纯数表/梯度/top5/投影检查','source':['checks/check_pure_functions.py','checks/step28_g_half_extracted.py','checks/step28_alias_ranking_extracted.py','checks/step28_g_half_run_extracted.py'],'inputs':['checks/handwritten_inputs.npz','checks/check_pure_functions.py中明确的非法输入及人工投影常量'],'outputs':['checks/raw_numeric_outputs.npz','checks/pure_checks_result.json','logs/pure_checks.stdout.txt','logs/pure_checks.stderr.txt','logs/pure_checks.exit_code.txt'],'outcome':'EXIT_0','environment':pure['environment'],'check_body_seconds_after_imports':pure['elapsed_seconds'],'full_process_wall_seconds':'NOT_MEASURED','model_instances':0,'memory_instances':0,'optimizer_calls':0,'formal_asset_reads':0,'project_qualification_seconds_charged':0,'note':'三个预期ValueError不计意外失败；不导入完整项目入口，不执行其tiny测试/训练/统计。'},
 {'id':'R07','activity':'中文报告、阅读范围、结构化发现与交付封装','source':['tools/build_review_metadata.py','tools/package_review.py'],'outputs':['report.zh.md','findings.json','reading_scope.json','execution_ledger.json','evidence/traceability.json','README.zh.md','MANIFEST.payload.json','MANIFEST.payload.tsv'],'outcome':'DOCUMENT_AND_ARCHIVE_GENERATION_ONLY','note':'交付封装身份检查不新增模型计算或项目资格。'}
]
write(O/'execution_ledger.json',{'review_environment':'网页隔离容器，非项目Linux py310','no_delegation':True,'no_network_or_project_server':True,'project_sources_modified':False,'original_input_zip_returned':False,'unexpected_reviewer_exceptions_count':1,'all_scientific_check_sources_inputs_outputs_retained':True,'interactive_viewing_note':'普通文件阅读/grep/sed显示不等同科学检查日志，未声称所有交互式显示均有逐命令单独转录。初次机械查看的留证范围已明确。','executions':ledger})
print(json.dumps({'reading_scope_members':len(entries),'findings':len(findings),'traceability_rows':len(trace),'execution_ledger_entries':len(ledger)},ensure_ascii=False,indent=2))
