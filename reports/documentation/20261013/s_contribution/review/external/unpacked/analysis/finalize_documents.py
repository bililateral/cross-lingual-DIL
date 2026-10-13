#!/usr/bin/env python3
"""Assemble static-review documents and provenance inventories; never execute project code."""
from __future__ import annotations
import hashlib
import json
import pathlib
import re
import shutil
import zipfile

OUT = pathlib.Path('/mnt/data/S_contribution_review')
SRC = pathlib.Path('/mnt/data/s_review_input')
ARCHIVE = pathlib.Path('/mnt/data/review_input(20261013-031814).zip')

def dump(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')

def digest(data):
    return hashlib.sha256(data).hexdigest()

p = OUT/'review.zh.md'
s = p.read_text(encoding='utf-8')
s = s.replace('一手阅读共 12 项，重点在方法／公式而非全文实验复核。', '一手阅读共 11 篇论文，其中 DER 与 DER++ 分别核查；重点在方法／公式而非全文实验复核。')
s = s.replace('MI Kernels 和 LUAR 的相关 PDF 页使用截图查看；', 'MI Kernels、LUAR 和 KDD 2019 卖家匹配的相关 PDF 页使用截图查看；')
s = s.replace('实读 §3、§4.1–4.2 及 §4.3 对应讨论；', '实读 §3、§4.1–4.2，§4.3 仅定位及相关片段，不算完整审读；')
s = s.replace('[L9] X. Tai et al.', '[L9] Xiao Hui Tai, Kyle Soska, Nicolas Christin.')
s = s.replace('实读 §5 开头、5.1、5.2，定位至 5.3 开头；', '实读 §5 开头、5.1、5.2，定位至 5.3 开头；PDF 第5页截图复核 5.1／5.2；')
s = re.sub(r'\n\[L12\].*\Z', '\n', s, flags=re.S)
s = s.replace('S 的可加结构另有本文第六节的限制。', 'S 的可加结构另有本文第七节的限制。')
s = s.replace('## 五、第 3.9 节是否公允：应保留比较，但不能设成循环门槛', '## 五、第 3.9 节是否公允：基本公允，但不能强读成循环门槛')
s = s.replace('公平结论应分两层：', '第 3.9 节明确承认这并非某篇论文已写出 S 全部代码，并保留任务化候选；这一点公允。报告并没有明确要求新增一个与 S 完全相同的实验臂。因此，下述循环门槛是需要防止的过度解读，不记为已经发生的实验或逻辑错误。公平结论应分两层：')
s = s.replace('去掉循环性必要条件。', '明确“构造性归约”不是新增对手要求；不要把风险说成已发生错误。')
s = s.replace('| F04 | 表述／推理门槛缺陷 | §3.9 可作归约说明，不能把补齐成 S 的构造当作必须胜出的新对手，也不能由可组合推得零贡献。 | 区分单篇实际方法、原则映射、与 S 完全相同的重述；明确“构造性归约”不是新增对手要求；不要把风险说成已发生错误。 |', '| F04 | 合理边界澄清；非已证错误 | §3.9 基本公允，报告没有明确要求 S 战胜一个与自身相同的对手；后续不可把原则映射强读为零贡献或循环实验门槛。 | 区分单篇实际方法、原则映射、与 S 完全相同的重述；明确不追加同义对手。 |')
s = s.replace('F03、F04 是本轮报告应落实的主要文献／表述修订；F06 是公式与引用精度补充。', 'F03 是本轮需补足的主要近邻覆盖；F04 是防止后续过度解读的澄清，不是已证逻辑错误；F06 是公式与引用精度补充。')
s = s.replace('这比泛称保持袋分布更具体，既能指导分配／权重，又有上述退化反例作约束。', '这是对主报告 §6.2–6.3 已有“共同条件与身份关系区分”动机的进一步量化，不是声称报告完全遗漏了关系结构。它比泛称保持袋分布更具体，既能指导分配／权重，又有上述退化反例作约束。')
p.write_text(s, encoding='utf-8')
(OUT/'review.zh.txt').write_text(s, encoding='utf-8')
findings = s.split('## 九、逐项发现、分类与最小修改\n',1)[1].split('\n## 十、',1)[0]
(OUT/'findings.zh.md').write_text('# 逐项发现与最小修改\n\n完整来源与推导见 `review.zh.md`。本表是正文第九节的逐字摘录，不是另一轮审查。\n'+findings+'\n', encoding='utf-8')

# Semantic read ranges are deliberately more conservative than displayed/extracted ranges.
scopes = {
'context/AGENTS.md': ('完整语义阅读', [[1,24]], '本轮纪律与服务器／数据权限；不是重新审核全部工程。'),
'context/RESEARCH_DISCIPLINE.zh.md': ('定向语义阅读', [[1,50]], '后续仅标题／边界定位，不主张全文复审。'),
'context/SELLER_ALIAS_RECORD_REPLAY_PILOT.zh.md': ('完整语义阅读', [[1,45]], '历史完整 C/S 合同背景，不重新诊断 C。'),
'current/report.zh.txt': ('完整语义阅读', [[1,239]], '全文阅读，关键 §2、§3.9、§5–8 定向复读；部分全文显示早于 read_source.py 日志。'),
'current_source/step28_continual_expression_data.py': ('定向语义阅读', [[84,136],[216,280]], 'render 与 make_group 的控制者／账号条件及逐记录使用；其余函数未完整读。'),
'current_source/step28_record_replay.py': ('完整差异核对', [], '全文件与冻结 core 比较；四个 diff hunk 实读，未变行依据冻结全文，不冒称第二次逐行复读。'),
'evidence/attribution_correction.zh.txt': ('完整语义阅读', [[1,17]], '原主审勘误及其适用范围。'),
'evidence/attribution_external.zh.txt': ('定向语义阅读', [[47,90],[129,155],[306,338]], '执行资格、结果归因与结论／新颖性边界；其他只标题或关键词定位。'),
'evidence/attribution_primary.zh.txt': ('完整语义阅读', [[1,89]], '闭合独立主审、结果与后续权重删除时点。'),
'evidence/attribution_summary.json': ('仅字节身份核对', [], '没有具体矛盾，未解析 JSON、未读取指标数值、未重算统计。'),
'evidence/crossfit_result_primary.zh.txt': ('完整语义阅读', [[1,78]], '仅复用 C_cf 结果边界与处置，不读评分载荷或反向证明 S 最优。'),
'evidence/record_static_external.zh.txt': ('定向语义阅读', [[9,29],[88,121],[263,294]], '有时间范围的背景结论、集合与重组相关讨论；其余不作本轮复审。'),
'evidence/record_static_primary.zh.txt': ('定向语义阅读', [[1,12],[64,116]], '历史定位及 S／自然分布相关条目，不恢复旧 C。'),
'evidence/record_static_reading_scope.zh.txt': ('仅字节身份核对', [], '没有把前人 1017 行阅读清单当成本轮实读。'),
'frozen_source/docs/SELLER_ALIAS_RECORD_ATTRIBUTION.zh.md': ('完整语义阅读', [[1,97]], '归因合同、三臂定义及结果解释边界。'),
'frozen_source/schema/step28_alias_ranking_policy.json': ('完整语义阅读', [[1,80]], '排序项与冻结口径，未启动任何合同测试。'),
'frozen_source/schema/step28_bge_continual_policy.json': ('完整语义阅读', [[1,199]], '冻结训练、模型与记忆配置；不是全项资格复演。'),
'frozen_source/schema/step28_chinese_base_policy.json': ('定向语义阅读', [[1,126]], '输入／模型与训练配置；请求全文输出曾截断，后半未完整审读。'),
'frozen_source/schema/step28_record_attribution_policy.json': ('完整语义阅读', [[1,22]], '真实归因三臂及更新范围。'),
'frozen_source/schema/step28_record_replay_policy.json': ('完整语义阅读', [[1,30]], '底层记录 replay 目标、维度、内存及 chunk。'),
'frozen_source/scripts/step28_alias_ranking.py': ('定向语义阅读', [[89,129]], 'BCE／查询排序／hard-top5 的直接路径；1–185 曾取出，未把全部范围列为逐行复审。'),
'frozen_source/scripts/step28_bge_continual.py': ('定向语义阅读', [[1,135]], '配置、阶段学习率、optimizer 步数及所需入口；后续未变功能未读全。'),
'frozen_source/scripts/step28_chinese_base.py': ('定向语义阅读', [[89,193]], 'BGE loader、record_texts、原监督与排序；余下代码未重审。'),
'frozen_source/scripts/step28_continual_population.py': ('定向语义阅读', [[1,40]], '模型、head 与 AdamW；1–130 曾取出但完整语义审读仅声明直接依赖段。'),
'frozen_source/scripts/step28_continual_population_data.py': ('定向语义阅读', [[66,104],[198,258]], '数据组、记录结构、Algorithm R、序列化与调度；不加载正式群。'),
'frozen_source/scripts/step28_record_attribution_run.py': ('定向语义阅读', [[1,275]], '加载、真实 update、train_stage、阶段缓存／保存／恢复检查、同初始权重与分支；276–630 未全读。'),
'frozen_source/scripts/step28_record_replay.py': ('完整语义阅读', [[1,330]], '本轮主要实现对象；全部函数，A0/A1、目标、VJP、更新、参考、缓存。'),
'input_manifest.json': ('完整结构解析／字节核对', [], '解析全部28载荷成员并逐一核大小与SHA；清单自身不自引用。'),
'request.zh.txt': ('完整语义阅读', [[1,18]], '包内请求；同时使用用户本条完整请求。'),
}
with zipfile.ZipFile(ARCHIVE) as z:
    manifest = json.loads(z.read('input_manifest.json'))
    origins = {e['member']:e['source'] for e in manifest['entries']}
    items=[]
    for name in sorted(z.namelist()):
        data=z.read(name)
        status, ranges, note = scopes[name]
        items.append({'member':name,'project_source':origins.get(name,'package manifest'),
          'bytes':len(data),'sha256':digest(data),'lines':len(data.decode('utf-8-sig').splitlines()),
          'byte_identity_checked':True,'semantic_status':status,'semantic_line_ranges':ranges,
          'notes':note,'executed_as_project_code':False})
    after=[]
    for name in sorted(z.namelist()):
        now=(SRC/name).read_bytes(); orig=z.read(name)
        after.append({'member':name,'bytes':len(now),'sha256':digest(now),'unchanged':now==orig})
assert all(x['unchanged'] for x in after)
assert digest(ARCHIVE.read_bytes()) == '3db153cbfefd597d2a35b6e53bdff5500fecd447a568d8a2ea1345078b92597c'
dump(OUT/'audit/source_integrity_after.json', {'scope':'all 29 extracted members byte-compared to original uploaded ZIP after review', 'all_unchanged':True,'archive_sha256':digest(ARCHIVE.read_bytes()),'members':after})
dump(OUT/'reading_scope.json', {'scope':'this review only; display logs do not prove full semantic review', 'input_members':29, 'project_execution':False, 'items':items})
lines=['# 实际阅读／未读范围','',
'本清单独立于旧审查人的阅读范围。全部29成员做字节身份核对；“全文取出”“逐字节散列”和“语义审读”不等同。`audit/reading_log.jsonl` 是部分取出操作日志，不是全部工具记录或逐行理解的证明。早期全文阅读、后续 sed/nl 定向阅读与本清单合并描述；未读部分不会被说成已核查。','',
'项目源码从未作为模块导入或执行；仅审查者自写的文本／ZIP核对和有理数手工例程序运行。','',
'| 成员 | 行数 | 语义状态／实读行 | 说明 |','|---|---:|---|---|']
for e in items:
    ranges='、'.join(f'{a}–{b}' for a,b in e['semantic_line_ranges']) or '无全文语义行号声明'
    lines.append(f"| `{e['member']}` | {e['lines']} | {e['semantic_status']}；{ranges} | {e['notes']} |")
lines += ['', '## 未做及无法独立重演', '',
'没有模型、正式记录、训练／评价标签、Memory与F/O载荷，也未取外部项目数据。既有Linux原生资格、参数实际位移及已完成结果来自明确标识的旧证据，不是本轮工作容器的实测。本包并非可脱离未附依赖与模型直接重跑的完整项目归档；本轮没有因而要求补齐整个工程。', '',
'`attribution_summary.json` 未作内容回查；没有重新计算三臂指标、bootstrap或任何统计。旧静态全文、旧阅读清单及当前生成器其余功能未递归重审。', '',
'论文的版本、实读小节、未读范围与站点失败见 `references.zh.md`、`references.json` 及 `audit/access_issues.json`。没有运行论文官方代码，没有用摘要或论文标题判数学等价。']
(OUT/'reading_scope.zh.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')

refs=[]
reftext=s.split('## 一手参考文献与本轮使用范围\n',1)[1].strip()
for line in reftext.split('\n\n'):
    match=re.match(r'\[(L\d+)\] (.*)',line,re.S)
    if match:
        urls=re.findall(r'<(https?://[^>]+)>',line)
        refs.append({'id':match[1],'entry_and_reading_scope':match[2],'primary_urls':urls})
assert len(refs)==11
versions={'L1':'author PDF, ICML 2002','L2':'arXiv 1703.06114 HTML','L3':'arXiv 2004.07211 HTML','L4':'2308.13324v1','L5':'2207.01805v1','L6':'2306.16180v3','L7':'2505.10649v1','L8':'ACL EMNLP 2021 proceedings PDF','L9':'author-hosted KDD 2019 PDF','L10':'2508.03081v1','L11':'1805.08289v3'}
for r in refs:
    r['read_date']='2026-10-13'; r['version_used']=versions[r['id']]; r['official_code_run']=False
    r['full_experimental_results_recomputed']=False
# Tool identifiers are in an audit field only; standalone prose uses stable bibliographic URLs.
webrefs={
'L1':['turn316836view0','turn421882view4'],
'L2':['turn421882view3','turn868110view3'],
'L3':['turn316836view1','turn406218view0'],
'L4':['turn317383view2','turn252195view1','turn413688view2'],
'L5':['turn316836view3','turn252195view2','turn198329view4'],
'L6':['turn421882view0','turn868110view1','turn198329view1'],
'L7':['turn421882view1','turn252195view3','turn413688view3'],
'L8':['turn421882view2','turn406218view1','turn406218view3'],
'L9':['turn406218view2','turn775589view2','turn198329view0','turn792468view0'],
'L10':['turn775589view0','turn920880view0','turn886120view3','turn989003view2','turn413688view0'],
'L11':['turn920880view1','turn886120view1'],}
for r in refs:r['audit_web_response_identifiers']=webrefs[r['id']]
dump(OUT/'references.json',{'independent_primary_papers':11,'date':'2026-10-13','exhaustive_literature_claim':False,'references':refs})
(OUT/'references.zh.md').write_text('# 一手文献、实际阅读与未读范围\n\n11篇独立论文；DER、DER++分别核查，但不虚增论文数。以下是外审正文书目的完整副本。仅作有范围的方法比较，不宣称截至2026-10-13的完备文献检索。\n\n'+reftext+'\n\n## 检索筛选但未作方法级判断\n\nCoMEL、AGLR-CL、Replay-free Multi-patches、FEAT和部分2024—2026作者验证线索只在检索标题／摘要层筛选，未据其标题认定与S等价或不等价。未读代码和未访问论文不能支持“没有其他先例”。站点访问问题另列，不隐藏在“已读”条目里。\n',encoding='utf-8')
issues=[
 {'stage':'attachment','kind':'unavailable_parse','detail':'Files.read 对ZIP返回 No readable content；改用已挂载附件解包。不是附件身份失败。','scientific_effect':'none; all payload identities subsequently checked'},
 {'stage':'source_display','kind':'truncated_output','detail':'若干合并容器输出截断，相关目标段单独重新取出；剩余未确认范围在 reading_scope 声明。','scientific_effect':'no claim of full semantic reading of all displayed files'},
 {'stage':'source_diff','kind':'expected_difference','detail':'diff -u 的退出码1表示冻结／活动core有四处接口差异；完整差异保存 core_diff.patch。','scientific_effect':'not a failed test'},
 {'stage':'web','kind':'access_failure','source':'ConSlide CVF ICCV2023 PDF','url':'https://openaccess.thecvf.com/content/ICCV2023/papers/Huang_ConSlide_Asynchronous_Hierarchical_Interaction_Transformer_with_Breakup-Reorganize_Rehearsal_for_Continual_ICCV_2023_paper.pdf','observed':'Internal Error','fallback':'read author arXiv:2308.13324v1; final-version byte equality not verified'},
 {'stage':'web','kind':'access_failure','source':'AKD/PMP CVF CVPR2025 PDF and landing page','url':'https://openaccess.thecvf.com/content/CVPR2025/papers/Li_Advancing_Multiple_Instance_Learning_with_Continual_Learning_for_Whole_Slide_CVPR_2025_paper.pdf','observed':'Internal Error; subsequent find/landing-page access also failed','fallback':'read author arXiv:2505.10649v1'},
 {'stage':'web','kind':'no_literal_match','detail':'LUAR定位词 2 Learning 与 DER定位词 Functional Regularisation 未命中；改读实际章节标题／来源正文。','scientific_effect':'not absence-of-method evidence'},
 {'stage':'web','kind':'unproductive_search','detail':'若干带引号的广泛检索返回词典、手术、软件服务等无关页面；后续改用方法主题与一手域名定位。无关页面未用作科学依据。','scientific_effect':'search not exhaustive'},
 {'stage':'web','kind':'anchor_not_external','detail':'DER参考文献锚点 click 只定位同一页；随后直接打开FDR作者arXiv并读§3.1，未把锚点点击当FDR原文阅读。','scientific_effect':'FDR method independently read'},
 {'stage':'illustrative_code','kind':'execution_summary','attempts':1,'failed_attempts':0,'exit_code':0,'detail':'手工算例首次成功；stdout、空stderr、源码、输入、输出及执行记录均保留。无项目模块导入。'},
]
dump(OUT/'audit/access_issues.json', {'nature':'reviewer-authored event summary, not a verbatim full browser/container session transcript','issues':issues,'hidden_scientific_retries':False,'project_training_runs':0})

(OUT/'input').mkdir(exist_ok=True)
shutil.copyfile(ARCHIVE,OUT/'input'/ARCHIVE.name)
readme='''# 完整 S 贡献定位静态外审交付包

日期：2026-10-13。正文：`review.zh.md`；相同UTF-8内容：`review.zh.txt`。主结论是保留S作为基础与强内部对照、保留任务化最终候选资格，但不宣布现有独立贡献已充分。

## 文件入口

`findings.zh.md` 为逐项发现。`reading_scope.zh.md` / `.json` 逐一列本次29个输入成员的实际语义范围和未读部分。`references.zh.md` / `.json` 列11篇一手论文及实读章节、版本和限制。原输入ZIP的未改副本在 `input/`；`audit/input_verification.json` 包含28载荷逐项大小／SHA核对，`audit/source_integrity_after.json` 证明抽出成员与原ZIP字节仍相同。

`audit/core_diff.patch` 是冻结与活动core的全部差异。`audit/read_excerpts.txt` / `reading_log.jsonl` 是部分源码显示记录，不是全部原始工具转录，也不能独自证明显示的每一行都已语义审读；以阅读清单为准。`audit/access_issues.json` 是审查者整理的失败／未命中／截断情况，不冒称原始浏览器日志。因文件工具未解析ZIP，本次不伪造文件内容citation。

## 独立手工算例

以下命令只运行附包的标准库代码；输入为手工索引与有理数，不含任何正式群、文本、标签、模型或项目模块。所有输出都是有限枚举或代数见证，不是新科研训练结果，也不是项目Linux/GPU原生资格。

在解压目录中使用本机 Python 3.10 或更高版本：

```sh
mkdir reproduced
python analysis/hand_examples.py analysis/hand_examples_inputs.json reproduced > reproduced/stdout.log 2> reproduced/stderr.log
python analysis/verify_input.py "input/review_input(20261013-031814).zip" reproduced/input_verification.json
```

原首次执行的6类代数核查结果已在 `analysis/hand_examples_output.json`，环境、源码/输入散列、退出码和耗时在 `analysis/hand_examples_execution.json`；stdout与stderr在 `logs/`。原执行首次成功，stderr为0字节；没有失败的手工算例重试。再次执行的耗时、Python版本及argv自然可不同，不应当成输出分数变化。

`analysis/read_source.py` 使用本次容器的绝对路径，是保存阅读来源用的原始辅助脚本；普通读者不必运行它。`analysis/finalize_documents.py`、`analysis/package_delivery.py` 同样用于本次交付编排，不属于研究方法或项目实现。

## 完整性与边界

`MANIFEST.json` 和 `MANIFEST.sha256.tsv` 都列出其余交付载荷的相对路径、字节大小与SHA-256；两个清单文件不自列，避免自引用。最终ZIP及两个清单的大小/SHA在ZIP外的交付回执中给出。外部回执不置入已封装ZIP，以避免循环散列。

没有运行项目训练代码、重新统计已有结果、访问项目服务器、读取封存资产、委派辅助审查者或改变原件。旧证据中的实际执行声明保持其形成时间；本次静态结论不新增训练授权。不包含论文全文副本，只提供一手出处及有限范围的分析。
'''
(OUT/'README.zh.md').write_text(readme,encoding='utf-8')

# Small textual QA, unrelated to project execution.
assert '[L12]' not in s and '共 12 项' not in s
assert '同义对手' in s and '非已证错误' in s
assert len(items)==29
assert (OUT/'review.zh.md').read_bytes()==(OUT/'review.zh.txt').read_bytes()
assert not re.search(r'\b(?:TODO|TBD|PLACEHOLDER)\b',s)
print(json.dumps({'status':'PASS','input_members':len(items),'primary_papers':len(refs),'review_bytes':(OUT/'review.zh.md').stat().st_size,'source_members_unchanged':True},ensure_ascii=False))
