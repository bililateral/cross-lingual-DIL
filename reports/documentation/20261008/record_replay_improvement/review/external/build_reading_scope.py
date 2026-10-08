"""Create a conservative, per-member reading ledger; no scientific experiment."""
from pathlib import Path
import csv
import hashlib
import io
import json
import zipfile

ROOT = Path('/workspace/scratch/fa11b6ce03cf')
OUT = ROOT / 'replay_review'
current_zip = ROOT / 'upload/review_input(8).zip'
common_zip = ROOT / 'review_current/background/common_evaluation_input.zip'
review_zip = ROOT / 'review_current/background/common_evaluation_review.zip'
original_zip = ROOT / 'review_background/common/background/original_result_input.zip'
history_zip = ROOT / 'review_background/original/history/implementation_review_input.zip'

readings = {}

def mark(container, names, status, note):
    for name in names:
        readings[(container, name)] = (status, note)


mark('P0', ['new/implementation.zh.txt', 'source/docs/SELLER_ALIAS_REPLAY_IMPROVEMENT.zh.md',
             'source/schema/step28_replay_improvement_policy.json', 'current/AI_RESEARCH_HANDOFF.zh.md',
             'new/authorization.json', 'new/proposal.zh.txt', 'current/AGENTS.md',
             'current/RESEARCH_DISCIPLINE.zh.md', 'request.zh.txt',
             'background/primary_disposition.zh.txt'], '全文', '逐文核阅；首读顺序为 implementation、合同、policy、HANDOFF，随后核对授权与直接依赖。')

full_scripts = ['step28_record_replay.py', 'step28_replay_diagnostics.py', 'step28_replay_improvement_run.py',
                'step28_replay_improvement_verify.py', 'step28_bge_continual.py',
                'step28_bge_continual_evaluate.py', 'step28_alias_ranking.py',
                'step28_alias_calibration.py', 'step28_continual_population_data.py']
mark('P0', ['source/scripts/' + s for s in full_scripts], '全文', '阅读全文，沿本次真实调用路径核对语义；未在网页执行其 Torch/GPU/正式训练路径。')
mark('P0', ['source/scripts/run_step28_replay_improvement_linux_20261008.sh',
             'source/tests/test_step28_replay_improvement.py', 'source/tests/test_step28_record_replay.py'],
     '全文', '完整阅读启动参数或测试实现；网页未执行 shell 启动器或 Torch 单元测试。')
mark('P0', ['source/schema/' + s for s in ['step28_alias_ranking_policy.json', 'step28_bge_continual_policy.json',
                                          'step28_chinese_base_policy.json', 'step28_record_replay_policy.json']],
     '全文', '完整读取 JSON 配置并对照直接调用；配置存在本身不作为执行正确性证明。')

partial = {
    'step28_record_replay_run.py': '1–245、398–486 行；Budget、旧训练/检查点、read_roles、fields/统计；fields 414–426 原函数另由 AST 提取执行。',
    'step28_bge_continual_run.py': '1–155 行；数组、save/RNG、parse_once 与旧训练入口。',
    'step28_er_weight.py': '1–155、218–295 行；实际 update 及相关保存/监督；与旧 LOGIT 冻结来源做函数 AST/差异核对。',
    'step28_chinese_base.py': '1–232、678–700 行；合同、分组、模型加载、通道池化、目标/分数及固定群计数。',
    'step28_continual_population.py': '1–74、157–210 行；模型/Adam 建立与 archive、完整状态保存恢复。',
    'step28_continual_population_run.py': '1–76 行；持久化空间预留/删除辅助；不将其旧 Budget 当作新运行 Budget。',
    'step28_continual_population_evaluate.py': '1–102、210–250 行；群指标、分类/排名指标实际定义。',
    'step28_continual_expression_run.py': '1–130 行；实际公开输入、标签入口。',
    'step28_record_replay_verify.py': '1–75 行；preflight 与被复用的原生前提。',
}
for name, note in partial.items():
    mark('P0', ['source/scripts/' + name], '节选', note)
mark('P0', ['current/RESEARCH_DISCUSSION.zh.md'], '节选', '757–775 行及相关关键词定位；仅本次五臂/诊断最新决定，未递归重审历史讨论。')
mark('P0', ['package_manifest.json'], '机器字段', '完整解析 64 条 path/bytes/sha256 并核对所有外层载荷；不等同 64 件语义全文审查。')
mark('P0', ['background/common_evaluation_input.zip', 'background/common_evaluation_review.zip'], '机器字段',
     '核对压缩包字节数、SHA256、中央目录并解包；内部成员阅读范围分别见 P1/P3。')
for i in range(1, 6):
    mark('P0', [f'evidence/cpu/attempt{i:02}/result.json'], '机器字段',
         '状态、测试计数、累计时间、资源、来源清单等目标字段；attempt05 的 28 个来源大小/SHA 与当前源码核对。')
    mark('P0', [f'evidence/cpu/attempt{i:02}/unittest.txt'], '全文', '完整阅读测试名称、通过/失败与错误栈。')
    mark('P0', [f'evidence/cpu/attempt{i:02}.console.txt'], '全文' if i in (1, 5) else '机器字段',
         'attempt01/05 完整阅读；其余提取外部 wall/RSS 字段。所有五次失败或成功均保留计账。')
mark('P0', ['evidence/cpu/attempt01/failed_test_source.py'], '节选', '133–161 行；首轮失效手写群及训练入口测试附近，追溯每账号仅一条记录造成的 Group.validate 错误。')

mark('P1', ['background/original_result_input.zip'], '机器字段', '核对 9,614,123 字节、335 成员及指定 SHA256；内部逐件范围见 P2。')
mark('P3', ['INCREMENTAL_EVALUATION_REVIEW.zh.txt'], '节选', '1–28、51–76、165–224 行及相关关键词；确认增量评价已关闭、旧结论与原始指标口径，不递归复做所有旧审查。')

mark('P2', ['result/source/docs/SELLER_ALIAS_RECORD_REPLAY_PILOT.zh.md',
             'qualification/report.zh.txt', 'qualification/native_disposition.zh.txt'], '全文',
     '复用旧正确合同及已关闭原生处置；这些是旧记录表实验证据，不是本次新原生结果。')
mark('P2', ['qualification/external/review.zh.txt'], '节选',
     '246–355 行及相关关键词；旧资源门槛 R1/R2 与估计边界，未重审已关闭路线。')
mark('P2', ['history/implementation_review_input.zip'], '机器字段',
     '读取 159 成员中央目录；仅抽取旧 LOGIT 冻结来源和清单，不解读其他历史路线；逐件见 P4。')
mark('P2', ['result/job/completion.json'], '机器字段',
     'status/access/budget，特别 budget.elapsed_seconds；不把旧完成状态解释为新运行完成。')
mark('P2', ['qualification/gpu/attempt01/result.json', 'qualification/native_gate.json'], '机器字段',
     '旧原生状态、时间/资源/资格相关目标字段；不挪作本次原生测量。')
mark('P2', ['result/job/evaluation/collected.json', 'reference/collected.json', 'reference/reference/collected.json'],
     '机器字段', '由真实 historical_replay 的保存角色查找读取组指标登记、路径、大小、SHA 等目标字段。')
mark('P2', ['result/source/schema/step28_record_replay_policy.json'], '机器字段',
     '读取旧参考采集入口及本次继承的策略字段；另与当前配置做字节比较。')

identity = json.loads((OUT / 'calculation/identity.json').read_text())
for item in identity['current_vs_frozen']:
    old_name = 'result/' + item['path']
    if ('P2', old_name) not in readings:
        mark('P2', [old_name], '机器字段',
             '与当前直接依赖作逐字节同源性比较；语义阅读范围以 P0 对应文件为准，未独立重审整个历史文件。')
mark('P2', ['result/source/scripts/step28_record_replay.py'], '节选',
     '与当前文件做差异比对，重点 objective 的原 C/S 分支和 VJP/Adam 调用；当前全文见 P0，未另行重审所有未变历史代码。')

saved = json.loads((OUT / 'calculation/saved_history.json').read_text())
for name in saved['read_matrices']:
    mark('P2', [name], '机器字段',
         '已开放的 60×22 群指标矩阵：大小/SHA/形状校验，实际历史角色入口同输入差和旧 C−S O MAP 配对复算；未读取原始标签/权重/Memory。')
timings = json.loads((OUT / 'calculation/supplementary.json').read_text())['historical_timing']['rows']
for item in timings:
    mark('P2', [item['path']], '机器字段', '仅 JSON 顶层键及 training_seconds；未重做旧更新轨迹。')

prefix = 'background/frozen/logit_low_source/'
mark('P4', [prefix + 'docs/SELLER_ALIAS_LOGIT_LOW.zh.md'], '全文', '核对旧 LOGIT0.1 正式含义和已固定系数。')
mark('P4', [prefix + 'docs/SELLER_ALIAS_BGE_CONTINUAL.zh.md'], '节选', '17–135 行，含 104–125 行校准约定；核对旧 BGE 与校准/评价继承。')
mark('P4', [prefix + 'scripts/step28_er_weight.py'], '节选',
     '实际 update 函数及与当前 er_weight 全文件差异；另机器比较 update AST 相等。整文件字节不相等，不能宣称整件完全未改。')
mark('P4', [prefix + 'scripts/step28_er_weight_run.py'], '节选', '43–166 行；旧 LOGIT 种子、更新和检查点入口。')

containers = [('P0', current_zip, '用户本次上传包'), ('P1', common_zip, '此前增量评价输入包'),
              ('P2', original_zip, '此前原结果包'), ('P3', review_zip, '已结束增量外审包'),
              ('P4', history_zip, '仅为追溯 LOGIT 来源而列出成员的旧实现包')]
rows, identities = [], []
for cid, path, description in containers:
    payload = path.read_bytes()
    with zipfile.ZipFile(io.BytesIO(payload)) as z:
        identities.append({'id': cid, 'description': description, 'local_path': str(path),
                           'bytes': len(payload), 'sha256': hashlib.sha256(payload).hexdigest(),
                           'members': len(z.infolist())})
        for info in z.infolist():
            status, note = readings.get((cid, info.filename), ('未读', '未作本次内容核阅；只列中央目录元数据。P0 的载荷 SHA 核验不改变此阅读分类。'))
            rows.append({'container': cid, 'path': info.filename, 'bytes': info.file_size,
                         'reading': status, 'scope': note,
                         'identity_check': '外层登记载荷大小/SHA 已核验' if cid == 'P0' and info.filename != 'package_manifest.json' else '容器身份已核对；成员身份仅按 scope 所述'})

known = {(r['container'], r['path']) for r in rows}
assert set(readings).issubset(known), sorted(set(readings) - known)
assert len(rows) == 600
counts = {}
for row in rows:
    counts.setdefault(row['container'], {})
    counts[row['container']][row['reading']] = counts[row['container']].get(row['reading'], 0) + 1
data = {'scope_definition': {
    '全文': '全文内容实际核阅，仍不等于执行其代码。',
    '节选': '仅列明行段、函数或差异/关键词；不宣称全文。',
    '机器字段': '结构化目标字段、数组计算、字节比较或容器元数据；不宣称逐文语义阅读。',
    '未读': '未作本次内容核阅；至多目录/容器身份或外层统一哈希检查。'},
    'note': '该台账是当前实例本次工作范围。所有历史层均列件，但未读历史不递归重审。没有项目服务器、正式输入/标签/模型、GPU 或 subagent。',
    'containers': identities, 'counts': counts, 'files': rows}
(OUT / 'READING_SCOPE.json').write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
with (OUT / 'READING_SCOPE.csv').open('w', encoding='utf-8', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader(); w.writerows(rows)
lines = ['逐文件实际阅读范围（UTF-8）', '', data['note'], '', '分类定义：']
lines += [f'{k}：{v}' for k,v in data['scope_definition'].items()]
lines += ['', '容器身份与统计：']
for c in identities:
    lines.append(f"{c['id']} {c['description']}：{c['bytes']} 字节；{c['members']} 成员；SHA256 {c['sha256']}；分类计数 {counts[c['id']]}")
lines += ['', '全部 600 个成员逐件台账（包括未读）：', '']
for i, r in enumerate(rows, 1):
    lines.append(f"{i:03d}. [{r['container']}] {r['path']}\n    {r['reading']}；{r['bytes']} 字节。{r['scope']}\n    身份核验：{r['identity_check']}")
(OUT / 'READING_SCOPE.zh.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')
print(json.dumps({'members': len(rows), 'counts': counts, 'containers': identities}, ensure_ascii=False, indent=2))
