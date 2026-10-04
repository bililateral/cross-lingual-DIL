"""Compose the external review from independently recomputed saved-result evidence.

This is document assembly only. It neither imports project code nor runs a model.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def table(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |",
                      "|" + "|".join("---" for _ in headers) + "|"] +
                     ["| " + " | ".join(map(str, row)) + " |" for row in rows])


def number(value, signed=False):
    return (f"{value:+.6f}" if signed else f"{value:.6f}").replace("-", "−")


def interval(value):
    lo, hi = value["conditional_95pct_interval"]
    return f"{number(value['mean'], True)} [{number(lo)}, {number(hi)}]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--evidence", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    ev, out = args.evidence.resolve(), args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    stats = read(ev / "statistics/run_v1/independent_statistics.json")
    stat_check = read(ev / "statistics/run_v1/machine_verdict.json")
    count_check = read(ev / "counts/run_v2_single_cpu/machine_verdict.json")
    logit = read(ev / "logit_path/run_v1/result.json")
    risk = read(ev / "risk_path/attempt_v1/result.json")
    provenance = read(ev / "provenance/run_v1/result.json")
    diagnostic = read(ev / "diagnostics/results_v1.json")
    identity = read(ev / "attachment_identity.json")
    with (ev / "counts/workpoint_summary/workpoint_summary.csv").open(encoding="utf-8", newline="") as f:
        workpoints = list(csv.DictReader(f))
    names = {"seq": "SEQ", "tenth": "ER0.1", "logit_quarter": "LOGIT0.25",
             "logit_tenth": "LOGIT0.1", "risk": "risk"}
    methods = ["seq", "tenth", "logit_quarter", "logit_tenth", "risk"]
    comparisons = {}
    for study in ("logit_low", "risk"):
        comparisons.update(stats[study]["comparisons"])
    compare_names = {
        "logit_tenth_minus_tenth": "LOGIT0.1−ER0.1",
        "logit_tenth_minus_logit_quarter": "LOGIT0.1−LOGIT0.25",
        "logit_tenth_minus_seq": "LOGIT0.1−SEQ",
        "risk_minus_logit_tenth": "risk−LOGIT0.1",
        "risk_minus_tenth": "risk−ER0.1",
        "risk_minus_seq": "risk−SEQ",
        "risk_minus_logit_quarter": "risk−LOGIT0.25",
    }
    abs_results = stats["risk"]["absolute_endpoints"]
    core_absolute = table(
        ["方法", "O MAP", "O R@5", "N MAP", "N R@5", "Z MAP", "Z R@5", "F_first MAP↓", "final_all MAP"],
        [[names[m], *[number(abs_results[m]["primary"][ep][metric]["mean"]) for ep, metric in
                      [("O", "map"), ("O", "recall_at_5"), ("N", "map"), ("N", "recall_at_5"),
                       ("Z", "map"), ("Z", "recall_at_5"), ("F_first", "map"), ("final_all", "map")]]]
         for m in methods])
    core_comparisons = table(
        ["固定比较", "ΔO MAP [95%区间]", "ΔN MAP [95%区间]", "ΔZ MAP [95%区间]", "原23项"],
        [[compare_names[k], *[interval(v["all_role_differences"]["primary"][ep]["map"]) for ep in ("O", "N", "Z")],
          f"{v['passed']}/23"] for k, v in comparisons.items()])
    other_endpoints = table(
        ["固定比较", "ΔF_first MAP↓", "ΔF MAP↓", "ΔG MAP↑", "Δfinal_all MAP↑"],
        [[compare_names[k], *[interval(v["all_role_differences"]["primary"][ep]["map"])
                             for ep in ("F_first", "F", "G", "final_all")]] for k, v in comparisons.items()])
    checks_labels = [
        ("old_map_improves", "O MAP差>0"),
        ("old_map_interval_above_zero", "O MAP差的95%下界>0"),
        ("old_recall5_improves", "O R@5差>0"),
        ("new_map_non_decrease", "N MAP差≥0"),
        ("new_recall5_non_decrease", "N R@5差≥0"),
        ("O_average_precision_non_degradation", "O AP差≥0"),
        ("O_roc_auc_non_degradation", "O ROC-AUC差≥0"),
        ("O_brier_non_degradation", "O Brier差≤0"),
        ("O_log_loss_non_degradation", "O log-loss差≤0"),
        ("O_brier_against_raw_reference", "O Brier≤对手raw Brier"),
        ("O_log_loss_against_raw_reference", "O log-loss≤对手raw log-loss"),
        ("N_average_precision_non_degradation", "N AP差≥0"),
        ("N_roc_auc_non_degradation", "N ROC-AUC差≥0"),
        ("N_brier_non_degradation", "N Brier差≤0"),
        ("N_log_loss_non_degradation", "N log-loss差≤0"),
        ("N_brier_against_raw_reference", "N Brier≤对手raw Brier"),
        ("N_log_loss_against_raw_reference", "N log-loss≤对手raw log-loss"),
        ("Z_map_non_degradation", "Z MAP差≥0"),
        ("Z_recall_at_5_non_degradation", "Z R@5差≥0"),
        ("Z_average_precision_non_degradation", "Z AP差≥0"),
        ("Z_roc_auc_non_degradation", "Z ROC-AUC差≥0"),
        ("Z_brier_non_degradation", "Z Brier差≤0"),
        ("Z_log_loss_non_degradation", "Z log-loss差≤0"),
    ]
    check_keys = list(compare_names)
    assert all(set(v["checks"]) == {k for k, _ in checks_labels} for v in comparisons.values())
    checks_table = table(
        ["项", "原判据", "L−E", "L−Q", "L−S", "R−L", "R−E", "R−S", "R−Q"],
        [[str(i), label, *["通过" if comparisons[k]["checks"][key] else "未过" for k in check_keys]]
         for i, (key, label) in enumerate(checks_labels, 1)])
    seq_difference = comparisons["logit_tenth_minus_seq"]["all_role_differences"]["primary"]
    failed_logit = table(["未过项", "观察差", "条件95%区间"], [
        [f"{ep} {label}", number(seq_difference[ep][metric]["mean"], True),
         "[" + ", ".join(number(x) for x in seq_difference[ep][metric]["conditional_95pct_interval"]) + "]"]
        for ep, metric, label in [("N", "average_precision", "AP"), ("Z", "map", "MAP"),
                                  ("Z", "average_precision", "AP")]])
    probability_table = table(["方法", "O AP", "O ROC-AUC", "O Brier↓", "O log-loss↓", "N AP", "Z AP"], [
        [names[m], *[number(abs_results[m]["primary"][ep][metric]["mean"]) for ep, metric in
                      [("O", "average_precision"), ("O", "roc_auc"), ("O", "brier"), ("O", "log_loss"),
                       ("N", "average_precision"), ("Z", "average_precision")]]]
        for m in methods])
    workpoint_table = table(["方法", "端点", "TP", "FP", "FN", "TN", "Precision", "Recall", "F1", "FPR"], [
        [names[w["method"]], w["endpoint"], *[w[k] for k in ("tp", "fp", "fn", "tn")],
         f"{float(w['precision'])*100:.4f}%", f"{float(w['recall'])*100:.4f}%",
         number(float(w["f1"])), f"{float(w['fpr'])*100:.6f}%"]
        for w in workpoints if w["method"] in ("logit_tenth", "risk")])
    full_workpoints = table(["方法", "端点", "TP", "FP", "FN", "TN", "Precision", "Recall", "F1", "FPR"], [
        [names[w["method"]], w["endpoint"], *[w[k] for k in ("tp", "fp", "fn", "tn")],
         f"{float(w['precision'])*100:.4f}%", f"{float(w['recall'])*100:.4f}%",
         number(float(w["f1"])), f"{float(w['fpr'])*100:.6f}%"] for w in workpoints])
    now = datetime.now(timezone.utc).isoformat()
    parts = []
    parts.append("""# LOGIT0.1与risk固定开发点：独立科研结果外审全文

日期：2026-10-04，日期与正式作业时刻按Asia/Shanghai（UTC+8）解释。

## 一、明确裁决

本次外审接受两项固定开发运行在可审证据范围内的有效性，并接受联合报告的主要科学结论。没有发现已证实的科学阻断，也没有发现影响本轮交付的复现缺陷。

LOGIT0.1相对匹配ER0.1、LOGIT0.25分别通过原23/23项，相对SEQ通过20/23项。它建立了明确的旧域候选排序增益，并保留新域/最终最新域的三个观察保护未满足这一事实。risk相对LOGIT0.1、ER0.1、SEQ、LOGIT0.25分别为4/23、5/23、7/23、4/23；它是有效运行得到的固定点负结果，不能作为本配置改进新旧能力平衡的证据。[R1–R6]

本裁决完成的是本次独立结果外审。原验收逐项结果保持不变；本回复不替主执行者写入其后续主审已关闭，不作自动模型替换，也不启动另一项实验。创新、独立最终确认和真实市场外推均未由本次结果建立。

### 模型身份分别记录

| 项目 | 本轮实际可知的信息 |
|---|---|
| 用户请求型号 | 网页GPT 6 Pro |
| 本轮网页可见选择 | 当前审查工具没有提供可独立核对的模型选择器截图或读数；不能确认。历史报告中的6/Pro不能代替本轮证据。 |
| 审查者自述／会话配置说明 | ChatGPT；会话配置说明为GPT-6 Astra Pro。此项属于配置/自述信息。 |
| 独立后台身份认证 | 没有独立可验证的后台模型、路由或服务标识，不能认证请求型号与实际后端相同。 |

我没有把来源标签、模型自述或既有外审PASS当作科学计算的替代证据。

## 二、附件身份、实际审读范围与来源分离

实际收到的review(5).zip为9,639,622字节、729个成员，SHA-256为：

0d5e45250361558d6b2d66bca8815103e6f0657eeab9fb9d63a4bbbec50c5053

三项均与用户给定值完全一致，ZIP CRC检查无失败成员。内部MANIFEST列728个载荷，加清单本身共729文件；逐项大小与SHA一致，没有未列、缺失或重复路径。用户本次明确授权上传，已取代包内历史“待上传授权”快照；未重复请求该授权。[R1、attachment_identity.json]

首先阅读了联合结果报告、LOGIT0.1和risk固定合同，并阅读继承的BGE合同中与输入、记忆、优化、校准、七端点和23项直接相关的规定。随后分别审读两套实际source的policy、更新函数、阶段/正式入口、恢复、盲门、对照接入和评价代码；直接复用已关闭实现外审/主审和CPU原生梯度证据。对未变父组件，核对适用来源与直接证据，不递归重审无关历史。每个分项子报告列出人工审读路径/行段；全量散列核对不冒充对所有历史行重新做语义审查。[S1–S7]

| 来源对象 | LOGIT0.1 | risk |
|---|---:|---:|
| 实际冻结科学来源 | 37份 | 32份 |
| 主返回inventory载荷 | 272份 | 313份 |
| 保存结果核验inventory载荷 | 228份 | 265份 |
| 新矩阵/计数集 | 18套 | 18套 |
| 复用矩阵/计数集 | 63套 | 81套 |

两来源树有22个同路径文件，其中19个相同、3个不同：step28_er_weight.py、step28_er_weight_run.py、step28_er_weight_evaluate.py。审查始终使用各作业自己的source；根目录两份audit脚本只作为新增保存结果核验来源。冻结source与authorization、execution、startup、manifest、collected、evaluation及适用CPU审计的来源链一致，显式本地导入闭包也被各自来源表覆盖。2306次大小/SHA比对包含重复绑定核对，不是2306个独立科学检验。[R1]

risk的LOGIT0.1对手在结果形成前已固定具体job、execution与policy。完成后绑定execution.json、run/manifest.json、run/partition.json、evaluation/collected.json、evaluation/evaluation.json五文件；它们与本附件LOGIT0.1实际job一致。63/81套复用矩阵及计数都能追溯到原collection的同一point、role、group/column身份，没有按效果更换对手。[S6、R1、R3]

## 三、实际目标和梯度接入

定义完整基础群目标B(g)=BCE+query_rank+0.5×known_top5_hard。BCE按378无向对均值，rank按28查询均值，hard沿用已冻结的查询/正例归约。LOGIT0.1执行：

L_LOGIT = B(current) + 0.1 B(history)
          + 0.5/378 × Σ_pair (z_history − z_reference)^2。

因此历史hard的有效权重是0.05；MSE独立为0.5，不再乘0.1，也不整体除以1.1。冻结代码中MSE进入历史objective之后才backward；目标数组无梯度。当前、历史各一次train前向并各反传一次，累加后统一clip和一次AdamW，编码器与评分头都进入训练。正式调用明确传入该分支的存活target及0.1/0.5。[S5：update 212–278；runner 50–96]

risk执行：

L_risk = B(current) + 0.1 B(history)
         + 0.5 × (D_rank + D_positive + D_negative)/3。

对查询q、正候选数m，三个风险分别为：全27候选logsumexp减正候选logit均值；正候选softplus(−z)均值；负候选softplus(z)均值。每群按m=1的16查询和m=2的12查询分层，三个通道分别排序。每通道D为两层全部查询的[当前有序风险−冻结有序参考]_+²之和除28，故按查询均匀权重16/28与12/28，不能误解为两层各占一半。[S6：step28_risk_replay.py 19–80]

当前风险与torch.sort(...).values保留计算图，参考以常量张量进入；保持项在history backward之前加到objective中。首更新隐藏头诊断使用同一图的autograd.grad，不替代正式反传、不新增历史前向。实际正式runner进入risk.update，最后统一裁剪及一次optimizer.step。这里没有发现“只有配置/日志、没有实际保持项”的矛盾。[S6：risk_replay 148–223；R3]

梯度有效性的依据分层处理：旧Linux原生手写证据已在编码器query权重和隐藏头上验证保持梯度增量及不同参数更新；原外审独立导数/Adam及变异检验已关闭未变机制。本轮核对其来源适用性、正式实际调用与所有保存更新分解。原生探针只涵盖说明的组件，且计数/参考激活夹具有明确标识，不将其当成正式288步或全编码器逐参数证书。网页此次没有重新加载BGE或复跑原生梯度。[S7、R2、R3]

从两项各1728行更新保存值独立重组目标：LOGIT完整total与记录完全一致；risk按三个保存float32通道重新取均值后重组total的最大差约8.79×10⁻⁹。基础BCE+rank+0.5hard分解最大差3.2782554626464844×10⁻⁷，符合原3×10⁻⁶容差。该容差仅用于浮点实现复核，不用于放宽成功线。两项保持项在全部1728步均为正；仅凭非零日志不够，故同时保留上述实际代码与已关闭原生梯度链。[R2、R3、R6]

## 四、共同起点、六群记忆、日程和完整盲门

三个顺序各从原共同首域完整model/Adam/RNG及首校准映射恢复，Adam计数为288。首域全60群盲分数恢复精确回放的正式记录、起点描述符及来源一致；LOGIT采用原LOGIT首缓存，risk采用原ER六群缓存并仅给这六群生成当前共享模型的eval风险参考。没有从ER0.1或LOGIT0.25末端模型续训的证据。[S5、S6、R2、R3]

本页按保存UID及种子独立重建Algorithm R、当前6轮群日程、历史逐步draw与随机流。六个新阶段均为48当前拟合群各6次、288历史draw，与原ER逐步一致；当前dropout流与历史dropout流分离。阶段局部学习率重新预热29步/衰减，头为0.001；Adam结束为576/864。第1、29、30、288步观察完整，末步编码器学习率0、参数不变而头继续更新，符合合同，不是漏训。[R2、R3]

阶段二末仅为本路径新入缓存群生成来源阶段eval参考；存活旧参考摘要不刷新、淘汰群不恢复。实际供给如下：

| 顺序 | 首域存活群 | 第二域新存活群 | 判读 |
|---|---:|---:|---|
| ABC | 4 | 2 | 按原固定随机供给 |
| BCA | 2 | 4 | 按原固定随机供给 |
| CAB | 0 | 6 | 首域恰无存活群；原机制不按域平衡，并与各对照配对 |

CAB的事实不能据结果触发重抽、补群或另设成功线。它说明“六群”不保证每个旧域都被直接重放，不是本轮已证实的采样错误。[R2、R3]

保存记录中含全部附属状态的最大记忆为LOGIT260,866字节、risk227,733字节，均低于1,048,576字节。核对了计量代码、成员/来源摘要及更新后的状态记录；因为真实序列化缓存正文不在附件，未在网页重新反序列化或重新计量原payload。模型和单份Adam训练状态按原合同另计，归档权重不是学习器免费教师。[S4、R2、R3]

两项完整盲门都要求全部1728更新、六个原生终点恢复核验和24个完整分数数组完成，之后才parse_once(valid)。before_valid的账为train=1、valid=0、heldout=0、owners=0；access/completion为train=1、valid=1、heldout=0、owners=0，来源在结束前再核对。校准每次只用刚到达域12群/4536对/240正，对所有域使用一个正斜率映射；旧校准群不再次拟合。[S5、S6、S8、R2、R3、R5]

24数组=每个终点的12×378校准raw分数、60×378 valid raw分数、60×378 stage-cal分数及60×378 first-cal分数，共六终点。前两类是float32，后两类是float64。每作业18套新评价矩阵来自六终点×三个保存角色；分数数组和指标矩阵不是同一种产物。[R5]

## 五、实际完成、预算、监听与保管

| 项目 | LOGIT0.1 | risk |
|---|---|---|
| 开始（UTC+8） | 10月3日16:01:03 | 10月3日21:23:46 |
| 完成（UTC+8） | 10月3日21:23:48 | 10月4日02:43:55 |
| 时间戳差 | 5小时22分45秒 | 5小时20分09秒 |
| 程序内耗时 | 19,363.136874秒 | 19,206.534378秒 |
| 训练/评价包装退出 | 0 | 0 |
| 新更新/群梯度呈现 | 1728/3456 | 1728/3456 |
| 最大观察作业产物 | 10,447,066,320字节 | 10,446,984,031字节 |
| 原硬限 | 12小时、16GiB | 12小时、16GiB（等待另计） |
| 六权重描述符合计 | 7,838,879,856字节 | 7,838,862,246字节 |

两项均无正式failure.json，完成记录与指标文件绑定一致。上述是附件可核对的正式记录，不是本网页连接服务器作出的新观察。[S8、S10、R2、R3]

risk监听52次资源采样，等待3064秒（51分04秒）后启动；启动样本满足原GPU/主存/磁盘/CPU门槛。21:29:31确认已落盘真实updates后自删；训练和监听最终退出0，cleanup_exit=0，无自动重试记录。旧外审发现的快退出漏自删、rm失败仍写成功回执两项监听问题已有保留原件与修正，当前实际记录与修正版SHA相符。[S7、S10、R3]

risk开始比LOGIT包装完成早2秒；资源门记录与已固定对照身份一致，不能仅由这2秒差断定重复启动。LOGIT0.1在risk统计前已完成，实际没有触发“候选完整矩阵保存后等待对照、另行finalize恢复”的故障分支。这个分支的已审实现与本次实际走过的路径分开记录。[R3]

权重边界尤其需要准确：保存结果audit中的native_weight_hashes_match核对的是Linux提供的payload_custody哈希与endpoint描述符，没有读取原生权重。此次网页也只核对这些记录；没有重新散列、载入或恢复12份模型。verification/execution.json为执行后整理的展开argv，LOGIT command.txt为已披露简写，不能把这些回执当成独立后台执行认证。[S10、R1]

## 六、独立评价计算方法与数值核验

新的统计参考没有导入提交者的audit、评价、训练模块或Torch。两作业合计按路径核对180套矩阵（81+99，包含跨作业复用），每套60×22 float64，群ID、实际域、列顺序一致且数值有限。按独特字节内容计96份，不能把180个槽位当成180批独立样本。原始三角色之外的primary是按冻结规则拼合评价列：曲线/排序取raw，概率/分类取stage-cal，没有第四套重新收集结果。[R4、R5]

参考用PCG64(20260930)独立生成(5000,3,20)整群抽样索引，与两个保存draws逐元素一致。第二轴固定实际域A/B/C；同一实际域的索引跨方法、阶段、顺序和角色复用。每次先直接索引20个抽中的完整群，再算实际域均值、顺序内端点、最后三个顺序等权均值。另用群出现次数/20乘原矩阵交叉检查；区间用线性分位与排序后显式邻项插值相互核对。[R4]

令R(t,j)为给定方法/顺序在学完阶段t后、到达位置j的域宏指标，各表达式外再对三顺序等权平均：

| 端点 | MAP等越大越好指标的公式 |
|---|---|
| O | [R(3,1)+R(3,2)]/2 |
| N | [R(2,2)+R(3,3)]/2 |
| Z | R(3,3) |
| F_first | R(1,1)−R(3,1) |
| F | {[R(1,1)−R(3,1)]+[R(2,2)−R(3,2)]}/2 |
| G | {[R(2,2)−R(1,2)]+[R(3,3)−R(2,3)]}/2 |
| final_all | [R(3,1)+R(3,2)+R(3,3)]/3 |

Brier/log-loss在F_first、F、G中反号，使遗忘量正值仍表示变差、学习增益正值仍表示改善。final_all=(2O+Z)/3在点值与每次bootstrap都核对。七端点×22指标、原三个角色及派生primary、所有方法、七组比较与逐顺序结果均重算；没有只核对摘要MAP。[S4、R4]

| 本网页独立核验 | 最大绝对差 |
|---|---:|
| 端点/区间/逐顺序与保存结果（56,364个数值） | 6.38378239159465×10⁻¹⁶ |
| 直接索引与出现频数法 | 7.771561172376096×10⁻¹⁶ |
| 线性分位与显式插值 | 1.1102230246251565×10⁻¹⁶ |
| 逐群六分类列、合并混淆整数及其率 | 0 |
| 全体/分域22指标宏均值 | 5.551115123125783×10⁻¹⁶ |
| 7,128+8,910行阶段CSV宏指标 | 2.220446049250313×10⁻¹⁶ |
| 24个校准仿射数组与保存值 | 0 |

数值比较数量是覆盖说明，不是显著性检验次数。所有方向判据仍直接用>0、≥0或≤0，没有加入数值容差。AP、梯形PR-AUC、查询MAP、MRR分别保留原定义，不混称“准确率”。[R4、R5]

从10,800行保存混淆整数重算precision、recall、F1、specificity、balanced accuracy、MCC；每群20正/358负，先加整数后计算合并率。36套新角色逐群score≥0的数量精确等于TP+FP。校准数组保存a·z+b这一logit，因此阈值0就是概率0.5；全部正仿射和精确并列保序均实际核对。[R5]

还利用保存raw NLL、正例总数和盲分数验证校准NLL及梯度：对n个分数z，Σyz=Σsoftplus(z)−n·NLL_raw，而NLL(a,b)=[Σsoftplus(az+b)−aΣyz−bΣy]/n。该恒等式及其导数核对了保存轨迹，误差约10⁻¹⁶；它以原NLL为条件，不能声称从真值重算原始NLL或重新拟合了校准器。[R5]

这些区间条件于现有固定模型、生成数据、一个s0、三个顺序和固定校准；只反映群抽样不确定性，不覆盖重新训练、重新校准或历史适应性开发。三顺序使用同60群，不是三个种子或三份独立评价集。[S4、R4]

## 七、核心绝对结果与七组固定比较

下表数值为0—1尺度。O/N/Z分别为最终旧域、刚学完的新域、最终最新域；F_first是遗忘量，越小越好。所有值从本次独立参考输出生成。[R4]
""")
    parts.append(core_absolute)
    parts.append("""
LOGIT0.1在所列五个固定配置中的O MAP、N MAP和final_all MAP观察均值最高，Z MAP仍低于SEQ。这是本批开发结果的有限比较，不是全局最优、生产可用性或创新认证。

### O/N/Z MAP及原验收

差值一律为候选减对手，区间为条件95%；乘100才是百分点。
""")
    parts.append(core_comparisons)
    parts.append("""
### 其余四端点的MAP结果

F_first和F的负差表示减少遗忘；G和final_all的正差表示改善。以下同样完整保留跨零与负结果。[R4]
""")
    parts.append(other_endpoints)
    parts.append("""
## 八、全部原23项：没有修改方向或成功线

L=LOGIT0.1，E=ER0.1，Q=LOGIT0.25，S=SEQ，R=risk。“差”均为左侧减右侧。概率损失用候选stage-cal；明确标为“对手raw”的四项另按原定义比较。通过/未过由未四舍五入数值决定。[R4]
""")
    parts.append(checks_table)
    parts.append("""
23项混合了观察值保护和一项区间要求；不是23个独立显著性检验，也不是全族同时显著性或总体非劣效证书。正向顺序数单独核对，未来三种子要求未被当前s0结果替代。

### LOGIT0.1相对SEQ的三项未过
""")
    parts.append(failed_logit)
    parts.append("""
三项95%区间都跨零。正确判读是：原零观察退化保护没有全部满足，但这三项并没有分别建立总体退化；同时也没有证明总体非劣效。N MAP的+0.000093微正差同样不能证明不损害新学习。Z R@5却增加0.015179，[0.000893,0.029762]，说明固定K的覆盖与MAP可以给出不同方向，不能混为一项。

LOGIT0.1相对SEQ的O MAP增益+0.044203有正向区间，ΔF_first=−0.043560，区间[−0.058671,−0.027082]。这里“减少0.043560”对应遗忘量差值为负，避免把正的减少幅度与负差值区间写成同一符号量。[R4]

### risk的负结果范围

risk相对LOGIT0.1的O/N/Z MAP条件区间都低于0，且三个顺序O、N观察差均负；Z在ABC为+0.001047，在BCA/CAB为负，平均结论不等于每个顺序都下降。ΔF_first、ΔF均为正且区间为正，ΔG与Δfinal_all均为负且区间为负。因而相对主要强对照，问题同时涉及旧能力损失和新学习不足，不能描述为更好平衡。[R4]

risk相对ER0.1的O MAP差−0.005080区间跨零，N/Z的负差区间不跨零。可以认定本点新群体学习有负向证据、旧域保持没有建立增益；不能把O总体下降说成已确定。risk相对SEQ的O观察值虽为+0.006029，区间跨零，N/Z则有负向证据。上述固定比较不相互替代。

risk对两个LOGIT仅通过O/N的Brier和log-loss相对对手raw输出的四项参照保护；对ER0.1额外通过O R@5，对SEQ额外通过O MAP点值、O AP、O Brier。这些项不能替代O MAP区间、新域保护或主要比较。此固定点不达目标是有效负结果，既不是无证据推定实现错误，也不是整个风险保持方法家族已被否定。

## 九、基础识别、概率质量与固定工作点

以下主概率列使用stage-cal，AP与ROC使用预定raw角色；正仿射下相关排序列已验证完全相同。[R4、R5]
""")
    parts.append(probability_table)
    parts.append("""
risk相对LOGIT0.1在O/N/Z的AP、ROC-AUC、Brier、log-loss观察保护均未满足。这支持“原基础识别/概率保护未实现”，不能进一步省略工作点结果。

固定stage-cal概率0.5的合并整数和工作点如下。O/N各有120个群-模型呈现，来自同一60群的重复使用；Z有60个呈现。它们不是新增独立样本，也没有形成新的验收标准。[R5]
""")
    parts.append(workpoint_table)
    parts.append("""
risk在这些固定工作点更保守：相对LOGIT0.1，误报率较低、precision较高，同时recall与F1较低。因此不能写“risk的每个识别指标都下降”。LOGIT0.1的23/23也不能解释为固定工作点FPR不增或自动同控判定合格；固定阈值FPR并不在这23项的直接保护清单中。

本研究的实际用途仍是马甲候选检索。MAP/MRR/Recall@K/NDCG@K与该用途直接相关；不把早期自动判定门重新移作本轮候选检索资格门，不根据这批结果选择另一个阈值或放宽成功线。实际候选规模和人工复核预算没有在本轮得到新的业务确认。

## 十、预定风险诊断：线索成立，因果归因未建立

联合报告第7节的诊断范围与原始保存记录一致。六阶段eval自比较均为[0,0,0]；同一未更新模型、每阶段一个固定存活群、四个固定train seed，总24次模式比较得到：

| 通道 | train相对同模型eval参考的惩罚范围 |
|---|---|
| rank | 0.0128307715—0.1047988385 |
| positive | 0.0062768841—0.0746508539 |
| negative | 5.28451949×10⁻⁸—1.13378892×10⁻⁵ |

这证明没有参数更新时也可产生保持惩罚，不能把正式训练中全部正惩罚解释为学习新域造成的遗忘。它只覆盖六个阶段各一群/四seed，未测全缓存噪声分布，不是因果消融。[R6]

六阶段首更新隐藏头weight上的加权分量范数为：

| 分量 | 范围 |
|---|---|
| 0.1×完整历史监督 | 0.0096834656—0.0164486449 |
| 0.5/3×D_rank | 0—0.0112479618 |
| 0.5/3×D_positive | 0—0.0228851512 |
| 0.5/3×D_negative | 1.92383297×10⁻⁷—3.53050541×10⁻⁶ |

negative在这些有限观测中很弱；ABC阶段二和BCA阶段二positive范数分别超过对应加权历史监督范数。该测量只覆盖每阶段第一步隐藏头，不是全编码器；没有梯度夹角，不能把分量范数相加当合成范数。模式诊断选择缓存第一个群，首更新梯度来自首次draw到的历史群，仅ABC阶段三两者相同，其余五阶段不同，因此两表也不是同群配对因果实验。[R6]

risk各阶段0.5倍保持项均值约0.005906—0.013722，LOGIT的0.5MSE约0.069328—0.130222；同为0.5系数不等于同约束强度，标量损失大小也不能直接等同梯度或更新强弱。两个方法各1728步的合成裁剪前norm都>1：risk约5.727—34.569，LOGIT约6.488—26.216。这是既定裁剪发生的证据，不是梯度冲突证明。[R2、R3、R6]

分别排序的边际风险不保留查询对应关系或通道间联合结构；前轮合法反例已证明缓存代理风险改善仍可出现MAP下降与FPR上升。该逻辑结论复用，不重新扩展文献检索。本次负结果并未证明分布排序、模式噪声、通道尺度或有限缓存中的哪一个是主因，也不支持“缓存已经保持很好，仅仅泛化不好”的说法：正式日志来自draw群的train模式，并非所有缓存阶段末eval独立评价。[S7、R6]

## 十一、研究问题、创新、外推和问题分类

本轮确实回答了限定的研究问题：在中文合成三域、不同新卖家群体分批到来、最多六完整群及附属状态≤1MiB、同一s0和三个固定顺序的条件下，比较保存结果中的旧分布未见卖家排序/识别与新群体学习。LOGIT0.1在当前固定对照里形成更强的局部基线，risk本固定配置未提供更好平衡。它没有研究同一控制者跨阶段新增马甲追踪，也没有评估真实市场全库检索。[S1–S4、R4]

范围限制继续保持：每群28账号、每查询27候选且有正候选；基础配置已经经过联合三域与valid开发，未来域不对研究设计者完全未知。ABC/BCA/CAB只覆盖三个循环顺序，不等于三个训练种子；已用基础test也不能重新称未见最终留出。没有大候选库、无正候选查询、真实市场或跨种子稳定性证据。平均保护不等于逐域、逐查询或逐更新保证。

创新判断不改变：LOGIT是已知输出保持思想的匹配适配；risk单向分位平方泛函已有直接先例，代理不保证目标指标的前轮结论保持。本次效果不能认证实质方法创新，单点失败也不否定整个家族。无需为接纳有效负结果而重新查一轮未变化文献。

| 分类 | 本次裁决与处置 |
|---|---|
| SCIENTIFIC_BLOCKER（已证实科学阻断） | 未发现。主要统计、方向、来源与实际接入一致；不以risk未达目标反推实现无效。 |
| REPRODUCIBILITY_DEFECT（影响本轮交付的复现缺陷） | 未发现。原服务器绝对路径/CPU编号是已披露布局，网页参考使用本地附件根和当前允许CPU，不把环境重定位当缺陷。 |
| 报告解释补充 | 明示固定工作点FPR/召回取舍；模式与首更新探针通常不同群；权重“hash匹配”只核对保管记录/描述符。当前联合报告主结论不因此反转。 |
| 后续研究建议 | 若未来另行授权重新设计，首先说明相对LOGIT0.1的任务收益和机制必要性；保持原始正负结果，再前瞻确定验证范围。当前结果不支持直接换γ继续搜索或改成功线。 |
| OUT_OF_SCOPE_OVERDESIGN | 为本轮现有结果追加全编码器梯度归档、全面恢复/监控框架或无具体矛盾的全历史重审，均不作为关闭条件。新的训练、消融、种子和数据不在本授权内。 |

不要求盲目重训，不重新开真值，不事后修改原验收。必要的工作点/诊断措辞已在本报告中补充，不修改冻结原报告、代码或正式结果字节。

## 十二、未验证边界与证据交付

本页实际做了：逐项附件/来源/对照绑定、保存UID日程与损失分解、全部矩阵归约、独立整群bootstrap与23项、混淆整数重建、分数仿射/工作点和条件NLL恒等式核验。参考使用本页Python3.12.14/NumPy2.3.5及标准库；数值参考单线程，允许CPU为0–8，主要数值运行固定CPU0。部分纯标准库身份/日志脚本记录了允许CPU集合，不冒称其亲和性都相同。没有安装额外依赖、导入训练/Torch或调用服务器。

明确没有做：重新读取正式train/valid真值；重新从逐对真值计算AP、PR-AUC、MAP、MRR、NDCG或原始Brier；重新拆分原始逐对TP/FP；从真实文本重生成盲分数；重新载入model/Adam/RNG；重新反序列化训练缓存；重新计算原权重文件的SHA；新训练、消融、参数搜索、test/owners或私有身份资产访问。保存结果真实性的这些原生环节由冻结代码、直接已关闭证据和正式运行记录共同支持，不能说本页完整重演了原生训练。

新增独立参考计算没有失败版本，原始stdout/stderr/退出码及全部输入身份保留；counts首次成功运行与随后固定CPU0的成功运行都保留。统计/来源探索时猜错局部路径的只读错误另记录为审查端探索错误，不冒充项目缺陷，也不伪造当时没有归档的原始进程日志。一次协作消息抄写Z MAP末位笔误已更正，程序生成的报告/JSON未受影响。原包中的既有失败、修正及主审原件以原字节一并保存。

交付包含本中文全文external_review.zh.txt、统一机器判定external_verdict.json和external_review_evidence.zip。ZIP内original/review.zip是收到的上传文件原样副本，保留用户给定SHA；review_evidence/含六个分项的独立源码、实际输入身份、输出、日志、机器判定、完整22指标/七端点/逐顺序CSV和bootstrap索引/分布。REPRODUCE.zh.txt提供只针对保存结果的重放入口；MANIFEST.json覆盖证据ZIP自身以外的全部成员，清单不自哈希。

### 原始来源索引

以下S路径均相对于original/review.zip解包根。L=reports/seller_alias_continual/20261004/logit_low_result；R=reports/seller_alias_continual/20261004/risk_result。

[S1] docs/SELLER_ALIAS_LOGIT_RISK_RESULT.zh.md；reports/seller_alias_continual/20261004/result_review/request.zh.md。
[S2] docs/SELLER_ALIAS_LOGIT_LOW.zh.md；L/source/docs/SELLER_ALIAS_LOGIT_LOW.zh.md；L/source/schema/step28_logit_low_policy.json。
[S3] docs/SELLER_ALIAS_RISK_PILOT.zh.md；R/source/docs/SELLER_ALIAS_RISK_PILOT.zh.md；R/source/schema/step28_risk_policy.json。
[S4] docs/SELLER_ALIAS_BGE_CONTINUAL.zh.md，第2–8节；两source中的同字节父合同及policy。
[S5] L/source/scripts/step28_er_weight.py，212–278行；step28_er_weight_run.py，50–178、181–422、454–526行；step28_er_weight_evaluate.py。
[S6] R/source/scripts/step28_risk_replay.py，19–80、83–145、148–245行；step28_risk_execution.py，35–97行；step28_risk_study.py，13–76行；R/source/scripts/step28_er_weight_run.py及step28_er_weight_evaluate.py。
[S7] direct_reviews/logit_external.zh.txt、logit_primary.zh.md、logit_cpu/；risk_algorithm_external.zh.txt、risk_algorithm_primary.zh.md、risk_execution_external.zh.txt、risk_execution_primary.zh.md、risk_cpu/。
[S8] L与R的job/execution.json、job/before_valid.json、job/access.json、job/completion.json；job/run/manifest.json、partition.json及points、maps、scores、updates、memory中的获准记录；dependencies/shared、tenth、logit_quarter及mapping.json。
[S9] L与R的job/evaluation/collected.json、evaluation.json、reference/、全部新npy与counts及stage_metrics.csv；R/job/evaluation/logit_reference_binding.json。
[S10] L与R的inventory.json、verification/execution.json、verification/payload_custody.json、verification/audit/；R/listener/全体记录；scripts/step28_er_weight_audit.py 137–154行。

### 本次独立证据索引

以下R路径相对于交付ZIP根：

[R1] review_evidence/provenance/provenance_review.zh.txt、provenance_verdict.json、run_v1/result.json、hash_checks.json、checks.json及source_differences.diff。
[R2] review_evidence/logit_path/logit_path_review.zh.txt、machine_verdict.json、independent_logit_path_v1.py、run_v1/。
[R3] review_evidence/risk_path/下的中文子审、机器裁决、risk_path_reference_v1.py、attempt_v1/及risk_binding_reference_v1.py、binding_v1/。
[R4] review_evidence/statistics/statistics_subreport.zh.txt、statistics_summary.json、independent_saved_matrix_reference_v1.py；run_v1/independent_statistics.json、all_23_checks.csv、full_absolute_endpoints.csv、full_comparisons.csv、regenerated_draws.npy、primary_map_bootstrap_distributions.npz及运行原件。
[R5] review_evidence/counts/subreview.zh.txt、subreview_machine.json、independent_counts_v1.py；run_v2_single_cpu/完整工作点、校准与machine_verdict；workpoint_summary/；source_excerpts.txt。
[R6] review_evidence/diagnostics/diagnostics_report.zh.txt、diagnostics_verdict.json、independent_diagnostics.py、results_v1.json及run_v1实际输出/退出。

## 附表：五个固定方法的O/N/Z合并工作点

全部为stage-cal概率0.5、描述性合并整数结果，不加入原23项判据。[R5]
""")
    parts.append(full_workpoints)
    text = "\n\n".join(x.strip() for x in parts) + "\n"
    report_path = out / "external_review.zh.txt"
    with report_path.open("x", encoding="utf-8") as f:
        f.write(text)
    verdict = {
        "schema_version": 1,
        "review_date_local": "2026-10-04",
        "timezone": "Asia/Shanghai",
        "assembled_at_utc": now,
        "external_review_status": "ACCEPT_WITH_DECLARED_VERIFICATION_LIMITS",
        "review_scope": "Two completed fixed developed-valid points: LOGIT0.1 and risk; saved-result external scientific review",
        "scientific_blockers": [],
        "delivery_reproducibility_defects": [],
        "attachment": identity,
        "model_identity": {
            "user_requested": "网页GPT 6 Pro",
            "visible_selector": None,
            "visible_selector_status": "NOT_INDEPENDENTLY_EXPOSED_THIS_REVIEW",
            "reviewer_self_description": "ChatGPT / GPT-6 Astra Pro (session configuration statement)",
            "independent_backend_attestation": None,
            "requested_backend_certified": False,
        },
        "authorization": {"specific_upload_authorized": True,
                          "historical_pending_upload_snapshot_superseded": True,
                          "new_experiment_started": False, "automatic_model_replacement": False},
        "executor_primary_review_claimed_complete": False,
        "reviewer_access": {"server_connections": 0, "native_model_loads": 0,
                            "formal_text_reads": 0, "real_training_cache_deserializations": 0,
                            "formal_label_parse_attempts": {"train": 0, "valid": 0, "heldout": 0, "owners": 0},
                            "new_training_updates": 0, "torch_imported": False,
                            "saved_score_arrays_read": 48,
                            "label_level_AP_MAP_recomputed": False,
                            "label_level_raw_Brier_recomputed": False,
                            "calibrated_NLL_verified_conditional_on_saved_raw_NLL": True},
        "source_verification": {"payloads": provenance["manifest_payloads"],
                                "files_including_manifest": provenance["actual_files"],
                                "hash_checks": provenance["hash_check_count"],
                                "all_hash_checks_passed": provenance["hash_check_count"] == provenance["hash_checks_passed"],
                                "source_counts": {"logit_low": 37, "risk": 32},
                                "same_path_source_count": 22,
                                "unchanged_shared_sources": 19,
                                "changed_shared_sources": provenance["changed_common_sources"]},
        "execution": {
            "logit_low": {"supported_valid_in_review_scope": True, "updates": logit["update_rows"],
                          "group_gradient_presentations": 3456, "new_endpoints": 6,
                          "new_metric_sets": 18, "reused_metric_sets": 63,
                          "maximum_recorded_memory_bytes": logit["maximum_recorded_memory_bytes"],
                          "runtime": logit["runtime"], "original_exit_status": 0},
            "risk": {"supported_valid_in_review_scope": True, **risk["counts"],
                     "lifecycle": risk["lifecycle"], "original_exit_status": 0},
        },
        "statistics": {"bootstrap": {"generator": "PCG64", "seed": 20260930,
                                     "shape": [5000, 3, 20], "actual_domain_axis": ["A", "B", "C"],
                                     "sampling_unit": "paired complete group", "quantile_method": "linear",
                                     "coverage": 0.95, "recreated_draws_exact_match": True,
                                     "single_training_seed": "s0", "orders": ["ABC", "BCA", "CAB"],
                                     "unconditional_noninferiority_established": False},
                       "matrix_sets_by_job": {"logit_low": 81, "risk": 99},
                       "matrix_shape": [60, 22],
                       "maximum_errors": stat_check["max_absolute_differences"],
                       "numeric_comparison_counts": stat_check["numeric_comparisons"]},
        "all_original_23_checks": {
            key: {"name_zh": compare_names[key], "passed": value["passed"], "total": 23,
                  "checks": value["checks"], "failed": value["failed"],
                  "positive_old_map_orders": value["positive_old_map_orders"]}
            for key, value in comparisons.items()},
        "primary_MAP_seven_endpoints": {
            key: {ep: value["all_role_differences"]["primary"][ep]["map"]
                  for ep in ("O", "N", "Z", "F_first", "F", "G", "final_all")}
            for key, value in comparisons.items()},
        "primary_absolute_results": {m: abs_results[m]["primary"] for m in methods},
        "working_point_summary": workpoints,
        "counts_maximum_errors": count_check["max_absolute_errors_by_category"],
        "findings": {
            "logit_low": "LOCAL_POSITIVE_RESULTS_AGAINST_MATCHED_ER_AND_LOGIT_QUARTER; ORIGINAL_SEQ_GUARDS_20_OF_23",
            "risk": "VALID_NEGATIVE_FIXED_POINT; NO_IMPROVED_OLD_NEW_BALANCE_ESTABLISHED",
            "risk_vs_logit_tenth_ONZ_MAP": "ALL_CONDITIONAL_INTERVALS_BELOW_ZERO",
            "risk_vs_ER_tenth_MAP": {"O": "INTERVAL_CROSSES_ZERO", "N": "INTERVAL_BELOW_ZERO", "Z": "INTERVAL_BELOW_ZERO"},
            "logit_vs_SEQ_failed_guards": [
                {"endpoint": ep, "metric": metric, **seq_difference[ep][metric]}
                for ep, metric in [("N", "average_precision"), ("Z", "map"), ("Z", "average_precision")]],
            "risk_failure_cause_proven": False,
            "risk_family_invalidated": False,
            "method_novelty_established": False,
            "independent_final_holdout_established": False,
        },
        "report_clarifications": [
            "23 mixed original checks do not protect fixed-threshold FPR; risk has lower FPR and higher precision but lower recall/F1 than LOGIT0.1 at the specified stage-cal threshold.",
            "Mode and first-update gradient probes refer to the same group in only one of six stages; no paired causal interpretation.",
            "Native weight hashes were compared as supplied Linux custody records to endpoint descriptors; no new webpage hashing or model restore.",
        ],
        "limitations": [
            "Saved group-level AP/MAP and other truth-dependent metrics are inputs, not independently recomputed from labels.",
            "Original native gradients, model/Adam/RNG restores and true cache contents were not reexecuted or reloaded in this webpage review.",
            "Bootstrap is conditional on fixed models, calibration and developed valid, not retraining, reselection or fresh final-test uncertainty.",
            "Three orders are not three seeds; the same 60 groups are paired and repeated across paths.",
            "No real market, unrestricted candidate corpus, no-positive query or same-controller cross-stage tracking claim.",
        ],
        "report": {"path": "external_review.zh.txt", "bytes": report_path.stat().st_size,
                   "sha256": hashlib.sha256(report_path.read_bytes()).hexdigest()},
        "evidence_roots": [f"review_evidence/{x}/" for x in
                           ["provenance", "logit_path", "risk_path", "statistics", "counts", "diagnostics"]],
    }
    with (out / "external_verdict.json").open("x", encoding="utf-8") as f:
        json.dump(verdict, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")
    assembly = {"script": str(Path(__file__).resolve()), "inputs": [],
                "output": {"report_bytes": report_path.stat().st_size, "report_characters": len(text),
                           "machine_bytes": (out / "external_verdict.json").stat().st_size},
                "status": "COMPLETE_DOCUMENT_ASSEMBLY_ONLY"}
    for relative in ["statistics/run_v1/independent_statistics.json", "statistics/run_v1/machine_verdict.json",
                     "counts/run_v2_single_cpu/machine_verdict.json", "counts/workpoint_summary/workpoint_summary.csv",
                     "logit_path/run_v1/result.json", "risk_path/attempt_v1/result.json",
                     "provenance/run_v1/result.json", "diagnostics/results_v1.json", "attachment_identity.json"]:
        p = ev / relative
        assembly["inputs"].append({"path": relative, "bytes": p.stat().st_size,
                                   "sha256": hashlib.sha256(p.read_bytes()).hexdigest()})
    with (ev / "integration/assembly_result.json").open("x", encoding="utf-8") as f:
        json.dump(assembly, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(json.dumps(assembly["output"], ensure_ascii=False))


if __name__ == "__main__":
    main()
