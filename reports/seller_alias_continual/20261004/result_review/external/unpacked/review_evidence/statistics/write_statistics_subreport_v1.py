#!/usr/bin/env python3
"""Render a Chinese report from the already-completed independent statistics."""
from pathlib import Path
import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument("--input", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
data = json.loads((args.input / "independent_statistics.json").read_text())
verdict = json.loads((args.input / "machine_verdict.json").read_text())
environment = json.loads((args.input / "execution_start.json").read_text())
names = {"seq": "SEQ", "tenth": "ER0.1", "logit_quarter": "LOGIT0.25", "logit_tenth": "LOGIT0.1", "risk": "risk"}
comparisons = {key: value for study in data.values() for key, value in study["comparisons"].items()}


def title(key):
    a, b = key.split("_minus_")
    return names[a] + "−" + names[b]


def interval(value):
    low, high = value["conditional_95pct_interval"]
    return f"{value['mean']:+.6f} [{low:+.6f}, {high:+.6f}]"


report = [
    "# 独立保存矩阵统计复核：中文子报告",
    "",
    "## 一、裁决与范围",
    "",
    "本统计子任务通过：从正式保存的群级矩阵独立重算，未发现会阻断本轮结论的统计错误或影响交付的统计复现缺陷。LOGIT0.1三比较与risk四比较的原23项判断全部与提交结果一致。此裁决只覆盖保存矩阵的完整性、指标角色、端点代数、配对bootstrap和判据；训练实现、首域恢复、盲门及来源真实性由其他并行审查覆盖。",
    "",
    "没有运行或导入提交者的audit、训练或评价模块；只读其冻结统计源码确定实际约定，再以新的显式阶段公式和索引实现计算。未连接服务器，未加载Torch、原生权重或真实缓存；没有读取正式文本、逐对真值、test、owners或身份资产，也没有训练、调参、重新校准或改成功线。",
    "",
    "## 二、实际审读与独立实现",
    "",
    "审读docs/SELLER_ALIAS_LOGIT_RISK_RESULT.zh.md、两项固定合同、原BGE合同第7—8节；分别读取两作业source内当前policy及继承的step28_bge_continual_policy.json、冻结step28_bge_continual_evaluate.py和step28_er_weight_evaluate.py。还只读提交者的两个独立audit文件，未把其PASS或计数作为判断依据。所有被计算使用的文件实际大小/SHA和用途见run_v1/input_inventory.json。",
    "",
    "矩阵输入为LOGIT0.1作业18新＋63复用、risk作业18新＋81复用，共180个实际文件路径，去重后96种矩阵字节内容。每份均为完整60×22、float64且有限；两作业的60个group_ids、实际域A/B/C及22列次序一致，每域20群。新作业各恰好六阶段点，raw/stage-cal/first-cal三角色完整。22列由名称精确匹配，没有把AP、梯形PR-AUC、MAP或MRR互换。",
    "",
    "primary明确取raw的曲线/检索列和stage-cal的概率/分类列。实际所有保存角色的四曲线列和十检索列逐元素完全一致，故本批primary数值与stage-cal一致；这来自核对，不是把二者定义混为一谈。",
    "",
    "独立参考脚本independent_saved_matrix_reference_v1.py先按group_ids实际行位置分出A/B/C，使用PCG64(20260930)一次生成(5000,3,20)索引；第二维严格为实际域，同一抽样在所有方法、角色、阶段、顺序上复用。两个job的保存draws均与新生成结果逐元素一致。随后直接取matrix[domain_rows][draws]、在被抽20群上平均，再分别组合各顺序的七端点并在三个顺序上等权平均。这与提交audit先构造端点加权场再乘频数的实现结构不同。",
    "",
    "另写出现次数算法：计算每次每域各群出现次数除20，再与该域矩阵相乘，逐元素对照直接索引生成的每域bootstrap均值。所有端点的95%区间使用线性分位，并另以排序后第q×4999个位置两邻项显式插值交叉核对。三阶段、三实际域均设不同数值的手算夹具检查了21个端点值及Brier/log_loss符号，共63个有具体期望值的检查。",
    "",
    "设R(t,j)为某顺序第t阶段结束时，第j个到达域的群宏指标。实际公式为：O=(R(3,1)+R(3,2))/2；N=(R(2,2)+R(3,3))/2；Z=R(3,3)；F_first=R(1,1)−R(3,1)；F=((R(1,1)−R(3,1))+(R(2,2)−R(3,2)))/2；G=((R(2,2)−R(1,2))+(R(3,3)−R(2,3)))/2；final_all=(R(3,1)+R(3,2)+R(3,3))/3。Brier和log_loss在F_first/F/G反号，使正遗忘表示损失变大、正学习表示损失下降。另逐次核对final_all=(2O+Z)/3。",
    "",
    "## 三、运行及数值一致性",
    "",
    f"真实运行环境为Python {environment['python'].split()[0]}、NumPy {environment['numpy']}；当前允许CPU0—8，实际固定CPU0，OPENBLAS/OMP/MKL线程各1。运行约{verdict['elapsed_seconds']:.3f}秒，退出0，stderr为空。实际完整argv、启动时环境、脚本大小/SHA、stdout/stderr/exit均保留。第一次正式独立参考运行即通过，没有计算脚本修订覆盖。",
    "",
    "| 对照内容 | 数值数 | 最大绝对差 |",
    "|---|---:|---:|",
]
labels = {
    "saved_results": "独立重算与保存端点/比较/跨作业复用统计",
    "direct_vs_occurrence_counts": "直接索引与出现频数逐域bootstrap均值",
    "quantile_linear_vs_manual": "线性分位与显式两邻项插值",
    "final_all_identity": "最终总体与(2O+Z)/3恒等式",
}
for key in labels:
    report.append(f"| {labels[key]} | {verdict['numeric_comparisons'][key]:,} | {verdict['max_absolute_differences'][key]:.17g} |")
report += [
    "",
    "这些数量只是复核覆盖记录。判断依据是明确的独立公式、同域同群配对索引及逐项结果一致；数值容差只用于实现重放，不用于放宽任何性能方向。原23项仍使用严格>0或≥/≤0且观察退化容忍为0。",
    "",
    "此前只读路径发现有两次审查者错误猜测：reused_collection.json实际应为reference/collected.json；source/src实际应为source/scripts。原工具错误与更正记录在pre_reference_browsing_failures.json。这是我方路径发现问题，不是附件缺件或实验失败；该文件如实标明系对工具输出的记录，不冒称原始进程日志。",
    "",
    "## 四、核心绝对结果",
    "",
    "下表均为原冻结端点的MAP；F_first/F越小越好，G越大越好。保留完整精度机器结果，不用表内六位小数作判据。",
    "",
    "| 方法 | O | N | Z | F_first↓ | F↓ | G | final_all |",
    "|---|---:|---:|---:|---:|---:|---:|---:|",
]
for arm, variants in data["risk"]["absolute_endpoints"].items():
    values = [variants["primary"][ep]["map"]["mean"] for ep in ("O", "N", "Z", "F_first", "F", "G", "final_all")]
    report.append("| " + names[arm] + " | " + " | ".join(f"{v:.6f}" for v in values) + " |")
report += [
    "",
    "## 五、七组固定比较",
    "",
    "差值为候选减对手，括号内为条件95%区间，单位为0—1指标的绝对差。乘100才为百分点。",
    "",
    "| 比较 | O MAP差 [区间] | N MAP差 [区间] | Z MAP差 [区间] | 原23项 |",
    "|---|---|---|---|---:|",
]
for key, result in comparisons.items():
    primary = result["all_role_differences"]["primary"]
    report.append("| " + title(key) + " | " + " | ".join(interval(primary[ep]["map"]) for ep in ("O", "N", "Z")) + f" | {result['passed']}/23 |")
report += [
    "",
    "| 比较 | F_first MAP差 [区间] | F MAP差 [区间] | G MAP差 [区间] | final_all MAP差 [区间] |",
    "|---|---|---|---|---|",
]
for key, result in comparisons.items():
    primary = result["all_role_differences"]["primary"]
    report.append("| " + title(key) + " | " + " | ".join(interval(primary[ep]["map"]) for ep in ("F_first", "F", "G", "final_all")) + " |")
report += [
    "",
    "LOGIT0.1对ER0.1和LOGIT0.25均通过原23项，是本次固定开发比较的局部正结果。对SEQ为20/23，未过的是N AP、Z MAP和Z AP的零观察退化保护：",
    "",
    "| 未过项 | 差值 [条件95%区间] |",
    "|---|---|",
]
seq = comparisons["logit_tenth_minus_seq"]["all_role_differences"]["primary"]
for ep, metric in (("N", "average_precision"), ("Z", "map"), ("Z", "average_precision")):
    report.append(f"| {ep} {metric} | {interval(seq[ep][metric])} |")
report += [
    "",
    "三者区间均跨零，故原观察保护未满足；这既不是三项总体退化已被证实，也不是非劣效成立。N MAP差虽然微正，其区间同样跨零，不能据此宣称新学习不受损。Z Recall@5差为" + interval(seq["Z"]["recall_at_5"]) + "，与Z MAP负观察值并存，指标含义应分别报告。",
    "",
    "risk对主要LOGIT0.1对照的O/N/Z MAP条件区间均在零以下，F_first/F增大、G和final_all下降亦有同方向条件证据。risk对ER0.1的O MAP区间跨零，N/Z区间均低于零；不能把O观察下降说成已证实总体下降。risk对SEQ的O微正但区间跨零，N/Z负向，因此不能用O点值包裹成解决了旧保持与新学习权衡。三顺序不是三训练种子，不能把这三条路径当训练随机性重复。",
    "",
    "## 六、原23项全部判断",
    "",
    "✓通过，×未过；这是原混合观察保护和一项区间要求，不是23个独立显著性检验。两项against_raw_reference比较候选stage-cal概率损失与对手raw概率损失，其余概率保护比较双方stage-cal。未把以后三种子资格加入当前23项，也没有用允许重放误差替代性能零容忍。",
    "",
]
keys = list(comparisons)
report += ["| 原检查 | " + " | ".join(title(k) for k in keys) + " |",
           "|---|" + "---:|" * len(keys)]
for check in comparisons[keys[0]]["checks"]:
    report.append("| " + check + " | " + " | ".join("✓" if comparisons[k]["checks"][check] else "×" for k in keys) + " |")
report += [
    "",
    "risk对两个LOGIT的4项通过全部为O/N的Brier、log_loss相对对手raw概率参照。对ER0.1另外通过O Recall@5观察改善；对SEQ另外通过O MAP观察值、O AP和O Brier。不能把4/23或7/23中不同性质的通过项汇总为核心排序目标已达到。",
    "",
    "## 七、分类结论、边界与证据文件",
    "",
    "已证实科学阻断：本统计范围未发现。影响本轮交付的复现缺陷：未发现。两处我方路径猜测失败已单独保留，不上升为实验缺陷。提交者关于两固定点的核心统计陈述与独立重算一致，无须因本统计检查盲目重训。",
    "",
    "后续研究建议：只在另外授权及事前冻结设计下检验训练随机性、独立最终数据和任务规模外推；当前结果可保留LOGIT0.1的强基线位置，并结束risk此固定点。本文不建议从当前valid反推新门槛，也不要求额外消融或加点。",
    "",
    "范围外过度设计：将本结果审查扩成新训练、多seed、调γ/λ、重新打开真值、原生权重重载、对所有既往历史重审，均非本统计裁决所需。",
    "",
    "标签级边界：AP、MAP、MRR、各曲线/检索指标及概率损失的群内原值来自正式唯一收集。本次可复算这些已保存群级值的域/顺序/端点/差值/区间，不能宣称根据隐藏逐对真值重新算出了群AP/MAP。独立混淆整数和工作点审查属于另一子任务，本报告没有冒认其覆盖。",
    "",
    "推断边界：群bootstrap条件于固定训练模型、一个s0、三顺序、校准映射、生成数据和已开发valid；不覆盖重训、重新校准、先前自适应开发或真实市场。没有多重比较校正或全家族显著性主张。23/23不等于总体非劣效、没有遗忘、创新证明或独立最终留出通过；risk负结果也不否定整个方法家族。",
    "",
    "完整结果文件：run_v1/independent_statistics.json保存所有方法×四角色×七端点×22列绝对统计和七组所有角色差值；full_absolute_endpoints.csv、full_comparisons.csv便于复核；all_23_checks.csv含161个判断；primary_map_bootstrap_distributions.npz含七比较×七端点的5000次MAP抽样结果；regenerated_draws.npy为实际再生抽样索引；input_inventory.json、matrix_identities.json、group_metric_identity.json保存输入身份。run_v1/machine_verdict.json、run_v1/execution_start.json及顶层run_v1.launcher.json、run_v1.stdout.txt、run_v1.stderr.txt、run_v1.exit_status.txt构成机器判定和原始运行记录。",
]
with args.output.open("x", encoding="utf-8") as f:
    f.write("\n".join(report) + "\n")
summary = {
    "status": verdict["status"], "scope": "saved-matrix statistics only",
    "script": "independent_saved_matrix_reference_v1.py", "result_dir": "run_v1",
    "comparisons": {k: {"label": title(k), "passed": v["passed"], "checks": v["checks"],
                         "primary_map_endpoints": {ep: v["all_role_differences"]["primary"][ep]["map"] for ep in ("O", "N", "Z", "F_first", "F", "G", "final_all")}}
                    for k, v in comparisons.items()},
    "numeric_comparisons": verdict["numeric_comparisons"],
    "max_absolute_differences": verdict["max_absolute_differences"],
    "scientific_blockers": [], "reproducibility_defects": [],
    "report": args.output.name,
}
with args.output.with_name("statistics_summary.json").open("x", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)
    f.write("\n")
print(str(args.output))
