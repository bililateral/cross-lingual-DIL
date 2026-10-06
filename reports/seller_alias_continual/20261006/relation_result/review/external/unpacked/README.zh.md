# 关系目标记忆固定候选结果外审：报告与独立证据

日期：2026-10-06。

## 裁决和首先阅读

完整报告为根目录 `SELLER_ALIAS_RELATION_MEMORY.result_external_review.zh.md`。

裁决：**接受本次为有效配置级开发负结果；相对LOGIT0.1六条件0/6，结束该固定配置，不进入追加匹配重放。** 未发现新增科学阻断或影响主要统计复现的缺陷。仅有一处末位展示勘误、开放对象澄清和概率评分术语限定；不需要重新训练、读取标签、重新校准或补齐整套消融。

## 输入身份

`inputs/relation_result_context_review.zip` 是当前用户附件的原字节副本：5,077,487字节，272个文件，SHA-256：

`3c11a9640c13438d9eb7ace1005183d83ccafbdca7036fa39e0689b06731625a`

其 `context_inventory.json` 的271载荷及实际解压文件已全部核对；输出在 `audit/identity/identity.json`。

`inputs/prior_pilot_review10.zip` 仅用于复现“当前23来源中只有正式入口和集成测试相对上一轮发生变化”的历史修复对照，来自本会话前一轮附件，319,319字节、45成员，SHA-256：

`ddebe7ca566cff689975ca083e3acd597710c815992aed224497b23c306eb43e`

它不包含本次正式效果，不可代替当前结果输入。当前包已包含本次需要的旧审查原文及基线矩阵。

## 证据目录和边界

| 目录 | 实际核验对象 | 输出 |
|---|---|---|
| `audit/identity/` | 当前原ZIP、271清单载荷、解压文件 | 逐项大小／SHA及封装安全／CRC |
| `audit/metrics/` | 候选28套、基线27套矩阵／计数 | 独立7端点、22列、4角色、均值／逐顺序／5000配对区间及6规则 |
| `audit/lifecycle/` | 冻结23来源、阶段调度、Algorithm R、盲分数、校准、完成链 | 实际可见文件核验、2592／4320、37分数及18仿射变换 |
| `audit/resources/` | 源绑定、正式控制台、包装时间、资源／失败账 | 108条日志、预算峰值、旧基线资格 |
| `audit/repairs/` | P1/P2/P3修复与准入身份；最窄故障分支 | 23来源比较及未改函数体的显式故障注入 |
| `audit/interpretation/` | 全部角色均值、合并计数／群宏、轨迹、解释 | 局部正项、校准作用、154行展示与勘误 |
| `tables/` | 从独立统计输出派生的完整表格 | `independent_all_endpoints.csv`，1386行，含3个分顺序值和条件区间 |

网页执行环境为Python3.12.14／NumPy2.3.5。各核验脚本未导入项目生产评价或训练模块；repairs的控制小例仅抽取未改函数体并注入明确的测试依赖。网页没有BGE/GPU／项目Linux实测，没有读取正式正文或逐对真值，没有载入9份模型或9份缓存，也没有训练或重新校准。

**候选37组盲分数存在；逐对真值不在附件。** 因此可以核对分数身份、仿射变换和排序／并列，可以重算保存矩阵之后的统计，不能从盲分数重新生成MAP/AP/AUC/Brier/log-loss或重新优化校准器。缓存／权重本体的SHA只按Linux直接记录交叉比对，未在网页重新计算。

当前源包中的 `scripts/step28_relation_memory_result_verify.py` 使用原Linux工作区及缓存本体哈希，本次没有修改后冒称完整脚本原生通过。当前基线的 `reference.linux_root` 在网页不存在；使用原合同的 `reference.local_root` 对应所附矩阵。无需构造Linux目录或修改冻结policy。

## 最小可移植复核

先将 `inputs/relation_result_context_review.zip` 解压到一个新目录，例如 `current_context/`。这是读取用户已提供的材料，无需访问正式数据或服务器。

核对附件身份：

```bash
python audit/identity/verify_context_package.py \
  --zip inputs/relation_result_context_review.zip \
  --extracted-root current_context \
  --out fresh_identity.json
```

独立复算全量保存矩阵：

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python audit/metrics/independent_saved_metrics.py \
  --package-root current_context \
  --output-dir fresh_metrics
```

依赖仅为标准库和NumPy；不需要Torch、BGE、正式文本、逐对标签、模型权重或缓存。使用新的输出目录，保留本包原始审查输出。此命令的主要结果应复现6项均为false、8316数值最大差约1.33e-15、完整统计文件SHA `2269c3b0d802ecb3166d57336e9c7aa3e8edc5ffd1a3d36e5ad22890aaac8356`。不同NumPy／BLAS或硬件的浮点求和可有末位差异，应依据报告1e-12数值容差核验，不要求跨环境输出文件必须逐字节相同。

其余脚本保留当次实际执行的源码与绝对输入位置，便于审计覆盖。若异地执行，可只更改这些**审查脚本**开头的ROOT／PACKAGE／OUT等路径常量；不得修改冻结项目来源。`audit/repairs/source_and_admission_check.py` 的old路径需指向 `inputs/prior_pilot_review10.zip` 解压目录；这个历史比较只服务修复范围核对，不参与效果统计。生成表格脚本也只处理已有独立输出，不是项目评价入口。

## 审查自身失败和修正保留

`audit/metrics/first_attempt/` 保存首次审查代码及KeyError输出：角色字典浅共享造成派生primary污染保存角色遍历。复制字典后已修复；输入包未改。这不是项目缺陷。

`second_attempt_before_metadata_correction/` 保存成功统计之后、尚未更正“分数缺失”措辞和加入路径参数之前的审查记录；最终docstring／limitations已经准确限定为缺逐对真值。`correction_identity.json` 证明前后完整统计文件一致。最终脚本已实际成功执行，stderr为空。网页审查代码的修复不消耗或重置任何正式实验预算。

## 清单

`manifest.json` 列出本证据包除清单自身外所有文件的包内相对路径、字节数及SHA-256，包括两个原始输入ZIP。清单不自引用；外层ZIP的实际大小与SHA在交付回执中另记。

本包不实施也不授权新训练、调参、换种子、模型删除、匹配重放、其他数据开放或新系统建设。主执行者应独立处置本报告意见，保留冻结历史和当前负结果。
