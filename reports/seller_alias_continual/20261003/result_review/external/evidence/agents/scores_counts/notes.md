# 保存分数、校准、计数与复用身份：独立审查分工记录

本记录是联合外审的一部分，不代替主审的完整结论。本分工先完整阅读根 `REQUEST.zh.md`，只读取获准小结果、代码、合同与元数据；没有读取正式文本、正式标签、缓存正文、模型权重、test 或 owners，没有连接服务器或重新训练。读取覆盖见 `read_coverage.json`。网页实际执行环境为 Python 3.12.14、NumPy 2.3.5、Linux x86_64，本次子进程 CPU affinity=[0]，线程环境设为1；这与原生 py310/CUDA 正式证据分开。

## 1. 这一部分的判定

**没有发现使本次两项比较失效的 SCIENTIFIC_BLOCKER，也没有在本分工范围发现影响交付的项目 REPRODUCIBILITY_DEFECT。** 两项计算使用的18+81、18+45套指标/计数、分数映射和保存计数之间可以对上。此处结论仅针对本文列出的保存结果一致性、复用身份与手写函数语义核验；不能替代正式标签有效性证明或模型重新执行。

162是两研究使用集合数之和，其中45套参考集合在两研究中重复。按端点×角色计算共有**117套唯一集合**，不能把162写成162套独立实验/新观察；9720也是含重复使用的群计数行，共7020唯一群计数行。185条校准核验记录含173条实际迭代轨迹和12条最终状态重复核对，不是185次拟合。

## 2. 独立执行得到的数值结果

独立脚本 `independent_scores_counts.py` 没有调用项目审计脚本及其计算函数。

| 核验 | 覆盖及结果 |
|---|---|
| 集合身份 | ER0.1新18、复用81；LOGIT新18、复用45。126个复用记录与原首轮/旧ER的原始collected描述符逐项一致；两研究共同45套的矩阵及计数完全相等。 |
| 评价矩阵 | 全部162个使用集合均为有限float64、60×22；60个群ID唯一，实际A/B/C各20群；群顺序、角色、指标列与原集合一致。 |
| 群计数 | 9720行均为非负整数，每群TP+FN=20、FP+TN=358、合计378；逐群precision、recall、F1、specificity、balanced accuracy、MCC的独立公式与保存矩阵最大差0。 |
| pooled计数 | 先合并计数再计算各域/全体FPR、Recall、Precision、F1，与保存absolute_stage_results最大差0；没有用群平均Precision代替pooled Precision。 |
| 群macro | 用Python `math.fsum` 重新合并每个实际域和全体22列，与保存绝对结果最大差5.551115123125783e-16。 |
| 新盲分数 | 48数组：两研究各6点×4类；development为60×378 float32、calibration为12×378 float32、stage-cal/first-cal为60×378 float64，全部有限。 |
| 工作点 | 新36套角色分数共2160群，`score>=0`阳性个数与TP+FP完全一致；本包实际数值上与`sigmoid(score)>=0.5`一致。 |
| 正仿射 | 24个校准分数数组与`a*float64(raw)+b`逐值差0；按Python独立排序核对全部60×378位置，严格排序和原始并列均保持。全部矩阵的14个曲线/检索指标列在三角色间完全一致。 |
| 校准绑定 | 12个新映射均绑定对应新点model_state_sha及calibration score记录；校准仅使用阶段当前实际域12群，4536对、240正对。first-cal始终复用所属顺序首域映射。 |
| 校准条件恒等式 | 用原保存initial_nll和positive_count构造两个充分统计量后，核对173条迭代轨迹+12个最终状态。NLL最大差8.326672684688674e-17，投影梯度最大差9.367506770274758e-17；最终投影梯度均≤1e-6，最终NLL均不高于初始。 |
| valid log-loss条件传递 | 新12点×2校准角色×60群=1440行，用保存raw log_loss和已保存群正对总数作仿射恒等式验证，最大差1.1102230246251565e-16。实际分数全部落在不会触发1e-15概率截断的区间。 |

新分数的0.5工作点与排序用途应继续分开报告。例如ABC路径两新方法的第二、第三阶段raw分数均没有任何预测阳性；这与保存raw计数一致，不能据此改写排序指标，也不能将排序主比较通过解释为0.5自动判定已满足召回目标。校准后的工作点已按原角色记录，未作valid事后改阈值。

## 3. 校准的独立性边界

设原校准logit为x，存档初始NLL为L(1,0)，存档正例比率为m_y。无需个体标签即可写出：

`m_yx = mean(softplus(x)) - L(1,0)`

`L(a,b) = mean(softplus(a*x+b)) - a*m_yx - b*m_y`

`dL/da = mean(x*sigmoid(a*x+b)) - m_yx`

`dL/db = mean(sigmoid(a*x+b)) - m_y`

独立实现使用`max(z,0)+log1p(exp(-abs(z)))`、分段sigmoid和`math.fsum`，不同于项目的`np.logaddexp`/均值路径；再按上下界投影梯度。**这验证的是保存分数、保存真值衍生标量与整个校准轨迹之间的条件一致性。m_yx来自原保存NLL，并非新独立标签证据；没有恢复、猜测或索要任何个体标签，也没有重新执行真实标签校准拟合。** valid仿射log-loss核验同样以原保存raw log_loss为条件；它不是从独立标签重算log_loss。

## 4. 必要手写CPU例与网页失败原件

`handmade_metric_reference.py`只通过AST提取项目中的纯数值函数，不导入项目模块，使用人为构造的28账号（4个三人组和8个二人组，共20正边、358负边）的全并列、混合并列、全负有序三例。参考实现以显式候选循环、阈值集合、Mann–Whitney逐正负对AUC、Python累加计算全部22指标。最大差4.440892098500626e-16。另一个五点手写校准例检查Bernoulli NLL和二参数梯度，中心差分最大差1.542065433679518e-11，并检查三个投影边界情形。

本分工的手写指标例与主审的手写指标覆盖有重叠，不应累计声称都是新独立案例。

首次`runs/scores_counts_handmade_v1`退出1：独立参考代码自身用`sum(labels)`得到NumPy uint8，执行`378-npos`触发`OverflowError`。失败发生在参考侧，尚未调用该例的生产指标函数。原stderr、原源码快照、命令和退出码均保留；仅将该行改为`sum(int(y) for y in labels)`，`runs/scores_counts_handmade_v2`同一组例退出0。该次修订是网页参考脚本类型处理修复，**不是项目生产失败或新的训练失败**。两次执行不是两批新增用例。

`runs/scores_counts_schema_v1`与`v2`只是同一结构查看脚本的输出调整，均退出0，不计测试。`runs/scores_counts_independent_v1`是保存小结果的完整独立计算，首次退出0。

## 5. 未验证范围及过度要求

1. 正式AP、梯形PR-AUC、ROC-AUC、MAP、MRR、Recall@K、NDCG@K依赖原一次正式标签收集。本分工验证保存值、排名不变关系、后续聚合与手写函数语义，**未从正式标签独立重算这些数值**。
2. 正式TP/FP具体哪些边属于真阳/假阳和正式Brier，未独立从真实标签重建；只能独立核计数代数、保存率值及盲预测阳性总数。
3. 共享首域完整model/Adam/RNG身份和首映射可从原点记录相互核对；权重不在包内，网页没有重新载模验证。原正式恢复证据要由主审结合源码及原生记录判断。
4. 原始完整基线矩阵没有在其历史路径重复存一遍时，按原collected的大小/SHA描述符核验本包reference副本；这是同源归档核验，不宣称另从服务器取得第二份字节证据。
5. O/N/Z/F_first/F/G/final_all、5000条件区间及23项比较由主审的另一路独立实现处理，本分工不冒称独立完成那些未执行的统计任务。

要求为这次审查追加正式标签读取、补训、重新校准、索要缓存正文、上传权重或增加服务器/环境审批体系，均属本次的OUT_OF_SCOPE_OVERDESIGN；当前保存结果足以完成本文的核验，无须追加。没有发现需要主执行者为本分工结果修改冻结代码或原结果的必要修正；保留上述证据边界即可。

## 6. 证据索引

- 独立结果：`numeric_results.json`；手写例：`handmade_numeric_results.json`。
- 独立源码：`independent_scores_counts.py`、`handmade_metric_reference.py`；读取结构源码：`schema_inspect.py`。
- 原执行流与失败修订：主证据目录`runs/scores_counts_*`。每个run.json记录精确argv/cwd、UTC起止、环境、退出码、stdout/stderr大小/SHA，source_snapshots保留实际执行源码。
- 原项目关键输入：两新job的`run/points/`、`run/maps/`、`run/scores/`、`run/partition.json`、`run/manifest.json`、`evaluation/{collected,evaluation}.json`及全部对应矩阵/计数；基线和旧ER原`evaluation/collected.json`、基线首域点记录。
- `read_coverage.json`区分程序实际内容读取与人工作代码/合同阅读范围。主审应将本分工和其他分工合并，不能只据本分工覆盖声称完整阅读两套31/34来源。
