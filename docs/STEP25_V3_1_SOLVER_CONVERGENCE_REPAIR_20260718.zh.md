# Step25-v3.1 求解器收敛契约修复

更新日期：`2026-07-18`

## 1. 修复原因

Step25-v3 的同步包、逐 pair 预测和 summary 内部完全一致，但模型 artifact 暴露了优化器终止缺陷。策略指定 projected-gradient tolerance 为 `1e-8`，实际部分模型最终 residual 达到 `0.52`，仍被标记为 `solver_converged=true`。

根因是旧实现采用以下任一条件即可终止：

1. projected-gradient/KKT residual 达到 tolerance；
2. 相邻迭代 relative loss change 足够小。

第二条只能说明目标函数在当前步长下变化很小，不能证明带方向约束问题满足 KKT 条件。旧 artifact 因此可能是提前停止解。

## 2. 旧 v3 结果边界

旧目录保持不可变：

`reports/step25_template_decontaminated_authorship/v3_copy_aware_dual_channel_20260718/`

其 `9/9` payload、`21/21` producer 和全部指标重算已经通过，但科研状态降级为：

`solver-termination-invalidated diagnostic`

旧结果可以说明当前分数和报告一致，也可以用于定位优化器问题，但在修复后完全复跑前，不能将 C2 正式冻结为方法负结果。

## 3. 唯一允许修改的内容

v3.1 只修改 constrained LR/L2 的数值求解与终止判据，不改变其凸目标函数、L2、样本权重或系数边界。普通 projected gradient 在共线特征的回归测试中运行满 `10,000` 次仍停在 `2.3e-6`，因此修复版使用低维 active-set projected Newton direction 加 Armijo backtracking，并继续以 projected-gradient/KKT residual 作为唯一收敛标准：

- `solver_converged=true` 只允许在 final projected-gradient residual `<=1e-8` 时出现；
- relative loss 仅记录为 diagnostic，绝不参与收敛判断；
- line search 停滞、达到最大迭代但 KKT 未达标时必须 fail closed；
- artifact 新增 termination reason、final objective、gradient infinity norm、projected-gradient residual、relative loss 和 accepted step size；
- sync manifest 遍历全部 repaired solver artifact，任何非 KKT 解都会阻断返回包。
- sync manifest 必须找到固定 C0-C3 矩阵产生的全部 `44` 个 repaired artifact，缺少任意 source-only、English OOF 或 Chinese OOF fit 都会阻断返回包；
- 新旧 English/Chinese pair-feature CSV 必须逐字节相同，防止“只修求解器”重跑意外改变科学输入。

## 4. 禁止修改的内容

以下内容与 v3 完全相同：

- canonical English/Chinese train 边界；
- `401 = 116/285` English 与 `573 = 229/344` Chinese；
- seller-component 五折划分和 seed；
- factorized sample weights；
- C0、C1、C2、C3 feature sets；
- coefficient directions；
- `L2=10`、无 class weight、feature standardization；
- grouped bootstrap 配置；
- 全部 D0-to-D1 gates；
- missingness closure；
- English-only operational identifier control；
- valid/test 禁用、D0 不选模、publication 和 Step11/17 hard false。

这不是重新调参，也不是根据旧 v3 结果选择新模型，而是对完全相同目标函数进行正确求解。

## 5. 输出隔离

修复结果写入：

`reports/step25_template_decontaminated_authorship/v3_1_solverfix_20260718/`

所有文件使用新的 `step25_v3_1_*` 名称。旧 v3 不覆盖、不删除、不重新解释。

## 6. 验证要求

Linux runner 必须依次完成：

1. v3.1 contract tests；
2. 四个 config-only preflight；
3. 冻结 feature join 重放；
4. C0-C3 KKT-only refit；
5. operational control 重放；
6. closed manifest；
7. bounded conclusion 输出。

最终 manifest 必须报告：

- 所有 repaired artifact 的 `solver_converged=true`；
- repaired artifact 数量必须精确等于 `44`；
- termination reason 只能是 `projected_gradient_kkt_tolerance`；
- maximum final projected gradient `<=1e-8`；
- relative loss used for convergence 为 `false`；
- 两个 pair-feature payload 均与旧 v3 byte-identical。

## 7. 结果解释

如果 v3.1 正确收敛后仍保持 C2 AP 下降、template tail 恶化且 gates 不通过，则正式冻结 Step25-v3 为负结果，并停止在 D0 上继续调整 copy penalty。

如果数值明显变化，也只能依据原 v3 gates 判断是否允许一个未来 D1 replication。v3.1 仍不能产生 publication promotion，不能进入 Step11/17，也不能把 C3 sensitivity 事后改成主模型。
