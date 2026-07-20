# Step 28：对抗式身份信息合成与佐证门训练方案

## 1. 这一步到底解决什么

Step 27 只对既有文本做重排、遮盖和轻量扰动，并且明确禁止制造新标识符、控制者关系和身份边。因此它最多增加“同一条旧监督的不同文本视图”，不可能补足当前训练集中稀缺的身份识别信息。

Step 28 独立新建虚拟身份世界，合成新的卖家、控制者、私有标识符以及公开/商品数据中的干扰标识符。v3 不把“出现过一个直接联系方式”视为充分证据，而是训练一个有限佐证加分机制：

```text
冻结的 identifier-redacted E5 内容分数
+
由生产解析器从原始合成文本中恢复的多标识符佐证证据
+
由多标识符佐证边构成的二跳路径
```

它不是继续改 Step 27，也不会覆盖 Step 27 产物。

## 2. 两条物理分开的轨道

### 2.1 成对出现证据轨道

每个反事实组固定同一对 train-only 冻结 source-score 参考卖家和同一个基础分数，只改变标识符在原始文本中的身份角色。v3 的身份句与 source-score 参考文本是因子化的两个通道，并没有把原卖家全文复制进虚拟商品；因此这里不把参考 UID 描述成统一的虚拟 profile。

三类成对世界如下：

1. 同一控制者共享至少两个私有标识符的正例 vs 不同控制者共享一个直接外观服务标识符的碰撞负例；
2. 多私有标识符佐证正例 vs 商品数据负例；
3. 多私有邮箱佐证正例 vs 公开帮助页引用负例。

第一类不是让解析状态直接泄露标签：正负臂都会被生产解析器判为 `verified_direct_both_sides`。模型必须利用“是否存在至少两个已验证共享标识符”区分它们。训练与审计均预注册20%的反向困难组：正例只有一个共享标识符，碰撞负例反而有两个。因此合成门不可能再靠固定模板规则取得100%一致率，学到的只能是生成分布中的统计佐证关系。

正负世界先生成标题和描述，再调用 `step3_build_seller_profiles.extract_item_identity_signals`。生成器不允许直接填写 `direct_identity_eligible`、`product_data_risk_context` 等标志。解析结果再交给 Step 25 的 `occurrence_evidence` 形成模型可见状态。

同一个反事实组的两个世界是互斥的替代世界，分别建立完整的 world-local occurrence index。这样可以在两臂中使用同一标识符字面量，又不会因为把互斥世界错误拼成一个语料库而造成 token DF 升高或 direct/risky 聚合污染。

world-local DF 只用于替代世界中的上下文判定，不等价于生产语料的全局 public-frequency 统计。v3 要求最大 world token DF 不超过阈值且 `high_frequency_public` 状态数为0，因此只检验 direct/risky/support 角色，不声称检验高频公共标识符机制。

### 2.2 身份组件图轨道

普通方向的正世界中，A-B 与 B-C 每条边各有至少两个独立共享私有标识符：

```text
A ==两项私有佐证== B ==两项私有佐证== C
```

普通方向的碰撞负世界中，每条边只有一个直接外观服务标识符，三个卖家属于不同控制者：

```text
A --单项直接外观碰撞-- B --单项直接外观碰撞-- C
```

A 与 C 均无直接共享标识符。普通成对模型看到的 A-C 状态都是 `no_shared_identifier`。两臂的 A-B、B-C 又都会被解析为 `verified_direct_both_sides`，所以单纯的二跳路径也不能区分标签；只有两条支持边都达到至少两个已验证标识符时，才产生 `corroborated_two_hop_path`。图轨同样有20%反向困难组，避免门槛成为构造性必过。

这修正了 Step 25/27 中“元数据知道有链，但模型输入看不见链”的根本矛盾。

## 3. 训练目标与对照

Step 28 v3 不再把573条中文 train 放进身份门的拟合。旧数据中 `risky_only_shared` 大量伴随正例，它能说明风险上下文不能作为可靠身份证据，却不应该被学习成负惩罚。继续混合拟合只会让真实 soft/silver 标签与合成机制互相抵消。

训练单位是完整反事实组，而不是单行。每组严格包含一正一负，使用相同 source-score 参考卖家、相同冻结 source probability 和相同权重。拟合目标为：

```text
sum_group weight * softplus(-(score_positive - score_negative))
+ L2
```

组内 source logit 完全抵消；禁止新截距，也不改变冻结 source 系数。

最终只有两个正向加分项：

- 双侧至少两个已验证私有标识符形成的佐证；
- 两条支持边都达到上述佐证要求的二跳路径。

单个直接外观标识符、普通 verified 二跳路径、商品数据、公开支持页、无共享标识符和风险二跳路径均在进入模型前结构性置零，必须精确回退 source；它们不加分，也不扣分。

pair 与 graph 两轨在目标函数中分别归一化为总有效权重1.0。这样系数方向和幅度不会由120:40的预注册组数比例机械决定；两轨的 log-loss 改善也分别设门，禁止 pooled 指标掩盖某一轨失败。

对照为：

- `S0`：冻结 source；
- `M1`：完全相同的反事实组，但身份特征遮蔽，系数固定为0；
- `M2`：正常方向与20%反向困难组共同进入拟合，训练两个有界的统计佐证 uplift；
- `N0`：按每种特征差分精确一半正向、一半反向，方向和为0，拟合系数必须回到0。

当前573条中文 train 只在模型冻结后运行描述性安全诊断：用保留文件中的共享哈希数量作为佐证覆盖率代理，报告 neutral 行是否逐位等于 source、被代理判定为佐证的行是否只升分。该代理诊断不参与拟合、调参、阈值或 GO/NO-GO，也不用于声称真实校准有效。

## 4. 数据量

预注册配置生成：

- train：3种成对配方各40组，加40个图配方组，共160个反事实组、320行；
- synthetic audit：3种成对配方各20组，加60个图配方组，共120组、240行。

每个组的正负臂属于同一个统计依赖组。560行不能声称为560个独立真实身份，更不能写成新增了560条真实 ground truth。

train 与 synthetic audit 的 source-score 参考卖家、虚拟控制者、虚拟卖家、实际解析标识符、反事实组和去标识符后的真实表面模板哈希必须全部零重叠。audit 使用不同措辞模板，不能靠虚构 template ID 自证隔离。

冻结的 E5 cache 虽然包含整个 `zh_target_strict` seller 池，但生成器不得从整个 cache 抽样。载体白名单只能由573条 canonical train pair 的左右 seller 推导；审计逐个检查实际载体属于该白名单。旧 valid/test seller 即使不读标签也禁止作为合成载体。

## 5. 模型能看什么、不能看什么

模型只接收冻结 source probability 和七个由解析器/图构建器得到的数值特征。其中只有 `corroborated_private_occurrence` 与 `corroborated_two_hop_path` 可训练；单项 direct、普通 verified path 及三类风险/支持状态都被结构性固定为0系数。

以下信息只保存在独立 lineage/audit 文件，不进入特征矩阵：

- 虚拟控制者、虚拟卖家和 world ID；
- 真实内容载体 UID；
- 配方、模板和随机种子；
- 标识符原文及其哈希；
- 生成器直接写出的预期状态；
- label 来源规则。

审计会从保存的原始合成标题/描述重新运行生产解析器，并逐行重建七个模型特征。预期状态与实际解析状态必须100%一致。

## 6. 通过门槛及其含义

主要合成机制门槛：

- 每组正负 source probability 差不超过 `1e-12`；
- pair 与 graph 两轨的 M2 成对排序一致率分别不低于0.90和0.75；
- pair 与 graph 两轨的 M2 相对 M1 成对 log-loss 都至少下降0.10；
- corroborated-private 与 corroborated-path 两个系数均不低于0.10且不超过预注册上限；
- 单项 direct、普通 verified-path、risk、support、risky-path 五列系数必须严格为0；
- M1 与 N0 的所有系数必须回到0；
- synthetic audit 中的全部 neutral 行必须逐位回退冻结 source；真实573仅报告同一诊断，不参与门禁；
- 全局 AP/ROC 只作描述，不再用跨组 source 波动否决组内因果机制。

通过后只允许得出：在模板和标识符均隔离、包含直接外观碰撞及20%反向困难组的合成世界中，production parser 与有限佐证门的实现一致；拟合结果对多标识符佐证呈正向统计关系，并对单项 direct、普通 path、risk/support/no-shared 保持严格中性。它不能被表述为发现了真实身份规律。

它不证明真实地下市场性能，不解决真实 prospective ground truth 的缺口。真正的下一步是冻结 Step 28 的 policy、代码、模型、阈值和 manifest，再采集新的 score-blind prospective real holdout；历史 valid/test 继续封存。

## 7. 运行

数值实验统一在 Linux 环境执行：

```bash
bash scripts/run_step28_causal_identity_synthesis_linux_20260719.sh
```

runner 先执行契约测试，再执行生成、因果审计、训练、synthetic development audit 和同步清单闭包。该 audit 不是一次性 prospective sealed evaluation；真正的 sealed real evaluation 仍须等新数据采集后另建流程。所有产物写入新的 `reports/step28_causal_identity_synthesis/v3_20260719/`；同名文件若内容不同会立即失败。
