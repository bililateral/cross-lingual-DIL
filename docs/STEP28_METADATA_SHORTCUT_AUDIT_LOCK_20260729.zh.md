# Step 28-v13 统计捷径审计实现锁

更新时间：2026-07-29

状态：`CORE_IMPLEMENTATION_LOCKED_FORMAL_EXECUTION_BLOCKED`

## 1. 审计目的与当前边界

本锁逐项转录 Step 28-v13 数据集合同和 parent policy 已预注册的
`metadata_shortcut_audit`，不新增或修改科学门槛。它只检测本应与
controller/label 独立的 14 个 null-nuisance 特征，不能证明不存在所有未知
捷径。

`dataset_smoke_v3` 仅为 development smoke。构建合同第 14 节禁止在 smoke
上输出 `feature×label`、AUC、系数或排序；其 development 只有 3 个 world，
也不满足固定 world 五折。因此：

- 可以用无标签人工夹具测试投影与统计实现；
- 不得在 `dataset_smoke_v3` 上运行数值审计；
- 不得打开其 Audit A/B oracle 或标签；
- 不得把实现通过写成 `PASS_DATASET_ONLY`。

正式执行还要求 parent policy 冻结、功效产物、formal 数据、Identity
experiment lock、独立 supervision/custody、精确 NumPy/scikit-learn 环境和
只读挂载全部就绪。

当前实现锁不会把“禁止文件打开次数”硬编码成 0。是否真的没有打开
raw/oracle/identity/M0 等文件，必须由以后独立部署的 OS custody/access-log
回执证明；该回执、formal 根/child manifest、精确 world-set 和输入文件哈希
尚不存在，因此当前锁在代码层保持不可启用。未来必须新建版本化 execution
lock，不能把本锁的 `enabled` 原地改为 true。

本版所有公开 release writer 自身也重复执行该总门，不能通过导入 Python
函数绕过 CLI。本版执行门无条件拒绝 release：即使调用者复制内存 lock，
同时伪造 version、status 和任意 64 位哈希也不能启用。未来正式版必须另行
实现并验证 formal release、
custody/access、execution environment 与逐 split 精确输入 binding，不能只
检查字段“非空”。

## 2. 固定数据流

### 2.1 无标签 nuisance projector

每个 split 只允许读取：

```text
observed/candidate_pairs.csv
observed/history_item_index.csv
observed/redacted_items.jsonl
```

输出精确 16 列：

```text
canonical_pair_uid
world_uid
absdiff__item_count
absdiff__title_missing_rate
absdiff__description_missing_rate
absdiff__time_bucket_probability_00
absdiff__time_bucket_probability_01
absdiff__time_bucket_probability_02
absdiff__time_bucket_probability_03
sum__item_count
sum__title_missing_rate
sum__description_missing_rate
sum__time_bucket_probability_00
sum__time_bucket_probability_01
sum__time_bucket_probability_02
sum__time_bucket_probability_03
```

seller 的 7 维依次为商品数、标题缺失率、描述缺失率和 4 个时间桶概率。
pair 先连接 7 个绝对差，再连接 7 个和；全部用 `.12f` 序列化。每个 world
必须恰有 40 个 C40 pair、28 个 seller；每 seller 必须有 2–8 个 item。
observed 三文件的 world/seller/item keyset 必须闭合，不允许额外 world。
缺失、重复、跨 world、非法桶、unknown column 或非有限值使整个 split
失效。

projector 禁止读取 raw `items.jsonl`、seller profile、oracle、controller、
mechanism、identity33、M0、candidate sampling audit、parser/slot、placebo
或标签。

### 2.2 独立 supervision sealer

每个 split 只允许读取：

```text
observed/candidate_pairs.csv
oracle/controller_membership.csv
```

唯一公式：

```text
label = int(controller(left) == controller(right))
```

输出精确两列 `canonical_pair_uid,label`。train/development 分别封存；
Audit A/B 必须由不同 custody 在正式解封流程中处理，不能由训练前进程打开。
label sealer 不得读取 nuisance projection、文本、identity、mechanism、M0
或 adapter。

封存后另由独立公式校验器读取同一 C40、controller membership 和两列标签。
它不调用 sealer 的逐行相等判断，而是先按 `(world, controller)` 枚举全部
卖家组合，再以正 pair 集合成员关系复算标签；只发布通过布尔值、行数和哈希，
不发布类别计数。

### 2.3 nuisance validator

每个进程只允许挂载一个 split 的 sealed 16 列 projection、sealed 两列
labels、独立标签公式通过回执和最小统计锁。禁止挂载数据集父目录、oracle、
文本、parser、category、market、style、mechanism、identity33、M0 或
adapter。

train 与 development 分开运行，均通过才满足 metadata shortcut 子门。Audit
A/B 只在各自盲预测、统计代码和锁均冻结后顺序解封。

label sealer、公式校验器和统计 runner 都必须取得对应 split/operation 的外部
授权回执；development 绑定 train 子门回执，Audit A 绑定盲预测冻结与 A
解封回执，Audit B 还必须绑定完整数据集状态机签发的 Audit A
`PASS_A_ONLY` 父回执和 B 解封回执。仅有“A 已完成”、metadata 子审计通过、
A=`INVALID` 或 A=`NO_GO` 均不能解封 B；仅在 CLI 中写
`--split audit_b` 也不能构成授权。

## 3. 固定统计方法

检测器固定为：

1. fold-train `StandardScaler` +
   `LogisticRegression(lbfgs,L2,C=1,max_iter=10000,tol=1e-10)`；
2. `HistGradientBoostingClassifier(max_depth=2,max_iter=200,
   learning_rate=.03,l2_regularization=1,early_stopping=false,
   random_state=2026072707)`；
3. fold-train `StandardScaler` +
   `SVC(rbf,C=1,gamma=scale,probability=false)`。

三者均无 class weight，使用 `decision_function`。所有 14 列必须全部进入，
不得选特征、调参或按结果换模型。

world 依完整
`SHA256(ASCII "2026072707" + 0x1f + world_uid UTF-8)` 排序，序号模 5
分折。同一 world 的全部 seller/controller/pair 必须留在同一折。空折、
fold-train 或 fold-test 单类、未收敛和非有限数均为 validity failure。

每个模型计算：

```text
AUC_sym = max(ROC_AUC, 1 - ROC_AUC)
```

点统计为三个 `AUC_sym` 的最大值。AUC 接近 0 表示可反向预测，不能当随机。

固定 OOF score 后，以 world 为单位做 9,999 次 bootstrap，不重训。随机数为
NumPy `PCG64DXSM`，base seed `2026072711`，split seed 为
`SHA256(ASCII seed + 0x1f + split_name)` 的前 128 bit。每次先取三模型
`AUC_sym` 最大值，再用 `method="higher"` 取 95% 上界。
bootstrap draw 矩阵的审计哈希固定按 C-order、无符号 64 位大端
(`>u8`) 字节计算，不依赖 CPU 本机字节序。

通过必须同时满足：

```text
max_model_auc_symmetric <= 0.52
world_bootstrap_95_upper <= 0.53
```

没有预注册 permutation p-value，不得临时增加并替代上述门。该 probe 不做
threshold、F1 或校准；这些不是独立性 probe 的 estimand。

## 4. 明确不进入随机性门的合法信号

以下信号只能单独透明描述，不能进入 14 维随机 AUC 门：

- 脱敏文本的长度、数字、标点、换行、繁体和大写：注册的 authorship 信号；
- category/product/attribute equality：注册的 high-semantic 困难负例；
- market difference：注册的 cross-market 正机制。

它们的可预测性不能被解释为 identity adapter 的新增贡献。

## 5. 发布、失败与结论纪律

- 每个阶段使用新 staging 目录，文件 fsync 后原子 no-replace 发布；
- 每个 manifest 绑定精确输入 allow-list、文件大小/SHA、源码闭包、schema、
  row/keyset、fold、OOF、bootstrap 和 self-hash；
- `parent_manifests` 按 role 的 UTF-8 字节序排序；父 content hash 必须直接
  使用已验证父 manifest 的 `canonical_self_hash`；
- projection、label 和公式回执必须匹配冻结的精确版本；公式校验器把 label
  manifest 作为 DAG 父节点前，须验证其完整 schema、identity、输入
  allow-list、物理文件集及标签文件哈希；
- 目标已存在、输入漂移、环境漂移或任一 validity failure 均不得覆盖旧产物；
- 失败也必须形成不可变 failure report；
- label formula、metadata shortcut 和 mechanism coverage 三个 validator
  全部通过后，父状态才可能是 `PASS_DATASET_ONLY`；
- 当前锁不包含 mechanism coverage 实现，因此绝不授予该状态。

固定环境为 NumPy `2.2.6`、scikit-learn `1.7.1`。当前 Windows
scikit-learn `1.7.2` 只能运行实现测试，正式 runner 必须 fail closed。
