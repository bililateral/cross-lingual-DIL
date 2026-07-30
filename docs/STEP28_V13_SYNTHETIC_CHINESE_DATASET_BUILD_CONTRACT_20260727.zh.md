# Step 28-v13 中文合成身份数据集正式构建合同

更新时间：2026-07-29  
当前版本：`2026-07-29-step28-v13-synthetic-chinese-dataset-v13-draft`
  
当前状态：`DRAFT_NOT_FROZEN`。只有本文档、formal policy、代码、模板、种子
清单和功效设计产物的 SHA-256 全部互相绑定后，才能改为 `FROZEN`。

## 0. 文档地位

本文档是 Step 28-v13 中文合成数据集的当前权威实施合同。根目录
`Build_Plan.md` 只保留为讨论草稿；两者冲突时以本文档为准。

本项目使用两把锁、四个不可倒置的阶段：

1. **Dataset release lock**：冻结本文、dataset policy、模板、parser 完整
   上下文夹具、风格参考、
   功效设计、dataset producers 与 custody deployment；之后一次性生成并
   验收正式合成数据。它不要求尚未运行的 GPU input hash。
2. **无标签结构验收**：只运行自动结构门，只暴露固定 schema 所需计数、
   boolean 和 hash；通过后状态是
   `STRUCTURALLY_ACCEPTED_PENDING_SUPERVISION`，不是
   `PASS_DATASET_ONLY`。
3. **Identity experiment lock**：冻结 identity child policy、静态
   label-free CPU/GPU policy、M0/adapter/统计脚本、诊断安全投影、盲报告
   schema 和全部成功门；必须在任何模型、人类诊断或 adapter capability
   打开 train/development supervision 前完成。child 只能逐项转录 Dataset
   release lock 已冻结的科研选择，不得根据可观察 Audit A/B 的协变量修改。
4. **有监督数据有效性验收**：单独的 validator 才能打开 train/development
   supervision，分别验证标签公式、真正独立的 nuisance shortcut 和机制覆盖；全部通过后
   才进入 `PASS_DATASET_ONLY` 并允许训练 adapter。validator 只输出冻结
   报告，不与 adapter 共享进程或权限。

每次 M0 run 的动态 chunks、opaque input 和 map commitment 由随后 immutable
sync manifest 绑定。Dataset parent 中由 identity child 拥有的未来
CPU/GPU policy、compatibility result 和 identity deployment 不留可回填的
path/hash；parent 冻结后永远不修改。

本文档在正式冻结前只能通过有记录的新修订版完善。每次更新必须增加日期、
原因、影响范围和上一版文档 SHA-256；不得删除旧结论或静默改写已运行实验
的含义。工程冒烟使用独立的 `v13_dev_smoke_*` 命名空间，绝不与 formal
命名空间共享 world、controller、seller、item、identity、template、pair、
seed、salt 或输出路径。

一旦当前科学版本冻结，任何可能改变数据或结论的修改——包括风格统计、
生成器、机制、模板、世界数、候选/query 规则、seed/salt、parser 合同、
M1 重连、33 特征、尺度、M0/C0、损失、L2、threshold、slice、metric、
bootstrap、置信区间、实质门或删行规则——必须建立新的科学版本、run_id、
outputs_root、命名空间和 manifest；旧版本保持原字节及原状态。
`lifecycle_stage` 可以记录 `DATASET_RELEASE_LOCKED`、
`STRUCTURALLY_ACCEPTED_PENDING_SUPERVISION`、
`IDENTITY_EXPERIMENT_LOCKED` 等内部阶段；它们不是科研结论。最终
`scientific_status` 固定为：

```text
PASS_DATASET_ONLY | PASS_A_ONLY | PASS_A_AND_B | NO_GO | INVALID
```

任何正式 observed audit 被用于调整设计，或其 label/qrels、类别数、标签
派生统计被解封后，修复版必须使用全新的 audit world/controller/seller/item/
identity/template/pair/seed 命名空间。开 audit 后发现实现或统计错误时，
保留原产物并发布撤回或勘误记录；不得静默替换。只有逐字节相同的确定性
重放可以保留同一 run_id。任何正式输出不得用不同字节覆盖。

## 1. 科研问题与结论边界

本数据集服务于以下受控问题：

> 在冻结的合成中文市场生成分布和标签盲候选策略中，使用正确身份历史训练
> 的适配器，是否比不使用身份历史的冻结英文基础模型，以及使用标签盲、
> 端点不重合的 33 维整行错配训练的匹配对照更好？

模型角色固定为：

- `M0`：冻结的英文来源完整分类流水线，不在合成中文上训练；
- `M1`：冻结 M0 加只用世界内、C40/非 C40 分层的 33 维整行错配矩阵拟合
  的身份适配器；
- `M2`：冻结 M0 加只用正确合成训练历史拟合的身份适配器。

数据验收通过本身只支持“数据合同通过”。仅 Audit A 通过时，本实验最多
支持：

> 在冻结的 `G_A`、C40、operational M0 和五个预注册标签盲整行错配
> placebo 所定义
> 的合成总体及生产 parser 成功合同下，正确合成身份—标签对齐具有跨世界
> 新增预测训练价值。

只有同一冻结模型在 Audit B 也通过时，才可以增加：

> 该价值在本合同预注册的机制移位 `G_B` 下仍然成立。

它不能证明真实地下市场中的因果规律，也不能证明真实中文跨语言性能已经
改善。真实结论仍需要新的真实中文快照和从未参与开发的独立人工标签。

## 2. M0 兼容合同

当前主基础模型固定为：

```text
M0_operational_primary = LightGBM + legacy18 + LaBSE
```

敏感性底座固定为：

```text
C0 = LightGBM + legacy18
```

这里的 M0 是用户决定用于推进 Step 28 的 operational baseline，不改写
Step 7-v4.2 中“尚未由独立英文数据认证为唯一最强模型”的历史事实。

旧方案中的以下路径全部撤销：

```text
五字段卖家画像
→ passage: 前缀
→ multilingual-e5-large
→ Step24 标准化逻辑回归
```

v13 必须兼容的真实 M0 前向路径分成两条，不能把同一份 redacted items
误画成全部 legacy18 输入：

```text
合成原始商品
├─ Step3 ensure_profile/update_profile/finalize_profile
│  → 冻结 top/signature/snippet/rounding 语义
│  → Step7-v3.1 build_clean_seller_record
│  → 冻结英文 reference 下的 legacy18
└─ Step7-v4 redact_raw_field
   → 清洗后的逐商品 title/description
   → 四 tokenizer 共同 256-token 分块
   → 冻结 LaBSE，无 prompt/prefix
   → 六项 LaBSE 商品集合聚合

legacy18 + LaBSE6
→ primary 按冻结 joblib 的 24 维顺序和 24 个 imputation medians
→ 冻结 LightGBM
→ M0 概率
```

身份历史与 M0 清洗是两条独立路径，不能互相冒充：

```text
Step3 extract_item_identity_signals → occurrence → Step28 33维历史特征
Step3 profile → Step7-v3.1 clean    → legacy18
Step7-v4 redact_raw_field           → 身份删除逐商品文本 → LaBSE6
legacy18 + LaBSE6                   → 冻结 LightGBM M0
```

Step3 parser 代码 SHA-256 固定为
`4b531367d3f81e863d1fbae8be1e5b9de0867e59425d2270ce88d02bff69a202`；
Step7-v4 common/redactor 固定为
`8acdac12a579314ddf3e863e3b1c19a026fc252fe520a4ded3f53fda6e765334`；
其 lazy import 的 Step7-v3.1 source/legacy 实现固定为
`5c4f607b5fd17dc378cc89dc93c2ac1865db8e22ca6623551db08c7014d44734`；
33 维历史实现固定为
`39fe4e952563dcbee4b300c89f8c1c1072357b5687e7d128009c1b951f228eab`。
不得依据 planned occurrence 直接删文本后声称重放了 v4 redactor。全部
`must_extract`/实际 parsed identity surface 必须经 v4 redactor 验证无残留；
`must_ignore` 按定义不是身份，禁止借 oracle 删除，并须逐字节保留后审计。

C0 独立按其 joblib 的 18 维顺序和 18 个 medians 前向。两者都不重新
标准化。legacy18 只允许
`MODEL_ELIGIBLE_TRANSFER_FEATURE_NAMES`；`same_market_bool` 和
`same_source_dataset_bool` 仍为 audit-only shortcut，禁止进入模型。

冻结指纹如下：

| 对象 | SHA-256 |
| --- | --- |
| primary joblib | `aae7f0e520c28006471cf1e1e518d6b0d383e6d14810a2c4aca4d28aa11d91dd` |
| C0 joblib | `e6f7c3a4cb262e067b7308e298d1b850ab237cb08de66ab85fcbaf1896157e1c` |
| Step7-v4.1 policy | `804b78478344b1c18340a2bf657f6c5c003b701c9cc655bc5723e7e441b2678a` |
| M0 producer | `f8651574d20ae424f62a68f34da2cdadc44a23ae547039886a5bca4b5707a472` |
| Step7-v4 parent policy | `41193c9957fb80024a0e376c76ebcfe0851fe9fee8c15f597c436824fc3b2327` |
| Step7-v4 public replay producer | `f72cf8e0008636124a6168ed7203ddc2ef4ca5eea3bc94e66371565bb09860ae` |
| Step7-v4 chunk/encoder producer | `7f7018f55e543ad809152d786d6d0e34722f18141ace89b21d7d1eb660f548dc` |
| LaBSE content | `391b10840ba616f47c4799f23ca5a0f511c7198bebfc9ba5927345f2b5fb8a21` |
| PCM tokenizer/model content | `90b7b87f49de527078c9023308eb6186685221ffe1f4a3478a75ecfda817b413` |
| mStyleDistance tokenizer/model content | `6671396e1cf27009b90584869e8a7d95ed3a055192edae897b478afd256fdb27` |
| multilingual-e5-large tokenizer/model content | `d1541dcd59047401678c3fa66d3793ffead9b46edacf29958e1a0848612be4f2` |
| Step3 profile/compression schema | `7338bf6d10a20fdf6bc5da9cbcf81a4f0014c1861178644a6acbdc07bf495c86` |
| Step7-v3.1 source/clean policy | `0b9f65b54a38c615bfc8ebe5b0b4757a87434f55535eb5fafec92148d7c55c67` |
| 英文 reference | `825cc0a42806388de8f4f016273ed83650082f757eea660189e97e48a57853eb` |
| 英文 582 seller UID 集合 | `b417fbe6ec1c146943657b00de973889adb0732fbe4aa996297b6462447f8c0e` |

禁止对合成数据调用 `FeatureFactory.design()`，因为它会重新拟合目标域
reference。不得重新拟合中位数、LightGBM、阈值或特征顺序。

synthetic seller profile 必须直接复用 Step3 的
`ensure_profile/update_profile/build_specificity_catalog/finalize_profile`
和冻结 Step3 compression policy。八个 numeric profile 字段固定为 item
count、title/description length median、digit/punct ratio mean、repeated
title/description share、max category share，保留 Step3 的 float 计算与
六位小数 round 语义。随后才用 Step7-v3.1 的 clean-text contract 构造
`clean_categories/clean_titles/clean_descriptions`。不得从 redacted items
另算 numeric profile，因为那会改变冻结英文 legacy18 的输入语义。

Step3 specificity catalog 只在每个 28-seller world 内从该 world observed
raw profiles 无标签拟合；不得跨 world 或跨 split 计算。缺失、增加或置乱
其他 world 时，本 world 的 finalized profile bytes 必须不变。该 per-world
选择规则在四个 split 完全相同，后续仍只使用冻结英文 reference 计算 IDF
和 numeric percentiles。

Step7-v4 synthetic redaction registry 也只从同一 world 的 label-free
observed Step3 finalized profiles 与实际 parser output 构造，planned
expectation/oracle 禁止进入。必须调用冻结 helper：

- parser rows 先构成 `literals_by_seller`；
- `global_identity_tokens(literals_by_seller, profiles)` 生成经过
  `canonical_identifier_token` 规则过滤的 global tokens，不能把任意 raw
  email/URL 直接塞入；
- 每个 seller 的 literals/phrase tokens 分别由
  `seller_identity_literals(profile)` 和
  `seller_identity_phrase_tokens(profile)` 生成；
- contextual aliases 由
  `contextual_global_alias_tokens(profiles,literals_by_seller)` 生成，再用
  v4 冻结的 matcher、denylist 和碰撞判定算法过滤；deletions 使用
  `v4_contextual_alias_deletion_tokens`；
- seller contextual collision tokens 使用同一冻结算法生成；
- audited global phrases 使用冻结 Step7-v3.1 常量。

这 7 组 registry 参数及 seller UID 映射逐组保存 canonical hash。其他 world
的身份值缺失、增加或置乱
时，本 world redacted bytes 必须不变；全部 parsed identity surface 必须在
clean text 中无残留。禁止用 planned occurrence 帮 redactor “预知”生产
parser 没抽到的值。

这里**不重放**绑定英文 frozen snapshot 的 expected seller/hash/count
collision contract；那些 expected 值不能套到 synthetic UID。v13 为每个
world 另行生成无标签的 collision audit，冻结匹配算法、允许规则集合为空，
发现任一 content collision 即 fail closed。该 audit 的输入只能是 observed
profile、parser output 和商品文本，不能读取 planned/oracle/controller。

legacy18 的 `build_clean_seller_record` 使用它自己的冻结 v3.1 registry
接口：同一 world profiles/parser literals 输入
`global_identity_tokens/contextual_global_alias_tokens/
contextual_alias_deletion_tokens/AUDITED_GLOBAL_IDENTITY_PHRASE_TOKENS`；
seller literals/phrases由 finalized profile 内部推导。该 registry 同样
per-world 构造、逐组哈希并禁止 planned/oracle。

M1 只改变 placebo train 的 33 维 `z`。它必须按 pair UID 复用 observed/M2
已冻结的 legacy18、LaBSE6 和 p0；placebo worker 禁止重新生成或覆盖这些
列。M1 从已验真的 M2 `z` 整行复制，禁止逐列拼接；因此“p0 相同”来自
显式 immutable join 合同，M1 根本不重新编码 raw profile。文本重渲染只保留
为开发期 parser/renderer 压力诊断，不再产生主 M1。另仍要求同类型身份值
具有固定 surface signature，并验证 v4 redacted item text，作为 M2 非身份
内容没有漂移的独立工程门。

LaBSE6 的商品集合聚合固定
`chunk_to_unique_text=l2_normalized_mean_of_unit_chunk_vectors_then_l2_normalize`、
`unique_text_weighting_primary=equal_weight_per_unique_clean_text`、
`top_k=3`。六列名称和顺序唯一固定为：

1. `raw_labse_field_equal_centroid_cosine`；
2. `raw_labse_field_equal_symmetric_top3_cosine`；
3. `raw_labse_title_centroid_cosine`；
4. `raw_labse_title_symmetric_top3_cosine`；
5. `raw_labse_description_centroid_cosine`；
6. `raw_labse_description_symmetric_top3_cosine`。

字段不存在时先写 empty CSV value，随后只使用冻结 primary joblib 已保存的
训练期 median 填补，不添加 missing indicator。GPU aggregate 先按 12 位
小数写入正式 CSV，CPU 再从该 CSV 解析 float64 并按 joblib feature order
组矩阵；禁止绕过落盘精度，直接用未舍入内存值评分。

数据集构建阶段不得加载 LaBSE、LightGBM 或读取 M0 分数。M0 精确重放属于
数据集通过后的独立评分阶段。

## 3. 真实中文参考边界

### 3.1 允许内容

风格参考只允许来自当前真实中文训练期的原始商品级或单卖家级信息：

- 每个卖家的商品数；
- 标题、描述字符长度和字段缺失率；
- 商品类别边际分布；
- 数字、标点、换行、简繁体和英文大写比例。

统计单位固定为 seller，禁止让商品多的 seller 获得更高权重。先对每个
seller 计算 item count、标题/描述长度中位数、字段缺失率，以及逐商品
数字/标点/换行/简繁/英文大写比例的 seller 内均值；再对 seller 统计量用
Hyndman–Fan type 7 线性分位数汇总
`{0.05,0.10,0.25,0.50,0.75,0.90,0.95}`。item count 先确定性 clip 到
`[2,8]` 后保存 7 桶频率。类别先按 seller 内相对频率归一化再做 seller
等权平均，seller support 少于 20 的类别合并为 `other`；随后只保存降序
概率向量，丢弃全部真实类别名称。向量固定 8 维：保留前 7 个概率秩，其余
尾部质量合并为第 8 秩；不足 8 项补 0，再归一化。生成器按顺序把 8 个概率
秩映射到模板库预先人工编写的 8 个通用类别词，不能复制真实类别字符串。

最终参考文件只保存上述聚合直方图、分位数和匿名类别秩概率。不得保存真实
seller 行、seller UID、完整文本、文本片段、真实类别名称、极值来源或可逆
统计。v13 不从真实数据提取联系方式、字段位置、市场或时间分布；这些全部
由第 4、6 节的合成合同独立决定。

### 3.2 禁止内容

风格提取和生成进程不得读取：

- 英文或中文 pair 标签；
- valid/test 分区的单独风格画像；
- pair UID、候选结论、证据类型或模型输出；
- Step24、Step27、Step28 或 Step7 分数；
- 真实跨卖家身份复用、图度数、轮换或连通分量统计；
- 真实标题、描述、短文本片段或罕见短语；
- 真实 Telegram、邮箱、BAT、QQ、微信、电话、钱包或网址值。

训练卖家范围如果需要从受保护 split 存储中物化，必须由单独的边界脚本只
输出 `seller_uid` allow-list；风格提取器仍不得打开原标签文件。边界产物、
输入列和访问日志必须写入清单。

### 3.3 强制证明

- 使用标签不存在、标签清空、标签置乱三种环境时，风格参考字节哈希相同；
- 仅此测试不够，还必须有文件 allow-list、禁止路径 deny-list 和上游血缘；
- 禁止打开次数必须为 0。

## 4. 世界、卖家和商品规模

每个世界固定：

- 12 个隐藏控制者；
- 8 个控制者各有 2 个卖家，4 个控制者各有 3 个卖家；
- 恰好 28 个卖家；
- 哪 4 个控制者拥有 3 个卖家由独立随机流确定；
- 所有卖家均至少有一个同控制者伙伴。

正式分区固定为：

| 分区 | 世界数 | 卖家数 | 分类候选数 |
| --- | ---: | ---: | ---: |
| train | 500 | 14,000 | 20,000 |
| development | 125 | 3,500 | 5,000 |
| audit_a | 250 | 7,000 | 10,000 |
| audit_b | 250 | 7,000 | 10,000 |
| 合计 | 1,125 | 31,500 | 45,000 |

每个卖家的原始商品数从冻结真实训练期边际分布独立采样，再确定性裁剪到
`[2,8]`。商品数不能依赖 controller、机制或最终标签。controller 的机制
映射也完全不读取商品数；所有 seller 至少两件商品，因此任一预注册机制都
具有最低槽位容量。不得因机制、标签或结果重采样商品数。预计正式商品总数
为 10 万至 15 万；实际值只在生成后报告，不用下游成绩选择。

每个 seller 在 v4 redaction 后必须至少保留一个非空 title 和一个非空
description；其余 item 仍可按风格参考模拟缺失。该规则只依 seller/item
结构，不读 controller、机制、pair 或 label。dataset gate 必须逐 seller
验证，从而保证 evaluation pair union 的两侧至少有 title 与 description
共同字段，不能等 M0 报错后删行或重造。

四个分区的以下集合交集必须全部为 0：

- world UID；
- controller UID；
- seller UID；
- item UID；
- 规范化身份值；
- 身份图连通分量；
- 模板定义/骨架哈希；
- 渲染后模板实例哈希；
- 完整规范化商品文本。

同一 v13 mode 内，任一 normalized identity value 也只能属于一个 world；
只有该 world 内预注册的复用、hub、collision 或 rotation 才能共享它。不得在
同一 split 的不同 world 重用值。development smoke 与 formal 使用独立 key，
二者的 identity value 交集也必须为 0。

ID 使用独立随机命名空间和不透明值，不得包含 split、controller、机制、
标签或其顺序。结构随机数、ID 随机数、文本随机数、候选随机数和重连随机数
必须使用独立派生流。

正式 v13 的 world、seller、identity、pair、template 和 seed 命名空间还
必须与 Step28-v6 至 v12 的正式合成命名空间零交集；不能只证明 v13 四分区
内部隔离。

## 5. 中文文本生成

真实文本不得进入模板或词库。文本只能来自：

- 人工编写的语法模板；
- 白名单通用类别词、功能词和常见短语；
- 合成商品属性、合成身份和合成噪声值。

风格使用有限共享原型，而不是控制者专属暗号：

- 全局固定 12 个普通风格原型；
- 每个世界标签盲随机抽取 4 个；
- 每个原型恰好分配给 3 个不同控制者；
- 12 个 controller 先按秘密结构 HMAC 排序，连续三人一组分给上述 4 个
  原型；每个 seller 继承其 controller 原型，再只用公开 text HMAC 选择
  恰好两个不同风格因子，各自在冻结域中前移一格；
- 完整模板不得只服务一个控制者；
- 不允许控制者专属罕见词、模板子库、标点组合或类别组合。

这里的共享作者风格是 **M0 被允许利用的预注册基线信号**；它不是 33 维身份
模块的证据。长度、数字、标点、换行、繁体和大写等 redacted 文本统计不得
再被同时放入“必须 AUC≈0.5”的 nuisance 门，否则会把合法基线信号误判成
生成捷径。类别/商品/属性相同也是 high-semantic 困难负例的预注册信号，
跨市场则是 cross-market 正机制；二者同样只作透明描述，不进入随机性门。

### 5.1 非身份商品 DGP

所有非身份选择必须先于身份槽位生成，并按机器 policy 的域分离 HMAC
逐字节重放：

- seller 商品数按冻结真实中文 train 风格参考的 2–8 PMF 抽取；
- title/description 缺失率由冻结 seller 等权分位曲线逆变换产生；每 seller
  对完整 mask 依次提案，接受首个“至少 1 个非空 title、至少 2 个非空
  description”的提案，最多 10,000 次；该条件对所有 seller 相同且不读
  controller、机制、候选或标签；
- 每 item 的 `time_bucket` 只保存 JSON 整数 `0..3`，各 `1/4`；不得再自由
  生成 timestamp/date；
- category 按匿名 8 类概率、product 按冻结 category→product 表、attribute、
  delivery、service 和 split-local skeleton 按冻结列表抽取；
- code 固定为 `Q` 加 item-domain HMAC 前 10 个 hex nibble；每个 nibble
  逐一映射为 `A..P`，因此 code 总长 11 个字符且不会被 Step3 当作长数字
  QQ；全 mode 冲突即失败；
- 每个 split 恰有 4 个 title skeleton 直接显示 code，另 4 个不显示 code，
  而是用 code 倒数第二个 `A..P` 符号无标签映射到 16 个共享自然款式词；
  非空 seller 英文 tag 只在 code 最后一个符号为 `A/B/C` 时追加。款式词和
  tag 均不能读取 controller、机制、候选或标签，也不能成为 item 唯一标识；
- 非空 description 的 byte 顺序固定为“style 后的 base + 一个 must-ignore
  noise 或空”；若没有 identity clause，到此结束且不得添加 guard；若有
  `N` 个 clause，则按 item UID 对 12 个中性 guard 做 SHA-256 排序，依次
  使用互不相同的 `N+1` 个 guard：第一个在首个 clause 前，其余分别位于
  每个 clause 后。每个 guard 长 92–103 code point，超过 parser 的
  90-code-point 上下文半径；
- style 只改变 base。noise、guard、identity surface 任一字节都不得被繁简、
  换行或重复标点转换。

每 seller 恰放一个 parser-safe、redactor-safe 的 must-ignore noise；12 位
纯数字 `123456789012` 虽不被 Step3 抽取，却会被 Step7-v4 generic redactor
删除，故只留在 parser 压力夹具，严禁进入主 DGP。exact-title clone 和
high-semantic 只能改预注册的 designated target：两侧、非空 item、共同
category/product/attribute、不同 skeleton 均按 policy 的 asset-index HMAC
唯一确定，禁止看 C40 或成绩后重选。

分区模板纪律：

- renderer 代码和语法规范可以共享；
- train、development、audit_a、audit_b 的完整模板定义/骨架库 SHA-256
  集合必须两两零交集，不能只替换 slot 值制造“模板隔离”；
- 渲染后模板实例文本哈希也必须两两零交集；
- audit_b 仍使用同一中文 renderer 语法，只改变下文预注册的身份机制组合
  和强度；
- 完全不同 renderer 只作为探索性压力切片，不混入 audit_b 主确认。

防复制门：

- 合成与真实规范化完整标题、描述交集为 0；
- 不存在长度不小于 24 字符的真实连续子串；
- 真实罕见 12–23 字符 n-gram 命中为 0；
- 所有真实规范化身份值命中为 0；
- 字符 n-gram MinHash 近重复率不得超过 policy 的预注册阈值；
- 不得根据 M0 或适配器成绩筛选“更像真实”的文本。

由于 M0 的英文训练数据存在精确文本克隆捷径，控制者身份不得由精确标题或
描述克隆唯一决定。必须报告完整数据与“去精确清洗文本克隆”切片。

## 6. 身份世界

八类身份固定为 Telegram、邮箱、BAT、QQ、微信、电话、0x 钱包和外部网址。
正式生成允许的“模板角色×身份类型”不是任意组合：

- `direct_or_private` 与 `high_frequency_direct`：除 external URL 外七类；
- `public_support`：只允许邮箱、电话、external URL；
- `risky_product`：全部八类。

不存在正式 public-support 钱包模板；“公共支付/托管”不作为本版本已覆盖的
独立负机制。每个 identity 的全部 occurrence 必须使用同一模板角色，并在
实际 parser 后落入同一个 observable nuisance class。

### 6.1 唯一标签熵与身份值

正式 train、development、Audit A、Audit B 使用四把互不派生的 256-bit
秘密结构钥匙。公开 policy 只保存各自原始 key 的 SHA-256 commitment；原始
key 仅进入对应 split 的独立 custody account。A/B key 不能进入同一进程，
生成并 seal 后撤销 generator 权限。早期 draft 中公开过的结构 key 已永久
作废，其 commitment 列入 deny-list。

四把正式 key 还必须执行一次性 key ceremony。每个 split 的独立 custody
account 只能调用 OS CSPRNG 一次取得 32 bytes，候选数必须恰为 1；在此前
不得物化或筛选该 split 的 formal 协变量、拓扑、C40、覆盖或任何关联统计。
生成后立即把 split、单调序号、UTC 时间、commitment、CSPRNG API、host/HSM
attestation、account 和前一 receipt hash 写入 append-only chain，再把
commitment 冻结到 release。任何第二个候选、丢弃 key、预承诺试跑、receipt
断链或换 key 都令整个版本 invalid，防止在提交 seed 前“挑一个好看的世界”。

公开 ID 流先为一个 mode 生成 `sum(split world counts)` 个 world UID 的全局
池，world ordinal 不得在 split 内重启；再用独立、标签盲
`id_namespace` HMAC permutation 分配固定 split 数量。随后每 world 产生与
标签无关的 12-controller UID 池和 28-seller UID 池；
seller ordinal 只是池内位置，绝不按 controller、机制或生成记录排序。只有
秘密结构流把 seller 池分成 8 个 dyad 和 4 个 triad，再分配机制。因此仅凭
公开 ID/text/candidate/query key 和 observed 文件不得重建 seller→controller。
formal freeze 必须通过“不提供结构 secret 无法产出 membership 表”的
canonical test。

### 6.1.1 独立私有 DGP 重放

正式结构验收必须拆成四个互不替代的 capability：

1. split generator 使用本 split 唯一结构密钥，封存 observed 与 producer
   private 决策表；
2. independent DGP replayer 在空工作区中只读取公开 policy、该 split 的
   **完整登记 world 集合**、`seller/all-item/nonempty-title/
   nonempty-description` UID-only 池和这一把结构密钥，自行重建 private
   决策；少一个、多个、跨 split 或乱序 world 都失败；
3. producer-private projector 在 producer custody 内把完整 oracle 缩成
   exact-schema typed projection；raw identity value、rendered occurrence
   identity UID、文本、slot/flow、parser 输出及非比较 solver 字段不得进入
   projection；精确比较身份拓扑所必需的 synthetic `identity_asset_uid` 保留；
4. no-key comparator 在结构密钥已卸载后，只读取两份最小 projection 及各自
   immutable parent manifest，先验证 policy、完整 world 集、文件/内容 hash、
   source closure 和 custody parent，再输出固定计数、SHA-256 与
   exact-equality boolean。

replayer 的决策实现只能使用标准库密码学、整数、Decimal、JSON/CSV 原语；
不得 import `step28_v13_structure`、`world_builder`、`identity_plan`、
`nonidentity`、renderer、parser、redactor 或 producer common 的随机抽样
helper，也不得读取 producer 的 controller、mechanism、asset、target、
override、solver、slot、AST、raw text、label、qrels 或模型文件。公开 policy
必须自包含 style ID、类别/商品/属性、类别概率和标题骨架数量等 choice domain，
且 producer 在生成前逐字节验证这些副本与已登记 template/style input 相同。

独立重放至少逐 world 产出并精确比较：

- controller membership、seller market、controller style group；
- mechanism 名称及完整 `#ordinal` slot UID；
- positive target、hard-negative distinct typed-membership topology 和
  negative target；
- identity asset topology、asset UID、identity type；
- stable identity repeat、single-hop path repeat 与 repeat side；
- high-semantic 和 exact-title-clone 的左右角色、具体 item；前者还包括
  category、product、attribute 与两侧 title skeleton index。

hard-negative 回退叶只按会改变 identity-type feasibility 的
private-collision/false-rotation topology 计数。对每个 topology，只在登记
HMAC 顺序中取第一个完整合法的 exact-clone/high-semantic override completion；
不得把同一身份 topology 的不同文本 override 组合重复当作“新叶”测试。
若第 0 个 topology 类型容量不可行，必须转到真正不同的第 1 个 topology。

ledger 自哈希只能发现传输损坏，不能证明它来自独立 replayer。正式证据必须
再由 capability launcher 的 immutable parent manifest 绑定 ledger、receipt、
公开 policy、全部 UID pool、replayer source closure、访问日志和运行账户。
producer projection 也必须有对称 parent。只同步替换 producer 与 ledger 后
重算 self-hash，不得通过 comparator。

同 producer 再生成一次只能发现内存或落盘产物被改，不能发现 producer
自身算法错误。它只允许标记为
`DEVELOPMENT_SMOKE_SAME_IMPLEMENTATION_NOT_FORMAL_SEAL`，
必须显式写 `independent_replay=false`，不得满足 formal gate。

当前 Windows 开发验证已在 23 个 smoke world（四 split、2318 items）上使
第二实现与 producer 全项一致，并通过替代结构密钥完整图攻击；这只是开发期
实现证据。正式 `FORMAL_SPLIT_PRIVATE_INDEPENDENT_DGP_REPLAY` 仍要求冻结的
dataset custody deployment、四个隔离账户、真实 key commitment、访问日志、
source-closure hash 和无密钥 comparator receipt，当前不得提前宣称通过。
现有 `step28_v13_generate_dataset.py`、development replayer 和 development
comparator 在函数入口即拒绝 `formal`；`--validate-config-only` 在打开密钥、
UID pool、oracle 或 ledger 前返回。它们不能通过改 policy 状态直接升级成
formal launcher。

本门的范围仍是 typed structure，不是完整文本世界的第二实现：identity
value、occurrence-to-item flow、最终 rendered text 与 parser output 仍由各自
独立结构门负责；exact-title-clone 在此只认证 source/destination seller 与
item 选择，最终标题逐字节相等由文本结构门认证。不得把
`full_typed_projection_exact` 表述为“完整 DGP/文本实现全部独立通过”。

同理，每 world 在结构分配前固定生成 96 个 `ias_` identity-asset UID。
结构层只把需要的 asset descriptor 通过秘密 HMAC 排序映射到该公共池；未用
asset 不落入 observed 文件。每个 asset 的可见值不得编码 controller、
mechanism、role 或生成顺序。值由 mode 独立 `identity_value_key` 对
`global_asset_index` 作 per-type 仿射置换，再按固定 grammar 编码；QQ/电话
等有限域也因此是严格无碰撞映射，而不是依赖“碰撞概率很小”。每 type 的
salt 从 0 起做 label/structure-blind 搜索，先冻结 smoke、再冻结 formal，
要求完整候选池与真实身份 deny set、v6–v12 及另一 mode 零交集。formal
世界数改变后必须重做 salt artifact 并发布合同修订，不能沿用不匹配的池。

下游身份主键不是 Step3 大小写敏感值，而是
`step28_history_common.token_key` 的
`(contact_type.strip().lower(), normalized_value.strip().lower())`；identity
UID 是该二元组 canonical JSON 的完整 SHA-256。正式值 grammar 固定：

```text
telegram      tg + 14位小写字母数字
email         u + 16位小写hex + @id.invalid
bat           bt + 14位小写字母数字
qq            9位数字且首位非0
wechat        wx + 14位小写字母数字
phone         11位数字，13开头，不含+号或分隔符
crypto_wallet 0x + 40位小写hex
external_url  s + 12位小写hex + .example/path/ + 8位小写hex
```

Base58/TRON 等大小写敏感钱包禁止进入 formal，以免 Step3 图和 33 维图把同一
值解释成不同实体。每个 identity 的 raw surface 固定，不作同值多写法变体。

完整 identity asset 按已分配 `ias_` UID 排序做全局 type DFS；每个候选 type
按秘密 HMAC 和 policy type 次序唯一排序，并同时满足 background/multi/
rotation 的 distinct-type 约束及 seller 非空 description 容量。随后用固定
整数 Dinic 在“occurrence→同 seller 非空 item/type slot”图上饱和匹配；边
kind、HMAC、UTF-8 tie-break、reverse-edge 插入顺序和 slot ordinal 全在
policy 固定。第三方 solver、运行时字典顺序或失败后换 seed 均禁止。

### 6.2 背景可交换脚手架

每个 seller 恰有两个 seller-unique 背景身份。56 条背景边按 seller UID 和
slot ordinal 排序，使用一个 world HMAC offset 在七个 direct type 上循环，
因此每 type 恰有 8 条边且同一 seller 的两类不同。每 type 的 8 条边又恰有
4 条 1-occurrence、4 条 2-occurrence；具体按 `1 + (edge_ordinal mod 2)`
交替，因此每个 seller 也恰有一条 count=1 和一条 count=2 背景边，避免
某个 seller 的两个 direct type 同时耗尽固定容量。重复 occurrence 必须在不同 item。
这些 identity 的 seller degree 均为 1，不给任一 pair 制造共享证据，但保证
五个 M1 在每个 type/count stratum 都有标签盲可交换容量。

### 6.3 `G_A`、`G_B` 与正机制拓扑

市场先从三市场各 `1/3` 的 seller-iid secret-HMAC proposal 生成；统一对
`G_A/G_B` 接受第一个“至少一个 multi-market dyad 且至少一个
multi-market triad”的完整 world proposal，最多 10,000 次。该可行性条件
对 controller UID 对称且在机制分配前执行，避免一次性 formal 数据约 26%
因极少数无可用 cross-market pool 而整体作废；耗尽才 invalid，不能换 world
UID 或 secret。随后才分配机制。`G_A` 的四个 triad 固定承载：1 个单跳 rotation、2 个
corroborated rotation、1 个 cross-market；其余机制在 8 个 dyad。`G_B`
四个 triad 全为 1+3 个 rotation，其余均在 dyad。cross-market slot 只在
相应 size pool 中有至少两个市场的 controller 内按 secret HMAC 选择；容量
不足直接令 split invalid，禁止重抽市场或换 seed。

| 机制 | `G_A` | `G_B` | 精确可见拓扑 |
| --- | ---: | ---: | --- |
| stable | 2 dyad | 1 dyad | 一个 token 连全部 seller |
| multi-type | 2 dyad | 1 dyad | 两个不同 type token 各连全部 seller |
| cross-market | 1 dyad+1 triad | 1 dyad | 一个 token 只连 HMAC 选中的异市场 pair |
| single-hop rotation | 1 triad | 1 triad | A 连 L-M，B 连 M-R，A≠B；目标 pair 仅 L-R |
| corroborated rotation | 2 triad | 3 triad | 两个 token 连 L-M，另两个连 M-R；目标仅 L-R |
| sparse | 1 dyad | 2 dyad | 一个 token 连两 seller，每边一次 |
| no-direct-share | 1 dyad | 1 dyad | 不生成 controller identity；允许公共/风险噪声 |
| zero-visible | 1 dyad | 2 dyad | 不生成 controller identity，且两 seller 排除全部共享 identity 噪声 |

stable/cross 对每个 identity asset 抽一次、multi 对两个 identity 分别抽，
概率为 `G_A=.80/G_B=.55`；同一 identity 的全部 seller edges 因而统一为
count 1 或 count 2。single-hop 先以 `G_A=.70/G_B=.45` 对整条 path 抽一次；
成功时只随机选择 L-M 或 M-R 一侧重复，所以每侧边际是 `p/2` 且绝不会两侧
同时 corroborated。corroborated 机制已由每侧两个 token 保证佐证，再对每条
identity asset 独立以 `.70/.45` 决定其全部 seller edges 是否加第二
occurrence。这样任一 identity 只属于一个 occurrence-count stratum。

triad 中只有上述 designated target pair 归因于该正机制。rotation 的
L-M/M-R 和 cross-market 第三 seller 形成的 pair 只是 supporting positive，
不得冒充机制覆盖。

### 6.4 困难负向 DGP

每个 `G_A` world 固定：support hub degree 4、6 各一个；direct high-frequency
hub degree 4 一个；两个 risky token 各 degree 3；两个 private-collision
target；一个三 controller false-rotation target；两个 exact-title-clone
target；四个 high-semantic target。`G_B` 对应为 support 6、8，direct hub
8，三个 risky token 各 4，三个 collision、两个 false rotation，clone 与
semantic 仍为 2/4。

共享 identity 的 seller 必须来自互异 controller，zero-visible controller
seller 全部排除。support 只从邮箱/电话/URL 取 type；direct hub 与 collision
只从七个 direct type；risky 可用八类。collision 的计数单位是“一 identity
+一 designated pair”；false rotation 是来自三个 controller 的 L-M token A
与 M-R token B，目标仅 L-R。clone/semantic/collision/false-rotation 的
designated pair 列表互不重复；seller 可复用，hub 引起的 incidental overlap
允许并保留多标签。

direct high-frequency hub 必须先通过固定背景身份的类型容量必要条件。对按
原 HMAC 顺序得到的 controller 子集，先尝试原规则为每个 controller 选择的
首位 seller；若这组 seller 在七种 direct type 中没有任何一种满足
`固定背景需求 + hub需求1 <= 每seller/type容量2`，则枚举该 controller
子集中“每 controller 一个 seller”的组合，仅保留满足上述必要条件的组合，
按 `asset_kind=high_frequency_direct_hub_capacity_fallback` 的同一完整 HMAC
消息排序并取首项；当前子集无解才进入下一个原顺序 controller 子集。该回退
不得读取 label、C40、文本、缺失率、模型分数或后续类型求解结果；所有子集
均无解则 world validity failure。生产器和独立重放器必须分别实现并逐字段
一致。

high-semantic pair 使用相同通用 category/product/attribute、不同 split-local
title skeleton 和不同 code；world-local Step4 lexical similarity 必须一次性
达到 `0.20`，否则本 split invalid，不得换 pair 或重渲染。

负 flag 顺序固定为 support hub、direct high-frequency hub、risky token、
private collision、false rotation、exact clone、high semantic、raw33 全零
negative。hub flag 归属于共享同一 hub token 的 pair；false rotation 只标
L-R；最后一项严格定义为 `label=0 && raw33==0`。

机制覆盖只计算 C40 中真正的 designated target pair：对每个 mechanism，
统计“至少选中一个相应 target 的唯一 world 数”。负例允许多标签。具体最低
独立世界数与每 split 的 zero-history 最低 row/world 数在不接触 formal
seed 的功效 artifact 中先冻结；未达门是 validity failure，不是效果
`NO_GO`，也不得补样本。

## 7. 解析器三态合同

计划身份 occurrence 必须带：

```text
parser_expectation =
  must_extract | must_ignore | stress_unconstrained
```

- `must_extract`：主数据身份，按 item/type/normalized_value 精确召回 100%；
- `must_ignore`：普通数字、无效账号和干扰文本，误提取为 0；
- `stress_unconstrained`：格式破损或未支持变体，只作压力切片。

主 M0/M1/M2 实验只使用合同明确的 parser 输出。若主实验全部使用
`must_extract`，结论必须写成“在解析成功条件下的身份历史价值”，不能外推
真实解析器召回率。

正式 parser wrapper 的唯一调用固定为：

```python
extract_item_identity_signals(
    meta,
    title_raw=item["title"],
    description_raw=item["description"],
    structured_snapshot="",
    extra_fields=None,
)
```

禁止向 `structured_snapshot`、`extra_fields` 或 `meta` 的非血缘字段注入身份、
机制、controller、模板或任何额外文本。`source_dataset/source_row_number`
只用于把 parser row 无损外连接回 item UID，且严格固定为
`source_dataset=step28_v13_<mode>_<split>`、`source_row_number=item_uid`；
该二元组与 item UID 必须一一对应。

`meta` 只允许：上述 lineage、`seller_uid`、独立采样的
`source_market_raw`，以及均逐字等于 seller UID 的
`source_seller_raw/source_seller_id_raw/alias_normalized`。不得增加字段、
跨 seller 共享 alias，或把 controller/mechanism/结构 ordinal 编入 profile。

formal 的 `must_extract/must_ignore` 只允许在 description，title identity
slot 数固定为 0。只有实际含 identity clause 的 description 才有 guard；
renderer 使用按 item UID 排序后互不相同的 `N+1` 个 92–103-code-point
中性 guard，因此 clause 前、相邻 clause 之间和最后均恰有一个 guard。
guard 不命中冻结 Step3 的 seller-contact、wallet-cue 或 product-risk regex。
style 的繁简、换行与重复标点变换只允许作用于 base skeleton 的
product/attribute/code/delivery/service 段；noise、guard 和 identity clause
在 style 处理后按 AST 追加，全部 176 种可达 effective style 下 guard
字节必须完全相同。

Dataset release lock 前，必须用冻结 parser 对全部 description skeleton、
176 种可达 effective style、全部 delivery/service/noise、16 个标题款式词、
25 个允许 role×type、全部相邻 role/type 有序对及 8-type 最大多 slot case
跑完整渲染夹具。不得只测孤立 clause。夹具同时重放 Step3 item normalization、
`step28_history_common.token_key` 以及 Step3 profile→Step4
`normalize_contact_value`；三条链的预期 key 均在 fixture 中逐项固定。

正式 M0/M1/M2 世界只允许 `must_extract` 与 `must_ignore`；
`stress_unconstrained` 必须进入完全独立、非确认的探索集，不能进入 45,000
条分类主样本或正式检索。`must_extract` 除 type/value 召回外，还必须核验
`seller_facing_context`、`product_data_risk_context`、
`direct_identity_eligible` 和 `support_only` 与 planned expectation
逐项一致。

`parser_expectations.csv` 必须由冻结 full-render fixture 和渲染结构规则在
production parser 运行前物化，不能复制同一次 parser 输出形成自证。parser
worker 不得挂载 expectation；独立 structural auditor 才比较 expected 与
actual。

正式 parser 门必须检查集合严格相等，而不是只算召回率：

```text
parsed_set ==
{(item_uid, source_field, contact_type, normalized_value, expected_flags)
 for every must_extract plan row}
```

不得存在未计划的额外 parser row。每个 item 内禁止重复
`(source_field,type,normalized_value)` 计划槽；同一 seller–identity 的重复
证据若用于 repeated feature，必须位于不同 item。`must_ignore` 的 raw
surface 不得命中任一 parser row。

Step3 parser 本身不输出文本 span，且会在单商品内按
`(source_field, contact_type, normalized_value)` 去重。因此 parser occurrence
表不得伪造 span；文本可编辑位置由 renderer 独立保存的稳定 `slot_uid`
承担。`signal_uid` 含 identity value，重连后会改变，不能当稳定 slot ID。
offset 单位固定为 Python Unicode code point，区间为 `[start,end)`；直接
替换最终字符串时必须按 start 逆序执行并重算全部 offset。正式实现应优先从
落盘的 `render_asts.jsonl` opaque placeholder 重渲染，再独立核对 offset。

must-extract 和 must-ignore 分表：`renderer_identity_slots.audit.csv` 才有
identity UID/type/value、planned role 与 expected flags；
`renderer_noise_slots.audit.csv` 只有 noise slot、item/seller/field/span/
surface 和 `must_ignore`，不得伪造 identity UID。M1 只接收从 actual parser
与 identity-slot 表一对一 join 得到的安全投影；其中只保留 actual flags 和
actual-derived nuisance class，不含 planned role、expectation、controller 或
mechanism。

## 8. 唯一生成顺序与标签边界

正式顺序不可改变：

```text
冻结风格参考与模板
→ 生成隐藏控制者
→ 生成身份资产和公共噪声池
→ 生成卖家、商品和身份槽位
→ 渲染文本并注入身份
→ 冻结原始世界及哈希
→ 运行生产解析器
→ 运行标签盲候选生成器
→ 固定分类候选和检索查询
→ 候选清单哈希冻结
→ 隔离 oracle 根据 controller equality 生成标签
```

唯一标签公式：

```text
controller(left) == controller(right) → positive
controller(left) != controller(right) → negative
```

共享身份、文本相似、候选触发、M0 分数或任何手工机制名称都不能直接决定
标签。

controller membership 可以在对应 split 的 custodial generator 私有内存和
只读 oracle custody 中先存在；但 train/development/A/B 必须由四个独立
进程运行，A 与 B 还必须使用不同 OS account/root/key，任何进程都不能同时
拿到两把 audit secret。从“原始世界哈希冻结”起撤销 generator secret
挂载，parser、candidate、query、M0、M1、feature 和 training worker 都不得
挂载 oracle 路径。候选不变性测试必须由只挂载 observed 的独立进程完成。

C40 与 query/gallery 清单冻结后，独立 supervision worker 才能物化 train
和 development 的 classification labels；train/development 不定义检索任务，
也不得生成 retrieval qrels。另两个互相隔离的 sealing worker 分别物化
Audit A 与 Audit B 的 classification labels 和 retrieval qrels。A、B 使用
不同目录、访问控制/密钥、访问日志和哈希清单；模型开发进程永远不得读取
`sealed_audit/`。

每次 labels/qrels 文件访问都必须写 append-only access log。创建封存文件
不等于解封；只有把标签值、类别数或任何标签派生统计交给评估进程，才算
解封并永久消耗该 audit。sealing worker 允许在隔离边界内重算 controller
equality，但在正式解封前只输出 `rowwise_consistent=true`、行数和标签文件
哈希，不输出类别数、标签值或任何标签关联统计。

## 9. 标签盲分类候选 C40

每个世界 28 个卖家共有 378 个无序 pair。候选进程无法打开 oracle、标签
或模型分数，并对每个世界固定选 40 个 pair。

“生产可观察候选逻辑”在 v13 中精确定义为冻结的
`Step4-derived v13` 标签盲引擎，而不是调用 Step4 `main`：

- 只包裹 `build_seller_profiles`、`compute_retrieval_weights` 和
  `build_candidates_for_pool`；
- 生产脚本 SHA-256 固定为
  `3f15b37453184b4c1e62de14efbbfbc0fc17ea3299184b3ab1033212d237c7af`，
  schema SHA-256 固定为
  `c5aca462daf8e0ba9557085b92bd3e6a2e53465d5a7603274c69576c617a3f7b`；
- lexical DF、clone cluster degree 和 retrieval weights 都只在当前
  28-seller world 内计算，使 C40 成为单一 world 的独立函数；四个 split
  使用逐字节相同的算子；
- 因生产中文 `min_df=2,max_df_ratio=0.025` 在 28-seller 小池会令
  `max_df=1`，v13 在生成数据前固定一个明确命名的 per-world derived
  config：`min_df=2,max_df_ratio=1.0,max_df_cap=8,
  top_terms_per_profile=140,top_neighbors_per_profile=15,
  min_cosine_similarity=0.14,strong_cosine_similarity=0.20`；
  因此本版本称 Step4-derived，不声称精确重放生产池的 DF 拟合；
- 精确联系方式候选只继承现有 Step4 支持的 email、telegram、wickr、
  wechat、qq、phone；BAT、钱包和 external URL 仍可进入 33 维身份历史，但
  本版本不得偷偷加入候选触发；
- items 到 Step4 profile 的适配必须先复用 Step3
  `ensure_profile/update_profile/build_specificity_catalog/finalize_profile`
  与冻结 compression policy；`stopwords` 和 content minimums 逐字读取冻结
  Step4 schema，`pgp_alias_map={}`（v13 不生成 PGP，也不得打开真实 PGP
  文件）。禁止手写另一套 profile 文本、style 或 contact 聚合；
- 此处 items 精确指身份删除前的 raw observed items。candidate worker 必须
  在单个 world 的私有内存中独立重建临时 Step3 profile；禁止复用给 M0
  使用的 `profile_safe_items` 或已脱敏 `seller_profiles.jsonl`，否则 exact
  contact trigger 会被预先删除。临时 raw profile、Step4 SellerProfile 和
  原始候选证据都不得落盘；
- 固定 `primary_trigger` 优先级为
  `shared_contact_exact > shared_description_clone > shared_title_clone >
  profile_lexical_neighbor > fallback_hash`。

冻结 Step4 只会在已被 exact/clone/lexical 创建的 pair 上附加
`structural_support`，因此它是 audit flag，不是可独立到达的 primary 层。
不得为不可达层创建 Hamilton 配额；大小为 0 的层不参加余数排序，quota 固定
为 0，design inclusion probability 记为空值而不是除以 0。

固定算法：

1. 运行生产可观察候选逻辑；
2. 按冻结优先级为多触发 pair 指定唯一 `primary_trigger`；
3. 生产触发 pair 数 `N_prod>=40` 时，按各 primary-trigger 层大小通过
   Hamilton 最大余数法分配 40 个名额；`N_prod<40` 时全部纳入并用
   `fallback_hash` 补足；
4. 层内按
   `HMAC-SHA256(candidate_key, world_uid || 0x1f ||
   canonical_pair_uid)` 排序；
5. 如果生产触发 pair 少于 40，从该世界剩余 pair 按相同哈希补齐；
6. 保存 trigger、随机键、层大小、名额和设计纳入概率到单独的
   `candidate_sampling_audit.csv`；这些字段不得进入模型；
7. 对最终 40 条再用独立 message
   `world_uid || 0x1f || "selected_global_rank" || 0x1f ||
   canonical_pair_uid` 在同一 candidate key 下排序，完整 digest 后以 pair
   UID UTF-8 打破碰撞，赋全局 `selected_rank=1..40`。

Hamilton 的相同余数按上述 trigger 优先级打破平局。`canonical_pair_uid`
逐字节定义为：按 UTF-8 字节序排序两个 seller UID 后，使用 ASCII
`"||"` 连接，即 `left_uid + "||" + right_uid`；seller UID 本身禁止含
`"|"`。排序键为完整
`HMAC-SHA256(key, world_uid || 0x1f || canonical_pair_uid)` 的 32 字节
big-endian 整数，再以 canonical pair UID UTF-8 字节序打破理论哈希碰撞。
`fallback_hash` 只从未被生产引擎选中的其余 pair 中抽取。

`candidate_sampling_audit.csv` 每 world 必须恰有全部 378 个 unordered
pair，而不是只保存入选 40 条；exact schema 为
`canonical_pair_uid,world_uid,primary_trigger,trigger_flags,
lexical_similarity,structural_support_flag,layer_size,layer_quota,
hmac_digest_hex,design_inclusion_probability,selected_bool,selected_rank`。
生产未触发 pair 的 primary 固定为 `fallback_hash`、trigger flags 为空；
lexical missing 固定序列化为 `0.000000`，概率固定 `.12f`，布尔固定
`true/false`，未入选 `selected_rank` 为空。trigger flags 按冻结 trigger
优先级排序后用 ASCII `|` 连接，且只能包含
`shared_contact_exact,shared_description_clone,shared_title_clone,
profile_lexical_neighbor` 的可达子集；`structural_support` 只进入独立布尔
列。零大小层的 null 概率固定写为空字符串。审计表按
`world_uid UTF-8,canonical_pair_uid UTF-8` 排序，digest 为 64 位小写
hex。安全
`candidate_pairs.csv` 必须等于该表 `selected_bool=true` 的四列投影并按
`world_uid UTF-8,selected_rank` 排序，每 world rank 恰为 1..40。
审计表的 `hmac_digest_hex` 始终是 primary 层内/补齐抽样 digest，不是最后
的 global-rank digest；global-rank digest 不落盘，只按上述独立 message
确定次序，二者不得混用。

Step4 返回的 shared values、clone 原文、preview、rank score、review 字段及
全部 raw evidence 只允许存在于 candidate worker 私有内存，投影后立即丢弃，
禁止落盘。dataset-public 的 `candidate_pairs.csv` 安全投影精确只有
`canonical_pair_uid,world_uid,seller_uid_left,seller_uid_right`；抽样审计表
精确只有
`canonical_pair_uid,world_uid,primary_trigger,trigger_flags,
lexical_similarity,structural_support_flag,layer_size,layer_quota,
hmac_digest_hex,design_inclusion_probability,selected_bool,selected_rank`。
M0、adapter、
identity-history 和 metadata-shortcut worker 的 allow-list 必须同时排除
`candidate_sampling_audit.csv`；除 join 所需安全投影外，不得读取候选证据。

若 HMAC 排序视为对固定层的均匀伪随机排列，某层的设计纳入概率固定为
`quota / layer_size`；背景补齐层同理。该值只描述 C40 设计，不用于主指标
IPW。主 estimand 始终是 C40 内的世界等权总体。

正式合同只锁定：

- 1,125 个世界；
- 每世界 40 条；
- 总计 45,000 条。

train/development 的正例数、负例数和阳性率可在其监督文件开放后报告；
Audit A/B 的类别数只能在各自正式解封时报告。候选冻结本身不授权打开
audit 标签。任何类别数都不得作为正式生成成功条件，不得据此删除、补抽、
重抽或更换种子。该候选总体是合同定义的设计型审核队列，不冒充现实市场
自然 pair 总体。

候选表必须在 oracle 不存在、oracle 清空和 oracle 内容置乱时字节一致。

## 10. 封闭世界检索候选

每个 audit 世界固定：

- 按 `HMAC-SHA256(query_key, world_uid || 0x1f || seller_uid)` 选择排名
  最小的 4 个卖家；
- 每个 query 的 gallery 是同世界其余全部 27 个卖家；
- query 选择器不得读取 controller、身份机制、分类候选或模型分数。

每个审计集恰好 1,000 个 query。由于每个控制者至少有 2 个卖家，任何 query
都自然至少有 1 个相关 seller，无需用 oracle 挑 query。

四个 query 由哈希一次性固定，即使它们来自同一 controller 也不得事后
替换。retrieval qrels 只能在 query/gallery 清单哈希冻结后生成；Audit A/B
的 qrels 分别进入各自封存目录，遵循与 classification labels 相同的解封与
永久消耗规则。

query 排序的实际 message 固定为
`world_uid || 0x1f || seller_uid` 的 UTF-8 字节，排序键为完整
HMAC-SHA256 digest 的 32 字节 big-endian 整数，再以 seller UID UTF-8
字节序打破理论碰撞。gallery seller 按 seller UID UTF-8 字节序固定。

标识符构造固定为：

```text
query_uid =
  "qry_" + SHA256_UTF8(world_uid || 0x1f || query_seller_uid).hexdigest()

relation_uid =
  "rel_" + SHA256_UTF8(query_uid || 0x1f || gallery_seller_uid).hexdigest()
```

`canonical_pair_uid` 始终复用第 9 节的 ASCII `"||"` 规则。所有 UID 在生成
后做全表唯一性和反向外键验证；禁止另一套检索 pair UID 或截断 digest。

分类监督唯一公式是
`label=int(controller(left)==controller(right))`，row universe 恰为 C40；
检索监督唯一公式是
`relevance=int(controller(query)==controller(gallery))`，row universe 恰为
冻结的 `4×27` directed relation。独立 validator 必须从 controller
membership 与安全端点表重新计算，不得读取已物化 label/qrels 后“自证一致”；
任一 missing/extra/duplicate/key/value mismatch 都是 split validity failure。

该任务称为“同世界 27 候选封闭检索”，不能写成原计划的 50 候选生产检索。

## 11. M1 标签盲整行错配与文本重连压力诊断

### 11.1 主 M1：端点不重合的 33 维整行错配

主 placebo 只对 train 构造，继续使用已经登记、且在打开 train 标签前冻结的
5 组 32-byte seed；不得因为任何结果更换 seed。输入只有 M2 的全 378
`identity33`、完整 pair 端点、四列 endpoint-only C40 和公开 policy。禁止
读取 controller、pair label/qrels、机制、candidate trigger/sampling audit、
M0 输入或分数。

每个世界把 378 对固定拆成 `primary_c40=40` 和
`secondary_complement=338`，分别构造 destination pair 到 source feature
row 的确定性完美匹配。合法 source 必须与 destination 的两个 seller 端点
完全不相交，因此不可能自映射。destination 次序和每个 destination 的 source
候选次序分别由 policy 登记的完整 HMAC-SHA256 消息及 pair UID UTF-8
tie-break 唯一确定；使用仓库内确定性增广路匹配，禁止第三方 solver。
不存在完美匹配即整版失败，不得放宽为只排除 fixed point。

33 列必须作为不可拆分整行从 source 复制到 destination；pair UID、随后由
supervision 加入的 label、p0 和世界权重仍属于 destination。每个
`(world, universe)` 必须逐字验证 source/destination 的联合 33 维向量
multiset 完全相同。这样 M1 与 M2 在 C40 和 full-378 的支持、均值、零比例、
分位数、协方差和相关性理论上相同，只打断“哪组身份规律属于哪个目标 pair”。
五组 smoke 实测二样本 OOF AUC 必须为随机水平，且七项预注册支持门全部通过。

### 11.2 v7 文本边重连：仅开发压力诊断

v7 的文本边重连已经证明 renderer→生产 parser→history projection 可精确
重放，但五个 seed 全部未通过 33 维支持门：它逐层独立保留一阶度数，却破坏
了跨 identity 的 rotation/corroboration motif；full-378 也同样失真。因此
它不得训练、认证或解释主 M1。以下合同仅保留为开发压力诊断和失败原因记录。

诊断只对 train 构造，使用同一 5 个冻结 seed；不得重新运行候选生成器。
重连器禁止读取 controller、pair label 或机制真值。若需要保持公共/私有
等干扰边际，只能读取一个单独的 `rewire_nuisance_ledger`；该表仅允许
identity UID 和不含 controller/seller membership 的 nuisance class。

identity UID 必须由 observed `(contact_type,normalized_value)` 的 canonical
hash 产生；唯一序列化为
`"id_" + SHA256(canonical_json).hexdigest()`，其中 canonical JSON 对象只有
`{"contact_type": type.lower(), "normalized_value":
normalized_value.lower()}`，即冻结 history `token_key` 的真正二元组；UTF-8、key
字典序、`ensure_ascii=false`、分隔符 `(",",":")`、无换行。禁止从 planned
role、controller 或 seller membership 派生 UID。原始
`bundle_uid="bundle0_"+SHA256(canonical_json(world_uid,seller_uid,
identity_uid))`，不得含 secret、mechanism 或 ordinal，并须独立重算相等。
`render_asts.jsonl` 保存不含 controller/mechanism/label 的 item 字段与 opaque
slot placeholder。`bundle_uid` 表示同一 seller–identity edge 的 occurrence
集合；同一 bundle 的重复槽必须位于不同 item。生成器必须保证每个
`(item_uid,field_name,identity_type)` 至多一个 `must_extract` slot，从结构上
排除重连后同 item/field/type 被 Step3 合并。must-ignore 位于独立 noise 表，
不进入文本压力诊断图、flow、demand 或可替换槽，重渲染时逐字节保持原值。

nuisance class 只从 observed parser flags 和 token seller degree 推导，词表
与优先级固定为：

```text
any risky flag                         -> risky_product
else any support flag                  -> public_support
else seller_degree > 3                 -> high_frequency_direct
else                                   -> direct_or_private
```

不得读取 planned role，也不得把 controller、机制或 seller membership 编码
进 nuisance ledger。

重连按 `(world_uid,identity_type,nuisance_class,edge_occurrence_count)`
分层。合法 double-edge swap
`(s1,i1,c),(s2,i2,c) → (s1,i2,c),(s2,i1,c)` 除要求
`s1!=s2、i1!=i2` 和两条交叉边不存在外，还要求两条边 occurrence count
完全相同。不同 count 的边禁止交换。

这样每次 swap 同时精确保留 seller/identity degree、seller 总 slot demand、
identity 总 occurrence supply，以及每个 seller 和 identity 的 per-edge count
多重集；不会再用一个两段 flow 任意把可行的 `2/2` 分成 `3/1`。背景脚手架
保证每个 direct type 的 count=1 和 count=2 均各有四条 degree-1 可交换边。

swap 完成后，每个 seller/type/nuisance 独立做一次 slot matching：

```text
source → identity                    capacity = 该新 edge 的固定 c
identity → (identity,item)           capacity = 1
(identity,item) → compatible slot    capacity = 1
slot → sink                          capacity = 1
```

全部 identity count 与全部 slot 都必须饱和；这保证一个 identity 在同一 seller
的重复 occurrence 位于不同 item。因为 seller 接收的 edge-count 多重集与原来
完全相同，原可行 item pattern 在 identity 重命名后仍存在。不得跨 edge
重新分配 occurrence，也不得在 flow 失败后换 seed。

随后按 AST 重渲染并重新运行生产 Step3 parser；开发实现的精确渲染方法固定
为：先用 item-local AST 验证 item 与 `identity_slot_uids` 全集，再只替换
safe projection 登记的原始 identity span，按原 offset 从左到右重算新
offset。title 与所有 span 外字节（包括 base text、guard 和 must-ignore）
必须逐字节不变。parser occurrence 永远不能被当作可编辑 span。

rewire worker 不读取私有 parser expectation 或完整 renderer audit 表。它
只用 observed parser row 与 safe identity-slot projection 在
`(item_uid,field_name,identity_type,downstream_canonical_value)` 上的一对一
inner join 识别可重连槽；safe projection 只有 actual flags、actual-derived
nuisance class 和编辑所需字段，planned role/controller/mechanism 全部禁止。
safe projection 必须显式带 independently FK-validated `world_uid`；seller
UID 不可反解 world。专用 label-free projector 只读 observed
item/seller/world membership、actual parser 和不含 derived UID/planned
字段的 edit map，独立重算 identity UID、bundle UID、全 world seller degree
及 nuisance class；ledger 与每条 safe slot 的 class 必须逐行相等。
未被 parser 命中的 `must_ignore` 位于另一张 noise 表，自然排除。join 非
一对一即失败。每个 slot 的 expected flags 来自该 slot
原始、已经独立验真的 parser row，并随 slot/context 保持，不从重连后的
parser 输出反推。重连后的 expected set 由独立 structural-audit capability
根据 rewired AST/slot assignment 加上述 slot-local frozen flags 物化；
parser worker 不得挂载该 expectation，比较时仍执行第 7 节严格集合和 flag
equality。任何 parser 合并导致的 occurrence 数减少、nuisance class 漂移或
`must_ignore` 字节变化均使该 seed 失败。

`layer_uid` 固定为
`"layer_" + SHA256(canonical_json).hexdigest()`，canonical JSON 对象只有
`world_uid,identity_type,nuisance_class,edge_occurrence_count`，序列化规则
与 identity UID 相同。每个 stratum 独立从 iteration 0 计数，`E`、20E
接受门、attempt 上限和 original-edge retention 都只针对该 stratum。
seed 固定为 policy 对应 hex 解码得到的 32 个原始字节，不作 UTF-8 二次编码。
`rewire_seed_id="rws_"+SHA256(raw_seed_bytes)`。
每条 edge UID 固定为 `seller_uid + 0x1f + identity_uid`。
重连后的 `bundle_uid` 固定为
`"bundle_" + SHA256(canonical_json).hexdigest()`，canonical JSON 只有
`rewire_seed_id,layer_uid,seller_uid,identity_uid`，沿用相同 JSON 规则。
`canonical_edge_pair` 固定为两个 edge UID 按 UTF-8 字节序排序后用
ASCII `0x1e` 连接。swap 候选边对按
`HMAC(seed, layer_uid || 0x1f || decimal_iteration || 0x1f ||
canonical_edge_pair)` 完整 digest、再按 canonical edge-pair UTF-8 排序；
`decimal_iteration` 是无前导零 ASCII 十进制，精确定义为当前 stratum
已接受 swap 数。每个 iteration 从当前图快照枚举全部 unordered distinct
edge-pair，按上述键扫描；每检查一对（接受或拒绝）attempt 加一，首个合法
pair 立即交换、iteration 加一并丢弃旧快照余项。完整快照无合法 pair 时立即
fail（只有已注册的非 direct structurally-fixed 豁免例外）；达到 attempt
上限前不得开启未定义的新 sweep。
slot flow 使用仓库内固定整数 Dinic。`slotflow_uid` 是只含
`layer_uid,seller_uid` 的 canonical hash；source/sink 分别是
`SHA256(slotflow_uid||0x1f||"source"/"sink")` 的固定前缀 UID。复合节点 UID
是只含
`layer_uid,seller_uid,identity_uid,item_uid` 的 canonical hash。所有 forward
edge 先按固定 edge-kind 次序，再按
`HMAC(seed,layer_uid||0x1f||"slotflow"||0x1f||edge_kind||0x1f||endpoints)`
完整 digest，最后按完整 directed-edge UTF-8 字节排序；reverse edge 紧跟
对应 forward edge 插入，BFS/DFS 只按该 adjacency 顺序。第三方 solver 和
另一套 UTF-8-only tie-break 均禁止。相同 policy/seed 必须逐字节重放。

精确保留：

- seller 按类型的身份槽位数；
- identity 的 seller degree；
- identity 总 occurrence 数；
- seller 商品、字段、上下文和时间槽位；
- 全局类型、字段、上下文和时间边际。

不声称同时精确保留每个身份的所有精细时间—上下文联合轨迹。该部分只在
预注册容差内比较。

其中 `E` 定义为该
`(world,identity_type,nuisance_class,edge_occurrence_count)` stratum 重连前
的唯一 seller–identity 边数。每个 stratum 从 iteration 0 开始；在每次 accepted swap
后检查条件。对 `direct_or_private` 层，在“accepted swaps 至少 `20×E` 且
该 stratum 全部原边保留率不超过 1%”首次同时成立时立即停止；其他 observable
nuisance stratum 只要求首次达到 `20×E` 时停止并报告原边保留率，不用 planned
private role 作判断。每个候选 edge-pair 的合法性检查计一次 attempt；
若 attempt 达到 `10,000×max(E,1)` 仍未同时满足则该 seed fail closed，不得
换 seed、继续无界搜索或降低门槛。不可交换的非 `direct_or_private` 层可以
原样保留并明确记 `structurally_fixed=true`；不可交换的
`direct_or_private` stratum 直接失败。任何停止/豁免判断都只读 observed nuisance
class，不读取 planned role。每个重连版本必须：

- 每个可交换 stratum 有效交换数不少于 `20 × E`，并逐 stratum 报告尝试数、接受数和
  最终原边保留率；
- `direct_or_private` 原始 seller–identity 边保留率不超过 1%；
- 无自环、重复边、跨类型值或非法 occurrence；
- 无法充分混合的分层直接失败，不得静默放宽；
- 不得根据标签相关性挑选五个种子。

主 M1 与 M2 必须满足：

- pair UID、标签、世界权重逐行一致；
- 非身份商品内容一致；
- 身份删除后的 title/description 逐字节一致；
- legacy18、LaBSE 六项和每个 M0 分数逐行一致；
- 在最终进入 adapter/M0 的模型可见矩阵中，唯一允许差异是 33 维身份历史
  特征；且每个 `(world,C40/non-C40)` 的 33 维联合向量 multiset 完全相同。

每个 M1 derangement worker 只允许挂载 M2 `identity33`、public complete
pair endpoints、四列 endpoint-only C40、公开 policy 和本 seed。其他 seed、
candidate audit/trigger、rewire lineage、oracle、labels/qrels、controller 和
M0 全部禁止，访问 hook 必须记录 0 次。模型 worker 只能读 destination-keyed
M1 matrix，不得读 source→destination mapping。

开发期文本重连诊断的私有输出 exact schema 固定在 policy：
`rewire_manifest_schema` 逐 accepted swap 保存两条当前边、两条新边、iteration、
累计 attempt 和 digest；`rewire_stratum_audit_schema` 保存每层 E、attempt、
accepted、原边保留数/率、fixed 标志和最终边 multiset hash；
`rewired_slot_assignment_schema` 一槽一行保存新 identity/bundle、重算 offset
与原 slot-local observed flags；`rewired_ast_schema` 等于原 item-local AST
前加 `rewire_seed_id`。每 seed 还产生同 observed item schema 的 rewired
items、同 production parser schema 的 rewired parser rows，以及只供诊断
replay 的 safe history occurrence projection。所有表按 policy 的
`rewire_output_order` 固定排序。

对每个预注册 M1 seed，还必须在不读取 label 的情况下比较其与 M2 train 的
33 维支持范围、逐维标准化均值差、零比例、协方差/相关矩阵、直接/两跳路径
计数分布及预注册二样本分类器 AUC。容差、分类器、超参数、分组折和随机种子
必须在 formal seed 前写入 policy。任一 seed 超界使整个正式版本失败，不得
换 seed 或只保留较匹配 seed。主 M1 还必须额外逐世界、逐 universe 证明
33 维联合向量 multiset 精确相同；数值门只能作为独立重算，不能代替该精确门。

33 维必须由每个完整世界的全部 parser occurrence 一次性建图后计算；禁止
按 pair 子集重建图。固定阈值为
`direct_token_seller_frequency_maximum=3` 和
`weak_graph_token_seller_frequency_maximum=12`。按当前代码逐项重算，
`history_feature_details` 实际产生 37 个字段；v13 明确排除 4 个：

```text
mixed_context_token_count_log1p
same_token_path_count_log1p
verified_x_high_frequency
rotation_external_url_edge_token_count_log1p
```

其余 33 列的名称和顺序必须在 policy 中完整枚举，不允许按字典顺序或运行时
发现字段推导。

## 12. 数据目录与权限

正式根目录：

```text
reports/step28_synthetic_chinese_dataset/v13_20260727/
```

建议结构：

```text
reference/
  chinese_train_style_profile.json
observed/
  train/
  development/
  audit_a/
  audit_b/
placebo/
  train_seed_01/
  train_seed_02/
  train_seed_03/
  train_seed_04/
  train_seed_05/
supervision/
  train/
    classification_labels.csv
    access_log.jsonl
  development/
    classification_labels.csv
    access_log.jsonl
sealed_audit/
  audit_a/
    classification_labels.csv
    retrieval_qrels.csv
    access_log.jsonl
  audit_b/
    classification_labels.csv
    retrieval_qrels.csv
    access_log.jsonl
oracle/
  train/
    controller_membership.csv
    controller_style_groups.csv
    mechanism_assignments.csv
    solver_audit.jsonl
    parser_expectations.csv
  development/
    controller_membership.csv
    controller_style_groups.csv
    mechanism_assignments.csv
    solver_audit.jsonl
    parser_expectations.csv
  audit_a_custody/
    controller_membership.csv
    controller_style_groups.csv
    mechanism_assignments.csv
    solver_audit.jsonl
    parser_expectations.csv
  audit_b_custody/
    controller_membership.csv
    controller_style_groups.csv
    mechanism_assignments.csv
    solver_audit.jsonl
    parser_expectations.csv
producer_projection/
  <split>_projector_custody/
    producer_typed_dgp_projection.private.jsonl
    producer_typed_dgp_projection_manifest.private.json
    capability_parent_manifest.json
independent_replay/
  <split>_replayer_custody/
    world_replay_ledgers.private.jsonl
    replay_receipt.private.json
    capability_parent_manifest.json
  <split>_no_key_comparator/
    world_comparison_receipts.private.jsonl
    aggregate_comparison_receipt.json
audits/
manifests/
```

主要表：

- `worlds.jsonl`：split、world UID、版本和配置哈希；
- `sellers.csv`：seller UID、world、market、locale、item count；
- `items.jsonl`：item UID、seller UID、时间、类别、title、description；
- `renderer_identity_slots.audit.csv`：仅 must-extract，含稳定 slot、
  item/seller/field/span、identity UID/type/canonical value/surface、bundle、
  private planned role、expected flags 和 time bucket；只供 generator/
  structural audit；
- `renderer_noise_slots.audit.csv`：仅 must-ignore，含 noise slot、
  item/seller/field/span/surface/expectation；不存在 identity 列；
- `rewire_safe_identity_slots.csv`：actual parser 与 identity audit 表严格
  一对一 join 后的最小投影，含 actual flags/actual nuisance，不含 planned
  role、expectation、controller/mechanism；只供 M1；
- `render_asts.jsonl`：无 controller/mechanism/label 的 base field 与 opaque
  slot placeholder，供 M1 确定性重渲染；
- `independent_replay_inputs/*.csv`：只含 world/seller/item UID 及
  nonempty-title/nonempty-description 集合成员资格，不含 market、文本、
  controller、mechanism 或 label；只供相应 split replayer；
- `controller_style_groups.csv`：每个 controller 的已选 style ID；属于
  split oracle，不进入 structural audit 或模型；
- `solver_audit.jsonl`：hard-negative typed-membership topology、type solver
  与 flow 的私有 trace；
  属于 split oracle，禁止再嵌入 `world_generation_audit.jsonl`；
- `producer_typed_dgp_projection.private.jsonl`：producer-private projector
  输出的 exact-schema 最小 typed decision projection；不得包含 raw identity
  value、rendered occurrence identity UID、文本、slot/flow、parser 输出或
  非比较 solver 字段；为精确比较 topology 而保留 synthetic
  `identity_asset_uid`；formal no-key comparator 不得直接打开原始
  oracle/AST/solver 现场投影。development comparator 只有严格 schema 拒绝，
  没有 OS 文件访问隔离，不能据此声称“绝不打开调用者指定的错误文件”；
- `world_replay_ledgers.private.jsonl`：第二实现独立生成的 private decision
  projection；不得由 producer oracle 派生，必须覆盖完整登记 split world
  集，并由 capability parent manifest 绑定；
- `aggregate_comparison_receipt.json`：无密钥 comparator 的固定计数、hash 和
  equality boolean；逐 world private receipt 不公开给模型或人工诊断；
- `parsed_identity_occurrences.csv`：通过
  `source_dataset/source_row_number` 外连接补回的 item UID、字段、类型、
  `raw_value`、normalized value 和 parser flags；不声称 parser 原生输出 span；
- `candidate_pairs.csv`：第 9 节定义的四列无标签安全投影；
- `candidate_sampling_audit.csv`：trigger 和抽样血缘；只供
  candidate-integrity sealer，structural auditor 只能读取其不可连接 aggregate
  receipt，不挂载给任何模型或特征 worker；
- `retrieval_queries.csv`：无 relevance 的固定 query 清单；
- `retrieval_gallery_pairs.csv`：`relation_uid,query_uid,world_uid,
  query_seller_uid,gallery_seller_uid,canonical_pair_uid`；每个 query 的
  27 条有向无标签 gallery 关系；
- `evaluation_pair_universe.csv`：分类与检索 pair 并集以及用途布尔列；
- `redacted_items.jsonl`：身份删除后的 title/description；
- `identity_features.csv`：覆盖 evaluation pair union 的显式固定 33 列；
- `rewire_manifest.csv`：原边、新边、分层和交换血缘。
- `classification_labels.csv`：精确两列
  `canonical_pair_uid,label`，keyset 恰等相应 split C40，0/1 全量一行一个；
- `retrieval_qrels.csv`：精确两列 `relation_uid,relevance`，keyset 恰等
  27×全部 query 的 relation 全集，正负均显式写 0/1，不允许 positive-only。

oracle 还必须拆为分区 custody 文件：

```text
<split-custody>/controller_membership.csv
<split-custody>/mechanism_assignments.csv
<split-custody>/parser_expectations.csv
```

`rewire_nuisance_ledger` 只允许
`identity_uid,nuisance_class`；由于 normalized value 在全版本中不跨 world，
identity UID 不会在 ledger 中碰撞。seller–identity membership 必须从 observed
parser 输出重建。Audit A/B controller custody 也必须互相隔离，不能与
train oracle 放在一个普通 CSV 中。

retrieval relation 是有向主键；同一 unordered seller pair 在双方都成为
query 时可以有两个不同 relation UID，绝不能按 canonical pair UID 去重。
33 维和 M0 只对 canonical pair union 计算一次，再由 relation UID 外键映射
回有向检索行；qrels 也以 relation UID 为主键。labels keyset 必须恰等 C40，
qrels keyset 必须恰等 relation universe，`keys(p0)=keys(identity33)=
evaluation_pair_universe`。任何隐式 many-to-many join、漏 key 或额外 key
都使 split invalid。

controller、机制、planned role、模板 ID 只能位于 oracle/audit；模型和 GPU
工作区不得接收。目录分层本身不算物理隔离，阶段 worker 必须从空工作区
创建，并只挂载该阶段 manifest allow-list：

- `observed/` 和已通过检查的 `placebo/` 是 dataset-public；
- `supervision/train` 与 `supervision/development` 只在相应监督阶段依序
  开放；
- `sealed_audit/audit_a` 和 `sealed_audit/audit_b` 在模型、阈值、统计
  代码、盲预测和锁文件冻结前分别禁止挂载；
- `oracle/` 永不进入 M0、adapter、候选器或 GPU 工作区；
- CPU M0 基础评分也不得读取 supervision、sealed_audit 或 oracle。

完整 identity/noise audit 表只有 generator 与 generator-private
render-integrity sealer 可以读取；
主 M1 只读 M2 `identity33`、complete pair、四列 endpoint-only C40 和本
seed；safe slot 与 `render_asts.jsonl` 仅供文本压力诊断。parser worker 与
redactor worker 必须是两个独立 capability：parser 不得读 expectation、
slot/AST 或 replay；redactor 不得读 expectation、slot/AST 或 replay。
structural auditor 不得读取 observed/redacted 文本、parser 行、raw render
AST 或 world/seller/item join key。generator-private render-integrity sealer
在原 custody 内完成逐字节重建与 parser/redaction 对照，只输出不可与任何行
重连的 split aggregate receipt：固定 boolean、计数、输入父 hash 和 aggregate
content hash；不得含 world/seller/item/style UID、文本、parser 行或 identity
值/UID。candidate 与 M1 derangement 也必须先由各自 sealer 变成同类不可连接 aggregate
receipt。最终 structural auditor 只验证这些 aggregate receipt、独立 DGP
aggregate comparison receipt 及外部 parent seals。这样即使公开 text key 和
seller perturbation 算法可逆，auditor 也没有 seller-linked 文本可以反推
controller-linked base style/group。render、candidate、M1-derangement
aggregate receipt、递归 schema validator 和 receipt-only structural auditor
已有 combined-process development smoke 实现，但它们没有独立 OS
custody、audit hook 或外部 parent seal，不能作为正式证据。上述 sealer 与
parser、redactor、auditor 的独立 formal launcher 当前仍未实现。parser、
redactor、legacy18、LaBSE/M0、candidate、query、
identity-history 和 adapter worker 的 allow-list 必须显式排除，并由
audit-hook 证明成功打开次数为 0。`candidate_sampling_audit.csv` 对上述全部
模型/特征 worker 同样禁止。这样生产清洗和历史特征都不能绕过 parser，直接
按 planned slot 或触发证据得到答案。

development receipt 中的 `*_absent_from_sealer_arguments` 只证明该函数的
显式输入 schema/参数不含相应字段，不证明同一 combined process 的地址空间
不存在这些对象；只有未来独立 launcher 的挂载白名单、audit-hook access
manifest 和外部 custody parent 同时通过，才能把它升级为进程级“输入不存在”。
其中 candidate sealer 的函数参数现已只接收无 secret policy projection、
公开 context 和单独 candidate key，不再显式接收含 structure key 的完整
policy；但 combined development generator 仍在同一进程中先构造这些对象，
所以仍不能冒充进程级隔离。

四类 nonjoinable receipt 的 schema 必须递归固定，不能只固定顶层容器名。
共同顶层只能是 `version/evidence_level/fixed_boolean_gates/fixed_counts/
input_parent_hashes/aggregate_content_hashes/canonical_self_hash`。deployment
逐类登记四个内部 object 的 exact key set；boolean 只能是布尔值，count 只能
是非负整数，hash 只能是小写 SHA-256。所有动态 key、list、下一层 map、
per-world/seller/item/pair/style/category/trigger breakdown、路径、文件名、
UID、文本、parser 行和 access event 一律失败。structural auditor 读取的外部
parent 也不是完整 manifest。由于 M1 严格只在 train 构造，train 必须恰好
读取四条 parent projection（render、candidate、M1-derangement、independent
DGP），development/A/B 各恰好读取三条（不允许伪造一条空 M1 回执）；各
split 均按固定 role UTF-8 排序。每条
只有 `role/file_sha256/content_sha256/source_closure_sha256/
access_summary_sha256`，不得含路径、文件名或逐事件日志。render、candidate、
M1-derangement 三个 sealer 的 read/write/forbidden roles 与 exact schema 均登记在
`schema/step28_v13_dataset_custody_deployment.json`；当前 formal entrypoint
仍全部为 null。正式 validator 还必须把 `version/evidence_level` 校验为本次
冻结 release 的精确常量，并从 sealed inputs 独立重算每个 boolean、count 和
hash；只检查字段类型或照抄 sealer 值不算通过。

每类 aggregate receipt 的输入父哈希必须同时包含 policy 文件的精确字节
SHA-256，以及由 `policy version + mode + split + 该 split 全部登记 world_uid`
构成的 `registered_split_scope_sha256`。structural auditor 必须重新计算这一
作用域并要求所有父 receipt 完全一致；因此即使 Audit A/B 的 world 数相同，
也不能互换 receipt。当前 combined development generator 只接受仓库登记的
默认 policy 路径，拒绝内容相同但路径或原始字节不同的替代 policy 文件；
formal launcher 未来必须改由外部 parent seal 显式传入并绑定实际 policy
文件。render-integrity sealer 必须对 sellers、items、
redacted、parser、slot/noise、AST 和 override 各表执行 exact schema 检查，
并用保留重复的 multiset hash 证明每一输入行恰好被某个登记 world 消费，
任何孤儿行或跨 split 行均失败。

candidate-integrity sealer 不能只验证 sampling-audit 内部自洽。完整 policy
不得作为其参数，因为 development policy 也含 structure key。上游必须先构造
exact-schema public candidate policy projection，精确只保留四个 Step3/Step4
pin、observed/pair schema 和 candidate design；不得含 `randomness`、任一
structure/id/text/identity/query/rewire key、controller/mechanism/solver、
labels/qrels 或完整 policy。这里的 exact 不仅是字段名：整个静态 payload
的嵌套值和类型必须匹配代码与 deployment 同时登记的
`56641833ffff904fce7fcc00c1cddcfbf6d4b244d37f249c4e539b1ea805061b`，
projector/context builder 还必须要求它与从仓库登记完整 policy 派生的投影
canonical bytes 完全相同；禁止把 structure secret 塞进一个合法字段的值中。
formal launcher 只能挂载由外部 parent seal 绑定该精确 hash 的投影。
sealer 读取该投影、本 split raw observed
sellers/items、完整 pair universe、冻结 Step3/Step4 bytes、仅含登记 world
集合和 receipt schema/hash 的公开 context，以及单独挂载的本 split candidate
key，逐 world 重新生成 C40 与全部
378 行 sampling audit，再与待封存输入逐字节相等比较。development sealer
还必须走第二条不调用 C40 producer 的实现路径：直接重跑冻结 Step3/Step4，
从其原始候选行独立投影 trigger flags、primary layer、六位 lexical
similarity 与 structural-support flag；即使 C40 producer 被整体替换为一套
“内部自洽但共同出错”的结果，这条独立投影也必须拒绝。candidate 的
development source closure 不再依赖手写的短列表：从登记 entry source 出发，
用 Python AST 递归追踪所有 repo-local 绝对 import 和 literal dynamic import，
再合并登记的非 Python schema，按 repo-relative UTF-8 path 固定成员并逐文件
哈希；角色专用模块改为函数内 lazy import，干净进程导入 candidate sealer
不得顺带加载 M1/DGP/render 模块。测试必须逐一替换每个发现成员的观测 hash
并证明 closure 改变。任何 relative import、相对 dynamic target、
`import_module(variable)`/`__import__(variable)` 等非字面动态导入、
`import importlib as il`/`from importlib ... as ...` 别名、普通
`import builtins`，或
非直接 `importlib.import_module` 属性使用、直接
`getattr/vars(importlib|builtins, ...)`、其他对象上的
`import_module/__import__` 属性必须立即 fail closed；只允许直接调用绝对
字符串字面量 target，不能静默跳过。直接使用 `__builtins__` 名称，或直接
调用 `compile/eval/exec/globals/locals`，也必须 fail closed。以上是
development AST 检查的精确语法边界；它不声称拦截经别名、自定义分派或其他
未登记语法隐藏的反射，也不把 AST 冒充恶意 Python 沙箱。当前冻结源码不得
含未登记反射；development 允许保守的 repo-local 超集；formal 还必须
冻结第三方环境并由全新进程给出 exact loaded-module inventory，当前尚未实现。
independent-DGP
aggregate receipt 同样不能只投影 comparison 的声明：它必须实际接收每个
登记 world 的独立 replay ledger，校验 exact envelope、自哈希、mode/split/
world/graph、typed table hash 及每个 component 的行数/hash，再与 producer
projection 和 comparison receipt 三方闭合。

train M1 落盘前不得只相信调用者传来的 `support_preflight`。writer 必须从
M2、五份 M1、endpoint-only C40 和完整 pair endpoints 重新执行支持可比预检，
要求重算对象逐字段相等；随后必须从这些实际 placebo/mapping/matrix 再完整
构建一次 M1 aggregate receipt，逐 seed 重放确定性映射，并要求整份 receipt
完全相等及其中 `support_preflight_sha256` 与本次重算对象一致。只更新 support
hash 和 self-hash 不得使“拿 seed1 冒充 seed0”的协调伪造通过。回归测试还
必须用不调用
production support helper 的另一套 NumPy/sklearn 实现，逐 seed 重算主 C40
及 full-378 的 range、SMD、zero rate、quantile、covariance、correlation、
world-grouped OOF AUC 和逐世界/逐 universe 联合向量 multiset。

所有 receipt 中的 multiset hash 统一使用 deployment 登记的 framing：先写
UTF-8 版本前缀与 NUL，再写 8-byte unsigned big-endian 行数；每一行按
`ensure_ascii=false, sort_keys=true, separators=(',', ':'),
allow_nan=false` 编成 UTF-8 JSON，按完整 row bytes 升序排序、保留重复，再逐行
写 8-byte unsigned big-endian byte length 和原始 row bytes，最后取小写
SHA-256。禁止把 set、输入顺序或未登记的 CSV 序列化误称为同一 multiset hash。

每阶段必须落盘独立 `input_access_manifest`。repo 外 GPU 工作区必须从空
目录建立，拒绝 `..`、绝对路径、盘符、symlink、额外文件和缺失文件。GPU
禁止接收 raw items、任何非 opaque seller/world/item/pair 标识（包括合成
UID）、split、candidate trigger、
retrieval role、identity features、rewire ledger、英文 reference、joblib、
controller、机制及任何 label/qrels。只允许回传一个封存 transport bundle；
其 exact members 只有 `labse.item_aggregates.csv`、
`runtime_manifest.json` 和 `bundle_manifest.json`。所谓 output bundle 是这
三个成员的传输封装，不是第四种任意 payload；禁止额外成员、回传或落盘
embedding matrix。

所有 CPU stage 必须经 capability launcher 启动：在 worker 导入前安装
Python audit hook，对 policy 列出的 sensitive roots 执行 exact file
allow-list；任何未声明 `open` 先记失败事件再拒绝。访问日志采用 O_APPEND
写入的 hash chain（每行含 previous hash、event 和 self hash），stage 结束
后立即 immutable seal 并进入 manifest。Windows 冒烟只能证明该 launcher
下的进程级强制和哈希链，不冒充操作系统 WORM；formal Audit A/B custody
还必须使用不同 OS account/ACL 或独立挂载点。禁止打开次数来自 audit hook
真实事件，不能写死为 0。

capability launcher 还必须拒绝 worker 的 `subprocess.Popen`、`os.system`、
shell、网络连接、动态 `ctypes` 加载和未在启动清单内的 native executable；
不得把 Python audit hook 冒充对恶意 native code 的完整沙箱。正式运行前
另建并冻结 `schema/step28_v13_dataset_custody_deployment.json`，其中逐字节绑定：

| capability | 可读 | 可写 | 强制禁止 |
| --- | --- | --- | --- |
| 四个 split generator | style、template/policy、仅本 split secret | 仅本 split observed 与 oracle custody | 另一 split secret/oracle、旧 labels、M0 |
| split-private independent DGP replayer | 公开 policy、五个 UID-only pool、仅本 split secret | 仅本 split replay-private staging | producer oracle/slot/AST/raw text、另一 split key/root、labels、模型 |
| producer-private DGP projector/sealer | 本 split 已封存的 comparison-required producer private 表 | 最小 typed projection、projection manifest 与 capability parent | 另一 split、文本、parser、labels、模型 |
| no-key DGP comparator | 两份最小 projection、replay receipt 与两侧 immutable parent | private world receipts 与固定 aggregate receipt | 原始 oracle/AST/solver、任一 structure key/env、raw text、labels、模型 |
| parser worker | observed 最小投影、公开 parser 合同 | parser staging | expectation、slot/AST、replay、oracle、supervision、sealed audit |
| redactor worker | observed、parser 输出、最小 registry、公开 redaction 合同 | redacted staging | expectation、slot/AST、replay、oracle、supervision、sealed audit |
| candidate worker | raw observed sellers/items、完整 pair universe、冻结 Step3/Step4 与公开候选合同、仅本 split candidate key | C40 四列安全投影与 sampling-audit staging | structure key、controller/style/mechanism/solver、labels/qrels、模型分数、另一 split key；raw profile/Step4 evidence 落盘 |
| render-integrity sealer | observed/redacted、parser 输出、完整 slot/noise audit、private expectation、raw item-local AST | 不可连接的固定 aggregate boolean/count/hash receipt | controller membership/style group、mechanism、solver、structure key、labels；任何行级 UID/text 输出 |
| candidate-integrity sealer | 本 split raw observed sellers/items、complete pair universe、candidate projection、candidate sampling audit、冻结 Step3/Step4、无 structure/oracle secret 的 exact public candidate policy projection、公开 receipt context、本 split candidate key | 不可连接的固定 aggregate receipt | 完整 policy、structure/id/text/identity/query/rewire key、任一输出 world/seller/item/pair UID、trigger/role/lineage 行、labels、oracle、模型 |
| M1-derangement integrity sealer | M2 identity33、endpoint-only C40、complete pair、私有 mapping、公开 placebo policy | 不可连接的固定 aggregate receipt | 任一 world/seller/pair UID 或 mapping 行、labels/controller、candidate trigger/audit、M0 |
| M1 derangement | M2 identity33、complete pair、四列 endpoint-only C40、仅本 seed | destination-keyed placebo identity33 staging | source→destination mapping 对模型开放、oracle、labels/controller、candidate trigger/audit、M0 |
| text-rewire diagnostic | train parser、safe slot、AST、nuisance ledger | 仅开发诊断 staging | 主 M1 staging、完整 slot audit、expectation、oracle、labels、M0 |
| structural audit | train 读取 render/candidate/M1-derangement/independent-DGP 四份不可连接 aggregate receipt；development/A/B 只读无 M1 的三份；另读同角色最小 parent projections | 仅 aggregate boolean/固定计数/hash | observed/redacted 文本、parser 行、world/seller/item/pair join key、derangement mapping、raw AST、structure secret、controller-linked style/group oracle、机制、solver trace、classification label |
| train/dev supervision | pair 安全投影、相应 controller custody | 相应 supervision staging | 另一分区 custody、M0 |
| Audit A sealer | A pair/query 安全投影与 A controller custody | 仅 A sealed staging | B custody/密钥/目录、模型 |
| Audit A evaluator | A labels/qrels、三份 sealed diagnostic projection、冻结盲预测/统计 lock | 仅 A report | A controller raw custody、B 全部 |
| Audit B sealer | B pair/query 安全投影与 B controller custody | 仅 B sealed staging | A custody/密钥/目录、模型 |
| Audit B evaluator | 已授权 B labels/qrels、三份 B diagnostic projection、冻结盲预测/统计 lock | 仅 B report | B controller raw custody、A 全部 |
| M0 public scorer | redacted/profile、pair union、frozen reference/joblib/aggregates | 仅 p0 staging | 全部 supervision、oracle、slot/AST、candidate audit |
| public-input projector | world/seller membership、redacted text | 全 378/world opaque pair、text/index、CPU maps/seal | C40、retrieval role、labels、oracle、identity |
| CPU chunker | opaque unique text、四 tokenizer、CPU chunk projection | shared chunks | CPU maps、full parent、GPU policy、supervision |
| sync builder | projector/chunk seals、静态 GPU policy、LaBSE pin | CPU-side sync manifest | CPU map 文件、labels/oracle |
| CUDA materializer | sync 与 exact source members | 空 CUDA root；receipt 写在 root 外 | CPU maps、repo 其余部分 |
| CUDA encoder | 最小 opaque CUDA workspace | exact-member bundle | repo 其余全部、网络 |
| CUDA collector | bundle、sync/materialization/identity-lock receipt、CPU map seal | canonical LaBSE6 staging | supervision、oracle、identity、joblib |
| adapter train | train p0/identity33/labels | 冻结 scale/beta | development、Audit、oracle |
| development threshold | 冻结 beta/scale、development p0/z/labels | threshold | 修改 beta/scale、Audit |
| blind predictor | 冻结模型/threshold、A/B observed inputs | A/B blind predictions | 全部 labels/qrels/mechanism projection |
| metadata/coverage validator | 对应 split labels 与 sealed diagnostic inputs | validity report | adapter artifact 写权限 |

Dataset release lock 只冻结 `dataset_custody_deployment`，覆盖生成、oracle、
sealer 与结构验收；Identity experiment lock 另冻
`identity_custody_deployment`，覆盖 M0/GPU/collector/adapter。前者不得为后者
保留将来回填的 path/hash。

deployment 文件必须给出 A/B 不同的 OS account、group、绝对 custody root、
ACL/挂载规则、密钥环境变量名和解封事务脚本 SHA-256，不能写
`TBD`。sealer 在限制权限的 staging 中写入、`fsync` 文件与目录、生成
hash manifest 后原子 rename；evaluator 只在模型/阈值/代码 lock manifest
存在时才由独立账户取得只读挂载。A/B 密钥永不同时注入同一进程。Windows
smoke 不满足此表，只能验证 schema 与拒绝路径，因此 formal generation
保持禁用直至 deployment 文件通过审计。

CUDA 还必须运行在独立 UID 或 rootless container 中：只读 exact mounts、
单一可写 output、repo/CPU maps 不挂载、network namespace/egress 关闭，并
冻结 image/runner/mount/ACL manifest。Python audit hook 不能冒充 native
CUDA 进程的完整沙箱。

每个 manifest 至少记录：

```text
step, version, stage, run_id,
policy_sha256, policy_contract_sha256, producer_sha256,
parent_manifests[{role,file_sha256,content_sha256}],
files[path,size_bytes,sha256], canonical_self_hash
```

`parent_manifests` 按 role UTF-8 排序，可为 0/1/多父；collector 必须同时
绑定 returned bundle seal 与 CPU map seal。`canonical_self_hash` 精确定义为
删除该字段后，对 sorted-key、`ensure_ascii=false`、逗号冒号紧凑、无换行的
UTF-8 canonical JSON 求 SHA-256；parent content hash 就是该值，file hash
则是包含 self 字段的落盘字节 SHA。manifest 文件字节 SHA 与 canonical
content SHA 必须区分。输出角色集合固定，
拒绝重复、缺少或额外路径；manifest 不把自身列入 `files`。同字节重放允许，
不同字节覆盖必须失败；失败也必须生成独立 immutable failure manifest。
`canonical_self_hash` 只证明 manifest 内部未损坏，不证明生产者身份。

development smoke 的完整发布以根目录 `release_manifest.json` 作为唯一完成
标记。该父 manifest 必须按固定顺序绑定 train、development、audit_a、
audit_b 四个 split manifest 的落盘文件 SHA-256、去掉 self 字段后的
canonical content SHA-256 和各自 split payload digest；缺任一 split、存在
额外根目录条目或混入另一 release 的 child 时均不得生成完成标记。父 manifest
还必须绑定权威 `parser_template_fixture_result_v3.json` 的精确文件 SHA-256、
232,241 条 case 数、runner SHA-256 和 outcome manifest SHA-256。生成器在
构建任何 split 前必须实际打开并验证该结果，不能只验证 fixture 输入 JSON。

所有文件临时发布及最终 staging-directory 发布必须使用操作系统原子的
no-replace rename；`exists()` 后再调用可覆盖的 replace 不算不可覆盖保证。
release parent 若预先存在，必须证明它是 output root 下的普通目录，并拒绝
symlink、junction、越界 resolve 或非目录。四 split 可以在父 manifest
生成前暂时存在，但消费者必须把“缺父 manifest”解释为不完整 release，禁止
单独拼接或使用。父目录 fsync 若在完成标记发布后失败，状态是
`published-but-durability-unknown`，只能先校验不可变 manifest 再单独恢复
父 fsync，不得重新生成或覆盖文件。

M0 只允许逐 split 精确挂载：

```text
observed/complete_model_pair_endpoints.csv
observed/redacted_items.jsonl
observed/seller_profiles.jsonl
```

整个 `observed/` 目录不得挂载给 M0，因为 `observed/items.jsonl` 是承载合成
身份信息的原始解析输入，按设计可暴露身份文本和槽位数量。
replayer 与 producer projection 的正式 parent 还必须来自冻结 deployment
指定的 capability launcher、账户、ACL/mount 和 append-only access chain；
comparator 必须拒绝只带 self-hash、没有外层 custody parent 的输入。

combined-process development writer 在发布边界不得继续使用调用者拥有的
可变对象。它必须先深拷贝并执行完整 split payload 校验，再从 policy 登记
路径重新加载模板、fixture 和风格参考，重新生成整个 split；只有调用快照与
新结果的 canonical JSON bytes 完全相同才允许写入，而且实际写入对象必须是
新生成结果。这样可拦截 receipt 封存后再追加 seller、篡改 candidate 或改动
DGP projection 的内存攻击。该重生成仍来自同一 development producer，只是
发布一致性门，不得冒充正式独立重放或外部 custody parent。

## 13. 数据集训练前验收门

训练前数据门只允许：

1. 四分区完全无标签的结构、解析、复制、候选不变性、M1 整行错配、特征支持和
   manifest 检查；
2. 仅使用 train/development 已开放监督文件的标签公式、元数据作弊检测和
   标签相关机制覆盖检查。

Audit A/B 的 label/qrels 不得被训练前验收进程打开。训练前必须满足：

1. 世界、卖家和分类候选数精确符合合同；
2. train/development 每个标签与隔离 supervision worker 的 controller
   equality 验证 100% 一致；
3. 四分区 world/controller/seller/item/identity/template definition/
   template instance/component/text 交集为 0；
4. 与 Step28-v6 至 v12 的正式合成命名空间交集为 0；
5. 合成与真实规范化身份交集为 0，防文本复制门全部通过；
6. `must_extract` 召回 100%，`must_ignore` 误报为 0；
7. 生成器、candidate、query、M1 derangement worker 的禁止文件打开次数为 0；
8. candidate 在 oracle 缺失、清空和内容置乱时哈希相同；
9. M1 五个 seed 全部满足端点不重合双射、逐世界/逐 universe 联合向量
   multiset 精确相同及 33 维支持可比合同；
10. M1/M2 去身份文本、候选和非身份特征哈希相同；
11. 33 个正式 train 特征没有全零列，并报告矩阵秩、重复列和条件数；
12. train/development 的元数据作弊检测通过；
13. train/development 的每个机制达到预注册独立世界覆盖；
14. 主样本重放断言不存在基于 label、parser state、feature support、机制或
    slice 的行删除；任一坏行使整个 split 失效；
15. 所有输入、代码、配置和输出都有 size、SHA-256、producer、父清单和
    self-hash；
16. 任一失败均落盘失败清单并停止，不能自动换正式 seed、删行或重造同版本。
17. 四个 split 各自的 independent DGP replayer 与 producer 在完整登记
    world 集、membership、market、style group、mechanism slot、
    hard-negative typed-membership topology、identity type/repeat 及
    registered override 决策上逐字节相等；两侧 parent manifest、policy、
    source closure、输入/输出 hash 全部闭合；comparator 进程的结构密钥输入
    和成功打开次数均为 0。

shortcut 随机性门只审计真正应与 controller/label 独立的 nuisance。每
seller 固定 7 维依次是 item count、title/description 两项缺失率和 4 个
time-bucket 概率；pair 固定为 7 维 absdiff 再接 7 维 sum，共 14 列、
`.12f`。raw identity-bearing 文本、33 维和 M0 分数禁止进入。

redacted 长度、数字、标点、换行、繁体和 ASCII uppercase 是预注册的
controller authorship baseline；category/product/attribute equality 是
high-semantic 困难负例；market difference 是 cross-market 正机制。它们
分别透明报告，但不接受 `AUC_sym<=.52` 的“随机”要求，也不得被解释为身份
adapter 贡献。功效设计必须用完整 DGP/C40 验证上述 14 维门的通过概率。

检测器固定为：

- fold-train `StandardScaler` + `LogisticRegression(lbfgs,L2,C=1)`；
- `HistGradientBoosting(max_depth=2,max_iter=200,learning_rate=.03,
  l2=1,early_stopping=false)`；
- fold-train `StandardScaler` + `SVC(kernel=rbf,C=1,gamma=scale)`。

每 split 单独用 SHA world 五折做 OOF，禁止合并 train/development；空 fold、
单类 fold、未收敛或 nonfinite 都是 validity failure。统计量是三模型
`max(AUC,1-AUC)` 的最大值。固定 OOF score 后，用独立 seed、PCG64DXSM
做 9,999 次 world bootstrap，不重训；每 replicate 先对三模型取 max，再以
`method=higher` 取 95% 上界。点估计必须 `<=.52` 且上界 `<=.53`。AUC
接近 0 不算无捷径。

train/development 在 Identity experiment lock 后运行；Audit A/B 的 14 列
projection 在 lock 前自动生成并封存，只在各自正式解封时交给 evaluator。
该失败一律是 `INVALID/B_INVALID`，不能写成效果 `NO_GO`。

Audit sealing worker 在解封前只允许输出逐行标签公式一致布尔值、行数和
文件哈希。Audit 的元数据作弊 AUC、类别数和 label-dependent mechanism
coverage 只可在其全部盲预测、统计代码和锁文件冻结后，随该 audit 正式
解封一次性计算；任何一项失败都使该 split 记为 `INVALID/NO_GO`，不得返回
生成器调参。

矩阵满秩不是标签成功证据，也不得在正式 audit 生成后为了满秩修改生成器。

## 14. 工程冒烟与正式生成纪律

先生成约 1/50 规模、永不进入正式训练或报告的工程集：

- train 10 worlds；
- development 3 worlds；
- audit_a 5 worlds；
- audit_b 5 worlds。

冒烟只允许输出：

- schema、行数和 UID 唯一性；
- parser expectation 的逐项布尔结果；
- 每个世界 C40 是否恰为 40；
- query/gallery 行数；
- 重连结构是否合法；
- 预先计划的机制槽位数；
- 分区交集、防复制和 manifest 布尔门。

标签公式可以在隔离进程验证，但只能输出逐行一致性的单个 boolean。禁止输出
pair 类别数、阳性率、任何 `feature×label`、`trigger×label`、M1/M2 proxy、
AP、AUC、系数或排序。冒烟只能修复结构实现，不能调整机制概率、可分性或
成功差值 `δ`。

冒烟阶段禁止：

- 训练 M1/M2；
- 查看 M0/M1/M2 性能；
- 根据模型成绩调整生成器；
- 将冒烟数据并入正式数据。

冒烟通过后冻结 policy、代码、模板、五个重连种子和正式主种子，一次性
生成正式数据。Audit A/B 一旦生成，不因阳性率、区间宽度或结果不好而重造。

功效设计必须在任何 formal world/seed 生成及本版本冻结前完成，只能使用与
正式数据完全无重叠的设计模拟和预先假定效应情景；不得使用 formal A/B
label、预测或关联统计。

固定 Monte Carlo seed 为 `2026072713`、5,000 replicates，audit world grid
为 `{150,200,250,300,350,400}`。目标是 Audit-A 单独通过概率和按 A→B
顺序的联合通过概率都至少 `.80`，且各自 Wilson 95% 下界至少 `.78`；选择
满足条件的最小共同 A/B world 数。null 场景必须使整套 A familywise 假通过
率点估计 `<=.05` 且 Monte Carlo Wilson 95% 上界 `<=.06`。

检验 margin 不是 power alternative。设计 alternative 必须严格高于 margin：
artifact 至少模拟 A 的 AP 差 `.06`、B 的 AP 差 `.04`，并给出五个 M1 的
paired correlation、M0/M1/M2 conditional score distribution/calibration、
world random effect/ICC、C40 prevalence、retrieval relevance、hard-negative
prevalence/FPR；必须能生成逐行 label+score，而不是只指定 AP 均值。每个
replicate 还要生成独立 development，真正执行固定阈值选择，再完整重放
7 项 AP max-family、TOST、Recall、FPR、metadata、coverage、zero fallback
和 A→B 状态机。

机制覆盖的独立科学下限先于 power 固定：train 每 flag 至少 20 个独立 world，
development 至少 5，Audit A/B 各至少 10；raw33 全零 slice 同样至少达到这些
world 数且至少各有同数 row。power 可以选择更大的 W，不能降低该下限、margin
或 SESOI。若 grid 到 400 仍不足，只能发布新文档修订或放弃 v13 formal。

若当前 250 world 不是最终选择，必须在任何 formal seed 冻结前修订第 4 节。
正式 seed 一旦冻结，不得增加样本、降低门槛或换情景。功效 artifact 必须
解释 AP `.03/.015`、Recall `.01`、FPR `.01` 和 TOST `.01` 的科研最小意义，
并绑定 policy。还必须预演固定 9,999 bootstrap world 索引；formal audit
若仍出现单类 replicate，记为 `INVALID`，不得跳过或换 seed。

## 15. 后续 M0/M1/M2 实验接口

数据集通过后才建立冻结英文预处理包。它必须：

- 从 `step7_v4_prepare_source_data.replay_parent_public(...)[1]["reference"]`
  重建 582 个英文训练卖家参考；
- seller UID 集合哈希精确等于
  `b417fbe6ec1c146943657b00de973889adb0732fbe4aa996297b6462447f8c0e`；
- 完整 reference 对象 canonical hash 精确等于
  `825cc0a42806388de8f4f016273ed83650082f757eea660189e97e48a57853eb`；
- 用该 reference 在已有全部 733 个 Step7-v4 pair 上调用
  `build_safe_pair_rows` 重算 legacy18，最大差异不超过 `1e-15`，并证明两个
  audit-only shortcut 没进入模型；
- 按冻结 pair manifest 的原始 row order 重放 151 条 blind valid，只筛
  primary/C0，并与
  `valid_predictions.blind.no_labels.csv` 逐位 `np.array_equal`；
- row order 来源固定为
  `reports/step7_v4_raw_item_authorship_selection/v2_20260723/
  pair_manifest.no_labels.csv`，文件 SHA-256 为
  `f9f996bdac4a69ce361fa23417ed47cf9216854ffc5558b0202539a311886cb9`；
- blind 对照固定为
  `reports/step7_v4_1_style_free_classifier_selection/v1_20260724/
  valid_predictions.blind.no_labels.csv`，文件 SHA-256 为
  `b0bd6c89f8542fb7eaa158b0d8f1a4b1dc7aaa347d6ec1deaad2ac9e914a90aa`；
- primary 输入矩阵/概率的 float64 bytes SHA-256 必须分别为
  `37ecfd7671335a1d377a62983062f524ae92afa9ef99c920f58aef9450c04133` 和
  `14a1b6f0c579b857818510b5d6c9847b21baa09c4241f0ac9efdaa2afe74a5a7`；
- C0 输入矩阵/概率的 float64 bytes SHA-256 必须分别为
  `9404e755bfdccc3b8a624cddd6380f6ebac8455a4a19877f0909008ea5c729f4` 和
  `4a9d5ac7bb3fe3edcd03d83596228c9a9c9fbc86901d4a45727feecb68dc54b2`。

这些哈希覆盖 C-order 连续 `<f8` 裸字节，不含 header/shape；行顺序是冻结
pair manifest 中 `split_name=="valid"` 的原 151 行，列顺序来自各 joblib
的 `candidate.feature_names`。矩阵 shape 分别为 `(151,24)`、`(151,18)`，
概率均为 `(151,)`。

不得调用 `FeatureFactory.design()`、重拟合 reference/medians、重训
LightGBM 或重选其英文阈值。primary/C0 的冻结英文阈值分别为
`0.2324060118538871` 和 `0.32706942161832925`；它们仅用于重放门，正式中文
阈值指标按下文 development 规则另行冻结。

后续适配器固定为：

```text
logit(p) = logit(p0) + beta^T (z/scale)
```

- 只训练 33 个 beta；
- 无额外截距；
- M0 系数固定为 1；
- `eps=1e-9`，仅为计算 logit 将 `p0` clip 到 `[eps,1-eps]`；
- 尺度只由正确 train 身份矩阵无标签拟合，只缩放、不中心化：
  `scale_j=sqrt((1/W)Σ_w (1/n_w)Σ_i z_wij²)`，若不大于 `1e-12` 则置 1；
- 五个 M1 与 M2 共用尺度；
- 原始 33 维全零时通过代码分支逐位返回原始 p0；
- primary 与 C0 分别拟合 adapter，不跨底座复用 beta。

目标函数唯一固定为：

```text
eta_wi = logit(clip(p0_wi, eps, 1-eps)) + beta^T (z_wi / scale)

L(beta) = (1/W) * Σ_w [
            (1/n_w) * Σ_i {
              logaddexp(0, eta_wi) - y_wi * eta_wi
            }
          ]
          + (2/2) * ||beta||_2^2
```

其中 `logaddexp` 必须调用稳定实现，概率必须用稳定 `expit`，禁止按公式
字面计算 `log(1+exp(eta))` 或朴素 sigmoid。train
`W=500,n_w=40`。M2 与每个 M1 完全使用这一损失尺度；M1 只在
各自 destination-keyed deranged train 的 33 维上拟合，但推断时与 M2 一样读取正常的
development/Audit A/B 身份特征。solver 固定 float64 L-BFGS-B、解析梯度、
`beta_0=zeros(33),maxiter=10000,maxls=100,ftol=1e-12,gtol=1e-8`；未成功
或最终解析梯度无穷范数 `||∇L||∞>1e-7` 均 fail closed，不使用未定义的
“归一化梯度”。

五个 M1 的主 placebo 指标是五个 seed-specific metric 的算术平均，不是把
五组概率平均成 ensemble，也不得在 audit 上挑一个 seed。定义：

```text
Δ20 = AP(M2) - AP(M0)
Δ21 = AP(M2) - (1/5) * Σ_r AP(M1_r)
Δ21,worst = min_r [AP(M2) - AP(M1_r)]
```

数据阶段先验收 redacted text 与非身份原始字段相同；M0 评分后再执行
24 维、primary/C0 p0 完全相同的 Phase-B adapter 前门，不能在“数据通过前”
虚构尚未计算的 M0 相等性。

## 16. 后续审计指标与成功规则

分类主指标为世界等权 pooled AP：调用固定版本的 weighted
`average_precision_score`，每个世界 40 行总权重相同；它是 PR 阶梯积分，
不得与梯形 PR-AUC 互换。另报告：

- ROC-AUC、PR-AUC；
- Precision、Recall、F1、Specificity、Balanced Accuracy；
- Brier、log-loss 和混淆矩阵。

检索报告：

- MRR；
- MAP@10；
- Recall@1/5/10；
- Hits@1/5/10。

所有模型在 development 上分别选择阈值：候选集合为 float64 唯一 score、
`nextafter(min_score,-inf)` 和 `nextafter(max_score,+inf)`；预测规则固定为
`score>=threshold`。最大化世界等权 F1，完全相同 F1 时选择数值更高的
threshold。阈值、slice 定义和所有盲预测必须在打开 Audit A 前冻结，A 后
不得重选。

使用 formal policy 固定的 seed `2026072709` 做 9,999 次配对
world-cluster bootstrap。每次按 world
有放回抽样，所有模型复用同一抽样索引，AP 在合并重采样世界后重新计算；
任一 replicate 若只有单类则整个 audit 失败，不得跳过。检索先在 query 内
计算，再在 world 内平均。所有 10,000 条分类行和 27,000 条 query-gallery
关系都必须进入主指标，禁止按 label、feature state、support、parser 状态、
机制或 slice 删除主样本。

AP 主比较使用固定 95% 单侧 simultaneous max-error bootstrap。比较族为
`Δ20`、`Δ21` 和五个 `AP(M2)-AP(M1_r)`。对 replicate `b` 重算所有差
`d_j^b`，令
`q_lower=Q_.95(max_j(d_j^b-d_j))`，同时下界为
`LB_j=d_j-q_lower`。经验分位数固定使用 NumPy `method="higher"`。不得在
运行后改用普通 percentile、BCa 或层级检验。`M1平均-M0` 的等效性使用同一
配对 world bootstrap 的 90% percentile 区间做 TOST；下端使用
`Q_.05,method="lower"`，上端使用 `Q_.95,method="higher"`，完整区间必须
落在 `[-0.01,+0.01]`，点估计落界内不算通过。

Recall@10 的比较族固定为 `M2-M0` 和 `M2-M1平均`，使用同一 world 索引，
按上述 `q_lower` 公式单独构造该二项 family 的 95% simultaneous 下界。
困难负例 FPR 的比较族同样固定为这两个差，但上界使用
`q_upper=Q_.95(max_j(d_j-d_j^b),method="higher")` 和
`UB_j=d_j+q_upper`。这两个安全 family 不与 AP 七项 family 混合，也不得
事后改变方向或拆分。

确认顺序：

1. Audit A 是同机制首要确认；
2. Audit A 通过后才正式检验 Audit B；
3. Audit B 只支持预注册机制移位稳健性。

在 Audit A 解封前，必须冻结并哈希：两份 observed audit、C40、query/
gallery、M0/C0 全部前处理与模型、33 维定义和尺度、五个 M1、M2、全部
分类与检索盲预测、threshold、slice、指标代码、bootstrap 索引生成规则、
报告模板及所有成功门。Audit B 盲预测也必须在看到 A 结果前完成并冻结。

Audit A 解封后不得修改上述任何字节。A 未通过时，v13 状态固定为 `NO_GO`，
B 保持封存且不作正式检验；若人为打开 B，B 只能作为描述性结果并立即永久
消耗。A 通过后，只有模型、预测、阈值、代码和门槛逐字节未变时才可解封
B。任何 A 后修改必须建立新科学版本及全新 audit，原 A 或已打开 B 不得
再次称 sealed confirmation。

状态转移固定为：

| A 结果 | B 结果 | 主状态 | B 子状态 |
| --- | --- | --- | --- |
| 未解封 | 未解封 | `PASS_DATASET_ONLY` | `B_SEALED_NOT_TESTED` |
| validity 失败 | 保持封存 | `INVALID` | `B_SEALED_NOT_TESTED` |
| 有效但实质门失败 | 保持封存 | `NO_GO` | `B_SEALED_NOT_TESTED` |
| 通过 | 未解封 | `PASS_A_ONLY` | `B_SEALED_NOT_TESTED` |
| 通过 | 有效但实质门失败 | `PASS_A_ONLY` | `B_NO_GO` |
| 通过 | validity 失败 | `PASS_A_ONLY` | `B_INVALID` |
| 通过 | 通过 | `PASS_A_AND_B` | `B_PASS` |

若 A 未通过却人为打开 B，主状态不变，B 子状态记
`B_CONSUMED_DESCRIPTIVE_ONLY`。

任何 audit 的 labels、qrels、类别数、label-dependent 诊断或主指标一旦
可见即永久消耗；后续版本只能把它当开发/历史诊断集。

预注册实质门：

- Audit A：`LB(Δ20)>0.03`、`LB(Δ21)>0.03`，且五个
  `LB(AP(M2)-AP(M1_r))>0`；
- Audit B：对应两项 margin 改为 `0.015`，五个 seed 门仍大于 0；
- 两个 audit 中 `M1平均-M0` 的 90% TOST 区间完整落在
  `[-0.01,+0.01]`；
- Recall@10 的 `M2-M0` 与 `M2-M1平均` 两个 95% simultaneous 下界均不低于
  `-0.01`；
- 困难负例定义为 label=0 且具有共享已解析 token、任一两跳身份路径，或
  属于该 world C40 中按
  `(-lexical_similarity, pair_uid UTF-8 bytes)` 固定排序的前 10 条 pair；
  fallback 的 lexical similarity 固定为 0，不计算运行后分位数。其 FPR 的 `M2-M0` 和
  `M2-M1平均` 两个 95% simultaneous 上界均不高于 `+0.01`；
- 零历史逐位回退 M0。

C0 是预注册敏感性而非主结论一票否决门：报告与 primary 完全对应的差和
区间；若方向不一致或下界不大于 0，只能撤销“跨底座稳健”的表述，不改变
针对 operational primary M0 的预注册主判定。

## 17. CPU/GPU 边界

Windows CPU 数据集阶段：

- 风格参考；
- 世界、商品和身份生成；
- 生产解析；
- 候选与检索表；
- 五个 M1；
- oracle、标签、审计和清单。

后续 M0 阶段：

- Windows CPU 可以计算 legacy18；
- v13 不重写 chunk/encoder/aggregate 数学。CPU wrapper 必须直接调用已固定
  SHA 的 `step7_v4_encode_item_models.build_shared_chunks`、
  `validate_shared_chunks` 及 tokenizer digest helper；CUDA wrapper 必须直接
  调用同文件的 model/tokenizer/determinism helper；六维聚合必须直接调用
  `step7_v4_common.compute_pair_score_rows`。这同时继承 1e-12 零向量门、
  encode `float32`→aggregate `float64` 路径、`min(3,n)` top-k、title/
  description field-equal、256-row block、unique-text 排序/去重和 12 位落盘
  语义。由于冻结 helper 要求 `multiplicity` 并返回 6 primary + 6 weighted
  audit 列，v13 wrapper 对每个已经去重的 opaque seller-text row 只在内存
  补常数 `multiplicity=1`，把 `opaque_pair_id` 临时映射为 helper 的
  `pair_uid` 后调用；随后断言对应 6 个 weighted audit 值与 6 个 primary
  值在冻结 `.12f` 序列化后相同（`np.mean` 与 `np.average` 的 float64
  求和路径可能相差约 1e-17，禁止要求 bitwise 相同或用 weighted 替代
  primary），只投影 `opaque_pair_id + 6 primary`，audit 列和常数
  multiplicity 均不落盘、不进 bundle。wrapper 的输入/输出安全投影可以
  变化，数学 helper 不得 fork；
- helper adapter 的 in-memory rename 唯一固定为：
  `opaque_pair_id→pair_uid`、
  `opaque_seller_left/right→seller_uid_left/right`、
  `opaque_seller_id→seller_uid`、`opaque_text_id→text_uid`。值本身仍是
  `pair_%08d/seller_%08d/text_%08d`，只改字段名。落盘 shared chunks 保留
  冻结 helper 的 `text_uid` 列并保证其值是 opaque text ID；进入/离开 helper
  的每次 rename 都须做 exact schema 和双向无损断言；
- 仅“直接调用”仍不足以防 wrapper 错接。Identity experiment lock 前必须在既有无标签
  Step7-v4 public corpus 上建立固定 compatibility fixture：按 policy 规定的
  text/pair UID 字节序选样，同时用 pinned 原实现与 v13 wrapper 生成 chunk
  rows、四 tokenizer token-ID digest、LaBSE float32 embedding bytes 和最终
  六列 CSV，要求逐字节/逐位一致；fixture 输入、选择清单、预期输出和运行
  manifest SHA-256 全部写入 identity-transfer child policy。dataset parent
  policy 的占位 null 永久不回填；当前 child 尚未建立，所以 formal
  M0/identity experiment 禁用，但不阻断已满足 Dataset release lock 的正式
  数据生成。151 条 blind M0 replay 不能替代此门；
- 共同分块只有在四个冻结 tokenizer payload 全部恢复并通过指纹后才能做。
  当前 Windows 缺少 PCM 与 mStyleDistance payload，不具备正式共同分块
  条件；必须先建 repo 外、label-free 的 CPU chunking workspace，挂载四个
  tokenizer，产出并封存 shared chunks 与 opaque index。不得只检查
  tokenizer 的若干常见文件：每个 tokenizer 所在的完整冻结模型目录都必须
  通过 file-list、size 和 content SHA-256；如未来改为 tokenizer-only 子集，
  必须另发 policy 并固定子集逐文件 manifest/hash；
- Identity experiment lock 同时冻结静态、label-free 的
  `schema/step28_v13_m0_cpu_chunk_policy.json`，只含 shared chunking 与 PCM、
  mStyleDistance、E5、LaBSE 四个 tokenizer spec（包括 E5
  `text_prefix="query: "`、native window、local path/content pins）。CPU
  chunker 只挂载该 projection，不挂载含 private path 的完整 v4 parent
  policy；该 CPU projection 与另外三模型 payload 绝不进入 CUDA；
- CPU chunker runtime 精确固定 Python `3.10.19`、transformers `4.46.3`、
  tokenizers `0.20.3`；版本不等即停止。CUDA 精确固定 Python `3.10.19`、
  NumPy `2.2.6`、PyTorch `2.9.1+cu130`、CUDA runtime `13.0`、cuDNN
  `91300`、transformers `4.46.3`、tokenizers `0.20.3` 和
  sentence-transformers `5.6.0`，并重放 pinned deterministic flags；
- CUDA 编码另建第二个空 workspace，不继承 CPU workspace。四 tokenizer
  payload 和 CPU ordinal↔UID 映射必须在 CUDA 前卸载；CUDA 只接收封存的
  opaque payload；
- CPU-only sealed map 精确拆为
  `opaque_seller_map.csv(opaque_seller_id,seller_uid)` 和
  `opaque_pair_map.csv(opaque_pair_id,canonical_pair_uid)`。seller 按真实
  seller UID UTF-8 字节序、pair 按 canonical pair UID UTF-8 字节序分别
  分配 `seller_%08d`/`pair_%08d`；两个 map 的 file/content hash 进入 parent
  与 sync manifest，但 map 文件绝不进入 CUDA。CUDA pair input 精确为
  `opaque_pair_id,opaque_seller_left,opaque_seller_right` 且按 opaque pair
  递增；seller-text index 精确为
  `opaque_seller_id,field_name,opaque_text_id`，每行是 unique clean text，
  不含 split、source lineage 或 multiplicity；
- unique clean text 按 `(SHA256(text UTF-8 bytes), text UTF-8 bytes)` 排序，
  逐项分配 `text_%08d`。CPU 输出
  `unique_clean_texts.jsonl(opaque_text_id,text,text_sha256)`，shared chunks
  和 seller-text index 只能引用同一 opaque text ID。sync 前必须证明：
  text ID/文本/SHA 一一对应、seller index 无重复引用、所有引用均存在、没有
  corpus orphan、shared chunks 覆盖每个 text 且 chunk index 从 0 连续；
- CUDA 不挂载完整 v13 dataset/experiment policy 或完整 v4 parent policy。
  Identity experiment lock 预先冻结静态
  `schema/step28_v13_m0_gpu_policy.json`；其字段只允许 shared chunking
  参数、LaBSE spec、六项 aggregation、runtime、producer/content hashes 和
  固定 opaque schema，不包含本次运行的路径/input hash。sync builder 不生成
  或覆盖该 policy，只把其 SHA 与本次 shared chunks、opaque inputs、CPU-only
  map hash 绑定到 immutable run sync manifest。任何 identity/mechanism/
  adapter/33 维定义、private-label 路径或 hash 都不得进入 GPU policy；
- CUDA allow-list 只含：上述最小 GPU policy projection、v13 GPU
  wrapper、pinned 旧 encoder
  `7f7018f55e543ad809152d786d6d0e34722f18141ace89b21d7d1eb660f548dc`、
  common
  `8acdac12a579314ddf3e863e3b1c19a026fc252fe520a4ded3f53fda6e765334`
  和其顶层 import 必需的 sync builder
  `bc452cb3643ee0476b004e9715cb0a4ecba53971e5d54780beb9f42eeeaed1f4`，
  以及 shared chunks、唯一 clean text validation corpus、opaque
  seller-text index、opaque classification+retrieval pair union、sync
  manifest 和完整 LaBSE payload。CPU chunk workspace 同样挂载这三份
  pinned 旧模块和 CPU wrapper；不允许从 repo/PYTHONPATH 隐式取其他副本；
- CUDA 不接收 raw items、任何非 opaque seller/world/item/pair 标识（包括
  合成 UID）、split、身份、controller、
  label/qrels、candidate trigger、retrieval role、机制、33 维特征、rewire
  ledger、英文 reference、joblib 或英文私有标签；
- 回传一个 exact-member transport bundle：六项 LaBSE pair 聚合、
  runtime manifest 和只列这两项的 bundle manifest；无其他 payload；
- `labse.item_aggregates.csv` schema 精确为
  `opaque_pair_id` 加第 2 节六列，row order 必须与 opaque pair input
  一致。collector 先在 opaque 空间拒绝重复、缺失、额外或乱序 row，再读取
  CPU-only sealed map 映回 canonical pair；禁止按当前位置猜 join；
- Windows CPU 完成冻结 M0 评分、adapter 和最终审计。

## 18. 计划实施文件

数据合同与模型实验必须拆开：

```text
schema/step28_v13_synthetic_chinese_dataset_policy.json
schema/step28_v13_identity_transfer_experiment_policy.json
schema/step28_v13_dataset_custody_deployment.json
schema/step28_v13_m0_cpu_chunk_policy.json
schema/step28_v13_m0_gpu_policy.json
schema/step28_v13_synthetic_text_templates.json
scripts/step28_v13_common.py
scripts/step28_v13_materialize_style_source_boundary.py
scripts/step28_v13_build_style_reference.py
scripts/step28_v13_generate_dataset.py
scripts/step28_v13_producer_dgp_projection.py
scripts/step28_v13_independent_private_dgp_replay.py
scripts/step28_v13_independent_dgp_comparator.py
scripts/step28_v13_run_independent_dgp_replay.py
scripts/step28_v13_compare_independent_dgp_replay.py
scripts/step28_v13_build_placebo.py
scripts/step28_v13_materialize_supervision.py
scripts/step28_v13_audit_dataset.py
tests/test_step28_v13_synthetic_dataset_contracts.py
docs/STEP28_V13_SYNTHETIC_CHINESE_DATASET_BUILD_CONTRACT_20260727.zh.md
```

模型评分、adapter 和最终统计脚本在数据集正式通过后再进入第二阶段实现，
不能为了提前看性能混入生成器。第二阶段接口预先保留为：

```text
scripts/step28_v13_m0_common.py
scripts/step28_v13_build_frozen_legacy_reference.py
scripts/step28_v13_prepare_m0_public_inputs.py
scripts/step28_v13_build_m0_gpu_sync_manifest.py
scripts/step28_v13_materialize_m0_chunking_workspace.py
scripts/step28_v13_build_shared_chunks.py
scripts/step28_v13_materialize_m0_cuda_workspace.py
scripts/step28_v13_encode_labse.py
scripts/step28_v13_collect_verify_m0_cuda_outputs.py
scripts/step28_v13_score_frozen_m0.py
scripts/step28_v13_fit_identity_adapters.py
scripts/step28_v13_freeze_blind_predictions_thresholds.py
scripts/step28_v13_unseal_audit_a.py
scripts/step28_v13_evaluate_audit_a.py
scripts/step28_v13_authorize_or_seal_audit_b.py
scripts/step28_v13_unseal_audit_b.py
scripts/step28_v13_evaluate_audit_b.py
scripts/step28_v13_world_bootstrap_statistics.py
scripts/run_step28_v13_m0_linux_20260727.sh
tests/test_step28_v13_m0_inference_contracts.py
tests/test_step28_v13_m0_result_contracts.py
tests/test_step28_v13_adapter_and_audit_contracts.py
```

旧 v4 materializer 绑定四模型和旧路径，不能直接当 v13 GPU materializer。

## 19. 修订记录

- `2026-07-27 v1`：合并 `Build_Plan.md` 中两套冲突方案；改用当前
  `LightGBM + legacy18 + LaBSE` operational M0；固定 12 controller /
  28 seller 世界；保留 45,000 条标签盲候选但取消精确正负配额；删除真实
  valid/test 风格、真实短片段和跨卖家身份图统计；增加五重 M1、同世界
  检索、世界级推断、parser 三态和正式版本更新纪律。
- `2026-07-27 v2-draft`：依据三名独立代码/科研审查，拆开 Step3 parser 与
  Step7 redactor；明确 Step4-derived 候选器、renderer slot、整数流 M1、
  retrieval pair union；将 Audit A/B 物理分封并禁止训练前读取；写死
  audit 永久消耗、A 失败时 B 保持封存、world-cluster max-error bootstrap、
  TOST、适配器损失尺度、M0 位级重放和 GPU allow-list。上一版文档
  SHA-256 为
  `68491ed94b47f715256c425006aa60b66a9ec961667498534ccac8f2e021804f`。
- `2026-07-27 v3-draft`：累计补齐完整 structure-draw registry、
  hard-negative membership DFS、固定容量 identity-type CSP、公开 asset UID
  池、occurrence-slot flow、完整 parser fixture、两阶段 release lock 与
  Audit A/B custody 纪律。该未冻结草稿当时漏写本条修订记录；没有生成
  formal 数据。其最终 pre-v4 文档字节由下一条 previous-version SHA-256
  补接，不能把这次补记解释为既往已完成的正式封存。
- `2026-07-28 v4-draft`：纠正“同 producer 重生冒充独立重放”的证据层级；
  将 parser/redactor structural chain 与 structure secret/controller oracle
  解耦；增加完全不 import producer 决策模块的第二套 DGP 实现、UID-only
  replay input、split-private replayer、无密钥 comparator、controller style
  oracle、registered override 精确选择重放和替代密钥完整图攻击测试；把
  solver trace 从 structural audit 移入 split oracle。开发期 23 worlds /
  2318 items 全部 exact，但 formal custody 仍未冻结、formal generation
  仍禁用。上一版文档 SHA-256 为
  `5bd4b1bd4c6340d8dcbab1a21a4c8dba7c9d2e696fe0f0e38f2a22b4e0f0ab84`。
- `2026-07-28 v5-draft`：依据三路独立边界/差分/攻击审核，要求完整 split
  world 集；把完整 oracle 投影从 no-key comparator 移到 producer-private
  projector；为 replay 与 producer projection 增加 parent manifest 绑定；
  给 ledger 全 envelope 加 self-hash 与 strict schema；拆开 parser、redactor
  和 structural auditor 的部署权限草案（尚未实现隔离 launcher）；所有
  development 入口硬拒绝 formal，config
  check 不再读取私有输入。另修复 hard-negative 回退叶定义：同一身份 topology
  的不同文本 override 不再重复计叶，第 1 叶必须是真正不同的身份 topology；
  确定性容量夹具已让 producer/replayer 同时从不可行叶 0 转到叶 1 并 exact。
  当前仍无正式 OS custody launcher/ACL/access-chain parent seal，formal
  generation 继续禁用。上一版文档 SHA-256 为
  `a120afff41a1435b58a3117df5c774f8733eacd3d1ea8de0bcdac69c4f24f9b8`。
- `2026-07-28 v6-draft`：根据第二轮攻击审核，纠正 development comparator
  “绝不打开完整 oracle”的过度表述，并固定输入布局、当前 source bytes
  closure、规范记录顺序、文件/目录 `fsync`、原子发布与失败 staging 清理；
  增加 CLI config-only、不同 typed topology、跨叶累计 type-node 预算和发布
  故障注入回归。修正 producer projection 的 UID 术语：排除 raw identity
  value/rendered occurrence identity UID，但保留 exact topology 比较需要的
  synthetic identity-asset UID。正式 structural auditor 改为只读
  style-stripped render-integrity projection，明确 parser、redactor、auditor
  后经最终组合泄漏审核，进一步收紧为不含任何 world/seller/item join key 的
  aggregate receipt；structural auditor 不再读取 observed/redacted 文本或
  parser 行，candidate/rewire 也必须先聚合封存。明确 parser、redactor、各
  sealer 与 auditor 目前只在合同中分权、尚无独立 formal launcher。development
  comparison evidence 名称从 `PARENT_BOUND` 改为
  `SELF_HASH_MANIFEST_BOUND`，避免与外部 custody parent 混淆。最终复核又
  固定三类 receipt 的递归 exact nested schema，以及 auditor 只能读取四条无
  路径/事件的 parent-seal hash 投影，堵住动态 key 和完整 access manifest
  形成的旁路。外部不可伪造 custody parent、
  OS 账户/ACL/挂载/audit chain 仍不存在，formal generation 继续禁用。上一版
  文档 SHA-256 为
  `db8208977e130841b301411081adf27eef8cdf0837b196cf10d1dd1285ebdf91`。
- `2026-07-28 v7-draft`：在实现 C40 前固定 raw-observed candidate 输入
  边界，禁止误用已脱敏 M0 profile；补上 candidate worker capability；
  明确 sampling audit 的全表排序、null 概率和 trigger flags 精确语义，并
  将 raw sampling audit 从 structural auditor 收回到 candidate-integrity
  sealer，最终 auditor 只读不可连接 aggregate receipt。实现 M1 时又发现
  生产器与独立 replayer 都把 56 条背景边错误地全部设成 1 occurrence，
  与第 6.2 节早已登记的“每 type 各 4 条 count=1、4 条 count=2”矛盾，
  导致 count=2 direct 层不可交换；初次修复又错误地按七条一组交替，使部分
  seller 的两条背景边同时为 count=2，并可耗尽四卖家 hub 的全部七种
  direct type。现已在两套独立实现中改为逐边交替，既保证每 type 为
  `4×count1 + 4×count2`，也保证每 seller 为 `1×count1 + 1×count2`；
  同时固定重连私有输出 schema、五 seed 合并回执计数语义和
  identity-level nuisance 优先级。上一版文档
  SHA-256 为
  `6e397a2a279c90eb9682f947973c1704479c2ace0fc43851c4df9d5c884332ce`。
- `2026-07-28 v8-draft`：开发 smoke 首次实际重算五份文本重连的 33 维矩阵
  后，五个 seed 全部违反预注册 range/SMD/quantile/covariance/correlation
  门，且 full-378 同样失真；二样本 AUC 虽为 `.5284–.5728`，也不能覆盖
  rotation/corroboration motif 被逐层独立换边破坏的事实。故 v7 文本重连
  降级为 parser/renderer 压力诊断，禁止训练主 M1。主 M1 改为在每世界
  C40 与非 C40 内分别执行 endpoint-disjoint 完美匹配，并把 33 维联合向量
  整行错配；不读取 label/controller/trigger/audit，继续使用原五 seed。
  smoke 中五份 M1 的 range/mean/zero/quantile 差均为 0，covariance/
  correlation 仅有 `1e-15` 量级浮点归约差，OOF 二样本 AUC 均为 `.5000`。
  同版还修正背景 occurrence 为逐边 1/2 交替，保证每 seller 恰一条 count1
  与一条 count2。上一版文档 SHA-256 为
  `12e3ca11d6c4a383788896f6fb3980f59f451dd014e01ec156bfa384824607f7`。
- `2026-07-28 v9-draft`：实现 render、candidate、M1-derangement、
  independent-DGP 四类固定 aggregate receipt、递归 exact-schema validator、
  development self-hash parent projection 和只接收 receipt/projection 的
  structural auditor；统一登记 duplicate-preserving canonical multiset
  framing。复核时发现旧草案要求所有 split 都有四份回执，但 M1 又明确
  train-only，现修正为 train 四份、development/A/B 各三份，禁止用空 M1
  回执制造虚假通过。以上实现仍处于同一 development 进程，没有 OS
  custody、audit hook 和外部不可伪造 parent seal，formal generation 继续
  禁用。最终攻击复核继续修正了五处实现边界：所有 receipt 绑定精确
  mode/split/登记 world 集；render 证明所有输入行均被消费；candidate 用 raw
  observed 和冻结 Step3/Step4 完整重放触发与抽样；independent-DGP receipt
  直接校验真实 replay ledger；train 五个 M1 输出在落盘前重新校验登记 seed、
  exact schema/keyset 与自哈希。对应 dataset custody deployment 升为
  `v7-draft`，仍未启用 formal。最后一轮边界攻击又要求 candidate worker/
  sealer 只接收无结构或 oracle secret 的精确 public projection 与单独
  candidate key；sealer 增加不调用 C40 producer 的 Step3/Step4 直接触发
  投影。M1 writer 在落盘点重新计算支持预检并与 aggregate receipt 三方绑定，
  dataset writer 则从登记输入完整重生成 split 后才发布，封存后内存变更均
  失败。独立测试实现同时复算五个 seed 的全部支持数值门。最终复攻又发现
  合法 projection 字段值可夹带 secret、手写 source closure 漏掉实际本地
  import，以及私有 M1 writer 可接受“合法 seed 整体冒充另一 seed”的协调
  重签；现分别用静态 payload 精确 hash+registered-policy 字节对照、递归 AST
  本地依赖闭包+lazy import、从实际 placebo 完整重建 M1 receipt 修复并增加
  攻击回归；最终测试复核指出非字面动态 import 的未来漏绑风险，现进一步
  固定 relative/nonliteral dynamic import、相对 target 及登记的五类
  alias/indirect 语法 fail closed；同时明确未登记反射不属于 development AST
  沙箱声明，须由冻结源码审查和 formal fresh-process inventory 处理。第三方环境锁、
  fresh-process exact inventory 与外部 custody
  parent 仍不存在，仍未启用 formal。
  上一版文档 SHA-256 为
  `70145b21fdcbef9a70ebc2512ce9787a2061094bddfb3e7c1e32da5842f2bd44`。
- `2026-07-28 v10-draft`：根据最终源码闭包攻击复核，把 development AST
  的能力边界继续收紧并精确登记：直接使用 `__builtins__` 名称、直接调用
  `compile/eval/exec/globals/locals` 以及普通 `import builtins` 均立即
  fail closed；六个直接反射原语和全部登记 alias/indirect 模式均加入逐项
  回归攻击。文档同时明确，经别名、自定义分派或其他未登记语法隐藏的反射
  不属于该 AST 检查的声明范围，不能把它解释为恶意 Python 沙箱。formal
  仍要求冻结源码审查、第三方环境锁、全新进程 exact loaded-module
  inventory 和外部 custody parent；这些条件仍未实现，formal generation
  继续禁用。上一版文档 SHA-256 为
  `cbc0f044957fd60c73c016ca4ed737db93e4901dc95b6a28b73f24269a971556`。
- `2026-07-29 v11-draft`：根据实际样本文本复核，撤销“所有标题都显示长
  编号、每条都追加英文 tag、所有身份位置重复同一隔离段”的旧 renderer
  设计。code 改为 `Q + 10 个 A..P`；每 split 恰有一半标题改用由 code
  无标签映射的 16 个共享自然款式词，英文 tag 只按最后符号的 `3/16` 门
  出现；交付与售后基础段扩写为完整句子。身份隔离改为 12 个自然中性段的
  item-UID 排序无放回选择：无身份时 0 个，有 `N` 个身份时恰用 `N+1`
  个不同隔离段。完整 parser/template 夹具由 159,025 例扩大到 232,241
  例并覆盖全部 16 个标题款式映射。另将 Windows 原子写临时文件名改为短
  固定前缀，并让写入、替换、哈希、`stat`、`fsync` 与失败清理统一使用
  Win32 扩展长度绝对路径，避免长 staging 目标触发 `MAX_PATH`；这只修复
  落盘可靠性，不改变数据语义。formal generation 仍禁用。上一版文档 SHA-256 为
  `c539d360d3ab42b15a4ef837aefe60e30c0f7bc1666a5265dbdbdae7fd154b82`。
- `2026-07-29 v12-draft`：三路最终只读审核确认 `dataset_smoke_v2` 的四份
  child manifest、自哈希、181 个文件字节和 M0 脱敏边界均完整，但发现其
  缺少完整 release 父 manifest，fixture outcome 未由 release 绑定，文件及
  最终目录仍有 check-then-replace 并发覆盖窗口，release parent 也未拒绝
  预置 junction/symlink。因此 v2 只作为“字节完整、发布证据不完整”的开发
  产物保留，formal/scientific 使用继续禁止；修复版固定为
  `dataset_smoke_v3`。本版增加四 child + 权威 232,241-case fixture 的父
  manifest 完成标记、跨平台原子 no-replace、release-parent resolve 门和
  M0 三文件精确挂载 allow-list，并补最终 directory rename、长路径实际
  fsync 命中和长路径失败清理测试。为防误用，删除 v2 preflight 下四个非
  权威调试产物：candidate
  `a465ffe5be69f54425ccf6b064cf201bb5dcf1c0a102a83deb64fa7d011396b9`、
  limited-2000
  `e89a9194fe123eac1c97f9d5d28939ab3cd66dccde0afc88b1ea02eae8a98d68`、
  limited-5000
  `2a9711d4af3ee0d09248abcf4b7daa4127b792ea45c42bc61525792fc1d79339`
  及旧 200-case `result.json`
  `6ce2903f02e26c52729d4ee75dbb7c649a320f7f3ab93aa06f324c86dfdded05`；
  仅保留权威 `result_v3`
  `ebdc519978db442e1483e6b63a83076b2a4608907fe64a3a5dd438146b207e9e`。
  formal generation 仍禁用。上一版文档 SHA-256 为
  `24773b6499afa6a072e6110e975e199405b45415ec4ccd2e0d1162fd488f9c9c`。
- `2026-07-29 v13-draft`：在 500-world Audit B 精确预检中发现第 2 个
  UTF-8 world 的 direct high-frequency hub 八个 seller 恰好分别耗尽七种
  direct type 的固定背景容量，导致类型求解器对所有 membership topology
  都在 0 个类型搜索节点处失败；旧实现又为第 `n` 个 topology 从第 0 个
  重跑 DFS，并在每层重复构造 HMAC 候选，形成约 40 分钟的伪“卡死”。本版
  将 producer 与 independent replay 分别改为候选表缓存的一次有序 DFS，
  并增加上述只读取固定背景容量的 high-frequency hub seller 回退。回退前
  后的快速基准证明不触发回退的 10 个 world 完整 payload 哈希不变；触发
  world 的首个 topology 在回退后以 33 个类型节点可行。正式私钥尚未生成，
  所有旧精确预检因依赖闭包变化而只保留为历史设计证据，必须在冻结前对四
  个 split 全部重跑。
