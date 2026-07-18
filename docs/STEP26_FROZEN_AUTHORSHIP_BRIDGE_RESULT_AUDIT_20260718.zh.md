# Step26 冻结作者风格同边界桥接结果审计

更新日期：`2026-07-18`

## 1. 结论

Step26-v1 在 Linux 上完整执行并同步成功，输出、pair universe、指标和 grouped bootstrap 均可独立复现。程序状态为 `pass`，但预注册科研假设未通过：

- frozen English source-only semantic+style primary 在 representative valid 上为 `ROC-AUC=0.736296`、`AP=0.508495`、`PR-AUC=0.498554`；
- matched Step15-v8 clean 为 `ROC-AUC=0.754074`、`AP=0.574855`、`PR-AUC=0.569806`；
- primary 的 valid AP delta 为 `-0.066360`，不是要求的 `>= +0.03`；
- component-grouped bootstrap 95% CI 为 `[-0.322642, 0.149181]`，跨零且 lower bound 远低于要求的 `0`；
- 六项 gate 仅通过三项，`eligible_for_one_step26b_experiment=false`。

因此 Step24 的中文 canonical-train D0 高分不能外推为 corrected valid/test 上的稳定提升。按照预注册协议，不能继续实现 Step26B，也不能将 Step24 frozen authorship fusion 晋级 Step11/17 或论文主结果。

## 2. 同步与完整性

同步目录包含 `14` 个文件，其中 sync manifest 绑定 `13` 个 generated outputs：

- generated outputs：`13/13` 存在，size 与 SHA-256 全部匹配；
- required code：`8/8` 匹配；
- required frozen inputs：本地可见的 `18/20` 匹配；
- 缺少的两个 input 是 Windows 上已主动删除的 Step24 模型目录中的 `step24_model_provenance.json`，不是 Linux 运行输出；两个 embedding metadata 已持久化 repo、revision、provenance 与 model-directory fingerprint，因此不影响本次结果审计，但 Windows 本地不具备完整重新编码条件；
- summary、clean-text manifest、embedding manifest、sync manifest 的内部 payload hash 全部复现。

## 3. 数据边界与覆盖

| Split | Pair | Positive | Negative | Seller | Component |
|---|---:|---:|---:|---:|---:|
| representative valid | 120 | 30 | 90 | 127 | 35 |
| internal development test | 200 | 50 | 150 | 312 | 126 |

两个 split 之间：

- pair overlap：`0`；
- seller overlap：`0`；
- component-ID overlap：`0`。

共编码 `439` 个 seller。clean-text replay 记录 `identifier_redacted=true`、`labels_or_evidence_read_before_encoding=false`；两个 encoder 都记录 `locally_finetuned=false`、`encoder_parameters_updated=false`。

预测 CSV 共 `2,880 = 320 pairs x 9 models` 行。每个模型都完整覆盖同一 pair universe，无重复 pair。独立从 prediction scores 重算全部 `18` 组 ROC-AUC、AP 和 PR-AUC，与 summary/metrics CSV 零差异。使用相同 seller-component、seed 和 tie-aware AP 定义重新运行 `5,000` 次 grouped bootstrap，也逐值复现原结果。

## 4. 整体结果

### 4.1 Representative valid

| Model | ROC-AUC | AP | PR-AUC |
|---|---:|---:|---:|
| Step15-v8 contextual | 0.766667 | **0.601670** | 0.597075 |
| Step15-v8 B0 | 0.754444 | 0.600546 | 0.595417 |
| Step15-v8 clean | 0.754074 | 0.574855 | 0.569806 |
| frozen source E5 LR/L2 | 0.732222 | 0.532578 | 0.523386 |
| raw redacted E5 | 0.732222 | 0.532578 | 0.523386 |
| frozen source semantic+style primary | 0.736296 | 0.508495 | 0.498554 |
| raw mStyleDistance | **0.782222** | 0.459583 | 0.438111 |
| frozen source style-only LR/L2 | 0.684815 | 0.384316 | 0.371520 |
| raw PCM | 0.657407 | 0.361766 | 0.349465 |

Raw mStyleDistance 的 ROC-AUC 最高但 AP 很低，说明它能在全局上区分一部分正负例，却把若干 template/public-format negative 排到最前面；它不适合作为高精度候选发现主模型。

### 4.2 Internal development test，仅诊断

| Model | ROC-AUC | AP | PR-AUC |
|---|---:|---:|---:|
| Step15-v8 contextual | **0.844533** | **0.620525** | **0.615462** |
| Step15-v8 clean | 0.828667 | 0.544139 | 0.534563 |
| frozen source semantic+style primary | 0.764133 | 0.517033 | 0.503136 |
| raw mStyleDistance | 0.782000 | 0.489373 | 0.474811 |
| frozen source style-only LR/L2 | 0.752533 | 0.485204 | 0.471070 |
| Step15-v8 B0 | 0.666800 | 0.460756 | 0.456230 |
| raw PCM | 0.675867 | 0.442579 | 0.428367 |
| frozen source E5 LR/L2 | 0.662267 | 0.411959 | 0.397445 |
| raw redacted E5 | 0.662267 | 0.411959 | 0.397445 |

该 split 已被历史分析消费，只能说明机制差异，不能替代 valid gate 或论文 final holdout。

## 5. Gate 审计

| Gate | Observed | Requirement | Result |
|---|---:|---:|---|
| valid AP gain vs v8 clean | -0.066360 | >= +0.03 | fail |
| bootstrap CI lower | -0.322642 | >= 0 | fail |
| direct/component AP delta | +0.030349 | >= -0.03 | pass |
| soft-positive AP delta | -0.081002 | >= -0.03 | fail |
| template top-budget intrusion increase | -4 | <= 0 | pass |
| public-noise top-budget intrusion increase | 0 | <= 0 | pass |

Bootstrap mean delta 为 `-0.057595`，95% CI 为 `[-0.322642, 0.149181]`，`P(delta>0)=0.3478`。这不能证明 primary 在统计上必然有害，但明确不能支持它优于 v8 clean。

## 6. 根因

### 6.1 作者风格特征仍与模板复制高度纠缠

Representative valid 的均值：

| Evidence type | E5 | PCM | mStyle |
|---|---:|---:|---:|
| soft positive | 0.9512 | 0.9469 | 0.9962 |
| template negative | 0.9385 | 0.9505 | 0.9915 |
| direct positive | 0.9323 | 0.9158 | 0.9381 |

soft positive 与 template negative 在 PCM/mStyle 空间几乎重叠，template negative 的 PCM 均值甚至略高。这解释了为什么 style-only valid AP 只有 `0.384316`。

Internal test 中 public-contact/URL negatives 的 raw mStyle 均值为 `0.9964`，`6/6` 全部进入 raw mStyle 的 top-50 positive budget；template negatives 均值为 `0.9618`。冻结作者风格 encoder 仍把公共格式、复制文本和内容组织方式编码为“作者相似”。

### 6.2 English source 系数没有在中文 evidence composition 上稳定迁移

在 internal test 上，semantic+style 相对 E5 control 提高约 `+0.105 AP`；但在 representative valid 上反而降低约 `-0.024 AP`。这说明并非完全没有跨语言 style signal，而是 effect 对 split/evidence composition 高度敏感。

Representative valid 只有 `4` 个 direct positive、`26` 个 soft positive和 `3` 个 public-noise negative；internal test 有 `22` 个 direct/component positive、`28` 个 soft positive和 `6` 个 public-noise negative。当前 valid 只有 `35` 个 component，bootstrap 区间因此很宽。

### 6.3 Step24 D0 高分包含明显的 train-silver/composition optimism

Step24 source-only semantic+style 在中文 canonical train D0 上是 `AP=0.802718`，但在 corrected valid/internal test 上只有 `0.508495/0.517033`。编码器和 English source artifact 均未改变，因此下降不能归因于重新训练或实现差异。最合理的解释是：canonical train 中 `213/229` 个 positive 为 train-only silver，且其 evidence composition 更容易被 style representation 排序；D0 对独立边界的可迁移性被高估。

### 6.4 Step15-v8 contextual 仍是当前边界上的最强诊断模型，但并未解决 public noise

Contextual 相对 v8 clean 的 direct/component AP：

- valid：`0.099055 -> 0.369231`；
- internal test：`0.348823 -> 0.547385`。

这说明 occurrence-level direct evidence uplift 有实际价值。但 internal test 的 public-noise score 与 clean 完全不变，六条 public-noise negative 仍全部进入 top-50。因此 contextual 不能据此升级为已经解决噪声问题的论文方法。

## 7. 科研状态与下一步

1. 冻结 Step26A 为严格、可复现的 negative bridge result。
2. 不实施 Step26B；不得通过换 primary、降低 gate 或查看 internal test 后重新设计同一边界实验。
3. Step24 作者风格可以保留为诊断和负消融，但不能进入 Step11/17 publication validation。
4. 当前值得保留的机制证据是：clean ranking 与 occurrence-level direct evidence uplift 有互补性；但 public/template noise 缺乏可靠 downgrade evidence。
5. 下一步必须建设 score-blind、seller-component-disjoint 的 D1：至少包含 `30` direct/component positives、`30` template negatives、`30` semantic-topic negatives、`20` public-contact/URL negatives 和 `30` ordinary negatives。
6. D1 只能作为下一方法的独立开发/复制边界；所有配置冻结后仍需 Step20 prospective final holdout 一次性确认。
7. 如果无法获得足够 proof-level positives，应停止追求新的性能提升主张，转向 evidence-type concept drift、数据集纪律和严格负结果论文。

本结果不能支持“作者风格模型解决了中文马甲识别”，但它提供了一个重要且可信的结论：当前冻结 style encoders 的跨语言信号真实存在但不稳定，并被 template/public-format similarity 严重污染；Step24 的 train-D0 高分不能替代独立边界验证。
