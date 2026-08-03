# Step 28-v13 身份迁移实验执行前冻结与可见文本捷径控制

更新日期：`2026-07-31`

## 1. 本次修改解决什么问题

正式中文合成数据集 v1.2 发布后，原 24 项可见代理在 development 上的
最高 AUC 约为 `0.577`；追加的字符三元组攻击探针达到更高分数。这说明
合成器产生的可见文本中仍有一部分与标签相关的模式，原 24 项代理低估了
这种关联。

本次不重生成数据，也不把所有作者风格信号抹掉。原因是同一控制者的写作
习惯本来就是 M0 应读取的共享基线信息；把它全部消除会同时毁掉研究对象。
正确处理方式是：冻结一个更强的可见文本诊断，并强制 M0、五个 M1 和 M2
使用逐 pair 完全相同的冻结 M0 分数 `p0`。这样，可见文本带来的收益被共同
抵消，身份模块的证据只能来自 `M2-M0` 和 `M2-M1`。

## 2. 冻结的 T_text 诊断

`scripts/step28_v13_visible_text_probe.py` 固定使用 train 拟合、development
评估。它只读取脱敏标题、描述及安全档案统计，不序列化 UID、`profile_text`
或原始身份字符串；字符三元组经固定哈希后，用 pair 绝对差和逐元素乘积训练
L2 logistic SGD。禁止调参、候选选择和读取 Audit A/B。

正式结果为：

| split | pair 数 | 正例率 | ROC-AUC | AP |
| --- | ---: | ---: | ---: | ---: |
| train | 20,000 | 0.40 | 0.663437 | 0.559696 |
| development | 20,000 | 0.25 | 0.627569 | 0.346028 |

此前一次性探针报告的 development `AUC=0.628126`、`AP=0.347326` 已在
锁中披露；正式值的轻微差异来自文档序列化固定方式，没有为追逐更高分数
进行参数搜索。T_text 使用合成中文标签训练，因此只是最坏情况的描述性
攻击探针，不是冻结英文 M0，不是模型候选，也不是数据集重新验收门。

私有基础风格变量约 `AUC=0.91` 是生成器内部 oracle，不对模型可见，不能
称为“可见文本泄漏”。当前有直接证据支持的可见关联强度是上述约 `0.628`。

## 3. M0/M1/M2 的统一前门

`scripts/step28_v13_phase_b_front_door.py` 在任何适配训练前执行以下检查：

1. 角色必须恰为 `M0`、`M1_r01` 至 `M1_r05`、`M2`；
2. 每个角色必须覆盖同一 C40 `canonical_pair_uid` 集合；
3. 按主键连接后，每一行 `p0` 的 float64 数值必须完全相同且位于 `[0,1]`；
4. 主键不得进入特征矩阵；该验证器不得打开 identity33 或标签；
5. M1 与 M2 唯一允许的训练差异，是五个整向量错配 identity33 与正确
   对齐 identity33。

后续主要结论只认 `M2-M0`、`M2-mean(M1)` 和最差种子
`min(M2-M1_r)`。继承构建合同的配对 world bootstrap：M1 平均与 M0 的
90% TOST 区间必须完整落入 `[-0.01,+0.01]`，仅点估计接近零不算通过。
未经增量比较，M2 的总分不得写成“身份信息收益”。

## 4. 冻结状态和结论边界

执行前锁为
`schema/step28_v13_identity_transfer_experiment_policy.json`，状态是
`FROZEN_PREEXECUTION_CONTROLS_FORMAL_EXECUTION_BLOCKED`，canonical
self-hash 为
`3d212d57e1bc0fed09d76fb188cd7c043ab0e66f38e330e2f34aa7eafdfb7cb7`。
它同时固定 v1.2 发布清单、科学合同、operational M0/C0 joblib、Step7
策略、LaBSE 内容指纹和本次实现源码。

正式探针报告位于
`reports/step28_synthetic_chinese_dataset/post_release_audits/formal_visible_text_adversarial_probe_v1_20260731.json`，
文件 SHA-256 为
`c3e91df7120604134a27fac0c3d44a129f29717c1b2b1ea9321c6c205369ff71`，
self-hash 为
`7c01abc11bae797db92162a11e7007c0e048633be902caf047f85254b82a9c5b`。

本次只完成可见文本诊断与 Phase-B 前门，尚未授权正式 M0 评分、适配器训练
或 Audit 解封。仍须完成 label-free 兼容夹具、CPU/GPU 静态策略、冻结 M0
打分链、适配器/阈值/重采样指标实现、Audit 授权流程，并提交后重新验证完整
基线。v1.2 的正式数据字节和发布资格均未改变；当前仍不能声称 M1/M2 实验
成功。

## 5. 本次验证

- 三个新增脚本及两个新增测试文件通过 `py_compile`；
- 身份实验控制的 7 项针对性测试全部通过；
- 既有发布后审计的 15 项测试通过、1 项按合同跳过；
- 独立发布树加固审计重新得到原 self-hash
  `fbfe00737077f6422d24030e65840c447f9e1f314597ba3664171515f16580a7`；
- 全仓 `python -m unittest discover -s tests` 共运行 388 项：381 项通过、
  7 项按既有声明跳过、0 失败，用时 819.420 秒。

## 6. 2026-08-01 训练启动澄清

本文件冻结的控制已经通过，不等于正式训练已获授权。当前机器锁仍保持
`formal_m0_scoring_authorized=false` 和
`formal_adapter_training_authorized=false`。为回答“为何不直接在合成中文
数据上训练”，正式矩阵还必须在版本化后继策略中预先冻结 M3-base 和
M3-joint；当前 v1 锁及其正式探针报告不得原地改写。最新状态、基线定义和
完整启动门见
`docs/STEP28_V13_MODEL_TRAINING_READINESS_20260801.zh.md`。

## 7. 2026-08-01 后继 v2（不改写本锁）

本文件及 v1 schema 继续保存“实现未完成、执行被阻断”的历史事实。登记的
blocker 随后已在独立后继合同
`schema/step28_v13_identity_transfer_experiment_policy_v2.json` 中关闭；
v2 状态为 `FROZEN_PREEXECUTION_IMPLEMENTATION_COMPLETE`，允许按冻结
顺序开始 label-free M0、train-only M1/M2/M3 和顺序 Audit A/B 流程。
v1 本身仍保持 blocked，任何 v2 结果不得倒写成 v1 当时已经就绪。最新
执行边界、环境和命令以
`docs/STEP28_V13_MODEL_TRAINING_READINESS_20260801.zh.md` 为准。
