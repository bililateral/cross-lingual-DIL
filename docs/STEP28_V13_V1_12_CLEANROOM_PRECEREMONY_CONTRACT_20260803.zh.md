# Step28-v13 v1.12 clean-room 执行前合同

日期：2026-08-03
状态：`DESIGN_VALIDATION_ONLY_NO_FORMAL_AUTHORIZATION`

## 1. 科研问题与当前边界

Step28 仍回答同一个问题：冻结的真实英文马甲识别流水线能否与只从中文合成
训练集学到的身份历史适配模块组合，并在独立中文合成审计集上显著优于冻结
英文基线和等数据量无身份信息对照。

本合同的后继运行 ID 为
`v13_training_ready_v1_12_cleanroom_20260803`。v1.3--v1.11 均为永久失败
谱系；禁止复用其 seed、私钥、UID、身份值、运行目录或专用代码。成功冻结的
v1.2 只保留历史证据，因 C40 捷径和 Audit custody 暴露而没有训练资格。

当前正式 seed 生成、正式数据生成和模型训练授权均为 `false`。本阶段只能运行
无正式密钥的设计级预检，不能把预检通过表述成数据集就绪或模型成功。

## 2. 冻结数据形状

正式候选设计保持四个相互隔离的 split：train、development、Audit A、
Audit B，每个 split 500 个 world。每个 world 固定 28 个 seller、12 个
controller，并使用完整 `K28` 的 378 个无序 pair：20 个同控制者正例和
358 个异控制者负例。每个 split 因而有 189,000 个 pair，其中 10,000 个
正例；随机 AP 基线为 `20/378`。

C40 不得进入训练、调参、阈值、主评估、公开 model mount 或 M1 分层。它若
保留，只能在所有预测冻结后由私有评估器生成机制诊断。检索使用每个 world
全部 28 个 seller 作为 query，每个 query 的 gallery 为其余 27 个 seller。

## 3. 模型和对照

- M0：已由真实英文马甲标签训练并冻结的
  `LightGBM + legacy18 + LaBSE` 完整流水线；对中文文本只做推理。
- M1：五个独立对照。在每个 train world 的完整 378 对上，对整行
  identity33 做 endpoint-disjoint 双射；不读标签、controller 或 C40。
- M2：与 M1 共用 pair、M0 概率、权重、变换、L2、求解器和阈值，只换成
  正确对齐的 identity33。
- M3-base/M3-joint：仅用合成中文 train 标签训练的直接监督强对照。

核心身份增益是 `M2 - mean(M1)` 以及五个 `M2 - M1_r`。`M2 - M0` 同时含
目标域校准影响，只作辅助解释。

## 4. 公开、私有与可见文本边界

M0 只能读取去身份后的标题、描述和 pair 端点。标题从未承载登记身份槽，因此
只做固定文本规范化，禁止送入全局身份替换器；描述只按生成器登记的公开上下文
边界删除身份后缀。join-only UID 只校验格式和血缘，不作为自然语言扫描对象。

train/development 的 full-378 标签可在数据门通过后公开；Audit A/B 的标签、
qrels、controller、机制、身份资产、原始身份文本、AST 和逐行私有审计必须
只在 Git 忽略的 `private_custody/` 中。master seed 永不挂载给生成器或模型；
各进程只获得用途受限的派生 capability。

## 5. 历史排除与九项强制回归

未来运行必须直接读取并完整校验 915,996 个排序去重身份值哈希和 90 个禁用
master commitment 的紧凑档案，不得恢复已删除的失败 payload。正式仪式前，
以下失败必须各有正测和负测：

1. 机制槽唯一键必须是 `(world_uid, mechanism_slot_uid)`；跨 world 复用合法。
2. Windows 长路径的写、读、stat、遍历使用同一实现；manifest 版本只有一个源。
3. 身份冲突按固定资产顺序逐值确定性解析，禁止要求单一 salt 同时覆盖整个
   192,000 候选全集，也禁止通过换 master seed 解决。
4. self-hash 必须在全部字段写定后计算；追加任何字段都必须使校验失败。
5. 固定捷径攻击须在仪式前以正式求解器参数收敛；警告仍 fail closed，不能
   在正式运行中临时加迭代次数重试。
6. 无身份标题逐字节保留，身份删除只按登记描述后缀边界执行。
7. 可见自然语言身份扫描与 join-only UID 血缘校验必须分离。
8. producer 与 consumer 共用同一冻结成员合同，禁止过期成员版本名。
9. 发布审计和生成审计必须调用同一 world-scoped 机制槽验证器。

## 6. 正式仪式前的 GO 门

先冻结源码闭包、policy、阈值、依赖版本和 capability 域；再完成全仓测试、
九项回归、两个完整 world/756 pair 的无正式密钥端到端预检、真实长路径写盘
回放和固定捷径求解器预检。预检不得读取正式密钥、生成正式 UID 或输出科研
效果指标。

上述证据全部通过后，另立不可回写的正式 pre-ceremony lock。随后按固定顺序
一次性生成四份新 master seed 和公开 commitment。任何正式失败都永久关闭
v1.12：先记录边界和不可逆排除哈希，再删除失败数据、缓存、workspace 与
版本专用代码；禁止同 seed 修补、续跑或筛 seed。

## 7. 后续报告指标

分类完整报告 ROC-AUC、AP、梯形 PR-AUC、Precision、Recall、F1、Specificity、
Balanced Accuracy、MCC、Brier、Log Loss 和 Recall@FPR=1%。检索报告 MRR、
完整 MAP、Recall@1/3/5/10、NDCG@1/3/5/10。阈值只由 development 冻结；
bootstrap、置乱和置信区间单位一律为 world，不得把 189,000 个 pair 当成独立
样本。合成 Audit 只支持内部机制结论，不支持真实中文外部有效性。
