审查证据包说明（UTF-8）

一、先读什么

REPLAY_IMPROVEMENT_REVIEW.zh.txt：完整中文审查正文及 E1–E9 证据索引。
READING_SCOPE.zh.txt / .json / .csv：逐件列出五层容器共 600 个成员的实际阅读范围。
MANIFEST.json：交付件大小及 SHA256。仅用于交付身份，不是科学有效性证明。

本次结论：未发现已证实 SCIENTIFIC_BLOCKER 或当前 REPRODUCIBILITY_DEFECT；可由主审关闭后转入已经批准的原生资格核验。尚无本次原生 GPU 通过记录，不能直接绕过原生前提训练。

二、实际执行与未执行

实际主核对：audit_review.py；输出在 calculation，执行记录在 execution/attempt01。
实际补充核对：supplementary_checks.py；输出 calculation/supplementary.json，记录 execution/supplementary_attempt01。
实际台账构建：build_reading_scope.py；记录 execution/scope_build01。
独立计时包装器：record_execution.py。

主核对与补充核对均一次通过，原始 stdout/stderr/exit code 已保留。
另有一次只读探索错误：read_matrices 是 dict，却被当成列表 [:4] 切片，产生 KeyError。原脚本、实际输出和说明见 execution/exploratory_inspection01。该次没有计时包装，不补造 wall/RSS；主核对输出未被修改。

没有委派 agent，没有项目服务器访问，没有正式数据/标签/模型/完整 Memory/凭据访问，没有 GPU、Torch 真实更新或训练。网页环境是 Python3.12.14、NumPy2.3.5；没有 Torch/psutil。所用旧矩阵是已开放的 60×22 群指标，不是原始标签。

audit_review.py 对缺少 psutil 的 runner 仅提取原样 AST 函数：old_runner.fields；new_runner.policy、point_name、specifications、historical_replay。具体源行记在 calculation/execution.json；只适配相同 SHA 旧归档在附件中的路径，不改函数体、不执行正式 main。这项执行不能冒称完整正式入口或原生资格。

手写 37 参数网络的 finite difference 对照用于教师项链式法则；Adam 数字是解析/有限差分两种梯度代入同一手写 AdamW 公式后的参数差。它不是一次 Torch Adam 的独立实现鉴定。

三、如何在本地附件副本重现允许的核对

包内 input/review_input.zip 是用户本次已发送的原包逐字节副本，外层 SHA256 为：
f24e5c147561fef9a37cc05842c0843207c95c859b1f076fb47d10f2033fedb7。
没有新增任何项目端数据。无需访问项目服务器、BGE、正式标签或模型。

prepare_inputs.py 是本次为交付新增的解包便利脚本，未在原科学计算中使用；只做上述原包 SHA 核对与目录解包。build_reading_scope.py 是本次实际阅读台账的生成记录，含当前工作区路径和人工范围标注；它不是供另一位审查者继承“已读”状态的证明，也不是科学核对的必要步骤。

在本证据包解压目录中，使用现有 Python 与 NumPy（无须 Torch/psutil），可以执行：

  python prepare_inputs.py --workspace ./attachment_recheck
  python record_execution.py ./reproduced_execution/main python audit_review.py --root ./attachment_recheck/review_current --background ./attachment_recheck/review_background --out ./reproduced_calculation
  python record_execution.py ./reproduced_execution/supplement python supplementary_checks.py --root ./attachment_recheck/review_current --background ./attachment_recheck/review_background --output ./reproduced_calculation/supplementary.json

上述脚本只读附件、运行 NumPy 手写算术/纯保存指标统计、写复核结果。不要运行输入包内的 Linux 启动器、GPU verifier 或正式 train 入口；它们不是本证据包复现步骤。重现的耗时和包内当前网页耗时当然可能不同，不属于项目 Linux 原生资格账。

四、主要输出

calculation/identity.json：64 项外层载荷、继承来源比较、五轮 CPU 时间/RSS及最终28来源对应。
calculation/handmade_chain_rule.json：四臂教师项解析/有限差分梯度、手写一次AdamW影响、A0零空间反例及梯度差恒等式。
calculation/diagnostic_statistics.json：非恒定手写曲线的独立 bootstrap 对照、已知方向答案、空缓存 N/A 和截断 log_loss 边界。
calculation/resource_counts.json：36/45/216/270 等数量、全部前向/保存恢复/CPU 指标计数及原生调用覆盖。
calculation/saved_history.json：72个旧矩阵具体路径/身份、81逻辑角色同输入差、旧 C−S O MAP 实际域配对复算。
calculation/original_23_equal_example.json：原23函数同臂20/23、整体false。
calculation/supplementary.json：实际旧LOGIT update AST一致、旧保存timing字段算术和其限制。
calculation/execution.json：环境、主核对状态、AST来源行和明确未执行范围。

五、失败账和证据保管

原项目 CPU attempt01 的非法手写样例失败保留在原输入；五次尝试均参与原账。80.86秒外部合计与86.684205993秒保守累计口径不同；后续从88秒接账。该失败已修正，不是当前路径缺陷。
本次网页探索 KeyError 是审查过程中真实发生的只读元数据错误；不把它删去或归因给被审项目。

本包不修改任何原输入、冻结结果或正式科学合同。结论不授权第二轮、调参、扩预算、改指标或自动删除依赖中的阶段末权重。
