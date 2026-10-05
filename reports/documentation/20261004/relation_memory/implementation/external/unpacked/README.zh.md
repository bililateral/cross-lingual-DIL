# 关系目标记忆：增量实现外审证据

配套主报告：SELLER_ALIAS_RELATION_MEMORY.implementation_review.zh.md。

## 来源与边界

输入为本次 review(9).zip，142,778字节、28成员，SHA-256为4bf6dcff5b5023ebd6fa9a9a06434ae0568029f54c66882aee4f04704c594490。input_identity.json记录实际大小、manifest以及Linux CPU所列三源码的核对结果。

submitted_linux_cpu/是用户提交包内的原始Linux结果拷贝，不是网页重新执行的结果。non_torch/是网页Python 3.12.14和NumPy 2.3.5中的独立检查；环境没有PyTorch，原六测加载失败日志保留。_FailedTest不计作科学测试，不能把失败加载写成六测执行。

本证据没有项目服务器、BGE、GPU、正式文本/标签、旧训练权重、owners或凭据。没有修改输入源码，也没有授权正式训练或新增预算。

## 实际执行与数学小例

- non_torch/run_numpy_checks.py：真实生产非Torch Group、Algorithm R、Memory和规范参考函数；独立NumPy方程/有限差分；静态AST核对。对应numpy_check_results.json。
- non_torch/extra_budget_checks.py：六群长短文本的完整字节量尺、阶段新增RNG后的越界。一般量尺H/b是序列化夹具，不是训练所得统计。超限全容器大小采用显式临时量尺额度；生产1MiB门的拒绝和恢复原门后拒绝均保留。这不是把超限状态认作合规。
- state/reproduce_stage_budget.py：主报告采用的目标统计一致反例，H/b/c由48群常数规范z手算，Algorithm R实际插入48手写群并保留6群。没有模型训练或BGE token核验。阶段末1,044,680 B，begin_stage新增6,719 B，变为1,051,399 B；生产to_bytes拒绝，但生产draw仍返回群。对应stage_budget_result.json。主审阅读脚本后再执行，获得相同结果。

## 复核路径

脚本保留实际执行时的绝对路径，便于审查原件，未包装成实验平台。若在其他目录复核，仅将脚本中输入根路径改为同身份review(9).zip的解压目录；non_torch/snapshot保留独立检查时的未改源码/配置/测试副本。不得将路径调整后的运行当作原始日志，也不得混入不同源码仍沿用本报告身份结论。

独立检查时设置CUDA_VISIBLE_DEVICES为空，OMP_NUM_THREADS、OPENBLAS_NUM_THREADS、MKL_NUM_THREADS为1，PYTHONDONTWRITEBYTECODE为1。未安装依赖。state反例仅需要NumPy和标准库。

## 报告优先

报告A1—A5是当前原生一步前需要对齐/接通的事项；B1/B2属于正式固定点前必要修复，不阻断当前count=1直接optimization_step夹具。可选测试增强不自动变成新的前置门。官方API链接见official_api_sources.json；它们不证明项目已安装某个Transformer版本或已通过原生核验。
