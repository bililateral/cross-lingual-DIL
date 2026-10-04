完整群隔离元更新独立外审：交付说明
日期：2026-10-04（UTC）

一、阅读顺序

1. group_meta_review.zh.txt：完整中文主审报告，含科学定位、源码逐段追踪、数学推导、实际检查、缺陷、未来最小比较及来源。
2. external_verdict.json：机器可读裁决；不是正式效果或GPU准入证书。
3. RUN_INDEX.json：本次独立参考运行与上传Linux运行分列。每次实际参考运行附argv、cwd、stdout、stderr、退出状态；两次本审查自设范数断言失败完整保留。
4. EVIDENCE_MANIFEST.json：包中每个文件的字节数和SHA-256，不自哈希。其作用是定位和来源一致性，不认证科学正确性。

二、目录边界

input/review_original.zip：收到的review(6).zip的精确字节副本。
source/：按原ZIP的103成员清单提取，原始程序、合同、旧外审、成功/失败原件均不改写；不包含运行时生成的__pycache__。
evidence/scripts/、evidence/inputs/、evidence/outputs/：主审实际编写并运行的标准库/NumPy参考、手写输入、结果和日志。
evidence/agent_meta/：从原源码提取的验证器与query表达式、反例、失败及修订记录。
evidence/agent_stage/：精确AST阶段/记忆路由、完整手写输入和逐步路由。所有神经update是明确的spy，不是真实288步训练。
evidence/agent_native/：上传Linux原件的独立标准库复算、来源和修复差异；不是重新运行BGE。
evidence/agent_prior_scope/：旧审查复用范围、已读原文目录与一个组合概率参考。
evidence/proposed/：两项窄修正的拟议补丁；未应用于source或上传Linux原件。
sources/：附件文件来源和直接原始文献目录，说明已读与访问失败范围；不打包论文全文。

本次环境：Python 3.12.14及NumPy；没有PyTorch、Transformers、SentenceTransformers、pytest。没有安装这些包，没有访问正式数据、标签或旧模型，没有原生BGE/GPU复跑，没有新方法效果比较。原Linux证据是用户附件提供的另一环境的原始记录。

三、复核与重跑

所有原始argv和本次绝对路径保留在run.json或command.json中，方便核对真正执行过什么。不要直接覆盖本包内的历史输出；若重跑，请先复制到一个新的工作目录，再将命令中的旧绝对路径替换为新路径。

数学参考只需NumPy。在证据包副本根目录可运行：
python evidence/scripts/10_independent_math.py

该程序从固定种子重建手写NPZ并写入evidence/inputs与evidence/outputs/10_math；它不会加载source中的Torch代码。文件与日志来自本次真实执行；重跑生成的新日志应另存。

元路径纯Python/NumPy参考支持显式路径：
python evidence/agent_meta/source_validator_and_coverage.py --source /新副本/source --input /新副本/evidence/agent_meta/inputs.json --output /新的输出目录/meta_findings.json

阶段/记忆参考支持显式路径：
python evidence/agent_stage/scripts/stage_memory_reference.py --source /新副本/source --evidence /新的输出目录/stage

原生证据复算：先复制evidence/agent_native/inputs.json，把其中source_root改为新副本的source绝对路径，再运行：
python evidence/agent_native/audit_native_evidence_v2.py --inputs /新路径/native_inputs.json --out /新的输出目录/native

保留的audit_native_evidence.py与agent_meta/run_01/source_at_failure.py是首次失败版本，不能冒充最终修订。两个失败源于自设的过强范数归约一致性断言，不是新BGE执行失败。

最小修正参考：
python evidence/scripts/20_minimal_fix_reference.py

该脚本只在evidence/proposed写拟议副本和补丁，复算返回表达式并比较训练函数AST；它不对原source打补丁，不运行Torch。源代码和验证范围见文件头。

组合概率参考可从agent_prior_scope/run_logged.py的实际argv重放；它只计算Algorithm R均匀样本的组合概率，不模拟正式训练效果。

00_verify_archive.py保留了最初执行时“审查根目录的上一级/upload/review(6).zip”的路径约定。本包把原ZIP放在input/review_original.zip。若要重放最初脚本，应在新工作目录按其原布局安置同一ZIP；不要为了路径方便改写已保留的历史记录。source与原ZIP逐文件的独立一致性结果已在archive_inventory.json及来源清单中保存。

RUN_INDEX中的python -c命令，其完整参考代码就在argv字符串中。stdout为空或stderr为空的文件仍予保留；空文件不是缺件。

四、打包边界

build_review_bundle.py是本次实际运行的打包程序，负责复制原件、生成来源和运行索引、制作清单及检验ZIP的CRC与逐项内容。它不是模型测试。
为避免自指，EVIDENCE_MANIFEST不收录自身的哈希，最终ZIP的哈希在打包完成后另行产生，不能递归写入同一个ZIP。各项科学参考运行的完成状态、stdout/stderr已在打包前冻结并完整收入。

包内文件比原128MiB授权更小；但文件大小、哈希或程序数量均不替代主审的实现与科学判断。
