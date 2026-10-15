完整S/G/F结果外审交付

入口：review.zh.md（完整中文审查）。findings.json为分类发现；reading_scope.tsv/json逐一列376个输入成员及实际处理范围。
本包不是新项目实现、资格、训练或正式标签再评价；不替代主执行者责任。

唯一原始输入（本返回包不重复打包输入ZIP）：
review_input(20261015-121313).zip
6123032字节，376成员
SHA256 07f941b2d689c4f9e2cd024364b57e3607bd9bb762868378a854b5be97ef83f6

主要隔离核算：
code/independent_review.py
raw/attempt02.log
raw/attempt02/audit_output.json
raw/attempt02/recomputed_statistics.csv：全部1980行，146行N/A，保留原算式。
raw/attempt02/all_diagnostic_curves.csv：126行，每行MAP、Recall@5、log_loss，共36次诊断的valid/fit/cache分解。
raw/attempt02/diagnostic_recalculation.json：全部严格标签的输入变化、区间及D0/D1。
raw/attempt02/classification_and_calibration.json：预测正例缺失、pooled分类和六映射。
raw/attempt02/loss_forgetting_direction_addendum.csv：18行确定性符号解释；不覆盖原结果、不做新统计检验。
raw/attempt02/input_members.tsv：输入376成员完整字节清单。
raw/attempt02/array_inventory.json、programmatic_read_scope.json：数值数组与代码读取用途。

保留的外部首轮失败：
code/independent_review_attempt01.py、raw/attempt01.log、raw/attempt01/。
失败是审查脚本误将S的MSE1日志也要求null；修正后，输入不变，项目来源与阈值未改。
不能将该失败记为项目Linux、模型或数值负结果；见完整报告第十一节。

其他保存证据核对：
code/supplemental_evidence.py及raw/supplemental_evidence.json/.log：只读资格摘要、旧失败定位及progress/manifest。
code/read_evidence.py及raw/manual_reads.log/.jsonl：实际提取的文本与行号；不是全项目语义重审。
code/build_reading_scope.py、raw/context_gate_bindings.json及raw/build_reading_scope.log：逐文件范围、11个在包引用及6个限定继承引用。
model_identity.json、isolation_run_notes.zh.txt：身份与环境边界。

隔离核算复现命令（需要Python 3和NumPy；不需要Torch/模型/标签/网络）：
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python code/independent_review.py --archive '/你持有的原始ZIP路径/review_input(20261015-121313).zip' --output replay_output
python code/supplemental_evidence.py --archive '/你持有的原始ZIP路径/review_input(20261015-121313).zip' --output replay_output/supplemental_evidence.json
脚本严格检查原ZIP大小与SHA；不把任意替代输入当成本次结果。

文档辅助脚本read_evidence.py和build_reading_scope.py按生成时目录约定使用同级../input/解包目录；这些不是核心统计复现的前提。

清单说明：
payload_manifest.tsv记录本返回包除该清单自身以外每个成员的大小/SHA；无法自包含自身SHA，故明确排除自身。
交付ZIP旁的 sgf_result_external_review_20261015_members.tsv 包含ZIP所有成员（也包括payload_manifest.tsv）的大小/SHA。
交付身份 sgf_result_external_review_20261015_identity.json 给出ZIP自身字节数、成员数、SHA和外部清单身份。
不包含原模型、Memory、正式文本/标签、凭据、输入ZIP副本或任何字体。
