# 审查交付说明

主原文：FUNCTION_MEMORY_RESULT.external_review.zh.txt，含完整中文裁决、证据边界、五条件、八端点全部22指标和标量附表。
original_input.zip 是用户提交的 review_input.zip 原字节副本；完整嵌套历史仍在其中，未修改。
audit/ 内是本次五份实际独立脚本、真实stdout/stderr、机器输出和阅读范围。旧历史脚本未在本轮重新执行。
manifest.json 登记本包除manifest自身外的每个文件字节数/SHA，既有子清单按原字节保留。

## 复算

将本包和original_input.zip分别解压。以下命令只读取已保存记录、分数、群指标和群计数；不导入项目模块，不安装环境，不运行execute/native，不拟合或载入模型，不读正文/逐对标签/Memory。
使用已有Python与NumPy，先创建一个新的输出目录。将INPUT_ROOT改为original_input.zip的解压根，OUTPUT_ROOT改为新的结果目录。

python audit/provenance/independent_provenance_audit.py --input INPUT_ROOT --out OUTPUT_ROOT/provenance
python audit/provenance/additional_record_checks.py --input INPUT_ROOT --out OUTPUT_ROOT/provenance
python audit/metrics/independent_metrics_audit.py --input-root INPUT_ROOT --output-dir OUTPUT_ROOT/metrics
python audit/mechanism/check_mechanism_logs.py --root INPUT_ROOT --output OUTPUT_ROOT/mechanism
python audit/root_counts/audit_saved_counts.py INPUT_ROOT OUTPUT_ROOT/root_counts/result.json

前一条provenance脚本创建输出目录，后一条补充脚本沿用该目录。不要使用python -O，以免关闭审查断言。
运行时间、环境字符串和阅读清单中的绝对路径随复算环境变化；应核对数值、固定来源、通过状态及记录范围，不要求整个输出JSON的字节恒等。
本轮数值容差1e-12；原实际输出最大统计差4.44e-16，分类/仿射/共享矩阵差0。

审查者在读取标题/路径时的纠正在audit/review_scope.json如实说明；没有改动项目来源，也不计为正式训练失败。
