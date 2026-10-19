审查证据包说明

先读 report.zh.txt；结构化结论在 findings.json；独立算术在 budget_arithmetic.txt 与 independent_arithmetic.json。

本包不含项目权重、Memory载荷、正式文本/标签或新科研结果，不是执行门或启动许可文件。
原始输入需使用用户此次冻结的 optuna_budget_review.zip（377084字节，SHA256 58c4e04cc7f397fda0909331f6e6eabd9b8e63090cfa6b3a1efcbdd7b786d1b5）。

static_audit.py仅使用Python标准库。若自行复核，输入是已经解包的原附件目录，输出是另一个审查目录：
python static_audit.py EXTRACTED_INPUT OUTPUT_DIRECTORY
它不会导入项目、运行测试、读取附件之外的模型/正式数据或连接网络。

input_integrity.json：71载荷大小/SHA核对。
archive_identity.json：原ZIP身份。
source_binding_checks.json：原native48来源及calculations16来源。
ast_static_checks.json：36份Python纯语法解析。
static_check_summary.json：本轮静态机械结果及未认证范围。
reading_scope.json：逐文件实际阅读/机械/未重审范围。
execution.json：实际操作和失败索引，非穷尽原始工具逐字转录。
review_context.json：用户最新授权的审查上下文，不是项目execution_authorization。
evidence/review_notes：必要并行静态笔记与真实辅助错误可见记录；综合主报告已处理其结论。

manifest.json列出本包每项载荷的大小和SHA256，清单本身不自引用；ZIP整体身份另附返回。
