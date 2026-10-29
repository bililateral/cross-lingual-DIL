VendorLink 三市场数据接线与限定原生资格方案定向外审
2026-10-29，UTC+08:00

请先读 report.zh.txt。数据实现本轮通过，无新增科学阻断或复现缺陷；F01—F04保持关闭。GPU资格提议的2100累计任务秒/24更新仅是待确认上限，尚无该资格入口实现或执行结果。本报告不授权正式导出、GPU或正式训练。

唯一新输入：vendorlink_data_native_plan_review.zip
106444字节，29成员（28载荷+MANIFEST）
SHA256 c7f59348077e7163f6ace756eaebfbbe388edf2375a4ec8c5a6c467c02247949

主要附件：
- package_and_arithmetic.json：输入及嵌套旧源身份、逐项MANIFEST校验、cpu05/cpu06来源桥和聚合算术。
- input_manifest.tsv：全部28载荷与MANIFEST的实际大小/SHA列表。
- source_adapter_checks.json：本网页手写CSV/tar/ZIP来源适配核对。
- semantics_checks.json：本网页新旧builder差分、共享helper和计数核对。
- source_diffs.txt：旧data/tests至当前版本的完整差分。
- disposition.json：本轮裁决及合理后续事项。
- reading_scope.tsv：本轮阅读及复用范围。
- OUTPUT_MANIFEST.json：除清单自身外全部输出载荷的大小/SHA。

检查脚本仅使用Python标准库；没有Torch、Transformers、模型或训练。实际审查环境为网页隔离Python3.12.14/Linux6.18.55，不是项目Linux py310。脚本不会读取source_config内指向的真实来源路径，不调用prepare.main。来源检查会在临时目录创建并删除少量手写tar/ZIP；语义检查只处理手写账号文本与手写关系。

如要独立复核，可在普通隔离Python环境执行（替换本地输入路径；不需要项目服务器）：

python3 -B checks/verify_package_and_arithmetic.py /path/vendorlink_data_native_plan_review.zip package_recheck.json
python3 -B checks/source_adapter_microcheck.py /path/extracted_input source_recheck.json
python3 -B checks/semantics_checks.py --input /path/extracted_input --output semantics_recheck.json

后两条命令的extracted_input须为经第一条身份校验后的本轮ZIP解压目录。脚本读取随包源码且只导入已审的标准库data/prepare模块；勿将它用于来源不明的替换代码。

脚本复现的是本报告范围内的字节、静态、手写数据和聚合算术证据。它不会重新盘点正式原数据、重跑cpu06、验证项目GPU、获取正式qrels或授予执行许可。
