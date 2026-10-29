VendorLink 新增原生实施定向外审交付说明

先读 report.zh.txt；机器可读裁决见 disposition.json。
本轮完成的是新增实施审查，发现 NI01—NI03 三项必修复现缺陷。
当前实施关闭门保持 false；该报告不是项目执行许可。

本附件独立对应输入：
vendorlink_native_implementation_review.zip
136445 字节；SHA256 0d2124d490d4048f635f6b15f2538a883a20d0c2995084364849b3d1bc9280ad
35 成员=34 载荷+MANIFEST.json。

核对材料：
- input_manifest_verification.tsv：输入逐文件清单，34个MATCH载荷；MANIFEST自身单独标识，不声称其自我哈希。
- package_checks.json：外层ZIP、嵌套cpu07、cpu08九来源、配置、预算算术和原件身份。
- cpu07_to_cpu08.diff.txt：旧export/tests到当前的准确差异。
- export_bridge_checks.json：4800条纯手写数据的真实builder/export及qrels接线检查。
- control_budget_checks.json：当前源码AST的假时钟/假进程及门表达式反例；出现这些预期缺口不表示检查器执行失败。
- reading_scope.tsv：本轮阅读和复用边界。
- OUTPUT_MANIFEST.json：本交付所有载荷的大小和SHA；不列自身。

可复核脚本仅使用 Python 标准库。
本次实际网页环境：Python 3.12.14 / Linux 6.18.55 x86_64 / glibc 2.39。
它不是项目服务器 Linux py310 原生资格，更不是 GPU 资格。
没有运行原 unittest、Torch/Transformers、模型、optimizer.step、训练或GPU，也没有通过本检查器启动真实资格worker或监督测试子进程；没有访问项目服务器/真实源/正式qrels/模型权重。
cpu07/08真实子进程和微型模型数值是包内原件，非网页重跑。

复核方式（在你自己的临时目录，将路径替换为实际本地位置）：

python3 -B checks/verify_package.py /path/vendorlink_native_implementation_review.zip /path/package_checks.new.json

第一条核对完成且确认路径安全后，将该输入解压到一个新的 INPUT_ROOT。后两条只读取已验证源码；不会调用真实来源或模型：

python3 -B checks/export_bridge_checks.py /path/INPUT_ROOT /path/export_bridge_checks.new.json
python3 -B checks/control_budget_microcheck.py /path/INPUT_ROOT /path/control_budget_checks.new.json

不要把这些脚本的通过改称项目原生GPU通过。环境字段、临时路径可随复核机器不同；核心身份、来源桥、作用断言与所列反例应保持。
export检查器最初的tuple/list表示比较已在本检查器修正并记入结果，没有修改项目源码。

本报告建议的最小修复只涉及现有有界监督、内部入口与active发布；不要求全面重验历史，不增加方法臂、HPO、更新次数或预算。任何新实验/数据/参数/超预算事项均待用户确认。
