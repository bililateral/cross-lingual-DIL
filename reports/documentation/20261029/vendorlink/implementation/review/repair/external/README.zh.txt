VendorLink F01—F04 定向增量复核交付

先读 report.zh.txt；closure_status.json 给出四项关闭状态及证据范围。没有新实质缺陷或剩余必修项；正式运行仍需另行合同。

input_and_nested_verification.json：本轮17成员、16载荷与嵌套旧源身份，以及首包/首轮报告身份复核。
source_and_cpu05_checks.json：五份当前来源、33项未变函数/类、H优化器与Settings依赖检查、cpu05费用及手写端点复制微例。
source_diffs.txt：cpu01旧源→首轮以及首轮→修订的实际差异。
data_metric_checks.json：240条手写字符串、fit不变、blind/qrels/ID/全同分与部分同分接线核对。
reading_scope.tsv：本轮实际阅读/复用范围。
checks/中的两脚本只做哈希/AST/纯数据NumPy接口/手写字节复制，不加载Torch或模型、不跑训练或已有测试套件、无网络/项目服务器访问。手写字节文件仅使用.pt后缀，不是模型权重。

复核命令（只对已解压附件及独立输出目录）：
python -B checks/data_metric_checks.py BASE_DIR REPAIR_DIR
python -B checks/source_and_copy_checks.py BASE_DIR REPAIR_DIR OLD_DIR INPUT_ZIP OUTPUT_DIR
BASE_DIR为首包解压目录，REPAIR_DIR为本轮增量解压目录，OLD_DIR为本轮cpu_transfer.zip解压目录。第二脚本会写核对JSON与diff到独立输出目录。逐位核对首包/首轮报告的额外记录来自本会话已保存原件，不是脚本伪造的原生证据。

OUTPUT_MANIFEST.json列出本交付其余载荷的大小/SHA，不自引用自身hash。未重新打包用户原始输入、旧源码、任何模型或Memory。
