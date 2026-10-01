# 本次方案外审证据

这里不是训练／实现测试包。没有读取正式标签、样本正文或模型，没有连接项目Linux，没有运行项目科研代码。

- `review.zh.md`：完整审查意见。
- `numbered_sources/`：来自本次21份原文件的行号副本；仅用于引用。读取范围见报告，不因复制文件就声称逐行审阅全部无关历史。
- `evidence/submission_verification.json`、`outputs/final_source_verification.json`：开始及结束文件核验。
- `evidence/audit_derivations.py`：Python标准库算术、组合概率及解析反例。不是新数据实验，不采样项目标签，不拟合模型。
- `evidence/derivations.stdout.txt`、`.stderr.txt`、`.exit_code.txt`：实际执行记录，首次退出0，无数值参考修订。
- `outputs/derivations.json`：预算、日程、随机保留的解析概率与校准／终点的逻辑例子。
- `evidence/finalize_evidence.py`及其日志：最终源文件检查与编号副本生成。
- `evidence/run_environment.json`：命令、环境、限制。没有测量算术执行耗时，不伪造开始／结束时间。
- `evidence/source_register.json`：本次原论文访问范围与失败记录；没有分发第三方论文全文。
- `package_inventory.json`和`SHA256SUMS.txt`：本包文件清单；二者互不自包含以避免循环散列，外部回执提供ZIP整体SHA。

正文中的例子全部标为理论构造，不是项目结果。所谓当前域完美校准的反例只做期望损失算术，没有随机生成正式数据或运行优化器。

Files的ZIP解析失败、两个论文PDF链接获取失败有记录；实际算术没有失败。工具前端曾显示终端环境信息，实际两个Python脚本的stderr文件均为空、退出码均为0；不把前端信息改写为科研脚本错误。
