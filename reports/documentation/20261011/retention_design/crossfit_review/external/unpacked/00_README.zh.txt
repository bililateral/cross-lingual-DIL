折外支持的历史排序间隔保持：静态科研外审回收包
日期：2026-10-12

结论：建议继续到“实现前合同／资格准备”。

值得检验的有限贡献：在完整 S 保持主要对照、六群 Memory 和主训练过程不变的条件下，用出生域内完整群折外来源与全域来源共同支持的单边正间隔下限，整体替代候选自己的 D0，检验旧域未见账号排序收益、新域学习及识别／校准和额外来源算力的代价；固定 C_perm 只提供一次可行赋值对照。

这不是已成立的独立原创方法或有效性结论。本轮没有新候选实现、项目原生核验、正式数据／标签、模型、Memory 载荷或 GPU 运行。只有既有代码静态阅读、公开一手方法阅读及单独的手写有限分数参考。网页工作区参考不构成项目 Linux py310 资格。主执行者应完整回收并独立核查；外审标签不自动成为最终裁决、数据／执行／上传许可或预算授权。

建议阅读顺序：
1. 01_REVIEW.zh.txt：完整中文报告，逐项回答八个审查问题；先读第一、十、十一节可掌握结论和待办。
2. 02_FINDINGS.json：12 项机器可读 finding；逐项给出位置、证据、严重性、影响及最小处置。
3. 03_READING_SCOPE.zh.txt 与 04_INPUT_IDENTITY.json：实际阅读者／范围、输入清单、实测大小和 SHA256。
4. 05_REFERENCES.zh.txt 与 06_SEARCH_LOG.json：五篇一手来源、直接方法阅读范围、检索和获取局限。
5. math/：独立参考的全部源、输入、结果、日志和执行元数据。

成员清单（共 14 个文件；没有显式目录成员）：
00_README.zh.txt
01_REVIEW.zh.txt
02_FINDINGS.json
03_READING_SCOPE.zh.txt
04_INPUT_IDENTITY.json
05_REFERENCES.zh.txt
06_SEARCH_LOG.json
input/review_input.zip
math/execution.json
math/inputs.json
math/reference.py
math/results.json
math/run.log
return_manifest.json

return_manifest.json 给出其他 13 个文件的字节数和 SHA256。清单明确不自含自身摘要，以避免循环定义；交付页面给出的外层 ZIP SHA256 覆盖该清单及所有成员。先验证外层摘要，再校验各成员。

输入原件：
input/review_input.zip 是原上传 review_input(20261012-040213).zip 的字节一致副本，未修改、未添加新项目实现。
157150 字节；22 个成员（21 项载荷和 manifest.json）。
SHA256：7f3917c75e166e0e7ebac2d30c0bb788dc261147d1cd2d4065d70095db4dce94
21 项载荷实际大小／SHA256 与内部清单一致；无意外成员、重名成员或 CRC 错误。
用户本条未给独立预期外层摘要，上述身份为实测，不冒称与另一外层预期值匹配。

独立参考的重查：
实际环境为 Python 3.12.14、NumPy 2.3.5；准确执行命令、一次执行时间及退出码见 math/execution.json。
在具备相同依赖的非项目工作目录解包后，可将结果写到另一个临时路径，避免覆盖回收证据，例如：
python3 math/reference.py --input math/inputs.json --output /tmp/crossfit_reference_check.json
reference.py 不导入本项目模块、不加载模型、不训练、不读取正式数据；其手写标签拓扑不是正式标签。
results.json 中的 PASS_FINITE_REFERENCE_ONLY 仅指这次有限参考断言通过，不是原生资格或科研效果 PASS。

公开论文只提供精确网址、方法位置和短述，没有附全文或把未读附录算入实际阅读范围。
