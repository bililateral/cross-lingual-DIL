G_half反对性静态审查返回包

主入口：review.zh.md（完整中文审查正文）。
结构化结论：findings.json。
逐文件实际阅读范围：reading_scope.json。
一手文献及实际阅读限制：references.json；引用正文也在review.zh.md末尾。
原始输入身份与逐成员哈希：input_identity.json。
input/review_input.zip是用户原始附件的逐字节副本，不是重打包。

本包仅支持静态审查结论。没有项目Linux连接、项目代码执行、候选实现、训练、GPU调用，
没有读取正式文本/标签或反序列化模型/Memory。运行环境是网页隔离Python3.13.5，
不是项目Linux py310资格。本轮自述GPT-6 Astra Pro；当前界面选择器和后台路由均未独立验证。

手写与保存字段核算：
  checks/check_static.py  实际运行的纯标准库源码。
  checks/inputs.json      实际手写输入、差分步长/容忍和原包预期身份。
  checks/run_logged.py    实际执行包装；保存真实stdout/stderr/退出码/环境。
  checks/output.json      结构化数值输出。
  checks/stdout.txt       原始实际标准输出。
  checks/stderr.txt       原始实际标准错误，成功时为空（如实保留）。
  checks/status.json      真实命令、起止时间、退出状态及源码/输入哈希。

便携重放命令（在另一个解包副本中，不覆盖本归档原件）：
  python checks/check_static.py --input-zip input/review_input.zip --toy-input checks/inputs.json --output replay_output.json
这只是同一手写算术和保存汇总字段的隔离重放，不提供或授权项目执行。

真实失败保留：
  logs/identity_attempt01/  首次ZIP核对在上下文已关闭后调用testzip而失败；源码/错误文本
                            从真实工具调用和响应转录，status明确说明，不冒称原始重定向。
  logs/identity_attempt02/  只修正调用位置后的实际成功检查，真实重定向输出和状态。
首次失败是审查者自己的bookkeeping问题，不是项目科学实现错误；未删除/掩盖。
checks/核算首次执行即成功；没有人为制造失败来充数。

清单规则：member_manifest.tsv列出除该清单自身之外的所有ZIP成员的路径、字节和SHA256。
清单自身的身份及整个ZIP身份在交付消息的完整逐成员清单中提供，避免自指哈希。
最终交付只提供一个ZIP；交付消息同时给出ZIP字节数、成员数、SHA256及完整成员表。
