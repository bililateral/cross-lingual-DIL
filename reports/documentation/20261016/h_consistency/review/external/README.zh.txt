H_cons 唯一静态草案外部审查产物
2026-10-16

入口：report.zh.md（完整中文正文）
裁决：支持只为固定H_cons形成新的受限实施合同；不构成合同批准、实现/CPU/GPU/正式数据/训练许可；不认定论文独立贡献已成立。

input/review_input.zip 是实际收到的原包字节副本，SHA和各载荷信息见 evidence/input_verification.json；本轮末尾只读复核证据见 evidence/reverify.stdout.json。
reading/reading_scope.zh.md、reading_scope.json和input_sources.tsv列明逐文件实际阅读范围、字节数与SHA。E编号在正文中引用这些原包成员，L为UTF-8文本行号。
review_findings.json为逐项分类裁决。evidence/identity_scope.json区分用户请求的选择、未观察的本次选择器、自述身份与未独立核实的后台路由。
evidence/public_sources.json列明定向核对的原始论文和实际方法段。未随包复制论文全文。

checks/make_inputs.py 生成完全手写的输入；checks/inputs.json 保留此次实际输入。
checks/math_checks.py 为独立数学核算源码，没有导入项目代码、模型或数据，没有optimizer或训练。
checks/results.json、gradient_evidence.npz保存标量、每视图查询梯度与共享K梯度。
checks/attempt01/保留实际命令、源码/输入SHA、耗时、退出码、stdout/stderr。两脚本exit均为0，stderr均为空。预期错误式的NaN和分支丢失是变异检测，不是生产错误或运行失败。
checks/run_review_checks.py 拒绝覆盖attempt01。复算应在单独副本中保留/移走旧attempt01，不覆盖本次原件。PyTorch/NumPy版本在results.json中固定记录，未安装任何新依赖。

本轮17项只是审查环境的手写数学检查。未执行项目head/encoder/optimizer、原生资格、GPU、正式数据/标签或Memory读取，未重算旧H正式数组。完整边界见报告第九、十节。

MANIFEST.json列出除自身外全部外层ZIP载荷的字节数与SHA；自身不能递归记录自己的SHA。外部交付的.members.tsv逐项列出全部外层ZIP成员（含MANIFEST.json），.receipt.json记录ZIP自身身份及核验结果。内嵌原输入ZIP的16成员另有其原manifest与reading/input_sources.tsv，未改字节。
