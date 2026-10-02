# 本网页失败与修订账

以下均为审查者自己的工具或参考错误；没有因此修改收到的28科学来源、训练数据或结果。原包项目失败／重试证据原样保留，未清理。复算计时日志只对实际使用包装器的进程精确计时；未包装的探索调用不伪造时刻。

## F0 ZIP文字解析

`files.read`对上传file_00000000de5881f59f9a13e4dbd7f6f5返回：`Indexed retrieval failed and direct file parsing also failed (HTTP 400). This is not evidence that the file is empty.` 随后直接读取已挂载的原ZIP，不把解析失败当空包。intake记录CRC、成员和SHA。

## F1 手工参考v1（真正的参考运行失败）

原源码 `source/hand_objective_reference.py` 与另存v1逐字节相同；`logs/hand_reference_v1/`保留完整命令／起止／退出1／stdout／stderr。全部数学断言后，最后JSON序列化np.bool_时TypeError。v2仅将`clipped=norm>1`转为`clipped=bool(norm>1)`，使用新输出目录；数学式、断言、容差不变。v2通过。v3仅增加project/output/cpu CLI以供脱离本机路径复现，数学未变；再次通过。两份diff在results。

## F2 探索性J/stderr.log误查

首次探索性读取误假设正式job有独立stderr.log，FileNotFoundError。真实shell把stdout/stderr合并在train.log；项目保存结果审计另有audit_stderr.log，不应混淆。

首次调用未由run_logged包裹，精确起止未保存；现有`logs/inspection_missing_stderr_replay/`是标记清楚的后续重现，退出1及完整stderr保留，其时间不是首次调用时间。source/inspection_failure_probes.py保留重现源码。没有将该路径错误认定为包缺失。

## F3 探索性point.training键误查

point JSON不含training键，训练记录由manifest.training引用、文件位于run/updates。探索调用KeyError后改为读取实际已知文件。`logs/inspection_point_training_replay/`同样明确是重现运行，不回填初次探索时间。没有修改point补键。

## F4 补充审计v1（真正的参考运行失败）

`logs/supplemental_v1/`保留完整命令／起止／退出1／stdout／stderr。ID日程、项目审计核对和335个报告数值已完成；到额外实际域明细步骤时，误读并不存在的evaluation/group_ids.json。程序未完整通过，不以局部成功覆盖其退出1。

v2从原collected.json现有group_ids读取，并显式断言等于partition及reference群序、列序。保留v1及其已写出结果、v2和diff。没有改输入文件、原指标、容差或期待方向。v2完整退出0。

## F5 探索性打印SyntaxError

一次仅用于展示结果的Python内联代码末行误写`print('quarter vs seq brier'):None`，在解析阶段退出1，未开始读取结果，也无输出文件被修改。此调用不是独立数值参考；未记录精确起止。原命令源码及工具返回的完整错误文本分别保存于`inspection/exploratory_print_failed.py.txt`和`inspection/exploratory_print_failed.stderr.txt`，不伪造运行日志或修绿。

## 项目证据里的传输重试

原S/return_sync.json说明审计结果传输第一次SSH连接关闭，同一已保存归档随后复制成功；没有新增训练或label访问。该项目记录与网页参考失败分开，未据此声称重新连接服务器确认。
