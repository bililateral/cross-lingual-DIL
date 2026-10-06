# 函数目标累计关系记忆：本轮独立审查证据

入口报告：`FUNCTION_MEMORY_IMPLEMENTATION.external_review.zh.md`。

## 范围

仅为网页容器中的源码/保存证据核对、手写CPU测试、独立数学、符号指标与隔离故障例。未连接项目Linux、未载入BGE、未运行GPU、未读正式正文/逐对标签/权重/Memory、未签发正式gate或执行正式训练。项目CPU原始回执与网页输出分开放置，不能混作同一次执行。

`original_input.zip` 为本次授权原附件的精确副本，含完整历史链。`input/`提供本轮源码、合同、授权、CPU日志副本便于阅读；其中庞大的`input/history/discussion_audit.zip`没有重复装入外层证据ZIP，它在original_input.zip内，重放第一步会恢复。本次外层manifest量的是实际装包文件，不声称省略历史大文件的input目录本身独立完备。

## 重放

只在已经具备Python、NumPy、Torch CPU、psutil等现成依赖的环境使用。不安装依赖。建议在新目录解压整个证据ZIP，再运行：

```bash
python audit/scripts/replay_all.py
```

该脚本按顺序核对并解开原附件、核对历史/源码、运行所附七个手写CPU测试，再运行独立数学、两阶段纯统计容器、符号矩阵区间与五项判据检查。它不调用项目核验入口的main/native或正式execute。运行会在本地新建input/history/audit临时副本，并更新解压目录中的输出；外层原ZIP和其中原始输出不变。

不要把`run_supplied_cpu_tests.v1_order_sensitive.py`当作重放入口：它是审查者第一次列表顺序比较错误的原件。错误发生在测试开始前；修正及原stdout均已保留，说明见`audit/outputs/reviewer_corrections.zh.md`。

## 数值及时间

公式、运输、符号指标和隔离故障结果可独立复核；实际耗时/路径/平台细节可因重放环境变化。首轮网页测试时间与项目py310时间均在报告中按各自记录列明。新增重放记录另存，不用它覆盖项目证据。

所附源代码为原字节，不含审查者修复。故障例只在内存中构造人工gate对象或提取未改watchdog AST，并将退出器替换为记录器；没有创建批准文件，也没有实际杀进程。

## 清单

`manifest.json`列出每一实际载荷的相对路径、字节数和SHA-256，不列自身；它不能作为算法正确性的替代。审查结论见完整报告，最小必要问题的触发证据见`independent_checks.json`。
