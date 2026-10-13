# 独立手写核对的失败与修正

## run01

实际源码为`independent_checks_run01.py`（运行时文件名为independent_checks.py）。运行8项，7项通过，R检查末尾失败，进程退出1；见完整results.json、stdout、空stderr和exit_code。

异常为`AttributeError: 'int' object has no attribute 'detach'`。函数起初以a保存R张量，后面循环以a表示原账号，覆盖了该变量；在FP32与float64标量比较时访问了整数的detach。此前该项的部分中间比较完成，不把它们算作整项通过。这是外审脚本缺陷，不是受审实现失败。

## run02

保留run01源码后，仅在R检查中将目标张量命名为actual_R/literal_R，并加入单项执行开关。追加该项FP32梯度比较，diff见`run01_to_run02.diff`。只执行`finite_population_R_and_symmetric_two_roles`，未重跑已经成功的另外7项。退出0，数值／导数／有限差分及FP32比较全部达原容差。

stderr保留一次UserWarning：对requires_grad张量转日志标量。它在输出数值格式化处，不参与loss或反传，不影响前面断言。没有为了消除这条日志再执行。

两个results.json中的elapsed_seconds从测试准备后计时，不是含导入的进程完整墙钟，不是GPU计时，也不冲抵项目CPU累计预算。源码、输入、候选、枚举、梯度、参数数组和原始日志保留。新脚本复跑请使用新的输出目录。
