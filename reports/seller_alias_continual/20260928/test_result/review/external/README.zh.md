# 本次固定s0 test结果外审核验证据

完整结论见 test_result_external_review.zh.md。
原输入：test_result_review(1).zip，SHA de68ee8309ff6e9865a77f3fdc418b19a2eabe4cc5c896010d9a7a129733f138。

本包不含正式标签、正文或大模型。独立数值实现只依赖NumPy及Python标准库，单CPU运行。

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
 taskset -c 0 python -B scripts/independent_saved_audit_v1.py \
 --root /path/to/extracted/submitted/project --out /path/to/new/independent_output
```

不得把上述命令替换成项目正式execute。out必须是新目录。元数据脚本按本次实际沙箱读取，重放时用它支持的--root/--out/--reference；诊断脚本内保留当时原绝对路径便于追溯。日志中的命令是本次实际执行值，不是事后编造。

数值参考v1第一次通过；两个元数据参考失败、诊断和最终v3通过都保留。没有正式重算标签或重新拟合。

计时表述轻微问题仅需现行结果报告加注，不修改冻源/原值/机器false。
