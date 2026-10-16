# H完成结果外部审查产物

首先阅读`report.zh.md`（`report.zh.txt`内容相同）。`findings.json`提供分级发现，`reading_scope.json`逐文件给出实际阅读范围与输入SHA，`evidence/evidence_index.json`将正文E01—E10定位到原包路径/字段。

`checks/audit_saved_v1.py`是独立保存证据核查，`checks/run01/`有1650行重算、注册抽样索引、九诊断、126曲线均值和数值差。输入是用户已提供的原ZIP安全解包目录，不是项目正式数据。脚本不导入项目模块或加载模型/Memory。

补充summary核对首版`checks/reconcile_summary_v1.py`曾因审查者误写文件名退出1；失败源码、stdout/stderr/result均保留在run02。`reconcile_summary_v2.py`改为读取collected.json的准确路径，run03成功；两版本差异见patch。不要运行错误v1作为推荐入口。

可复现成功核对的命令（需Python、NumPy，输出应指向新的空目录）：

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python checks/audit_saved_v1.py --input /path/to/extracted_input --output /path/to/new_output/checks/run01
```

补充v2需要审查目录有上述run01输出与`evidence/`目录：

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python checks/reconcile_summary_v2.py --input /path/to/extracted_input --review /path/to/new_output --archive /path/to/original_review_input.zip
```

主脚本固定使用已登记PCG64(20260930)5000×2×20，不重新挑统计方案。不要在用户项目原生环境中把此脚本当成新的资格任务执行，也不据此重新计费/清零项目账。

本ZIP不重复包含用户原输入ZIP或任何模型/Memory/标签载荷。内部payload_manifest.json列出其余载荷；外部members.tsv列出ZIP全部成员（包括内部manifest）大小/SHA，避免自引用哈希。交付JSON提供ZIP字节数/SHA。
