# S_pos/H_pos静态外审返回包

先读`review.zh.md`。完整逐条意见为`findings.json`，实际逐文件阅读范围为`reading_scope.zh.md/json`。

`originals/`保留用户三个原附件，字节未改；`input/`为原ZIP的原字节解包副本。路径证据中的proposal/context均相对原ZIP根；报告中另外标明的checks/scripts/logs相对本返回包。

`checks/input_identity.json`是初始核对，`checks/input_identity_final.json`是收尾只读核对。它们证明实收身份和解包字节对应，不证明项目数据/结果或服务器现场。`request_context.zh.txt`只是本次用户消息的审查者范围记录，不能当项目回执。

`checks/handwritten_inputs.json`是全明示的手写分数输入；`scripts/handwritten_math.py`是实际运行源码，`checks/handwritten_pre_execution_identity.json`为执行前身份与固定容差来源；结果、stdout/stderr、环境及内外计时分别在`checks/handwritten_results.json`、`logs/handwritten_math.*`、`checks/handwritten_execution.json`。本次数值脚本一次exit0，没有修改/重跑或项目模型/优化器调用。四种错误公式被预期拒绝，不是失败实验。

数学脚本可在本目录解包后使用Python和NumPy运行：

```sh
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python scripts/handwritten_math.py
```

它只检查这里的手写分数，不调用项目代码，也不是项目py310原生资格。重新运行会写入新的手写结果，应先另存本包原件以保留本次实际输出。

原`verify_input.py`保留的是本次实际执行的绝对路径源码，不承诺不改路径即可在其他机器运行。它的输入是本会话原挂载的三个附件；原始命令与范围见`execution_ledger.json`。最终打包脚本也保留实际源码，打包是文档/文件操作，不是新研究实验。

`external_sources.zh.md`给出5份一手HTML的具体段落。它不是完整网页原始档案，不声称读完整论文或复现文献基准。报告所有正式结果只继承附带H_cons结果外审及主审，不重复重采样。

包内`payload_manifest.json`覆盖除它自身外的全部成员，记录字节数/SHA256。ZIP外另提供覆盖全部成员（含该manifest）的TSV及整包身份JSON，避免自引用哈希。没有字体、PDF、权重、正式文本/标签、Memory、owners或private载荷。
