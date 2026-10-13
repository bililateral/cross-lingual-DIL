# C_q针对性实现外审返回包

结论：**当前字节仍需修复后才可进入原生资格；当前不可正式运行。** 原生模型身份绑定为具体阻断；Q−S手写实际更新证据为必要最小补正。没有新效果裁决。

`review.zh.md`为完整中文报告；`review.zh.txt`与它同字节。`minimal_edits.zh.md`是未应用的最小修复建议；`findings.json`保存结构化发现。代码定位均对应`reference_input/`的原输入路径和行号。

`audit/reading_scope.zh.md`及JSON列80个输入文件已读／未读范围；`read_events.jsonl`为实际阅读范围记录，不是全部工具转录。`input_integrity.json`与`evidence_chain.json`保存输入及29来源、额外授权文件和CPU回传身份核对。旧静态材料和旧GPU记录只按报告所述范围复用，未全部重审或执行。

`repro/`保留本次外审自编源码、人工输入、枚举、双角色梯度、轻量更新参数／梯度数组、实际stdout/stderr/退出码。首轮7项通过、1项因审查者变量遮蔽失败；仅该项修正后单独重跑通过，原始失败、源码和diff完整保留。另一个机械归档成员假设错误在audit内单独记录。均不属于项目Linux CPU/GPU失败。

`reference_input/`为当前输入80成员的原字节副本，含29项冻结源码及包内已提供的授权、CPU和静态背景。没有新取得的项目权重、Memory、正式文本或标签；repro中的npz只是现场构造的小网络数组。

## 复现外审小算例

以下只运行审查者小算例；**不要执行reference_input里的正式或原生入口**。使用具备NumPy、PyTorch的独立环境和新的输出目录。原实际环境、命令边界及退出码在`repro/run_manifest.json`；本次使用Python3.13.5、torch2.10.0+cpu、NumPy2.3.5，不是项目py310。

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
CUBLAS_WORKSPACE_CONFIG=:4096:8 TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1 \
python repro/independent_checks.py --source-root reference_input --out external_new_run
```

原run02实际只执行受影响项，等价参数为：

```bash
python repro/independent_checks.py --source-root reference_input \
  --out external_new_R_run --only finite_population_R_and_symmetric_two_roles
```

这些命令是复現说明，不是本包已经额外执行了全套第二次检查，也不是项目训练／原生许可。

## 文件身份

根`payload_manifest.json`与`FILE_SIZES_SHA256.tsv`列全部返回载荷的字节和SHA256，但两份根清单互不列入、均不自哈希；ZIP内的旧参考清单仍作为普通载荷列入。ZIP自身SHA在单独交付的`.zip.sha256`文件中，避免递归哈希。报告、阅读范围和算例失败记录都属于被列载荷。

请求模型：GPT 6 Pro；界面选择器未独立取得；审查实例自述GPT-6 Astra Pro；后台路由不能独立确认。没有委派、项目Linux连接、正式数据读取、BGE载入、外发或新训练。返回包不是授权／原生PASS／正式gate。
