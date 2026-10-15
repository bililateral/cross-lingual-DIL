F 终端 Q 外审返回包（2026-10-15）

入口：review.zh.md（与 review.zh.txt 完全相同）；verdict.json 与 findings.json 是结构化裁决，不是项目执行 gate。
reading_scope.json/tsv 逐文件区分实际原文阅读、结构字段核对、AST 和仅身份校验。input_inventory.tsv 包含收到 ZIP 的全部 66 成员；attempt01_archive_inventory.tsv 为首轮源包的 37 成员。

checks/ 保存实际执行的只读机械核对、差异、保存数组算术源码及输出。local_checks/ 保存两个既有 integration fixture 的隔离执行；attempt01 为预期 FAIL，current 为 PASS。脚本退出码只表示审查脚本完成，应读取 result.json/test_success 和 unittest.txt 判断被测测试结果。evidence/project_cpu/ 是原包两轮 CPU 小记录副本，必须与 local_checks 区分。

本包没有项目模型、正式文本/标签或 Memory。handwritten_scores.npz 仅含本实例生成的固定手写评分数组。source_bindings/ 只保存来源清单；生产源码仍使用用户已经持有且本包已验证身份的原输入 ZIP，不复制为新实现。

复现说明（只是记录实际用法，不是新的科研执行要求或项目 Linux 许可）：
1. 将原输入 ZIP 解压为一个 INPUT 目录，保持全部原路径；另将其中 BP/qualification/cpu/attempt01/source.zip 解压为 FIRST 目录。BP = reports/documentation/20261015/balanced_pooling_repair。
2. 静态核查在具备 Python、bash 和 patch 的隔离环境执行：
   python checks/mechanical_audit.py INPUT OUT_CHECKS
   python checks/reference_arithmetic.py INPUT OUT_ARITHMETIC.json
   第二项需要 NumPy，不调用生产求解器或模型。
3. 本次两次固定小测试的实际命令形态为（分别使用 INPUT 和 FIRST；脚本只选择一个已有测试，不运行资格 main 或 formal execute）：
   CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 python -B local_checks/fixed_fixture_check.py INPUT OUT_CURRENT
   同一命令将 INPUT 换为 FIRST、输出换为 OUT_FIRST，即首轮来源。
4. 本次环境为 Python 3.13.5、Torch 2.10.0+cpu、NumPy 2.3.5、Linux 6.18.44；并非项目 py310/Torch 2.9.1+cu130。依赖版本、源码大小/SHA、实际基础输入K、输出、时间和CPU线程数见各 result.json。实际固定输入由所钉住的源码生成，未保存每一张中间完整表，不声称拥有原正式失败K。

重要解释：fixed_fixture_check.py 的 exact_optimum_plan_L1_perturbation_bound 和 fixed_plan_value_suboptimality_bound 为精确熵最优计划在给定残差下的条件性公式，残差自身是普通浮点测量，不是区间证书；请结合正文第四节。teacher_A0 的 top5 为辅助观察，实际 teacher 不进入排名损失，不应据此判其失败。

manifest.json 列出返回包所有载荷（除清单自身）的大小和 SHA256；output_inventory.tsv 列出除自身与 manifest 之外的载荷。ZIP 自身身份写在外部同名 .sha256.txt，避免自指哈希。不把哈希校验、测试数、模型名称或本报告视作原生/正式执行许可。
