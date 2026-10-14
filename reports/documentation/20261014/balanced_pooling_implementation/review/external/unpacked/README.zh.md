# S／G／F 合并审查返回包

先读 `review/report.zh.md`。完整中文审查结论为：核心数值与训练/评分接线有继续推进依据，但当前仍有1项科学阻断和2项必要复现缺陷；不是BGE原生通过、正式可运行或性能成功。

`review/findings.json` 是三项发现的结构化版本；`review/reading_scope.csv` 和 `.json` 逐项记录输入69个成员的实际阅读层级、范围、字节和SHA256。原输入按字节保存在 `authorized_input/review_input.zip`，不是新来源或改写副本。

`evidence/independent_checks.py` 与同名JSON/stdout/stderr/exit_code保存本轮网页容器实际手写核对。`evidence/na_pipeline/` 的56个文件全部来自手写数据调用受审 collect/finalize 的真实输出，不含正式标签，不是项目实验结果。矩阵/源码引用的输入根为解包后的原授权包。

`evidence/audit_input.py`、`verify_source_identity.py` 和对应结果记录ZIP、68载荷、29科研来源、两次CPU和22接收文件的机械核对。`metadata/executed_commands.txt` 为实际命令记录，`metadata/failures.zh.md` 区分预期拒绝、检查辅助错误和项目证据；不得把网页环境计入项目py310资格。

包内 `manifest.json` 列出除其自身外所有返回成员的路径/字节/SHA256，以避免自指哈希；ZIP旁的完整成员清单另包括 manifest.json 本身。ZIP外摘要给出压缩包整体字节、成员数与SHA256。清单只验证字节一致，不证明模型后台路由或项目服务器实际执行。

受审源码没有修复或替换。主执行者应独立处置最小修复；新字节须与相应受影响核验证据/来源绑定，不能把本次旧PASS自动移植。这里不新增预算、权限、种子、方法或重试。
