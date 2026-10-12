# C_cf 定向外审返回包

请先读01_REVIEW.zh.txt。该文件为完整中文正文；02_FINDINGS.json列出2项复现缺陷、3项非阻断建议及3类越界扩展，科学阻断为0。主执行者应独立复核，不将本报告或参考自述当作最终资格。

输入身份：review_input.zip；网页名review_input(20261012-061828).zip；570664字节、64成员；SHA256 f5e48cf6d4bd06dd2eb546a171b41ebc34fe2d5a40651c86fdbd72466242a477。逐字节原包副本位于input/review_input.zip，没有修改项目源码或原始回执。

阅读顺序：
1. 01_REVIEW.zh.txt：结论、八项问题的定向分析、F01—F12处置、finding、已有CPU证明范围及最小推进条件。
2. 02_FINDINGS.json：机器可读的分类、文件／函数／行号、触发、后果和最小处理。
3. 03_READING_SCOPE.zh.txt／.json：逐成员阅读范围及主审／静态分工的实际边界。
4. evidence/04_INPUT_IDENTITY.json、CPU_EVIDENCE_AUDIT.json、CPU_ATTEMPT01_TO_CURRENT.diff：输入、63载荷、29来源闭包、两轮冻结来源及精确累计时间。
5. 05_REFERENCE_SCOPE.zh.txt和reference/：本次官方实现引用与网页独立参考。

重要状态：原生GPU没有执行，正式模型效果没有产生。R01按默认PyTorch语义为具体恢复夹具缺陷，网页只运行语义模拟；另附真实Torch小探针未执行。R02运行的是原Budget类在小文件／受控资源探针下的计数检查。网页不是项目Linuxpy310。

复核命令（先把本返回ZIP解压到一个独立目录，从该目录运行；输出写到另一个新目录，保留随包原件）：

    python tools/audit_input.py --input-zip input/review_input.zip --output reproduced_identity
    python reference/reference.py --input-zip input/review_input.zip --inputs reference/inputs.json --output reproduced_reference

上述两项仅需标准库。reference/torch_restore_probe.py则需要实际PyTorch，本网页未运行；主执行者可将相关窄断言纳入既有手写CPU套件，而不必另跑该独立文件。若使用探针，其stdout也应保存，不能把本包NOT_EXECUTED标记改成无现场证据的PASS。

审查方机械核对脚本曾因tar中有四个正常目录项而触发自己的断言；修复后核对通过，evidence_audit.log保留两次原始输出。这不影响原ZIP身份，也不属于项目资格失败。

RETURN_MANIFEST.json列出除自身外全部返回成员的大小／SHA256。该文件不循环计算自身或最终ZIP的哈希；最终ZIP的总大小、总成员数、未压缩总字节和SHA256在交付消息中列明。原输入ZIP的内部64成员另在04_INPUT_IDENTITY.json完整列出。
