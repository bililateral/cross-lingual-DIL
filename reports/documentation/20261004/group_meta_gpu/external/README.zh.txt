完整二阶群隔离元更新：GPU资源准入增量外审交付

先读main_report.zh.txt；机器裁决为verdict.json。
核心结论：本次增量审查已完成，当前原生GPU资源准入未通过。
Tiny一致性不等于BGE通过；原生资源失败不等于算法效果失败。

目录
- evidence/received/review7_original.zip：收到的唯一审查包精确副本，输入未修改。
- evidence/supplied/：本轮三次尝试的重要原始记录，offload未产生result.json，故没有该文件。
- evidence/independent/receipt_run/：独立身份/来源/时间/预算/AST复算及实际执行输出。
- evidence/independent/hvp/：无Torch的NumPy多块梯度/Hessian独立计算。
- evidence/independent/gpu_semantics/：真实静态比较与Decimal语义反例。
- evidence/independent/doc_patch_dry_run/：文案补丁dry-run，原合同未应用改动。
- proposed/clarify_tiny_mutant.patch：仅更正比较对象的文案建议。
- proposed/staged_exact_hvp_suggestion.py：v2工程建议；未运行Torch/BGE，未实现formal runner。
- proposal_history/：v1原件、交叉审查和v1→v2修正；缺陷属于审查建议稿，非上传代码。
- reviews/：分项审查记录，主报告统一裁决；不要把分项建议当作额外训练授权。
- sources.json：主审实际访问的官方/原始公开资料及使用范围。
- EVIDENCE_MANIFEST.json：全部其余文件的大小/SHA，不自哈希。

可复算的独立检查（在解压后的本目录运行）

python evidence/independent/verify_review_package.py evidence/received/review7_original.zip rerun_receipt

上述脚本只需Python标准库及Bash，不导入Torch，不打开正式数据。

NumPy数学脚本可在一份复制目录运行以保留本次原始输出：

python evidence/independent/hvp/exact_hvp_numpy_check.py
python evidence/independent/hvp/exact_hvp_nonlinear_numpy_check.py

它们会在自身目录写相应result.json；若要保留本包原始输出，先复制该目录再运行。两个脚本仅需NumPy。七维、28人工账号和378关系仅是数学夹具，不是七参数BGE，也没有复现全部生产排序损失。

禁止把proposed文件直接描述成生产已跑通。
建议接入仍要核对完整未裁剪梯度、None/零掩码、所有模型/Adam状态、角色随机流、buffer及连续第二次原生更新；整个方案在现有预算内能否执行仍未知。

本交付不请求或使用正式数据/私有身份/外部凭据，不启动训练，不要求先做四条件机制对照。既有失败、建议首稿及修正均保留，输入没有被覆盖。
