# 本次网页独立补测证据

审查日期：2026-09-25。
冻结输入：pooling_review(1).zip，541531字节，SHA-256 e01f3a7e440c8238ee35813a4f56fb997aa9f3d4fa17b6d48f29e54be06b26b7。

本目录不是项目Linux服务器原生BGE运行记录。环境是网页Linux/Python 3.13.5/PyTorch 2.10.0+cpu/NumPy 2.3.5/单CPU线程。

- integrity.json：附件及56份来源文件、10份CPU小型证据的实际复核。
- source_checks.json：冻结历史来源、分区日程复算及CPU原始证据摘要；实际大模型权重未检查。
- pooling_tests.log：附件11项商品汇总测试，本次全部通过。
- base_tests.log：附件12项直接继承基础测试，本次全部通过。
- reviewer_checks.py/.log/.json：本审查新增7项测试及原始输出、数值结果。
- contract_status_only.diff：CPU合同原字节与现行合同的状态文字差异。

新增测试仅用手写数据和小CPU模型。12份模型文件门测试中的模型文件是显式写有“NOT A REAL MODEL”的收据夹具，只验证实际校验入口行为；不宣称检查了12份真实模型。checkpoint测试调用实际生产函数，但对象是小模型，不是BGE。完整科研报告在ZIP外的pooling_review_report.zh.md中，并在证据包打包时附入。

本次已完成测试，无需用户运行。为记录可复核路径，脚本顶部ROOT指向本次冻结附件解压目录/mnt/data/pooling_review_extracted。复核者在别处运行时可将该一处路径改为同一附件解压根目录；脚本不修改提交源文件。原始执行命令为：

```text
python /mnt/data/pooling_reviewer_evidence/reviewer_checks.py
```

没有正式文本/标签/owners/test载荷、模型权重或私密资产。本次未启动正式训练；当前暂停保持。
