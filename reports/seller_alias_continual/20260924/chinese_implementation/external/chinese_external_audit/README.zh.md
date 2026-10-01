# 本次外审执行证据

本目录来自网页外审的独立执行环境，不是项目Linux原生BGE报告。所有执行均为手工CPU用例；没有正式数据/标签读取、真实BGE加载或GPU计算。

- provided_contracts.log：附件原12合同复跑，12通过、0跳过。
- independent_checks.py / .log / independent_results.json：额外9组独立检查，含成功复现附件R1缺陷的一项断言测试。不是9项正式实验。
- check_suggested_fix.py / suggested_fix_checks.log / suggested_fix_results.json：隔离建议函数验证。建议尚未合并生产源码。
- suggested_evaluation_fix.diff：R1/R2最小修改建议。R3未在此补丁实施。
- reviewer_proposal_first_attempt.log：审查者首版辅助函数schema错误的保留失败记录，不是附件缺陷。
- environment_and_final_integrity.json、all_50_payloads_verified.json、attachment_inventory_check.json：环境、入口、语法及逐文件哈希检查。
- review_reply.txt、native_review_reply.txt：为审读而从用户包内既有JSON提取的旧审查正文，不是本次新审查结论。
- official_reference.json：唯一外部模型用法核对来源，不作任务性能证据。

脚本默认输入解压根目录为 /mnt/data/chinese_review，输出本证据目录为 /mnt/data/chinese_external_audit；在其他机器复核时需调整脚本顶部ROOT/OUT。只对用户提供压缩包和手工夹具执行；不要将其指向正式监督数据。独立检查使用现有torch/numpy/sklearn，不使用假冒这些库的替代包。

full report: chinese_external_audit_report.zh.md（位于包根）。
