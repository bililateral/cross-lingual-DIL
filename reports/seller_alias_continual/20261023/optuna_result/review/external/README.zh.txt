Optuna 结果独立科研外审交付说明
日期：2026-10-23（UTC+08:00）

裁决：本批结果可按“有效完成、未达到预定双域六格主目标”接收。
未发现需要追加训练、重新开放正式标签或重开已关闭资格才能结案的新缺陷。
S_pos为4/6格正，H_pos为5/6格正，均未通过预定六格门。
H_pos旧A最终排序有条件优势；新B和主MAP遗忘改善尚未确定。

一、阅读顺序
1. review_report.zh.txt：完整中文主报告，含方法、独立裁决、证据索引与边界。
2. findings.json／findings.csv：14条结构化发现；明确区分科学主张限制、已核对事实、轻微文字勘误和非阻断边界。
3. review_summary.json：精确数值、六格结果、选择、预算、核对数量。
4. scope/actual_read_scope.csv：7679成员逐文件大小/SHA及实际阅读/程序处理记录。
5. evidence/：执行、搜索、指标、统计的核对输出；reviews/：三个分项报告及原结构化发现。
6. scripts/：本次外审自写的四个只读复算程序及交付文件校验程序。

二、输入身份
实收ZIP：optuna_result_review.zip，103538522字节。
SHA256：be13255ad50a0fe814f58a150574f8403181cf3a9e7aa0af37293834188baac6
7678清单载荷全部匹配；清单自身另计一个ZIP成员。
输入原ZIP未重复装入本交付包；evidence/submitted_request.zh.txt是其根请求的精确副本。
报告E编号和 result/、context/ 等路径指输入包；evidence/、scope/、reviews/ 指本交付。
不能从哈希通过推导逐字阅读全文或独立重建标签级指标。

三、实际处理范围
直接阅读涉及81个不同文件；机器内容核对涉及4878个不同文件，两者重叠。
按最强处理层级互斥分类：{"direct_targeted_or_fields": 51, "direct_full_text": 30, "identity_only": 2752, "machine_content_check": 4846}。
所有7678载荷另有统一身份核对。阅读记录中human表示直接文本审阅，与程序处理区分，不表示另有人类审稿。
历史六份raw副本、底层正式标签和完整诊断bootstrap不在本次重新核算范围；见报告第二、四、七节。

四、复核方法
使用Python 3.12.14、NumPy 2.3.5；其余依赖为标准库。
以下参数 INPUT_ROOT 是原输入ZIP已经安全解压的目录；OUTPUT_DIR 应是新建空目录。
这些命令只读已开放保存证据，不导入原项目模块，不运行训练/推理，不连接项目服务器：

python scripts/audit_execution.py INPUT_ROOT OUTPUT_DIR/execution
python scripts/audit_search.py INPUT_ROOT OUTPUT_DIR/search
python scripts/independent_metrics_check.py INPUT_ROOT OUTPUT_DIR/metrics
python scripts/audit_statistics.py INPUT_ROOT OUTPUT_DIR/statistics

自写程序输出elapsed_seconds属于本次审查工作区，不能添加到原项目累计科研账。
scope/的直接阅读清单是本次审查记录；复跑程序只会再生其机器阅读清单。
本交付并不要求用户复跑；程序用于必要时核查已有数值。

五、交付完整性
file_list.txt列出全部文件路径与用途分组。
file_manifest.json记录除清单自身以外每一文件的字节数与SHA256，避免自引用哈希。
解压后可运行：python scripts/verify_delivery.py .
也可运行：python scripts/verify_delivery.py optuna_result_external_review_20261023.zip
交付ZIP整体SHA256在交付消息中给出。原始输入材料均未被改写。
