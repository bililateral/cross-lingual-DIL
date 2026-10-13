C_cf三臂固定A→B实验：2026-10-13针对性结果外审交付包

先读01_REVIEW.zh.txt。结论：接受有效完成与“未建立相对完整S的排序改进”的有限结论；本轮0项SCIENTIFIC_BLOCKER、0项尚未关闭的REPRODUCIBILITY_DEFECT，没有必须先修改的正式数值或重跑要求。

主要文件

01_REVIEW.zh.txt：完整中文审查正文、数学统计说明、证据等级、结果和逐项处置。
02_FINDINGS.json：结构化finding与最小处理；非阻断建议和超范围事项分列。
03_READING_SCOPE.csv／json：原附件303成员逐文件实际阅读／未读范围、大小及SHA。
04_FAILURE_LEDGER.json：项目及网页审查辅助脚本失败记录、处置和证据路径。
05_MAP_REVIEW.png：仅用独立复算的已保存结果绘制的主要差值和固定六点曲线。
06_PRIMARY_RESULTS.csv：原固定主要MAP绝对值、两项比较与旧域变化的独立重算表。
07_REFERENCE_SCOPE.zh.txt：独立代码、输入、输出及执行日志导航。
FILES_MANIFEST.json／csv：交付包逐文件大小和SHA；两份清单自身排除以避免自引用。

原始材料

input_archive.zip：实收4,285,123字节原ZIP的逐字节副本，SHA256 261b82d9b7d26b1e93c38cdc6ac63512acc8876fd78329e7b1dfe6a2c7ded3f0。
input/：原ZIP的全部303成员，保留原字节；没有修改其中报告、源码、资格、失败或数值。

独立主核对

源码：independent/recompute.py及run_logged.py。
完整成功：independent/attempts/run03/。
输出：run03/output/all_metrics_recomputed.csv包含全部990行、22指标；classification_recomputed.json、diagnostics_recomputed.json、result_summary.json、reference_checks.json、input_reads.json等给出其他复核。
140,432个标量比较最大差3.3306690738754696e-16；这些是机械核对数，不是独立科学检验次数。

independent/supplementary/：ZIP CRC与303成员、精确十进制资格时间、启动／完成来源、日志和Linux参考修正diff。
independent/attempts/run01和run02、supplementary_attempt01：本网页自身失败尝试的原源码快照、stdout／stderr、执行回执与部分输出，不作为成功结果。
independent/exploration_failure_replay/：早期JSON展示错误的明确标识回放。首次工具显示被截断，未保存首次完整原始stdout/stderr；这里不冒充原日志。

边界

本网页只对保存材料运行Python/NumPy。没有项目Linux连接、BGE载模、PyTorch/GPU核验、训练更新、正式标签解析、Memory或F/O载荷访问。本包不证明已从真值重生成所有原始指标；项目真实运行依来源对应的保存回执及已关闭资格审查支持。
原GPU attempt01和Linux分析attempt01的完整收到材料仍在input/。更早实施外审的tar读取错误只有当前包所提供的叙述，不虚构其未提供的原stderr。

下载ZIP外部大小／SHA随聊天答复提供。交付清单不包含自己，原输入清单同样不自包含；这不改变全部实际成员都在ZIP中的事实。
