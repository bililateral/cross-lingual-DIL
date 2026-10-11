LOGIT0.1／R0／S 已完成归因实验独立外审：交付说明
日期：2026-10-11

一、先读什么

review_report.zh.txt 是完整中文外审正文：结论、资格／72小时修订、实际运行、独立统计、六轮诊断、概率与阈值、成本、历史边界、真实阅读范围及未核验事项。main_report_erratum.zh.txt 给出唯一明确发现的非阻断勘误。

总体判断：接受固定开发条件下的有限归因结论；已证实SCIENTIFIC_BLOCKER为0，阻断性当前REPRODUCIBILITY_DEFECT为0。当前非阻断REPRODUCIBILITY_DEFECT为1：原report.zh.txt第37行R0的G从0.052435519订正为0.052435186。勘误不改变保存评价或方向判读；原上传包没有被覆盖。

二、输入身份及范围

唯一授权输入原名review_input.zip，网页上传名review_input(20261011-022206).zip。
字节数：28,532,122
成员数：1,266（均为文件）
SHA256：9aaf0f2a82de854741313ee1b948432d335132085f5bc92e21e3ae9f9a2ff29a
展开总字节数：40,039,402

本交付包不重复附带原输入。所有计算仅使用其已开放保存的分数、矩阵、计数、日志和来源记录。没有新训练、候选、独立留出或新验收门，没有解析正式文本／标签、导入项目模块、载模、反序列化Memory、连接项目服务器或访问其他外部材料。没有委派agent。

三、文件用途

review_report.zh.txt                 完整中文结论与证据说明。
main_report_erratum.zh.txt           RD-01的原格、正确值、影响及最小处置。
findings.json                       机器可读的问题分类与明确边界。
reading_scope.zh.txt                人工／程序阅读范围的中文说明。
reading_scope.csv                   1266个输入文件的实际阅读范围、字节数和散列。
RETURN_MANIFEST.json                本交付ZIP其他全部成员的字节数与SHA256。
README.zh.txt                       本说明与复算方法。

verify_saved_results.py             一次性只读独立参考；原输入之外输出；不导入项目fields/summarize_field。
verify_report_claims.py             126个报告显示单元格、29来源AST闭包、原生保存时间的预算算术。
plot_saved_review.py                从独立复算输出绘制4张静态科学图。

outputs/verification_summary.json   包身份、来源、执行、矩阵、诊断、误差及网页环境总账。
outputs/execution_details.json      27阶段、7776更新、存储、资源及衔接的复核细节。
outputs/input_file_inventory.csv    主复算程序的1266文件读取模式原始记录；扩展说明见reading_scope.csv。
outputs/recomputed_endpoints.json   全部三绝对方法／三差值、四角色、八端点、22指标的参考值与区间。
outputs/recomputed_endpoints.csv    上述4224行端点记录、共25344个标量。
outputs/recomputed_stage_domain.csv 全部243行阶段／实际域／角色均值。
outputs/pooled_classification.csv   332行合并混淆计数及派生指标；不混同群宏均值。
outputs/recomputed_diagnostics.json 完整诊断曲线、条件区间、标签及N/A。
outputs/recomputed_curves.csv       153条六轮指标曲线、train/valid六点和诊断差值区间。
outputs/report_table_checks.json    126格逐项核对；其中唯一RD-01与AST／预算核对。
outputs/derived_findings.json       获得／最终MAP、全部最终实际域、O阈值计数、成本、N代数等报告点值。

figures/map_attribution.png         三组O/N/Z的MAP点差与条件95%区间。
figures/all_six_epoch_map.png       全部固定有效fit/cache面板的MAP六轮曲线。
figures/all_six_epoch_recall_at_5.png 同一完整面板集合的Recall@5六轮曲线。
figures/all_six_epoch_log_loss.png  同一完整面板集合的raw log_loss六轮曲线。

verification_console.txt            最终主复算输出；保存矩阵统计与运行证据一致。
report_claims_console.txt           最终主报告文本核对输出；记录非阻断抄录错误。
reviewer_computation_notes.json     审查者自身计算修正记录，不是原项目缺陷。
first_reference_console.txt         最初参考运算的回执；浮点归约零附近描述标签差异及inventory类别统计。
first_report_parser_console.txt     最初文本解析器越过表尾的异常回执；已仅修正审查脚本。

delivery_receipt.json 在ZIP外另行提供，记录外层ZIP身份及完整成员表；不把外层ZIP自己的散列放回ZIP，避免自指。RETURN_MANIFEST.json不包含自身散列，其身份由外层ZIP与交付回执覆盖。

四、如何复算保存结果（可选，不是要求新实验）

已实测环境是本网页会话附带Linux计算容器：CPython3.12.14、NumPy2.3.5。主脚本使用POSIX resource统计，未声称Windows原生兼容。绘图另需Matplotlib。没有Torch依赖。原项目Linux/Torch/CUDA/RTX5090的实测只作为输入包保存记录；此处运行不替代它。

先将原授权ZIP解压到独立目录。以下路径是使用者自行替换的示例；输出目录必须在原输入目录之外。命令按顺序执行：

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python3 verify_saved_results.py --root /path/to/extracted_review_input --archive /path/to/review_input.zip --out /path/to/review_rerun_outputs

python3 verify_report_claims.py --root /path/to/extracted_review_input --results /path/to/review_rerun_outputs

python3 plot_saved_review.py --results /path/to/review_rerun_outputs --out /path/to/review_rerun_figures

第一脚本检查保存证据与独立统计的一致性，最终failure_count=0；这不是新的实验总通过门，也不是对全文没有错误的声明。第二脚本将原报告的那一格错误记录在inconsistent_cells中；该已知非阻断勘误不会使它异常退出。应阅读检查明细，不以exit0或PASS代替结论。

独立参考采用显式端点线性系数、实际A/B/C域内整群配对和bootstrap频数矩阵，不调用项目fields/endpoint_fields/summarize_field；固定PCG64(20260930)、5000次。主统计最大误差1.3322676295501878e-15。误差可能因NumPy／BLAS与归约顺序产生末位差异；所有保存数值仍受同一1e-12数值比较容差约束。诊断的零附近描述标签按冻结的“先逐群相减、再取均值”归约顺序，未新增事后容忍阈值。

outputs/derived_findings.json 是在已复得CSV/JSON上用直接均值、相减和计数相加得到的解释点值；不是额外bootstrap或独立确认。原式均在主报告第六、七、九、十节给出。reading_scope及本交付manifest属于本次审查的记录，不是通用审计系统。

五、解释边界

单seed、三个固定循环顺序、重复开发数据和条件区间的边界保留。顺序不是seed，辅助区间不是独立多重确认。S严格过拟合信号计数为0不等于无过拟合；N收益主要来自Z，A2与G方向未确定；负遗忘必须和首学、最终绝对值一起解释。raw log_loss、校准概率及固定阈值误报代价不能相互替代。历史C/C_plus负结果与R1不翻转。overall_pass_gate=false意味着本轮没有新总通过门。

正文及reading_scope列明未独立核验的载荷、原始真值、校准重拟合、GPU恢复、项目系统调用、历史原始实验与后台模型路由；这些未核验事项不被自动升级为本轮补实验前提。
