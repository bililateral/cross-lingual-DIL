增量评价独立审查复核材料
2026-10-08（Asia/Shanghai）

一、交付内容
INCREMENTAL_EVALUATION_REVIEW.zh.txt：完整中文审查正文，UTF-8。
READING_SCOPE.zh.txt / .json：逐文件实际范围，506条嵌套记录；全文/节选/机器字段核对/未读明确分开。
audit_saved_arrays.py：本次独立数组复核脚本；不导入项目代码，不执行训练。
record_execution.py：一次性记录命令、起止、stdout/stderr和退出码，无自动重试。
input/supplement/：本轮附件原脚本、原统计、原表和原日志的逐字节副本。
execution/：两次实际保存数组计算的原始执行记录；均首次退出0，stderr为空，没有修正后重跑。
supplied_replay/：附件原脚本在本次网页工作区的实际完整输出。
independent_audit/：独立路径完整25,344统计数、全部69项规则和过拟合证据字段核对。
*_identity*.json：已提供输入的身份核对；历史嵌套成员的语义范围以READING_SCOPE为准。

二、输入与重放
为了不重复打包已提供的历史归档，本下载包不再嵌入9.6MB原结果ZIP。
唯一计算输入取自用户本次review_input(7).zip中的background/original_result_input.zip：
9,614,123字节；SHA-256 dbed856f06482e29b444ce45b5b44973083bbd28964516ae2cb5ccf5c8be3c38。
将该文件复制到本包根目录的input/background/original_result_input.zip即可。
不要连接项目服务器、加载模型或补取任何标签/完整Memory来重放这些保存统计。

在本包根目录、已有Python与NumPy环境下，以下两条命令分别复现对应计算。
新输出目录须尚不存在；原输出应保留。它们是可选的复核说明，不是要求新增项目运行。

python record_execution.py new_execution supplied_once python input/supplement/step28_record_replay_compare.py --input-zip input/background/original_result_input.zip --output new_supplied_replay

python record_execution.py new_execution independent_once python audit_saved_arrays.py --input-zip input/background/original_result_input.zip --supplement-dir input/supplement --output new_independent_audit

本次实际环境Python3.12.14/NumPy2.3.5；是网页审查工作区，非项目原生py310复跑。
source脚本SHA、输入ZIP SHA及实际命令均在随包记录中。

三、结果解读提醒
原脚本交付保留既有21,120浮点统计的原表示；独立脚本给出直接按行抽样所得值，末位浮点差见audit_summary.json。
新增S−LOGIT区间来自逐群配对差；原五项判据及overall=false未改。
all_69_checks.csv的判定值使用各行statistic/operator/threshold；CI字段对应该端点的primary−primary差。
四条候选cal−对手raw保护只要求均值，其CSV行的ci_role_note明确说明附带CI不是该跨角色保护的区间。
过拟合字段核对仅说明现有日志的可用性，不以训练目标标量代替同口径fit/缓存/valid曲线。

本次没有计算失败或修复。工具长文本截断后的补读不是数值失败；历史外审/CPU失败不计入本次。
MANIFEST.json只登记本下载包载荷的实际大小与SHA，不把身份核对当科研有效性的证明。
