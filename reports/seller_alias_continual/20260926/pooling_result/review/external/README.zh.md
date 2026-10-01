# 本次正式结果外审证据

判定：结果有效性外审通过；冻结方法改进验收失败（13项5通过、8失败）。不是训练前外审续签。

## 主要文件

- `pooling_result_review_report.zh.md`：完整中文报告，含代码行号／记录路径、全22指标、13检查、统计解释及未验证范围。
- `verify_submission.py`、`submission_integrity.*`：实际附件及165来源身份核验，与前次附件代码比较；该脚本保留本网页实际路径，依赖两次原ZIP。
- `submitted_analysis_entry.py`：从本次附件复制的原始分析源码；实际在原项目路径执行，未修改。
- `submitted_entry.*`：原入口网页实际执行日志、资源日志及退出码。
- `submitted_entry_output/`：原入口在新reviewer_analysis目录生成的三个输出副本；原analysis未覆盖。
- `independent_result_audit.py`：审查者独立NumPy核查，不导入任何项目模块。
- `independent_audit.*`：修正后独立核查的实际日志、资源日志及退出码0。
- `independent_output_pass/`：全部独立统计，包含22指标、36矩阵分域摘要、固定／校准计数、轨迹及5000×22 bootstrap输出。
- `independent_result_audit_attempt1.py`、`independent_audit_attempt1.*`：审查者首次canonical JSON漏结尾换行的失败尝试；项目文件未改。错误原因和纠正范围见报告2.3节。
- `final_scope_check.py`、`final_scope_check.*`：最后重验165来源不变、原Linux与网页CSV相同，以及零预测阈值边界。
- `input_source_inventory.json`：用户附件来源清单副本。
- `build_report.py`：只把已有独立输出排版为报告，不进行训练或读取正式数据。

## 复核方式

使用用户原ZIP完整解压目录，在已有含NumPy的Linux环境运行；不需要安装其他包、模型、正式文本或标签。
先按报告附录A1运行原项目`step28_alias_pooling_result.py`至新的`Q/reviewer_analysis`；再按A2运行`independent_result_audit.py --root 项目根目录 --out 全新输出目录`。所有统计脚本为已有证据重算，没有训练、正式文本／标签／owners读取或模型加载。

原分析入口的ROOT由其脚本路径决定，不能直接把本证据包里的`submitted_analysis_entry.py`作为项目入口运行；它是原始源码留档副本。

独立脚本要求既有的`Q/analysis`和新生成的`Q/reviewer_analysis`，便于核对Linux与网页报告一致。线程环境变量及全部原命令见资源日志；独立脚本另固定一个CPU亲和性，实际Threads=1。

## 证据边界

本证据包不含正式文本、正式标签、模型或真值。保存矩阵的归并和区间可重放；真值依赖指标不是从标签重算。正式GPU更新／恢复及远端删除只能核对运行回执与来源绑定，不是本网页再次执行。原D和本轮候选权重按用户指令均已删除；预训练不在删除范围。

## 清单

`evidence_inventory.json`列出本包除两个清单文件外全部文件的大小和SHA；`MANIFEST.sha256`列出相同载荷SHA，并额外覆盖JSON清单。清单自身不作自引用。两份清单由打包时重新计算，不是科研输出。
