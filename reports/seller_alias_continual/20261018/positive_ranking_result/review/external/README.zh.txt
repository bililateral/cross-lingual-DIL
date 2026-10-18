S_pos/H_pos 结果外审包使用说明

先读 S_pos_H_pos_external_review_20261018.zh.txt，再读 findings.json。

裁决：本轮固定执行与保存统计可采信；S_pos−S双域raw MAP主目标未达到。无新增使结果失效的SCIENTIFIC_BLOCKER或REPRODUCIBILITY_DEFECT。不要把无新增缺陷解释为性能通过。

input/positive_ranking_result_review.zip 是用户本次唯一获准输入的原始字节复制，未重打包。
身份：5481875字节，389 ZIP成员，SHA256 4eeeb5df0e93b6c32bf352f42b656b945a8d4e4f71078b73fa4df0e015ae63f7。
该内层ZIP的manifest.json描述388载荷；外层artifact_manifest.json描述本次审查交付文件的大小与SHA256，自身不自哈希。

主要内容

1. 完整中文主报告：裁决、源/执行/原生/目标接线、主次比较、A末/遗忘、单双正例、三概率角色、24/18诊断、标签复核边界、成功失败原件及资源；附全部22指标的三角色端点绝对值、48组汇总计数和48验证曲线点。
2. findings.json：结构化结果、解释边界与越界扩展，主失败独立于缺陷列表记录。
3. independent_statistics/：主审另写脚本重算的全部2640统计行（含176 N/A）、16分组、固定draws及92项实际读取身份。没有重新生成标签派生原值。
4. supplied_replay/：原样随包审计在隔离环境的重放结果与图形；与项目原来analysis目录分开。
5. agents/execution/：来源/执行/预算核对，包括attempt01审查脚本失败及attempt02成功的确切脚本快照、完整stdout/stderr、exit和资源。
6. agents/wiring/：实际目标、VJP、原生梯度/参数/Adam保存证据及正式分项算术。
7. agents/diagnostics/：独立诊断、概率、旧A变化/遗忘和分组核对，含全部18判读与曲线。
8. reading_scope.json 与 root_manual_read_scope.json：区分全文、选段、机械解析和仅哈希；保留各分审自己的详细清单。
9. execution/、resource_and_execution_record.json、各分审执行记录：实际命令、环境、成功/失败与限制。核算数量有重叠，不是新增样本量。
10. scripts/：实际使用的审查代码与报告生成代码；未改任何输入代码。

在既有依赖的独立环境重放原样随包审计

先将input/positive_ranking_result_review.zip解包到一个新的review_input目录。以下命令仅产生新的保存统计分析，不应换成训练、资格或模型恢复入口：

OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 python -B review_input/workspace/scripts/step28_positive_ranking_audit.py --workspace review_input/workspace --output newly_created_audit_output

--output必须是尚不存在的新目录。依赖只有本审查用到的numpy和matplotlib及Python标准库；本次实际版本Python3.12.14、numpy2.3.5、matplotlib3.10.8。未安装新环境。原项目py310分析记录保留在内层输入，两者不得混称。

独立诊断脚本也支持可移植参数：
python -B agents/diagnostics/independent_diagnostics.py --workspace review_input/workspace --output another_new_directory

主审independent_statistics.py、执行证据和接线脚本保留了实际执行时的绝对工作区根，准确路径在各execution记录；如在其他目录使用，只调整审查脚本的输入/输出路径常量，不修改内层输入或计算规则。这样保留了实际命令/脚本身份，而不把未经执行的新包装器冒充本次证据。

失败记录说明

执行子审attempt01的exit2源于审查脚本错选清单相对根并误解analysis_inventory顶层类型，不是正式项目失败或真实缺文件。其快照、stdout/stderr、断言与费用完整保留；修正后attempt02通过。另一次早期current_uid字段打印KeyError只有工具错误摘录，明确未保存完整stdout，不伪称完整原件。主审原样audit、独立统计、接线和诊断核算均成功。

数据和行为边界

包中仍无正式权重、Memory载荷或标签原表；只含用户已开放保存分数、查询、计数、指标及标签派生正例数。没有联网搜索、没有连接项目服务器、没有新训练、没有运行生产finalize/blind_gate、没有从标签重算AP/ROC/Brier/log-loss，也没有实时检查服务器保管状态。

原项目12664/57600秒账与本审查环境核算分别记录；外层审查包SHA256见伴随的SHA256文本及交付消息。
