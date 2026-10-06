# 网页独立审查的执行与修正记录

## 边界
这是本轮网页工作区的保存结果/源码绑定/数值审查，不是项目Linux、BGE/GPU或训练回执。本轮没有重跑K1、原生GPU、完整微型2592更新或正式训练；当前环境存在CPU Torch，但仅在03脚本中导入冻结模块检查全局绑定和来源列表，没有调用训练、模型加载、评分、标签解析、校准拟合或项目原生核验器。

## 审查侧失败与修正
- `05_report_crosscheck.py`初版未去除主报告Markdown单元格两边空格，导致未识别数值行，断言 `len(checked)==22`失败，结果为0行。这是审查解析器问题，不是主报告数据错误。
- 原版脚本和当时stdout/traceback完整保存在 `reviewer_failures/05_report_crosscheck.v1.py` 与 `.v1.stdout.txt`。
- 唯一修复为对分割后的每个表格cell调用strip；修后实际识别22行，并逐列按原显示精度核对通过。未修改项目源文件、矩阵、counts、图或报告原件。
- 早期交互查看JSON时曾误把initial当成collected顶层，以及误用counts_by_domain键，分别得到KeyError；改为读取实际schema中的points.initial和pooled_fixed_half_classification。这里仅记录交互探索的错误摘要，未将其说成项目失败，也未伪造未另存的完整交互stdout。

## 便利修改，不是科学定义变更
01脚本新增“优先从解压包根目录review_input.zip读取”的便携路径，保留原/mnt/data回退；03脚本增加对包内三份旧LOGIT资格回执的显式一致性检查。此前实际版本保存在revision_history中；新版本均已实际重跑。

## 最终重放
`run_all.py`实际顺序执行五份当前脚本，全部exit0，详细时间、stdout路径与退出码见replay_run.json。五份脚本只读原件/保存统计，写审查输出，不修改冻结原件。输出时重新展开原附件只在本网页审查目录内发生。

脚本04的手写标签完全新造，只有合法28账号结构，用于目标值例证；不来源于未开放逐对标签，不尝试恢复真实标签。原生/正式日志的数值在报告中标为提交证据，不伪装本次模型实测。
