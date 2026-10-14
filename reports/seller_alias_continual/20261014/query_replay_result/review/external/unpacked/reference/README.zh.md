# 参考核对说明与复现

本目录记录当前审查实例在网页工作区执行的标签不可见参考核对。它不是项目Linux原生资格，不是新训练，不是新的科学验收门。原项目失败／补核在`../input/original_input.zip`内保持原字节。

## 输入与主脚本

输入：`../input/original_input.zip`，4,808,851字节，SHA256=9013c992a925bac6e69ba603438d5ca62692fcc38d5306a89f8a942f05973c96。完整357成员身份位于`run01/input_members.tsv`。不需要解压模型或标签；本包本来没有这些载荷。

主脚本`check_saved.py`独立编写，只使用Python标准库和NumPy，不导入项目代码，不执行所上传的Python／shell／pickle，不提取或访问项目正式标签、模型和Memory。读取公开UID、保存分数、计数、指标、映射、诊断及执行记录。使用原固定seed／群抽样索引，未新增种子、统计设计或模型更新。

主检查原运行环境：Python 3.13.5、NumPy 2.3.5；OPENBLAS_NUM_THREADS=OMP_NUM_THREADS=MKL_NUM_THREADS=1，详见run01/environment.json。源脚本1次执行完成，exit=0、stderr为0字节，输出187,168标量核对／1452行固定统计／27标签。误差记录在run01/check_result.json。

从解压后的返回包根目录，可在安装NumPy的Python环境执行以下等价命令（输出目录必须不存在，防止覆盖原证据）：

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python reference/check_saved.py --input input/original_input.zip --out new_reference_check
```

这是可复现命令；run01启动的原始shell逐字命令未另存，不将此称为shell现场录屏。实际参数、输入身份、源脚本、环境、输出与退出码均保留。运行时间／平台字段可能随环境改变，数值检查仍使用脚本列明的算术容差，不是新的科学非劣容忍。

## 附加两项检查／澄清

```sh
python reference/clarify_reference_reporting.py \
  --run01 reference/run01 --out new_reporting_clarification
python reference/check_custody_records.py \
  --input input/original_input.zip --out new_custody_record_check
```

前者只澄清我们自身run01把S缺失response默认汇总成0.0的报告字段。应使用reporting_clarification/execution_evidence.corrected.json解释S的mean_R=null／NOT_LOGGED；原始run01不删不覆写。该澄清没有改变科学统计或冻结输入，也不是一次失败训练。

后者逐项核对predelete的5权重记录、242份实际可见非权重字节及5份Memory的manifest记录；不连接服务器、不确认删除后现场、不读取Memory。其结果不能升级成删除回执。

## 文件说明

- run01/all_statistics.tsv：1452行绝对值、比较、旧A变化，保存全浮点精度。
- run01/diagnostic_decisions.tsv：27项训练／valid变化、差距、三个区间和判读；diagnostic_curves.tsv：162行六轮原量。
- run01/classification.tsv：五端点×三个角色×all/A/B共45行计数及汇总／宏率。
- run01/diagnostics.json：重建判读及六轮共同残差统计；不含真实T或K载荷。
- run01/calibration_records.tsv：五次保存校准记录；只验证映射，不重新拟合。
- logs：主检查、澄清、保管记录检查的stdout/stderr/exit及失败账。
- read_range.py与../evidence/reading_events.jsonl：审查定位工具及读取请求日志；具体语义阅读以保守范围表为准，非实验运行器。
- build_report.py：从已核对输出组织中文报告和表格，不执行项目实验。

原生资格失败总账5+6=11步、201+268=469秒保留在原ZIP。主参考检查及两项附加处理均退出0，无执行失败；不能把参考成功说成项目失败没有发生。这里也没有用已有analysis的violations_total=0作任何科学约束判定。
