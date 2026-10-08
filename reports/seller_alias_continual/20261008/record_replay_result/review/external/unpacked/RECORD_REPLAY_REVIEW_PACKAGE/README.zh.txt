记录作用表与同控重组 C/S 结果独立审查资料包

先读 RECORD_REPLAY_RESULT_REVIEW.zh.txt（完整中文正文；不是仅摘要）。
RECORD_REPLAY_READING_SCOPE.zh.txt / reading_scope.json / reading_scope.tsv 为494项逐文件实际范围。
independent/statistics 保留73套保存矩阵重算的全部统计、3520行端点表、1782行阶段域指标表、计数和五条件结果。
independent/records 保留来源、调度、分数、资源、清理核对及R1/R2实际差异。
execution 保留四次独立脚本调用的真实stdout/stderr、退出码、起止时间和两个失败版本。
scripts 为本审独立脚本；均不导入项目模块，不加载模型或Memory，不读原始文本/逐对标签。
MANIFEST.json 列出本审包内载荷的字节与SHA；manifest不自引用。原送审ZIP不重复放入本审ZIP。

输入身份：review_input(6).zip，9614123字节，SHA-256 dbed856f06482e29b444ce45b5b44973083bbd28964516ae2cb5ccf5c8be3c38。
原附件为复核输入，本包不修改它。必要输入路径对应INPUT_PACKAGE_MANIFEST.json及INPUT_ACTUAL_IDENTITIES.json。

复核方式（只对已有保存结果；不调用项目执行）：
将原送审ZIP解压到单独input目录，保持其相对路径。
在已有NumPy环境中运行：
python scripts/audit_saved_results.py --input /absolute/path/input --output /absolute/path/new_statistics
python scripts/audit_run_records.py --input /absolute/path/input --output /absolute/path/new_records --archive /absolute/path/review_input.zip
两次--output均使用新的目录；不要指向输入目录。archive是原review_input(6).zip的实际路径，文件名可由用户自行填写。
本次实际环境：Python 3.12.14 / NumPy 2.3.5；详见statistics.json。没有安装或运行项目训练依赖。

范围限制：AP/MAP/Brier等标签到矩阵步骤不重新生成；矩阵以后统计独立重算，计数到六分类指标独立重算。
Memory本体/已删模型/远端保留例外只核对所附身份记录，不将回执视为外审再次远端实查。
历史实现包仅按本次必要范围读取；既有原生数学/梯度证据复用。
结论：运行/评价证据有效；C–LOGIT五条件5/5，C–S为4/5，原整体false。方法创新和额外teacher分配收益未建立。
