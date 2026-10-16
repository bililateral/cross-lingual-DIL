# H_cons固定结果独立外审返回包

先读 `review.zh.md`（或同内容UTF-8 `review.zh.txt`）。核心判断：本固定结果及限定解释可保留；H_cons对S双域MAP主门false，对H新B直接门true，同时对H旧A排序及主概率系统存在代价。不存在用审查通过替代性能通过的含义。

## 文件导航

`findings.json`：逐项科学判断、未发现项目阻断/实质复现缺陷、审查者已关闭缺陷及超范围项目。

`reviewer_independent_v3/independent_all_metrics.csv`：独立直接索引重算的全部1650统计行；含106行N/A。`reviewer_checks.json`同时含9诊断、两根对照、正式目标分项、计数和预算。

`reviewer_analysis/`：原样audit.py本次真实输出；不是项目原Linux结果替身。`report_table_check.json`核对正文108表格行、471比较单元。

`scripts/`、`logs/`、`commands.md`、`execution_ledger.json`：实际核算/身份源码、已执行命令和结果/计时。`failures/`包含三处主脚本真实失败、一次临时诊断失败、补丁和诊断原件；失败未改写为成功。

`reading_scope.md`、`manual_reading_scope.json`、`source_excerpts/`和程序读取清单：区分手动审阅、程序读取、只哈希及未执行，不宣称重审整个项目。

`input_original.zip`：本轮实收原输入，3,187,528字节，SHA256 588e79e3a265319b9025fe94d16b4cc583f2d35cf4a51aa7fbc9b1fe54e4fb15；内容未改。`input_manifest.received.json`是其清单字节副本；身份结果见input_identity*.json。

`return_manifest.json`列出本返回包除清单自身以外的每个载荷的路径、大小和SHA256。清单不对自身作循环哈希；外部交付的ZIP整包身份和逐成员TSV（含该清单）另提供。空失败输出目录无有效文件，因此不占ZIP成员，不等于没有失败。

## 边界

独立核算使用Python3.13.5/NumPy2.3.5/Matplotlib3.10.8隔离环境，只读已保存数组/元数据。未连接项目Linux、未读模型/Memory/正式标签、未重新拟合校准、未跑CPU/GPU原生资格或新训练。AP/ROC/Brier/log-loss等原子标签级评价和部分资格原始receipt并未在本轮重新验证，具体见正文。

原项目正式6043秒+原本地分析2秒=6045/28800；CPU51/180、GPU9调用418/600保持原账。当前外审执行/失败另行保存，不把剩余预算当新授权。

全部文件由当前会话独立处理，无子agent或并行审查委派。
