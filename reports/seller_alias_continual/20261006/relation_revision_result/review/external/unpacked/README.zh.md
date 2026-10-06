# 本轮独立结果外审交付

先读 `RELATION_REVISION_RESULT.external_review.zh.md`。本包面向20261006_135937已完成B＋Q＋0.1R结果，不是启动许可。

## 内容与范围
原附件review_input.zip保持原字节；包含其内所有必要历史。reviewed_sources是本次31冻结来源另加主报告/context的方便阅读副本；figures是提交图原字节。audit保存本轮实际使用的脚本、输出、审查侧失败及修正；未包含正式正文/真值/权重/Memory本体，也没有新增训练。

## 重放
在解压根目录、有既有Python/NumPy/CPU Torch环境时执行：

```bash
python audit/scripts/run_all.py
```

第一步解开review_input.zip到input并按context解开三层历史；后四步只做本地身份、数值和源码绑定审查。不安装包，不连接项目服务器，不调用BGE、训练、评分、标签解析或完整原生结果核验器。重放会更新audit/outputs中的运行输出；应先保留原交付以验证原manifest。时间与环境字段可随重放环境变化，数值断言保持冻结定义。需从项目提供的矩阵而非真值重算MAP/AP聚合。

`06_package_report.py`是本次打包脚本，默认以本次网页工作区为路径组织副本；不是重放统计所需入口。报告来源位置以原包RR/source及RR/job为准；RR=reports/seller_alias_continual/20261006/relation_revision_result。恢复后的input目录保持原项目相对路径，不修改冻结JSON中的Linux根目录。

manifest.json覆盖除其本身之外的所有文件，记录字节数与SHA-256。不得把“脚本通过/哈希吻合”替代报告对证据范围和机制归因的限制。
