F求解失败分析与最小修复提案：外审回收包

入口：review.zh.md（完整正文）；review.zh.txt为相同UTF-8正文。
reading_scope.md：逐文件实际阅读／身份检查范围。
audit_checks.json：输入包、29生产来源、失败残差、A_F汇总和预算的只读核对。
handcheck.json：ChatGPT隔离CPU上原求解器的两个固定解析例；不是修复实现资格。
rounded_reference.json：同两个固定例子的实际FP64／FP32 K的80位闭式参考；不是区间证书。
各JSON附对应同名Python程序。原solver／依赖来自用户原ZIP，不在本回收包中复制。
artifact_manifest.json：本回收包自身文件的大小／SHA，不包含清单自身或外层ZIP散列。

自愿复算时，程序仅面向已经允许的隔离工作区；它们不授予访问项目Linux或训练的权限：
  python audit_checks.py INPUT_ZIP EXTRACTED_INPUT_DIR OUTPUT_JSON
  python handcheck.py EXTRACTED_INPUT_DIR OUTPUT_JSON
  python rounded_reference.py OUTPUT_JSON
第一项仅需标准库。手写计算使用已安装的numpy、mpmath、torch；不会下载库或模型。
handcheck.py断言原求解器SHA，只有原solve固定手写调用和零控制，optimizer调用0。

结论：有条件同意窄终端验收修订方向；原运行仍失败，新实现／新资格／新正式运行均未通过本包而自动获准。
未复制整个输入ZIP、正式更新原表、模型、Memory、正式文本／标签或盲预测数组。
