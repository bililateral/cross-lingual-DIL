# 审查者自身修正记录

第一次网页测试包装在执行测试之前，把source_inventory.json与Linux sources()列表作顺序完全一致断言，失败。逐path/bytes/SHA比较后确认23项完全相同，仅Windows清单与Linux排序不同。改为path排序比较，同时仍要求实际sources()与Linux CPU result.json的source_files列表完全相同；随后7个手写测试实际执行并通过。原脚本和原失败输出保留为v1_order_sensitive文件。

这是审查脚本的比较口径错误，不是项目源码不一致、项目测试失败或项目Linux重试。没有改动送审生产源、测试或合同。

正式gate反例只在网页内存构造字段并调用validate_gate，未落地任何正式授权文件、未执行execute、未创建其job。watchdog反例执行未改函数AST的隔离环境，使用替代写入失败和退出记录器，没有真实进程退出、服务器故障或项目运行。

## 新目录重放的审查者比较器修正

七份重放脚本全部退出0后，首版包装比较器错误要求全部浮点JSON逐字节相等，在`independent_checks.json`处拒绝。直接差异显示：符号评价输出精确相等；数学有限差分/残差与两阶段运输的误差末位有约1e-13变化，非整数/布尔/资格结果变化。此拒绝不是项目用例失败，未修改原数学脚本或所附生产源码。

保留`check_portable_replay.v1_exact_json.py`与原始失败输出。修正的独立重放比较器对字典/数组结构、整数、布尔和字符串仍严格相等，对浮点记录使用固定绝对1e-11比较（各原数学脚本自身断言仍全部执行并通过）。两次实际输出都保留，最终最大差见`portable_replay_check.json`。不宣称跨CPU执行的浮点结果逐位一致，也没有重跑项目Linux。
