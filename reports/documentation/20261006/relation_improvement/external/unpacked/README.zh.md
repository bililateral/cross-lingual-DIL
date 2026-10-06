# 交付说明：新候选讨论，不是生产实现包

先读根目录`RELATION_FUNCTION_MEMORY.discussion.zh.md`。`original_input.zip`是本次用户附件原字节；多层历史可由`audit/scripts/01_identity.py`重建。`audit/sources/reading_scope.json`说明实际阅读范围；`selected_sources.json`为方便核对而另存的未改来源；`literature.json`为本轮原论文定位。

## 重放

在已有NumPy与PyTorch的CPU环境中执行：

```bash
python audit/scripts/run_all.py
```

不下载或安装依赖，不导入项目模块，不连接服务器，不调用BGE，不读取真实文本/标签/模型/Memory，不训练候选。脚本在当前解包目录生成input和history副本，并改写`audit/outputs`中的复算结果，因此先核对交付manifest，再在副本目录重放。环境变化可能影响浮点末位、警告文字或Torch行为；断言会显式失败，不自动修复。

`02_mathematics.v1_with_logging_warning.py`、`02_mathematics.v2_before_component_guard.py`及对应输出保留早先真实执行版本，不属于默认重放入口。所有输出标明web-only。

根manifest涵盖除自身外全部文件的字节/SHA；不自哈希。原输入的旧manifest只是其各自历史身份，最新用户许可与records仍以本次context为准。没有新增项目许可、数据授权或性能承诺。
