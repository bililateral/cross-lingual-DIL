review(8) 分阶段完整梯度增量外审交付

先读main_report.zh.txt；机器裁决为verdict.json；实际执行范围为execution.zh.txt。

主结论：本次新增staged实现的精确链式梯度得到静态和微型证据支持；当前原生BGE在首个支持群create_graph=True求导时OOM，零成功元更新，资源及正式训练均未准入。本轮停止，无新资源尝试、无效果结论、无新优化路线。

目录
- input/review8_original.zip：新上传review(8)原字节。792243字节、96成员，SHA-256 aab6c2470c76a6d66985246310f0af6bb983e6527e88372778af089d83d90ca6。
- submitted/：上述ZIP完整展开；含当前26来源、三个Linux尝试、初始部署/回传原ZIP、直接背景prior。首次失败源码和日志没有覆盖。
- independent/：本次实际本地核验的可运行标准库脚本、JSON输出、精确diff及必要证据。
- process_notes/：分项审查笔记与原机械核验脚本，保存审查过程。其绝对本地路径为当时执行记录；以主报告/机器裁决为最终结论。
- EVIDENCE_MANIFEST.json：交付包每份载荷的大小和SHA-256，不对自己自哈希。

可复算的小检查（仅Python标准库，不需要Torch、CUDA、网络或原服务器）

在本交付ZIP展开目录运行：
  python independent/verify_increment.py input/review8_original.zip
  python independent/staged_scalar_oracle.py

第一个程序只读输入ZIP及其内嵌背景ZIP，核对来源、关键AST、尝试和预算；第二个程序计算既有标量夹具及解析错误向量。输出可与independent/verify_increment.stdout.json和staged_scalar_oracle.json对应比较；本地Python版本等环境字段可能不同。它们不会运行生产脚本或向GPU提交任务，机械检查成功不表示BGE通过。

*.repair.diff记录提交者本轮已经发生的一次tiny夹具修复。当前没有新生产算法补丁，不应把diff或本包中的原生包装源码当作继续尝试指令。

证据边界
tiny_first是7通过+1ERROR；tiny_repair才是8通过。两步为逐步同起点比较，不是两条独立轨迹累计漂移验证。原生四个一阶done后在hvp_0构建可微梯度失败，结果保存后包装134。25新增秒+101旧秒=126；余量不授权再运行。主报告明确区分SCIENCE/RESOURCE阻断、已修复夹具缺陷、后续建议和超范围事项。

没有正式文本/标签/缓存、旧实验权重、卖家私有身份或凭据。旧基础体系仅作为原件保留，未递归重审。本交付不声称审查者运行过Torch/CUDA/BGE或连接过服务器。
