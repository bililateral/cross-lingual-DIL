# 关系目标记忆实现与CPU手写证据

2026-10-04，Asia/Shanghai。用户明确按已提出顺序推进。当前已完成核心实现、执行定义及Linux CPU小模型核验；GPU原生一步尚未执行，正式训练/评价编排尚未接入，不将局部完成写成正式运行就绪。

## 实际实现

- scripts/step28_relation_memory.py：新32维关系头及w；仅BGE autocast；Dropout关闭；完整查询平方目标及H/b/c；可微式(9)；独立历史梯度；串行反传一次Adam；统计迁移、Algorithm R与刷新；无压缩完整字节容器、参考对应和恢复；288步train_stage入口。
- tests/test_step28_relation_memory.py：独立标量查询定义、梯度/更新/生命周期及盲区构造。所有文本/标签为手写生成，没有导入真实群。
- scripts/step28_relation_memory_verify.py：CPU受限核验和待审GPU完整一步入口，共用正式optimization_step；GPU形状448段×256 token，含首次Adam分配，不伪造首域步数。
- docs/SELLER_ALIAS_RELATION_MEMORY_EXECUTION.zh.md：输入、精度、随机流、计数、计量、原生准入及正式运行尚缺事项。

共享基线文件未改；复用BGE加载、标题/描述编码、群结构、配对顺序和日程。未增加LOGIT/风险/元学习项。训练阶段只接收当前拟合群和Memory，不接收Archive；正式调度器及端点评价仍待接入，不能用此局部API声称数据资格已自动落实。

## Linux核验

执行环境：yongpeng@10.201.109.111，/home/yongpeng/cross-lingual，既有py310；Python3.10.19、Torch2.9.1+cu130、NumPy2.2.6，taskset CPU0、CUDA_VISIBLE_DEVICES为空，各数学库线程1。22:14:56资源实查GPU0空闲32,063MiB、主机available251,621MiB；实际核验没有使用GPU或BGE权重。资源查看未发现本候选/项目训练或监听活动，不动其他服务。

22:16:36启动核验，进程退出0。6项检查全部通过，失败0、错误0、跳过0；核验主体3.778287秒，峰值RSS690,233,344字节（约658MiB），30分钟外层timeout保护包含启动开销。该3.78秒不包含导入前的解释器启动开销，不能冒称整个shell墙钟精确值。

| 检查 | 有区分性的实际断言 |
|---|---|
| 显式损失/统计 | 逐查询独立标量目标、H/b/c及w梯度；错误的1016项统一平均确实与正确查询平均不同 |
| 迁移梯度 | 显式T和式(9)一致；Y/w全梯度与解析式、选定有限差分；直接epsilon w项非零；岭偏差、恒等映射；错误列配对改变结果 |
| 盲区 | 有界完整378边反例保持历史代理却让缓存正负次序反转；迁移后的二次式与联合变换表达一致 |
| 真更新 | 历史项单独对小encoder/head/w都有梯度；串行与联合所有参数梯度及一次Adam更新在预定容差内一致 |
| 记忆 | 两次48群入账，统计与reservoir最终96，迁移并加入统计与独立期望一致；完整字节≤1MiB、同字节恢复、288次后续抽样一致，重复固化/身份错配/超预算拒绝 |
| 阶段入口 | 真执行288次小模型更新和阶段固化；model/Adam/Memory恢复后继续实际第289步，逐参数与抽样同结果，没有把计数改到288冒充训练 |

原件：[result.json](cpu/result.json)、[unittest.txt](cpu/unittest.txt)、[console](cpu_console.txt)。控制台包含测试将requires_grad张量转成数值进行断言的PyTorch警告；被测优化路径没有因此断梯度，未隐藏该警告。既有证据回传后大小/SHA已核对。

本轮CPU已通过。核验前本地/远端新增三源码SHA相同，精确源码身份见result.json；基线直接依赖bge_continual、chinese_base、population_data、alias_ranking四文件也实查相同。核验没有修改原数据、旧权重或共享环境。

## 边界与下一步

小模型不认证原生显存、原生历史梯度或正式效果。手写容器字节通过不认证将来真实六群容器必然合格；正式运行仍需实际完整计量。恢复测试使用内存state_dict和真实Adam状态，正式磁盘原子checkpoint及完整PyTorch RNG保存/恢复尚未接入。

增量外审集中检查核心执行路径、手写证据盲点及GPU准入入口，不重审未变研究主线/已认可代数，不认证创新。外审通过并由主执行者处置后，执行一次待审最大形状BGE GPU核验；若通过，再形成正式固定点的具体预算、判据和运行/评价入口。正式数据/标签、固定点效果、正式训练及匹配对照均未发生。
