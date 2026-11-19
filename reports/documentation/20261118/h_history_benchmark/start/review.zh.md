<!-- {"experiment":"chinese_trial3_six","review_closed":true,"manifest_sha256":"792d6f85863de3286ba5e7f008a0ea35642397bab034c9c321ab9fa3dd5af028","qualification_sha256":"a426d54db36235abaab6a8e02a7bcb5c88537b3739c8dd0d7bccdb9aec621570","fix_qualification_sha256":"61cffdbd0002d2d7c26822b36b041894524cf09dd76ae1d10b840ee03582b1fd"} -->
# 中文六方法条件启动增量主审

11 月 19 日回收网页审查原件 25,727 字节，SHA-256 为 `3a87663c371c368c39e9960a58e915e4b7a3c7b7e0b1208b0a1134565799e457`。专用页已关闭。网页所读版本为 `501bca3d`，意见为暂缓 arm。以下处置适用于 `e1219ac0d520cf8310d13de35646e132482098d5` 的固定六方法部署。主执行者据实际源码、单核运行和现场输入核对关闭本增量，可以部署唯一条件监听。[一次授权与外发包](submission.json) · [网页原文](external_report.zh.md) · [生产清单](source_manifest.json)

| 审查项 | 独立判定与实际证据 |
|---|---|
| R1 前序末次结算 | 已核对正在运行的英文生产版本。发布 completion 后仍有一次 handoff 结算。监听在英文控制器和相关进程退出后核对预算上限、各桶用量及最后三个内部成功事件依次为 analysis、dispatch_monitor、controller_setup_or_handoff。外部操作另具 ID。少末笔和超预算均在单核实测被拒。英文科学判定值无需打开。 |
| R2 启动授权与主审 | prepare 解析本原件首行独立 JSON，严格要求布尔 true、对应实验和三份文件的 SHA，并核两次资格的 passed 布尔值。负面文字、假值、字符串及旧版摘要的合成用例均被拒。正式门绑定收到的主审、用户授权及两份资格回执。 |
| R3 FOREVER 首更 | FOREVER 在原 learner 完成 current 优化、核对 Adam 步数后，通过原生回调输出第一步的 run_id、stage、completed 和当前计数。其余五臂原入口已经输出原生标识与步数。消费端同时核对 worker 的内核身份、attempt、claim、轨迹及计数，并在回执保留原行偏移和摘要。单核真实 SGD 子进程调用同一 FOREVER 报告函数。零、无标识和错轨迹均未触发，正确第一步才触发。科学更新未改变。 |
| R4 自删与中断 | 先持久化首更，再核门、控制器、唯一派工、原行与文件 inode 后删除。两步之间以及删除后落末回执失败均已模拟。按原首更凭证可补齐回执，无需新进程做训练。相同内容的新文件和符号链接均保留，前者 inode 不符，后者不属授权对象。 |
| 首派资源回落 | 单核经过正式 wait_then_start、start_campaign、training_phase 和 ChineseRunner.run_many，显存替身依次给够、不够、再够。中间留下拒绝准入观察，零进程、零控制台，恢复后同一队列只派 attempt 1，训练费仅一笔。该场景替换模型 worker、NVML 与正式输入，测试真实创建和释放 Linux 子进程。 |
| 输入与费用 | 历史 H 的实际 config 是 five.config 的显式别名，预检现也直接调用 history.config。核对 BGE 和五份具名训练文件、两项公开元数据摘要及父目录，授权入口再次核对链接。旧 CPU 为 78 秒，初版监听单核 8 秒，审后两次各 4 秒，共 94 秒。其中首次测试把已存在的准入回执目录误当作已启动，已改查 process、console、实际队列和收费，失败原文与成本均保留。GPU 原账 47 秒。60 秒运营费为提前预记，准备另计实耗，等待时只计活动。ready 写入至交接验证的活动也随原账结算。 |

累计合同仍为 720 小时。活动检查累计到控制桶，首笔未建账错误有具名记录。运营预记仅覆盖例行具名操作，发生额外维护时据实记录超出部分。手动具名 CUDA OOM 一次重试、其他健康槽继续领取任务沿用原增量审查及既有试验。此次没有另做 GPU 实验或打开正式数据。本结论只签署本版条件启动；Linux 启动、真实首更及删除时点以各自运行回执记载。[原生 CPU](../cpu.json) · [原生 GPU](../gpu.json) · [原调度主审](../dispatch_review/primary_disposition.zh.md) · [监听初验](qualification.json) · [审后单核成败原件](qualification_costs.json)
