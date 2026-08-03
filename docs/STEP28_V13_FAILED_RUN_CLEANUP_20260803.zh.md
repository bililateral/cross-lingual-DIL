# Step28-v13 失败运行清理与科研边界（2026-08-03）

## 1. 当前结论

`v13_training_ready_v1_3_full378_fresh_20260801` 至
`v13_training_ready_v1_11_20260803` 均为永久失败运行。它们没有产生可发布的
fresh full-378 数据集，也没有产生 M0/M1/M2/M3 科研结果。任何开放 split
上的预检、M1 独立性或捷径审计通过，只是失败谱系证据，不能升级成数据就绪
或模型成功。

成功冻结的旧 v1.2 发布树不属于本轮删除范围，但其训练资格已因公开 C40
捷径和 Audit custody 问题撤销，只保留为历史发布证据。

## 2. 各版本失败边界

| 版本 | 首个不可恢复失败 | 结果边界 |
| --- | --- | --- |
| v1.3 | 最终校验器把 world 内模板槽误作 split 全局唯一 | train 未发布；其余 split 未生成 |
| v1.4 | Windows 长路径复核误报缺失，且 manifest 生产/消费版本不一致 | train 未发布；其余 split 未启动 |
| v1.5 | 全局 salt 被错误要求同时覆盖 192,000 个候选，搜索不可行 | 在 development 身份值计划阶段失败 |
| v1.6 | M1 标签公式回执在追加字段前计算 self-hash | 标签公式和行本身未证伪，但运行不可发布 |
| v1.7 | 正式 UID 捷径攻击的固定逻辑回归未收敛并 fail closed | 未产生捷径指标；Audit 未授权 |
| v1.8 | 第 476 个 train world 触发 identity-free 标题重写碰撞 | train 未完成，无部分发布 |
| v1.9 | join-only UID 的随机子串被错误当成可见身份泄漏 | development 边界校验失败 |
| v1.10 | 冻结授权源码仍要求过期的 premodel 成员版本名 | Audit 文本生成前失败 |
| v1.11 | 跨版本审计器再次把 `mechanism_slot_uid` 当成全局键 | 四 split 已生成但未发布；未评分、未训练、未解封真值 |

v1.11 的数据内 `(world_uid, mechanism_slot_uid)` 在每个 split 均为
`6,000/6,000` 唯一，world 内重复为 `0`；12 个模板槽跨 500 worlds 正常复用，
因此全局重复行数必然为 `5,988`。失败来自冻结审计合同的作用域错误，不是
观测到的数据实体碰撞；按 one-shot 纪律仍必须废弃整个运行，禁止原地修补。

## 3. 删除前保留的最小证据

失败身份值已压缩为不可逆、排序去重的 SHA-256 排除集合：

- 文件：`reports/step28_synthetic_chinese_dataset/failure_records/step28_v13_failed_identity_exclusions_through_v1_11.private.json`
- 文件 SHA-256：`f70611a4b5df7ddbded6784820026352c92952a0245fcb184b4e7c282c1447a0`
- canonical self-hash：`6f60e294bdbcae1d1da3802acdf4095d605c8e633755fe689d4a71b67fece4d3`
- 禁用身份值哈希总数：`915,996`
- v1.11 新增：train/development/Audit A 各 `42,000`，Audit B `44,500`，合计 `170,500`

该文件不含原始身份值、标签、控制者、qrels 或私钥。未来 fresh 运行必须直接
读取并哈希核对该压缩排除集合，不得为了重建排除库而恢复已删除的失败数据。
四份 v1.11 master commitment 及更早禁用 commitment 也已写入其中。

## 4. 清理纪律

本轮删除 v1.3–v1.11 的 private custody、构建 stage、Audit 文本/真值、公开
仪式中间件、失败版本专用 runner、policy draft/lock、记录器和诊断脚本。
保留本文、现有历史文档、必要的小型失败回执、压缩排除集合、成功 v1.2
发布树及 Git 历史。

从本轮起，任何实验第一次触发不可恢复失败后必须：先记录运行 ID、失败阶段、
根因、未授权事项、关键哈希和禁止复用边界；再删除失败 payload、缓存、临时
workspace 和该失败版本专用代码。不得以“以后可能审计”为由长期保存大体积
失败产物。只有继续执行未来实验所必需的不可逆排除哈希或小型回执可以保留。

## 5. 实际清理回执

清理完成后，`private_custody/` 和
`reports/step28_v13_identity_transfer/` 均为空，v1.3–v1.11 的九个公开运行根
也已删除。删除前 private custody 合计约 `20,592.23 MiB`（`20.11 GiB`）；
另删除了 19 个公开运行或 preexecution 根、两个 Python cache 根。

失败代码/配置/测试的逐文件清单为
`reports/step28_synthetic_chinese_dataset/failure_records/step28_v13_failed_file_cleanup_manifest_20260803.json`：

- 文件 SHA-256：`c6fbe552a67e533165e43d991ce5f2b2bee312ebf9455139ed61928159dccf19`
- canonical self-hash：`ab7c80ae69af314609759c6da57de3b9959072031be485dd43ddd8d5f7dd333e`
- 已核对后删除：`223` 个文件，共 `3,466,609` bytes

七份小型失败回执已按原字节转存到 `failure_records/`；其 SHA-256 不变。
仓库中仍保留 Git 已跟踪的 47 个 Step28-v13 成功 v1.2 构建/独立审计脚本，
它们不属于失败版本专用代码。大体积身份哈希归档使用 Git LFS；`AGENTS.md`
按用户约定更新但不得加入 Git。

清理重放返回 `PASS_FAILED_RUN_CLEANUP_REPLAY`：915,996 个哈希排序、去重、
格式和 self-hash 全部重算一致；223 个清单成员全部不存在；7 份失败回执
自洽；失败 private/preexecution 根为空；成功 v1.2 根仍存在。随后全仓
`unittest` 运行 381 项，374 项通过、7 项按既有声明跳过、0 项失败，用时
848.857 秒。
