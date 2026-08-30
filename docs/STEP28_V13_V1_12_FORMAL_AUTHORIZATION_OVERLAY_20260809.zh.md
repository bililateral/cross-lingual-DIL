# Step28-v13 v1.12 正式授权补充合同

> **历史关闭声明（2026-08-30）**：V1.12 已永久质量失败，本授权合同和已消费能力均不得复用。其专属策略、脚本和测试已从当前工作树清理，精确字节仅由 Git 历史保留；失败结论和保留边界以终止清理文档为准。

日期：2026-08-09

状态：`AUTHORIZATION_OVERLAY_DRAFT_NO_FORMAL_AUTHORIZATION`

## 1. 适用范围与冻结关系

本文件只补充正式密钥前的授权证据和执行纪律，不修改数据生成、标签、拆分、身份机制、阈值或模型设计。

`docs/STEP28_V13_V1_12_FORMAL_DATASET_BUILD_PLAN_20260803.zh.md` 是 `schema/step28_v13_v1_12_formal_build_draft.json` 按字节固定的原始合同，必须保持 11,174 字节和 SHA-256 `d57d36eae97b4771ab1b9f10961dfd522468b84a0e9f4980c4cc315aa89be375`，不得用本文件覆盖或改写。原始合同与本文件冲突时，仅在“正式授权证据、文本捷径口径、历史测试豁免和预检重跑纪律”四项上以本文件为准；其余仍以原始合同和冻结 draft 为准。

本文件、预锁证据验证器、修改后的预锁器和密钥仪式脚本必须进入正式 source closure。它们不属于最终文本预检回执已经绑定的 23 个成员，不得借此修改那 23 个成员。

## 2. 当前唯一文本捷径结果

当前唯一规范结果为：

`reports/step28_synthetic_chinese_dataset/design_preflights/v1_12_cleanroom_20260803/text_shortcut_preflight_receipt.json`

文件 SHA-256 为 `e371a460d65560f06e071a6961bf67c22d1f39a5e5e3a37431af5b1c88dd152e`，规范自哈希为 `ffe86c9ad07f77609c3463e831108e73504493b591c73815359c0419e58c4a20`，23 个来源成员的规范闭包哈希为 `75c3e940a873a2674300337d0d3f1042fa41626e7e0aa2e97f640744175da2a3`。

原始描述分支在设计训练集和设计开发集各使用 `500 × 378 = 189,000` 对；反事实机制中性文本硬门排除每个世界 6 个登记负例，各使用 `500 × 372 = 186,000` 对。不得把反事实硬门写成 189,000 对，也不得扩展为设计 Audit A/B 已完成文本预检。

第一次运行的外部中断由 `external_interruption_receipt.json` 固定；唯一一次精确替代已经启动并成功，替代资格已经消耗。v1.12 禁止第三次文本预检。若上述 23 个来源成员任一字节变化，当前结果立即失效，必须升新版本重新冻结，不能在 v1.12 下重跑。

## 3. 两层证据的唯一含义

预锁必须同时包含两层，不得任选其一：

- `design_evidence`：只作为旧版 `formal_common` 兼容所需的先决证据，本身不足以授权正式密钥；
- `authorization_evidence`：当前正式密钥授权必须通过的证据，精确包含最终文本捷径回执、外部中断回执和历史清单豁免回执；
- `authorization_overlay_contract`：本文件的文件大小和 SHA-256 固定项。

机器字段必须分别写明：

`LEGACY_COMPATIBILITY_PREREQUISITES_NOT_SUFFICIENT_FOR_SEED_AUTHORIZATION`

和：

`MANDATORY_CURRENT_SEED_AUTHORIZATION_EVIDENCE`

缺少、增加、替换或篡改任何一项都必须关闭授权。预锁写成后必须由复合验证器同时重放旧兼容层和新授权层，不能只调用旧 `formal_common.load_and_validate_prelock()` 就宣称合格。

## 4. 历史测试豁免

全仓测试仍必须真实执行完整发现。唯一允许的非零原始失败是：

`test_step28_v12_application_contracts.Step28V12ApplicationContracts.test_sync_manifest_is_closed_and_hashes_match`

它必须由 `historical_manifest_waiver_receipt.json` 精确绑定，且同时满足：原始失败数恰为 1、错误数为 0、失败完整名称唯一匹配、历史清单/历史测试/当前 `PROJECT_PROGRESS.md` 字节未漂移、当前 Step28-v13 v1.12 定向测试全部通过。任何第二失败、解析歧义、非预期成功、错误、子测试歧义或固定字节漂移都拒绝写入测试通过回执。

兼容投影必须保留原始计数和返回码为权威事实。`skipped_count` 可以等于 `raw_skipped_count`（仅已启动测试中的普通跳过）加 1 个已接受豁免，但必须同时保存 `raw_*` 字段、豁免完整名称、`count_semantics` 和 `status_semantics`，不得把该失败冒充原生跳过或普通通过。

Python `unittest` 在 `setUpClass` 或 `setUpModule` 阶段发生声明跳过时，会把它登记为跳过事件，但不会调用 `startTest`，也不会把它计入 `testsRun`。正式结构化结果必须因此分别保存“已启动测试中的普通跳过”和“未启动的夹具级跳过”，不得把二者相加后伪装成测试总数。当前只允许六个完整名称和原因均固定的 Step28-v11 普通跳过，以及一个完整名称和原因均固定的旧 metadata-shortcut 类级跳过；任何新增、缺失、改名、改原因或跨类别重分类都关闭授权。兼容口径的 `skipped_count` 只等于六个普通跳过加一个历史失败豁免；类级跳过另列，不进入 `testsRun` 或兼容计数恒等式。

## 5. 不可逆执行前验证

密钥仪式必须在写 start receipt、创建私有 stage、读取随机源或产生任何其他持久副作用之前，先完成：

1. 旧预锁验证；
2. 授权补充合同 pin 验证；
3. 三份授权证据的文件 pin、自哈希和语义重放；
4. 最终文本回执 23 个来源成员逐文件复核；
5. 全部门、500+500 世界、189,000/186,000 对、八类跨拆分零交集、可见禁止残留为 0 和七项正式授权为假；
6. 历史豁免与最新完整测试回执一致。

任一步失败时，start receipt 必须不存在，随机源调用次数必须为 0，私有 stage 必须不存在。

首次 start receipt 之前必须重新解引用历史豁免绑定的持续更新文档。start receipt 已存在后的合法恢复只验证冻结预锁、补充合同、豁免回执 pin、自哈希和正式 source closure，不再要求持续更新文档保持首次授权时的字节；否则正常更新科研进度会错误破坏已经消费的一次性恢复。

## 6. 当前授权边界

本文件本身不授权任何不可逆动作。当前正式密钥、正式数据行、审核真值解封和模型训练仍全部为 0/假。只有实现、测试、完整源码闭包提交、最新完整测试回执和网页端最终运行资格审查全部通过后，才能发布只授权一次 seed ceremony 的 prelock。
