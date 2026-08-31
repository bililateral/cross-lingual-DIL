# Step28-v13 v1.13 V9.4.1 可复制公开投影 V2 合同

日期：2026-08-31

状态：`FROZEN_LABEL_FREE_FOUR_SPLIT_PUBLIC_PROJECTION_TRAINING_INPUT_READY`

## 目的

Windows 生成无标签公开投影包，用户将包直接复制到 Linux；Linux 不使用 Git、不联网，只计算四分片的 LaBSE 六项分数；结果复制回 Windows 后，与保留的 legacy18、M0/C0 和 identity33 合并。正式输出结构和科研含义不变。

## 保留的必要检查

1. 复制包和返回文件的大小、SHA-256 与行顺序一致，防止复制损坏或错包。
2. Linux 包不得含标签、控制者、成员关系、检索相关性或审核真值。
3. Linux 使用冻结分块规则和 LaBSE，输出必须为四个 `189000×6` 有限矩阵。
4. Windows 合并时，base24 与 identity33 的四拆分行键必须逐字一致。

不再使用 Git 提交检查、历史 V6/V7 结果钉、私钥、一次性签发/消费状态机、隔离工作区、便携清单或重复结果状态。已有 `transfer_manifest.json` 负责检查复制输入，已有 `gpu_return_manifest.json` 负责检查返回结果。Linux 失败后删除不完整返回，可以在同一复制包上修复环境后重跑；不得改变包内数据。

## 三条命令

- Windows：`python -B scripts/step28_v13_v1_13_v9_4_1_public_projection_portable_v2.py prepare-windows`
- Linux：`bash scripts/run_step28_v13_v1_13_v9_4_1_public_projection_portable_v2_linux_20260831.sh`
- Windows：`python -B scripts/step28_v13_v1_13_v9_4_1_public_projection_portable_v2.py finalize-windows`

本合同只允许无标签公开投影，不授权模型训练、阈值选择、审核预测或任何真值读取。

## 正式执行结果

三阶段流程已于 2026-08-31 完成并通过独立 `validate-output`。训练、开发、审核甲、审核乙各有 `189000` 对，共 `756000` 对；正式输出含 36 个文件，共 736,034,647 字节。发布清单规范自哈希为 `1614e70bd84b76c292098d41f8b8aa0a51666c26f73631b943bf42576b9bba28`，Linux 返回清单规范自哈希为 `d1c81b4612c31eb0a58dae584a7257eb380750d14183f883dc7be40964ac2c9b`。

正式输出位于 `reports/step28_model_experiment/v9_4_1_public_projection_v1_20260831/`。中央处理器阶段、复制包、图形处理器返回和临时块／向量均已清理，只保留正式输出及小型图形处理器回执。全程没有读取监督或审核真值，没有更新模型参数或选择阈值。该结果只使无标签公共训练输入就绪；`m0_m1_m2_m3_training_authorized=false`，下一阶段仍须单独闭合训练授权。
