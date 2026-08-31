# Step28-v13 v1.13 V9.4.1 可复制公开投影 V2 合同

日期：2026-08-31

状态：`IMPLEMENTATION_NOT_YET_EXECUTED`

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
