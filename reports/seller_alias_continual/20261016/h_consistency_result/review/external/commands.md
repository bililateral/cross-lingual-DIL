# 实际命令与可重复运行说明

## 实际执行的必要核算命令

原输入只读解包于 `/mnt/data/hcons_review/input`，返回输出单独写入 `/mnt/data/hcons_review/return`。无项目服务器连接。所有科学核对均消费已保存小证据；没有模型、Memory、正式标签、test/owners/private_custody载荷。

### 1. 原样运行audit.py（一次成功，未改原文件）

实际完整shell保留于 `scripts/run_submitted_audit.sh`。核心命令：

```bash
cd /mnt/data/hcons_review/input
export CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/mnt/data/hcons_review/mplconfig
/usr/bin/time -v -o /mnt/data/hcons_review/return/logs/submitted_audit.time.txt \
  python audit.py --workspace evidence \
  --output /mnt/data/hcons_review/return/reviewer_analysis \
  > /mnt/data/hcons_review/return/logs/submitted_audit.stdout.txt \
  2> /mnt/data/hcons_review/return/logs/submitted_audit.stderr.txt
```

实际exit0；原脚本副本 `scripts/submitted_audit.py` 与输入audit.py SHA一致。输出内的PDF由原脚本例行生成，未额外作PDF审阅；没有以网页核算冒充项目py310。

### 2. 审查者独立补核

完整成功shell是 `scripts/run_reviewer_saved_checks.sh`：

```bash
export CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
/usr/bin/time -v -o /mnt/data/hcons_review/return/logs/reviewer_saved_checks.time.txt \
  python /mnt/data/hcons_review/return/scripts/reviewer_saved_checks.py \
  --input /mnt/data/hcons_review/input \
  --output /mnt/data/hcons_review/return/reviewer_independent_v3 \
  > /mnt/data/hcons_review/return/logs/reviewer_saved_checks.stdout.txt \
  2> /mnt/data/hcons_review/return/logs/reviewer_saved_checks.stderr.txt
```

最终exit0。v1/v2同入口在当时不同源码版本及不同输出目录运行，输出目录分别 `reviewer_independent` / `reviewer_independent_v2`，两次exit1。对应原版本代码与当时shell/日志、补丁均在failures各子目录；不能拿最终脚本配历史shell当成执行了历史版本。两个失败输出目录为空，因此ZIP不列空目录；失败完整信息在failure快照而非伪造部分成功结果。

临时计数字典诊断实际源码在 `failures/reviewer_v2/diagnose_count_comparison.py`，有stdout/stderr/JSON。更早的stdin诊断max(empty)异常已从实际工具记录转存 `exploratory_empty_max.py` 与 `.record.txt`，未为生成失败原件再运行。

### 3. 正文表格核对

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 \
python /mnt/data/hcons_review/return/scripts/check_report_tables.py \
  --input /mnt/data/hcons_review/input \
  --return-root /mnt/data/hcons_review/return \
  > /mnt/data/hcons_review/return/logs/report_tables.stdout.txt \
  2> /mnt/data/hcons_review/return/logs/report_tables.stderr.txt
```

首次所有具体值检查完成后错误计数断言退出1；源/日志保留。随后只读诊断通过compile/exec执行该源码、仅把最后计数断言替换成print实际计数，产生471单元/108行诊断原件；再按diff改为准确覆盖计数，最终exit0。没有更改任何统计规则/数值阈值，未单独计时。

### 4. 末次输入身份复核

```bash
python /mnt/data/hcons_review/return/scripts/verify_input_identity.py \
 --zip /mnt/data/h_cons_result_review_input.zip \
 --extracted /mnt/data/hcons_review/input \
 --output /mnt/data/hcons_review/return/input_identity_final.json \
 > /mnt/data/hcons_review/return/logs/input_identity.stdout.txt \
 2> /mnt/data/hcons_review/return/logs/input_identity.stderr.txt
```

实际exit0。逐成员比较ZIP原字节、清单和当前解包字节；没有重新提取或改动输入。首次身份核查的结果另存input_identity.json；末次完整可重复源码/日志如上。报告组装 `python .../scripts/build_review_report.py` 和支持记录组装 `python .../scripts/build_supporting_records.py` 均已执行，不产生新的科学统计。

## 在新隔离目录重复本返回包的核算

以下是供接收者复用的等价路径示例，**不是声称已在这组新路径再执行一轮**。须使用空的工作目录；所有输出目录由脚本新建，避免覆盖既有证据。所需依赖仅numpy、matplotlib；原样脚本附带画图，补核脚本不导入torch或生产包。

```bash
# 在本返回包解包根执行；work应为新的空目录。
mkdir -p work/input work/return
python -m zipfile -e input_original.zip work/input
export CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
python scripts/submitted_audit.py --workspace work/input/evidence --output work/return/reviewer_analysis
python scripts/reviewer_saved_checks.py --input work/input --output work/return/reviewer_independent_v3
python scripts/check_report_tables.py --input work/input --return-root work/return
```

核验统计键/数值及N/A位置，不要求墙钟、观察时间或PDF元数据相同。原审计器频数均值和本补核直接索引可能有浮点末位差，实际最大5.55e−16，小于原已用1e−12核对阈值。精确FP32分项、UID/身份和恢复分数检查没有以这个统计容差放宽。

## 日志与计时的诚实范围

`execution_ledger.json`分别记录原项目6045秒和本次隔离执行。只有time日志存在的项有可靠独立墙钟；其余为null，不编造秒数。不把本次审查失败计入项目原CPU/GPU历史账，也不清除其历史失败。

普通只读文本查看/源码定位命令不逐条重放；实际内容阅读范围与带行号节录由reading_scope.md/source_excerpts记录。首次Files两次尝试均无ZIP可解析正文，不产生任何内容推断。终端清理TERM消息和一次只读grep无匹配与科学核对失败分开记录。
