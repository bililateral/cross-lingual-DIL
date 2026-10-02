# 实际命令、环境与原始流索引

日期均为2026-10-02。UTC加8小时即Asia/Singapore时间。命令使用日志中保留的环境覆盖；这不是建议用户执行正式任务的指令。

```text
PYTHONPATH=scripts:tests
PYTHONDONTWRITEBYTECODE=1
OMP_NUM_THREADS=1
MKL_NUM_THREADS=1
OPENBLAS_NUM_THREADS=1
NUMEXPR_NUM_THREADS=1
CUDA_VISIBLE_DEVICES=
```

|记录|开始 UTC|结束 UTC|墙钟秒|退出码|
|---|---|---|---:|---:|
|`01_identity`|2026-10-02T06:24:53.906768+00:00|2026-10-02T06:24:54.522471+00:00|0.615721|0|
|`02_unittest`|2026-10-02T06:25:27.466793+00:00|2026-10-02T06:26:00.168668+00:00|32.701895|0|
|`03_run_help`|2026-10-02T06:26:11.368988+00:00|2026-10-02T06:26:12.034886+00:00|0.665918|0|
|`04_check_help`|2026-10-02T06:26:12.578353+00:00|2026-10-02T06:26:13.294288+00:00|0.715954|0|
|`05_run_bash_syntax`|2026-10-02T06:26:13.902678+00:00|2026-10-02T06:26:13.906506+00:00|0.003830|0|
|`06_check_bash_syntax`|2026-10-02T06:26:14.553043+00:00|2026-10-02T06:26:14.556823+00:00|0.003784|0|
|`07_math_v1`|2026-10-02T06:32:38.408612+00:00|2026-10-02T06:32:41.182015+00:00|2.773423|0|
|`08_diff_v1`|2026-10-02T06:33:02.373969+00:00|2026-10-02T06:33:03.039785+00:00|0.665834|0|
|`09_statistics_v1`|2026-10-02T06:34:59.564944+00:00|2026-10-02T06:35:08.902696+00:00|9.337771|0|
|`10_path_v1`|2026-10-02T06:37:25.473211+00:00|2026-10-02T06:37:26.038944+00:00|0.565749|1|
|`11_path_v2`|2026-10-02T06:37:39.145755+00:00|2026-10-02T06:37:50.441088+00:00|11.295353|0|
|`12_access_identity_v1`|2026-10-02T06:39:47.382066+00:00|2026-10-02T06:39:49.203695+00:00|1.821649|1|
|`13_diagnose_source_order`|2026-10-02T06:40:05.248791+00:00|2026-10-02T06:40:05.864279+00:00|0.615506|0|
|`14_access_identity_v2`|2026-10-02T06:40:26.181501+00:00|2026-10-02T06:40:28.701497+00:00|2.520014|0|
|`15_final_integrity`|2026-10-02T06:44:11.472919+00:00|2026-10-02T06:44:12.138666+00:00|0.665767|0|

### 01_identity

```bash
python -B /mnt/data/logit_audit/independent/verify_identity.py
```

工作目录：`/mnt/data/logit_audit/submission`。开始：2026-10-02T06:24:53.906768+00:00；结束：2026-10-02T06:24:54.522471+00:00；退出码：0；超时：False。

原 stdout：[stdout.txt](evidence/01_identity/stdout.txt)，1743 字节，SHA-256 `44c594f074f429680a26b39a3fd6139f7a346dd0ccae7bdbc4fce515b71f111a`。

原 stderr：[stderr.txt](evidence/01_identity/stderr.txt)，0 字节，SHA-256 `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`。

### 02_unittest

```bash
python -B -m unittest test_step28_er_weight_contracts -v
```

工作目录：`/mnt/data/logit_audit/submission`。开始：2026-10-02T06:25:27.466793+00:00；结束：2026-10-02T06:26:00.168668+00:00；退出码：0；超时：False。

原 stdout：[stdout.txt](evidence/02_unittest/stdout.txt)，3088 字节，SHA-256 `3f5bf42507a9504e354ae81e5ce4eae424dceb93b3d6a92424ede06981a0f9df`。

原 stderr：[stderr.txt](evidence/02_unittest/stderr.txt)，4602 字节，SHA-256 `77b200bf5024763cb757486cd7e2078a9e36a77f77794ef9444a04697d550a52`。

### 03_run_help

```bash
python -B scripts/step28_er_weight_run.py --help
```

工作目录：`/mnt/data/logit_audit/submission`。开始：2026-10-02T06:26:11.368988+00:00；结束：2026-10-02T06:26:12.034886+00:00；退出码：0；超时：False。

原 stdout：[stdout.txt](evidence/03_run_help/stdout.txt)，520 字节，SHA-256 `671664e11b19efc7390bf2bd68bd903b55a1f558faa19fea670d31984bc851a0`。

原 stderr：[stderr.txt](evidence/03_run_help/stderr.txt)，0 字节，SHA-256 `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`。

### 04_check_help

```bash
python -B scripts/step28_er_weight_check.py --help
```

工作目录：`/mnt/data/logit_audit/submission`。开始：2026-10-02T06:26:12.578353+00:00；结束：2026-10-02T06:26:13.294288+00:00；退出码：0；超时：False。

原 stdout：[stdout.txt](evidence/04_check_help/stdout.txt)，264 字节，SHA-256 `045bebc10b95643cdb02b6b27e5a0e6b6309868da0c2adebbd0a5c534a635a66`。

原 stderr：[stderr.txt](evidence/04_check_help/stderr.txt)，0 字节，SHA-256 `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`。

### 05_run_bash_syntax

```bash
bash -n scripts/run_step28_er_weight_linux_20261001.sh
```

工作目录：`/mnt/data/logit_audit/submission`。开始：2026-10-02T06:26:13.902678+00:00；结束：2026-10-02T06:26:13.906506+00:00；退出码：0；超时：False。

原 stdout：[stdout.txt](evidence/05_run_bash_syntax/stdout.txt)，0 字节，SHA-256 `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`。

原 stderr：[stderr.txt](evidence/05_run_bash_syntax/stderr.txt)，0 字节，SHA-256 `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`。

### 06_check_bash_syntax

```bash
bash -n scripts/run_step28_er_check_linux_20261001.sh
```

工作目录：`/mnt/data/logit_audit/submission`。开始：2026-10-02T06:26:14.553043+00:00；结束：2026-10-02T06:26:14.556823+00:00；退出码：0；超时：False。

原 stdout：[stdout.txt](evidence/06_check_bash_syntax/stdout.txt)，0 字节，SHA-256 `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`。

原 stderr：[stderr.txt](evidence/06_check_bash_syntax/stderr.txt)，0 字节，SHA-256 `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`。

### 07_math_v1

```bash
python -B /mnt/data/logit_audit/independent/math_reference_v1.py
```

工作目录：`/mnt/data/logit_audit/submission`。开始：2026-10-02T06:32:38.408612+00:00；结束：2026-10-02T06:32:41.182015+00:00；退出码：0；超时：False。

原 stdout：[stdout.txt](evidence/07_math_v1/stdout.txt)，10998 字节，SHA-256 `a31cc370c514d23db1aaee4bed57a592697a6e000fbae9b3910ac4e8c23987aa`。

原 stderr：[stderr.txt](evidence/07_math_v1/stderr.txt)，0 字节，SHA-256 `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`。

### 08_diff_v1

```bash
python -B /mnt/data/logit_audit/independent/diff_reference_v1.py
```

工作目录：`/mnt/data/logit_audit/submission`。开始：2026-10-02T06:33:02.373969+00:00；结束：2026-10-02T06:33:03.039785+00:00；退出码：0；超时：False。

原 stdout：[stdout.txt](evidence/08_diff_v1/stdout.txt)，2984 字节，SHA-256 `eba5581e910692f07e572848e729f67de29c350181ed796a1abd0e058fd2c125`。

原 stderr：[stderr.txt](evidence/08_diff_v1/stderr.txt)，0 字节，SHA-256 `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`。

### 09_statistics_v1

```bash
python -B /mnt/data/logit_audit/independent/statistics_reference_v1.py
```

工作目录：`/mnt/data/logit_audit/submission`。开始：2026-10-02T06:34:59.564944+00:00；结束：2026-10-02T06:35:08.902696+00:00；退出码：0；超时：False。

原 stdout：[stdout.txt](evidence/09_statistics_v1/stdout.txt)，5819 字节，SHA-256 `42fef6015b4ee2b5cff08ce72c03af555d3cbceda777fbc427bb376bff8ea667`。

原 stderr：[stderr.txt](evidence/09_statistics_v1/stderr.txt)，0 字节，SHA-256 `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`。

### 10_path_v1

```bash
python -B /mnt/data/logit_audit/independent/path_reference_v1.py
```

工作目录：`/mnt/data/logit_audit/submission`。开始：2026-10-02T06:37:25.473211+00:00；结束：2026-10-02T06:37:26.038944+00:00；退出码：1；超时：False。

原 stdout：[stdout.txt](evidence/10_path_v1/stdout.txt)，0 字节，SHA-256 `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`。

原 stderr：[stderr.txt](evidence/10_path_v1/stderr.txt)，583 字节，SHA-256 `ecdadb7bca76dbe0c1f7341150c16c03e3626e1990e2da0bde35b356be5f113e`。

### 11_path_v2

```bash
python -B /mnt/data/logit_audit/independent/path_reference_v2.py
```

工作目录：`/mnt/data/logit_audit/submission`。开始：2026-10-02T06:37:39.145755+00:00；结束：2026-10-02T06:37:50.441088+00:00；退出码：0；超时：False。

原 stdout：[stdout.txt](evidence/11_path_v2/stdout.txt)，13922 字节，SHA-256 `7d2081077a895c857570334afab7e6ad6d23f8c978df631659a164105e862703`。

原 stderr：[stderr.txt](evidence/11_path_v2/stderr.txt)，0 字节，SHA-256 `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`。

### 12_access_identity_v1

```bash
python -B /mnt/data/logit_audit/independent/access_identity_reference_v1.py
```

工作目录：`/mnt/data/logit_audit/submission`。开始：2026-10-02T06:39:47.382066+00:00；结束：2026-10-02T06:39:49.203695+00:00；退出码：1；超时：False。

原 stdout：[stdout.txt](evidence/12_access_identity_v1/stdout.txt)，0 字节，SHA-256 `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`。

原 stderr：[stderr.txt](evidence/12_access_identity_v1/stderr.txt)，295 字节，SHA-256 `0a3f4f791986666d620ba528149b549a3acd39c89fc75f6a38bb9fcaecf112d3`。

### 13_diagnose_source_order

```bash
python -B /mnt/data/logit_audit/independent/diagnose_source_order.py
```

工作目录：`/mnt/data/logit_audit/submission`。开始：2026-10-02T06:40:05.248791+00:00；结束：2026-10-02T06:40:05.864279+00:00；退出码：0；超时：False。

原 stdout：[stdout.txt](evidence/13_diagnose_source_order/stdout.txt)，3401 字节，SHA-256 `7dc46d3355f6dab11b75f3844234b89b1b19eeb18fa121622debe6a8d45ef442`。

原 stderr：[stderr.txt](evidence/13_diagnose_source_order/stderr.txt)，0 字节，SHA-256 `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`。

### 14_access_identity_v2

```bash
python -B /mnt/data/logit_audit/independent/access_identity_reference_v2.py
```

工作目录：`/mnt/data/logit_audit/submission`。开始：2026-10-02T06:40:26.181501+00:00；结束：2026-10-02T06:40:28.701497+00:00；退出码：0；超时：False。

原 stdout：[stdout.txt](evidence/14_access_identity_v2/stdout.txt)，1899 字节，SHA-256 `626ce7bd686096b5e6555a2d6f9a0fead8a39e1967d019dd98cd2b3efcff4b49`。

原 stderr：[stderr.txt](evidence/14_access_identity_v2/stderr.txt)，0 字节，SHA-256 `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`。

### 15_final_integrity

```bash
python -B /mnt/data/logit_audit/independent/final_integrity_check.py
```

工作目录：`/mnt/data/logit_audit/submission`。开始：2026-10-02T06:44:11.472919+00:00；结束：2026-10-02T06:44:12.138666+00:00；退出码：0；超时：False。

原 stdout：[stdout.txt](evidence/15_final_integrity/stdout.txt)，879 字节，SHA-256 `4e1e0b624e62033c465cf1bf707cfdc1151ec3ba3238415ed0804aaf22b34128`。

原 stderr：[stderr.txt](evidence/15_final_integrity/stderr.txt)，0 字节，SHA-256 `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`。
