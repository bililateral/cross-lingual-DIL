# 本次实际命令及原始输出索引

所有记录日期为2026-10-02。

| 运行标识 | UTC开始→结束 | Asia/Shanghai开始→结束 | 退出码 |
|---|---|---|---|
| 00_verify | 04:00:03.811442 → 04:00:05.802344 | 12:00:03 → 12:00:05 | 0 |
| 01_submitted_unittest | 04:00:15.919727 → 04:00:36.895330 | 12:00:15 → 12:00:36 | 0 |
| 02_run_help | 04:01:21.930874 → 04:01:22.502487 | 12:01:21 → 12:01:22 | 0 |
| 03_cpu_help | 04:01:23.054596 → 04:01:23.735630 | 12:01:23 → 12:01:23 | 0 |
| 04_bash_cpu_syntax | 04:01:24.317265 → 04:01:24.322644 | 12:01:24 → 12:01:24 | 0 |
| 05_bash_run_syntax | 04:01:24.909018 → 04:01:24.915325 | 12:01:24 → 12:01:24 | 0 |
| 06_diff_and_binding | 04:03:44.980213 → 04:03:45.650525 | 12:03:44 → 12:03:45 | 0 |
| 07_independent_gradient_v1 | 04:04:49.782921 → 04:04:53.065696 | 12:04:49 → 12:04:53 | 0 |
| 08_independent_statistics_v1 | 04:10:38.586038 → 04:10:46.307379 | 12:10:38 → 12:10:46 | 1 |
| 09_independent_statistics_v2 | 04:11:06.968651 → 04:11:16.552770 | 12:11:06 → 12:11:16 | 0 |
| 10_independent_paths_v1 | 04:12:43.238698 → 04:12:52.832826 | 12:12:43 → 12:12:52 | 1 |
| 11_independent_paths_v2 | 04:13:17.977663 → 04:13:27.452277 | 12:13:17 → 12:13:27 | 0 |
| 12_final_source_recheck | 04:15:35.266314 → 04:15:35.902000 | 12:15:35 → 12:15:35 | 0 |
| 13_delivery_quality_check | 04:20:48.942003 → 04:20:50.258129 | 12:20:48 → 12:20:50 | 0 |

## 00_verify

```text
{
  "name": "00_verify",
  "command": [
    "python",
    "-B",
    "/mnt/data/er_low_audit/independent/verify_submission.py"
  ],
  "cwd": "/mnt/data/er_low_source",
  "start_utc": "2026-10-02T04:00:03.811442+00:00",
  "end_utc": "2026-10-02T04:00:05.802344+00:00",
  "elapsed_seconds": 1.9909383929999933,
  "exit_code": 0,
  "cpu_affinity": [
    0
  ],
  "environment": {
    "PYTHONPATH": "scripts:tests:/mnt/data/er_low_audit/independent",
    "PYTHONDONTWRITEBYTECODE": "1",
    "CUDA_VISIBLE_DEVICES": "",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "TOKENIZERS_PARALLELISM": "false",
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1"
  }
}
```

完整输出：`logs/00_verify.stdout.log`、`logs/00_verify.stderr.log`。

## 01_submitted_unittest

```text
{
  "name": "01_submitted_unittest",
  "command": [
    "python",
    "-B",
    "-m",
    "unittest",
    "test_step28_er_weight_contracts",
    "-v"
  ],
  "cwd": "/mnt/data/er_low_source",
  "start_utc": "2026-10-02T04:00:15.919727+00:00",
  "end_utc": "2026-10-02T04:00:36.895330+00:00",
  "elapsed_seconds": 20.975639237000024,
  "exit_code": 0,
  "cpu_affinity": [
    0
  ],
  "environment": {
    "PYTHONPATH": "scripts:tests:/mnt/data/er_low_audit/independent",
    "PYTHONDONTWRITEBYTECODE": "1",
    "CUDA_VISIBLE_DEVICES": "",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "TOKENIZERS_PARALLELISM": "false",
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1"
  }
}
```

完整输出：`logs/01_submitted_unittest.stdout.log`、`logs/01_submitted_unittest.stderr.log`。

## 02_run_help

```text
{
  "name": "02_run_help",
  "command": [
    "python",
    "-B",
    "scripts/step28_er_weight_run.py",
    "--help"
  ],
  "cwd": "/mnt/data/er_low_source",
  "start_utc": "2026-10-02T04:01:21.930874+00:00",
  "end_utc": "2026-10-02T04:01:22.502487+00:00",
  "elapsed_seconds": 0.5716769820000422,
  "exit_code": 0,
  "cpu_affinity": [
    0
  ],
  "environment": {
    "PYTHONPATH": "scripts:tests:/mnt/data/er_low_audit/independent",
    "PYTHONDONTWRITEBYTECODE": "1",
    "CUDA_VISIBLE_DEVICES": "",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "TOKENIZERS_PARALLELISM": "false",
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1"
  }
}
```

完整输出：`logs/02_run_help.stdout.log`、`logs/02_run_help.stderr.log`。

## 03_cpu_help

```text
{
  "name": "03_cpu_help",
  "command": [
    "python",
    "-B",
    "scripts/step28_er_weight_check.py",
    "--help"
  ],
  "cwd": "/mnt/data/er_low_source",
  "start_utc": "2026-10-02T04:01:23.054596+00:00",
  "end_utc": "2026-10-02T04:01:23.735630+00:00",
  "elapsed_seconds": 0.6810789379999846,
  "exit_code": 0,
  "cpu_affinity": [
    0
  ],
  "environment": {
    "PYTHONPATH": "scripts:tests:/mnt/data/er_low_audit/independent",
    "PYTHONDONTWRITEBYTECODE": "1",
    "CUDA_VISIBLE_DEVICES": "",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "TOKENIZERS_PARALLELISM": "false",
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1"
  }
}
```

完整输出：`logs/03_cpu_help.stdout.log`、`logs/03_cpu_help.stderr.log`。

## 04_bash_cpu_syntax

```text
{
  "name": "04_bash_cpu_syntax",
  "command": [
    "bash",
    "-n",
    "scripts/run_step28_er_check_linux_20261001.sh"
  ],
  "cwd": "/mnt/data/er_low_source",
  "start_utc": "2026-10-02T04:01:24.317265+00:00",
  "end_utc": "2026-10-02T04:01:24.322644+00:00",
  "elapsed_seconds": 0.0054024500000195985,
  "exit_code": 0,
  "cpu_affinity": [
    0
  ],
  "environment": {
    "PYTHONPATH": "scripts:tests:/mnt/data/er_low_audit/independent",
    "PYTHONDONTWRITEBYTECODE": "1",
    "CUDA_VISIBLE_DEVICES": "",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "TOKENIZERS_PARALLELISM": "false",
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1"
  }
}
```

完整输出：`logs/04_bash_cpu_syntax.stdout.log`、`logs/04_bash_cpu_syntax.stderr.log`。

## 05_bash_run_syntax

```text
{
  "name": "05_bash_run_syntax",
  "command": [
    "bash",
    "-n",
    "scripts/run_step28_er_weight_linux_20261001.sh"
  ],
  "cwd": "/mnt/data/er_low_source",
  "start_utc": "2026-10-02T04:01:24.909018+00:00",
  "end_utc": "2026-10-02T04:01:24.915325+00:00",
  "elapsed_seconds": 0.006325612000011915,
  "exit_code": 0,
  "cpu_affinity": [
    0
  ],
  "environment": {
    "PYTHONPATH": "scripts:tests:/mnt/data/er_low_audit/independent",
    "PYTHONDONTWRITEBYTECODE": "1",
    "CUDA_VISIBLE_DEVICES": "",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "TOKENIZERS_PARALLELISM": "false",
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1"
  }
}
```

完整输出：`logs/05_bash_run_syntax.stdout.log`、`logs/05_bash_run_syntax.stderr.log`。

## 06_diff_and_binding

```text
{
  "name": "06_diff_and_binding",
  "command": [
    "python",
    "-B",
    "/mnt/data/er_low_audit/independent/verify_diff_and_binding.py"
  ],
  "cwd": "/mnt/data/er_low_source",
  "start_utc": "2026-10-02T04:03:44.980213+00:00",
  "end_utc": "2026-10-02T04:03:45.650525+00:00",
  "elapsed_seconds": 0.6705690379999965,
  "exit_code": 0,
  "cpu_affinity": [
    0
  ],
  "environment": {
    "PYTHONPATH": "scripts:tests:/mnt/data/er_low_audit/independent",
    "PYTHONDONTWRITEBYTECODE": "1",
    "CUDA_VISIBLE_DEVICES": "",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "TOKENIZERS_PARALLELISM": "false",
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1"
  }
}
```

完整输出：`logs/06_diff_and_binding.stdout.log`、`logs/06_diff_and_binding.stderr.log`。

## 07_independent_gradient_v1

```text
{
  "name": "07_independent_gradient_v1",
  "command": [
    "python",
    "-B",
    "/mnt/data/er_low_audit/independent/test_independent_gradient.py"
  ],
  "cwd": "/mnt/data/er_low_source",
  "start_utc": "2026-10-02T04:04:49.782921+00:00",
  "end_utc": "2026-10-02T04:04:53.065696+00:00",
  "elapsed_seconds": 3.2828209970000444,
  "exit_code": 0,
  "cpu_affinity": [
    0
  ],
  "environment": {
    "PYTHONPATH": "scripts:tests:/mnt/data/er_low_audit/independent",
    "PYTHONDONTWRITEBYTECODE": "1",
    "CUDA_VISIBLE_DEVICES": "",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "TOKENIZERS_PARALLELISM": "false",
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1"
  }
}
```

完整输出：`logs/07_independent_gradient_v1.stdout.log`、`logs/07_independent_gradient_v1.stderr.log`。

## 08_independent_statistics_v1

```text
{
  "name": "08_independent_statistics_v1",
  "command": [
    "python",
    "-B",
    "-m",
    "unittest",
    "test_independent_statistics",
    "-v"
  ],
  "cwd": "/mnt/data/er_low_source",
  "start_utc": "2026-10-02T04:10:38.586038+00:00",
  "end_utc": "2026-10-02T04:10:46.307379+00:00",
  "elapsed_seconds": 7.721382409000057,
  "exit_code": 1,
  "cpu_affinity": [
    0
  ],
  "environment": {
    "PYTHONPATH": "scripts:tests:/mnt/data/er_low_audit/independent",
    "PYTHONDONTWRITEBYTECODE": "1",
    "CUDA_VISIBLE_DEVICES": "",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "TOKENIZERS_PARALLELISM": "false",
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1"
  }
}
```

完整输出：`logs/08_independent_statistics_v1.stdout.log`、`logs/08_independent_statistics_v1.stderr.log`。

## 09_independent_statistics_v2

```text
{
  "name": "09_independent_statistics_v2",
  "command": [
    "python",
    "-B",
    "-m",
    "unittest",
    "test_independent_statistics_v2",
    "-v"
  ],
  "cwd": "/mnt/data/er_low_source",
  "start_utc": "2026-10-02T04:11:06.968651+00:00",
  "end_utc": "2026-10-02T04:11:16.552770+00:00",
  "elapsed_seconds": 9.584152348999964,
  "exit_code": 0,
  "cpu_affinity": [
    0
  ],
  "environment": {
    "PYTHONPATH": "scripts:tests:/mnt/data/er_low_audit/independent",
    "PYTHONDONTWRITEBYTECODE": "1",
    "CUDA_VISIBLE_DEVICES": "",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "TOKENIZERS_PARALLELISM": "false",
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1"
  }
}
```

完整输出：`logs/09_independent_statistics_v2.stdout.log`、`logs/09_independent_statistics_v2.stderr.log`。

## 10_independent_paths_v1

```text
{
  "name": "10_independent_paths_v1",
  "command": [
    "python",
    "-B",
    "-m",
    "unittest",
    "test_independent_paths",
    "-v"
  ],
  "cwd": "/mnt/data/er_low_source",
  "start_utc": "2026-10-02T04:12:43.238698+00:00",
  "end_utc": "2026-10-02T04:12:52.832826+00:00",
  "elapsed_seconds": 9.594163966999986,
  "exit_code": 1,
  "cpu_affinity": [
    0
  ],
  "environment": {
    "PYTHONPATH": "scripts:tests:/mnt/data/er_low_audit/independent",
    "PYTHONDONTWRITEBYTECODE": "1",
    "CUDA_VISIBLE_DEVICES": "",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "TOKENIZERS_PARALLELISM": "false",
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1"
  }
}
```

完整输出：`logs/10_independent_paths_v1.stdout.log`、`logs/10_independent_paths_v1.stderr.log`。

## 11_independent_paths_v2

```text
{
  "name": "11_independent_paths_v2",
  "command": [
    "python",
    "-B",
    "-m",
    "unittest",
    "test_independent_paths_v2",
    "-v"
  ],
  "cwd": "/mnt/data/er_low_source",
  "start_utc": "2026-10-02T04:13:17.977663+00:00",
  "end_utc": "2026-10-02T04:13:27.452277+00:00",
  "elapsed_seconds": 9.474667820000036,
  "exit_code": 0,
  "cpu_affinity": [
    0
  ],
  "environment": {
    "PYTHONPATH": "scripts:tests:/mnt/data/er_low_audit/independent",
    "PYTHONDONTWRITEBYTECODE": "1",
    "CUDA_VISIBLE_DEVICES": "",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "TOKENIZERS_PARALLELISM": "false",
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1"
  }
}
```

完整输出：`logs/11_independent_paths_v2.stdout.log`、`logs/11_independent_paths_v2.stderr.log`。

## 12_final_source_recheck

```text
{
  "name": "12_final_source_recheck",
  "command": [
    "python",
    "-B",
    "/mnt/data/er_low_audit/independent/final_source_recheck.py"
  ],
  "cwd": "/mnt/data/er_low_source",
  "start_utc": "2026-10-02T04:15:35.266314+00:00",
  "end_utc": "2026-10-02T04:15:35.902000+00:00",
  "elapsed_seconds": 0.6357284240000354,
  "exit_code": 0,
  "cpu_affinity": [
    0
  ],
  "environment": {
    "PYTHONPATH": "scripts:tests:/mnt/data/er_low_audit/independent",
    "PYTHONDONTWRITEBYTECODE": "1",
    "CUDA_VISIBLE_DEVICES": "",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "TOKENIZERS_PARALLELISM": "false",
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1"
  }
}
```

完整输出：`logs/12_final_source_recheck.stdout.log`、`logs/12_final_source_recheck.stderr.log`。

## 13_delivery_quality_check

```text
{
  "name": "13_delivery_quality_check",
  "command": [
    "python",
    "-B",
    "/mnt/data/er_low_audit/independent/delivery_quality_check.py"
  ],
  "cwd": "/mnt/data/er_low_source",
  "start_utc": "2026-10-02T04:20:48.942003+00:00",
  "end_utc": "2026-10-02T04:20:50.258129+00:00",
  "elapsed_seconds": 1.3161595949998173,
  "exit_code": 0,
  "cpu_affinity": [
    0
  ],
  "environment": {
    "PYTHONPATH": "scripts:tests:/mnt/data/er_low_audit/independent",
    "PYTHONDONTWRITEBYTECODE": "1",
    "CUDA_VISIBLE_DEVICES": "",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "TOKENIZERS_PARALLELISM": "false",
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1"
  }
}
```

完整输出：`logs/13_delivery_quality_check.stdout.log`、`logs/13_delivery_quality_check.stderr.log`。