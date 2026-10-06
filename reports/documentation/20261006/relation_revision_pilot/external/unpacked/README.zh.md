# 独立增量外审交付说明

首先阅读 `RELATION_REVISION_PILOT.external_review.zh.md`。裁决为：可按本轮已经批准的固定候选范围正式运行；实际项目gate仍须主执行者签发，本包没有给予新的数据、训练或重试权限。

## 文件组织

- `input/relation_revision_pilot_review.zip`：用户本次原附件，原字节保留；其中 `history/relation_revision_review.zip` 包含完整历史原件。
- `sources/current/`：供直接查看的本轮文本、源码、合同与证据副本。为避免再次存储5.26MB同一历史ZIP，此镜像没有重复复制 `history/relation_revision_review.zip`；完整原本在input中。运行00工具可补回完整current/history树。
- `code/01_*`：外层和嵌套载荷、源码身份和网页环境；`code/02_*`：实际网页CPU测试；`code/03_*`：真实导入、全局绑定、gate分支和部署／回传身份；`code/04_*`：独立符号保存矩阵路由／统计；`code/05_*`：一次尝试和资源失败保护；`code/06_*`：最终AST／Bash语法和源字节检查。
- `outputs/`：实际对应的输出。`04_fixture/`内全部是手写符号指标及合法计数夹具，绝不是新正式实验结果，也不声称由未提供标签重新计算MAP/AP。
- 带 `.failed_v*` 的源码和输出是网页审查工具自身的失败原件；修正原因见报告§10。它们不是项目失败，更不是隐藏的正式重跑。
- `manifest.json`：交付ZIP中除自身之外每个文件的字节数和SHA-256，按路径唯一列出。

## 证据边界

项目LinuxCPU/GPU回执与网页运行分开。网页本次有Torch2.10.0+cpu，实际通过K1、两项新接入测试及一项资源保护用例；没有BGE/transformers/sentence-transformers/CUDA，没有连接项目Linux，没有真实文本、逐对标签、正式权重或缓存的读取。

gate测试用的批准形状只是本地临时 `fixture_only` schema对象，已删除；并未签发可用于项目运行的gate。

## 离线复核

只核对输入和源文件，不执行任何训练，可在本包目录运行：

```bash
python -B code/00_prepare_sources.py
python -B code/01_identity_environment.py
```

00只安全解压当前原包和历史ZIP，不运行项目代码。`sources/current/manifest.json` 是用户本轮清单；外层本交付manifest是另一份清单。

为保留随附输出，重新执行审查脚本前先在包的**副本**中把 `outputs` 改名为 `outputs_delivered`，再创建空 `outputs`。随后03、04、05、06可分别复核；04会主动拒绝覆盖已经存在的夹具目录。它们不会运行正式execute或原生native。

02执行真实微型CPU更新，须使用已经具备torch/numpy/psutil的环境；不下载包、不安装依赖。该步骤不是核对清单的必要条件，也不是项目Linux运行命令：

```bash
timeout --signal=TERM --kill-after=5 240 python -B code/02_run_web_cpu_tests.py
```

不要运行本包 `scripts/run_*linux*`、不要把手写gate或人工基线当作正式资产。项目原生／正式运行仍遵守用户本轮新许可和主执行者审核流程。
