# 可移植独立参考 v4

v1/v2/v3 保存原始实际执行版本。v4 只把源码输入根目录改为相对路径或 `RISK_AUDIT_INPUT` 环境变量，没有改变数学公式、数据夹具、梯度/AdamW 预期值与断言。

默认目录关系为同一审查目录下 `input/`（原附件解压内容）和 `reference/portable_v4/`（本文件）。若另行放置原附件，用 `RISK_AUDIT_INPUT` 指向包含 `scripts/`、`schema/` 的解压根目录。不要把它指向正式数据目录。

## 环境

本次实际版本：Python 3.12.14、NumPy 2.3.5、Torch 2.7.1+cpu、SymPy 1.13.3、mpmath 1.3.0、filelock 3.32.3、fsspec 2026.7.0、Jinja2 3.1.6、MarkupSafe 3.0.3、NetworkX 3.6.1。Torch 及新增依赖来自官方 CPU 索引，完整安装命令和日志见 `../environment_v1/`。无 sentence-transformers/BGE 依赖。

使用已带这些依赖的 Python 即可。新建可重复环境的一种方式：

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install numpy==2.3.5
.venv/bin/python -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.7.1+cpu sympy==1.13.3 mpmath==1.3.0 filelock==3.32.3 fsspec==2026.7.0 jinja2==3.1.6 MarkupSafe==3.0.3 networkx==3.6.1
```

## 完整执行

从审查目录运行下列命令，结果目录必须尚不存在。记录器会明确禁用 CUDA，并把线程限制为 1，完整保存命令、环境、stdout、stderr、退出码和哈希。

```bash
.venv/bin/python reference/portable_v4/run_recorded.py reference/portable_v4/my_run .venv/bin/python reference/portable_v4/oracle.py --output reference/portable_v4/my_data
```

如果 Torch 通过 `pip --target` 安装到独立目录，可以设置 `RISK_AUDIT_TORCH_PATH` 指向该目录；记录器会保留原 PYTHONPATH 并追加该依赖入口。普通虚拟环境不需要此变量。

八个单缺陷生产源码变异已保存在 `../mutants/`。例如，仅检查阶段 γ=0 变异：

```bash
.venv/bin/python reference/portable_v4/run_recorded.py reference/portable_v4/my_gamma0_run .venv/bin/python reference/portable_v4/oracle.py --module reference/mutants/stage_gamma_zero/step28_risk_replay.py --tests stage --output reference/portable_v4/my_gamma0_data
```

此变异的预期退出码为 1，原模块完整验证预期为 0。现有成功/失败证据不是建议覆盖的输出目录。这里的所有数据是源码中生成的手写 Tiny 夹具；步骤计数重设、参考风险缩放和较低裁剪阈值均是明确人工激活探针，不是正式训练结果。
