# 本次修订外审交付与复算说明

主报告为 `SELLER_ALIAS_RELATION_REVISION.external_review.zh.md`。裁决是：**保留定义与生产实现，最小补强关键核验后推进**；没有新原生BGE准入、正式效果或训练授权。

## 文件与身份

- `input/relation_revision_review.zip`：用户本次授权输入的原始字节，5,257,725字节，SHA-256 `480371343ea0be50e4f8fcc614ec1aee5b35b20593151b7fbb3932e75669a9b5`。
- `input/prior_result_frozen_source/`：此前已审结果包的23份冻结来源，供本次跨轮来源比较复查；来自上轮解压副本，不是从本轮同名文件替代复制。
- `input_provenance.json`：上述输入来源和逐文件身份。
- `revision_identity_audit/`：本次整体ZIP、根manifest304载荷和解压字节的实际检查。
- `revision_math_audit/`：人工28账号群的独立NumPy／标准库公式、有限差分与反例；含原stdout和空stderr。
- `revision_code_audit/`、`revision_lifecycle_audit/`：实际AST／来源闭包检查、合法标签排列和首次Adam手算；未执行Torch故障变异。
- `revision_resource_audit/`：实际提交资源记录核对及监督器AST受控分支检查；Popen／psutil／时间替身的结果不是真实Linux停机测试。
- `primary_sources.zh.md`：本轮查证的一手论文URL、节／公式；不附长段论文抓取内容。
- 根 `manifest.json`：本交付包全部其他文件的大小和SHA-256；manifest自身不自包含。

所有计算在网页工作区完成。Python3.12.14、NumPy2.3.5，无Torch；没有BGE／GPU／项目Linux／正式数据访问。输入包中的Linux五测记录只是提交证据，已核对来源及回传，不宣称网页重跑。

## 可直接复算的两项

在空目录解压本交付ZIP，再将 `input/relation_revision_review.zip` 解压到自选目录，例如 `input/extracted_revision`。该输入zip已检查路径安全，但仍建议使用常规解压工具，避免覆盖其他工作。

身份检查（标准库）：

```bash
python revision_identity_audit/verify_revision_input.py \
  --zip input/relation_revision_review.zip \
  --root input/extracted_revision \
  --output recomputed/identity.json
```

公式和人工例（仅NumPy，输出到新目录）：

```bash
python revision_math_audit/independent_revision_math.py \
  --output-dir recomputed/math
```

这些命令是复算说明，本文没有替主执行者再次执行或消耗项目Linux预算。原网页实际结果保持在原路径，复算输出应另存，勿覆盖原件。

## 其他实际检查脚本的路径映射

为保留实际执行源码，以下脚本未追改其绝对输入路径。移机运行时，在工作副本中映射这些路径即可：

| 原路径或变量 | 移机对应 |
|---|---|
| `/workspace/scratch/f4d639b3b473/relation_revision_review`；`ROOT`或`root` | 本次输入ZIP的解压根 |
| `source_and_coverage.py`的`PRIOR` | 本交付 `input/prior_result_frozen_source` |
| `check_supervisor_control.py`的`SOURCE` | 本次解压根 `scripts/step28_relation_revision_verify.py` |
| 各脚本`OUT=Path(__file__).parent` | 放工作副本的新输出目录，避免覆盖本次原证据 |

`static_and_optimizer_probe.py`将JSON写stdout，应重定向到新文件。其他检查脚本按源码写JSON；其原输出已经随包保存。`source_and_coverage.py`只读取和解析项目源码，不导入项目模块。监督器分支脚本有意让受控子例返回FAIL／OSError，以验证真实`main`分支；这不是项目五测失败账。

本交付没有提供已运行的Torch修复用例，也没有生产代码补丁。主报告K1是具体补强要求及理由，执行和采纳由主执行者按实际授权独立决定。

## 清单核对

解压后，对根manifest中每个路径的原始字节计算SHA-256并核对bytes；清单恰覆盖所有其他文件。输入ZIP有自己的根manifest，两个清单职责不同。历史上下文中的旧context_inventory继续按原时点理解。
