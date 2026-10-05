# 关系目标记忆：正式接入增量外审独立证据

先读根目录的 `SELLER_ALIAS_RELATION_MEMORY.pilot_review.zh.md`。结论为“最小修复后运行”，范围只限本次固定单候选正式接入。

## 内容

- `input_identity.json`：本次 review(10).zip 的319,319字节/45成员/SHA及44载荷、23来源、原生3来源核对。
- `source/`：23项CPU源清单的原字节文件，另附必要的提交CPU/GPU记录、合同处置和原manifest。这个目录是审查快照，**不是可直接执行的完整项目**；没有正式文本、标签、缓存、真实LOGIT矩阵、权重或凭据。
- `results/`：外审本次实际取得的独立小例结果。不是项目Linux、PyTorch或BGE实测。
- `original_checks/`：取得上述结果的原始核验脚本；原路径保留以便查验。
- `checks/`：仅把源目录/输出目录改为便携路径的副本；`portability_edits.json`列出前后身份。没有改动被审生产函数体或source文件。

## 安全的独立重放

仅需已有Python及NumPy。这些小例不会运行正式CLI、execute、train、载模、GPU或正式标签loader。不要执行 `source/scripts/` 中的正式运行或原生核验入口，也无需安装Torch。

在本包根目录依次执行：

```bash
python checks/recheck_state_repairs.py
python checks/check_endpoint_metrics.py
python checks/independent_checks.py
python checks/finalize_budget_check.py
python checks/check_resource_edges.py
```

默认源目录是本包 `source/`，输出写入各自 `replay_results/` 子目录；原始 `results/` 不覆盖。首次之后再次重放finalize小例，应使用一个新的 `RELATION_REVIEW_OUTPUT` 路径，因为该小例刻意拒绝覆盖既存合成夹具。需要时可用 `RELATION_REVIEW_SOURCE_ROOT` 指定另一份完整解包快照。

资源控制小例只执行Budget/watchdog的真实AST，torch/psutil为明确的查询替身（CUDA未初始化、RSS=0），检验文件名与异常分支；绝非Torch/GPU核验。finalize小例使用真实指标/文件验证和未改函数体，仅参考路径/来源提供者指向合法手写矩阵。其既存超时账是合成反例，不代表实际耗尽过24小时。

便携副本路径不同，生成评价文件中的绝对参考路径及其大小/SHA可能不同；应核对状态、六项条件、拒绝/回滚和数值结论，而非要求这些路径相关输出的字节身份相同。

## 已证实的必要接入问题

1. 完整28套形成后，独立finalize不继承原作业剩余预算/资源失败资格，仍可输出COMPLETE与worth=true。
2. 正式失败资源快照不完整，watchdog写失败账再出错时会跳过退出；停止与留证应最小闭合。
3. job有后缀时控制台路径计量不同；一行统一路径或强制无后缀名称即可消除条件。

低秩盲区、单群/联合目标差异、迁移近似继续作为已知局限。配置收益不是机制创新证据。没有新数据/预算授权，没有正式训练，没有自动后续匹配重放。
