# 必要最小补正（建议，未修改受审源码）

## I01：把原生资格绑定到实际固定BGE

位置：`scripts/step28_query_replay_verify.py:84–86,202–208`；`scripts/step28_query_replay_run.py:345–349,636–646`。

在原生现有L操作内，先复用`core.model_files(run.base.model_config(c, 'split_rank'))`，比较`file_count`、`total_size_bytes`、`content_sha256`与`c['models']['split_rank']`。不匹配立即停止且不调用`run.fresh`。匹配才进行已有加载和初始化，将实际archive及已经算出的initial随native结果保存。示意如下；这不是已落地补丁或资格回执：

```python
def checked_fresh():
    archive = core.model_files(run.base.model_config(c, "split_rank"))
    expected = c["models"]["split_rank"]
    keys = ("file_count", "total_size_bytes", "content_sha256")
    if any(archive[k] != expected[k] for k in keys):
        raise ValueError("Pinned BGE archive differs before native loading")
    model, optimizer, initial = run.fresh(c)
    return model, optimizer, initial, archive

model, optimizer, initial, archive = timed("L", checked_fresh)
# 现有native result增加 pretrained_archive=archive、initial_state=initial。
```

正式`validate_gate`在已有检查之外，拒绝native结果缺失archive、三个固定字段不匹配的情况；这一步只读已允许的policy和资格小记录，不读正式数据。正式`train`将现有347–349行移到345行`public_inputs`之前；沿用这一次archive结果供manifest记录，不新增重复模型扫描。

关闭证据仅需人工文件身份测试：正确身份继续；错误／缺失身份拒绝；身份拒绝时替代的模型加载器和公共输入读取函数均未被调用。真实archive和其加载耗时由之后唯一获准原生执行取得，不能用人工身份记录充当真实BGE身份。来源变化保留旧CPU记录并明确增量承接，不重写旧PASS的source_files。

## I02：补已有S/Q共同状态的作用断言

位置：`tests/test_step28_query_replay.py:239–279`。

保存已经执行的S更新行及完整`original`状态。已有Q步之后，比较这两个已存在结果的目标，并复用`component_effects(model, original)`核对encoder/head裁剪后梯度及参数差异。将其另记为`Q_minus_S`，与现有`C_q_minus_Q`分开；可保存最大差值，不只写一个总PASS。

这是已知人工活跃样例的要求，不是“所有输入、所有参数、每一步都必有差”的新门。若目标发生抵消，只修人工样例并保留原因，不放宽公式容差、不引入正式数据。之后原生S与Q两步已经在六步安排中，可复用它们作相应记录，不能再加更新或额外GPU尝试。

## 不新增的动作

不改损失系数／候选分配、不加训练臂、不重开旧C、不额外训练、不开封真值、不建立通用变异平台或跨目录签名状态机、不重复全部未变CPU测试、不预建监听。包内旧CPU累计收费39秒保留；必要增量核验继续消耗原CPU300秒总额。

上述补正完成后仍须主执行者独立处置，才进入原合同规定的原生六步与实测投影。此文不是授权文件、正式gate或已执行补丁。
