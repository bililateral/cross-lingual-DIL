# 分阶段完整梯度外审：主审处置

2026-10-04。用户告知“外审已结束”，通过本地Playwright回收同一会话对review(8).zip的正式报告，并保存随后用户在网页追问“怎么解决呢？”的补充答复。结论：**接受当前分阶段完整梯度的限定实现结论；没有发现必须修正的生产梯度错误；BGE原生资源仍未准入，正式训练未启动。** 本轮仅核查既有代码与证据，没有新增科研计算。

## 原文与身份

- [正式外审全文](external/main_report.zh.txt)、[机器裁决](external/verdict.json)、[审查者执行说明](external/execution.zh.txt)。
- [完整原包](external_review.zip)：1,621,526字节、121成员，SHA-256 `0df9c944872c73cb2073fd01a1e1a9268ca9340816893b981d2a8ff04f59439c`。主执行者用PowerShell/.NET流式复核清单120份载荷，大小和SHA全部匹配；仅展开必要阅读文件，其他证据保留原ZIP。
- [网页本轮正文及后续答复](external_page.json)保存当前review(8)请求至最后回复；补充建议发生在正式ZIP报告之后，不改写该报告的停止结论。
- 上传原包仍为792243字节/96成员、SHA `aab6c2470c76a6d66985246310f0af6bb983e6527e88372778af089d83d90ca6`。网页可见6/Pro；没有独立后台模型身份认证。原文保存后已关闭专用页。

## 独立核对与处置

| 项目 | 核对依据与处置 |
|---|---|
| 完整梯度是否实际接入 | 沿staged_check的update_function追至staged.update、gradient、staged_gradient和共享apply_gradient。支持梯度先形成数值fast，query实际用fast求导，再以全参数VJP补回两个支持群的二阶校正；最后一次clip和一次AdamW。接受，未发现仅拆前向、漏二阶或未调用的伪实现。 |
| 数学顺序与目标 | `Dphi=I-AHs`，故梯度为`gs+beta*(v-Hs.T@(A*v))`。源码先乘固定率再VJP，输入为全部参数，历史项仍为0.1基础损失+0.5完整MSE。原objective/lookahead继续直接参考。未改算法或创新判定。 |
| 图、随机流、None | role_loss诊断转float，每角色fork_rng及固定seed重放；fast在HVP前删除，one_hvp局部图在返回后释放；独立used掩码保留完全unused的None。当前buffer仅支持只读并拒绝检测到的修改，不外推任意可变模型。 |
| 首次夹具错误 | 对照tiny_first与修复后测试：故意unused旧Adam状态保持288，其他到289，触发直接参考既有全状态计数门。分离标量夹具后八项通过，生产梯度与容差未改。缺陷已关闭；原7通过/1ERROR不追改。 |
| 原生失败 | stdout四个一阶done之后仅有hvp_0_start；result堆栈停在one_hvp第一条create_graph=True求导。最终VJP、外层clip/Adam均未发生，成功元更新0。S1资源准入阻断保留，不写成算法效果负结果。 |
| 退出及资源 | result已保存OOM，stderr随后terminate，包装134；峰值区间从首次meta前重置起算，并非单独HVP峰值。不能判定只缺20MiB、唯一碎片根因或所有等价实现不可行。累计126/3600秒不变，本轮新计算0秒。 |

本次主执行者读取实际生产核心、直接参考及共享更新、新增测试与Tiny模型/计数夹具、原生stdout/result、正式外审全文/裁决/执行说明；机械身份核验不替代上述判断。外审新执行的是静态与标准库数学核验，没有Torch、BGE、CUDA或服务器重跑。其解析错误向量支持标量对漏Hessian、错乘率、丢交叉项、只留Gauss–Newton有检出余量；不声称实际执行了四份变异生产程序。

## 现有证据的精确边界

接受外审对原实现报告的以下补充，冻结原件不改：

1. 两步是**沿staged连续两步路径的逐步同起点比较**。每步先direct更新，再恢复该步起点执行staged；第二步起点来自staged第一步。因此验证两次连续staged更新及逐步全状态一致，没有验证两条各自连续推进轨迹的累计漂移。
2. 八项并非全部CUDA：新增完整群比较使用CUDA；标量为CPU float64，旧六项主要CPU。Tiny每群70商品/140文本、microbatch4；原生99商品/198文本、microbatch32。两者均28账号/378关系。
3. Tiny没有注册buffer，不构成非空可变buffer保护的动态覆盖。模型/Adam按容差比较，CPU/CUDA RNG及None掩码精确比较。物理前向次数还需从真实循环与原生日志理解，不能只用返回计数字段作证明。
4. Tiny一次warm后把Adam计数置288，是明示夹具；没有验证真实旧首域完整model/Adam/RNG恢复。正式stage目前仍为直接版，本次仅手写native显式接入staged，formal runner尚未完成。

这些限定没有新增当前生产缺陷，不追加测试或GPU运行。已有MLDG/MER/La-MAML创新限制与未见卖家效果未知继续保留。

## 网页后续建议的独立判断

用户在网页追问“怎么解决呢？”后，审查者建议保持staged，只将支持HVP从两次反向改成`torch.func.jvp(torch.func.grad(loss))`。这是另一种精确求导实现建议，未在项目执行，也未提供已验证生产补丁；不能把询问解决方式理解成授权立即再试。

主执行者核对[PyTorch官方HVP教程](https://docs.pytorch.org/tutorials/intermediate/jacobians_hessians.html#computing-hessian-vector-products)及[2.9版jvp说明](https://docs.pytorch.org/docs/2.9/generated/torch.func.jvp.html)：前向/反向组合有一般内存依据，但算子可能不支持forward AD。教程当前显示2.14，不能冒充服务器2.9.1原生兼容性证据。对于固定随机流和同一局部光滑分支，JVP给出`H(Av)`，Hessian对称时等于所需`H.T(Av)`；Top5/ReLU/abs等实际分支必须另核对，不能只凭公式承诺等价。

若以后决定实现，应只改这一求导环节，并保持None/零掩码、完整MSE/交叉项、角色随机流与外层更新；损失核心和日志/检查需要适应函数变换。它对准了当前构建可微梯度图的失败点，有明确理由，但资源成功、算子覆盖、全部实际梯度尚无证据。本次未改实现、未启动该路线、未扩大预算或正式数据访问；不自动叠加checkpoint/offload等其他路线。
