# 增量外审：同一目标的分阶段精确梯度及单次资源尝试

延续前轮会话。用户最新原话：“保持当前算法的数学定义，把完整梯度分阶段计算，避免同时保存多个群的大计算图。尝试一下”。已按此仅实现一条staged路线，保持原四群/全参数/固定点/FP32/eager/microbatch32/28GiB/RSS128GiB。新消费25秒，前轮101秒累计126秒；无正式数据/旧权重，无效果结果。没有切换offload或新机制。

请先核对review_manifest.json与26来源、Linux运行快照，读取docs/SELLER_ALIAS_GROUP_META_STAGED.zh.md及本报告implementation.zh.md；再沿实际入口检查staged_check→原native夹具的update_function→staged.update→gradient→staged_gradient→共享apply_gradient。直接objective/lookahead未改，只提取外层更新公共主体；原GPU夹具仅加显式更新函数入口。

重点审查：

1. 分阶段是否真实得到gs+beta*(v-Hs.T@(A*v))，而非只拆前向或漏掉二阶链；全参数交叉项、完整MSE、随机重放、buffer、图释放和None语义是否正确。具体已证错误和通用风险分开。
2. 独立检查新增测试的检出能力与两步模型/Adam比较。tiny_first的测试夹具错误已留证；随后使用原批准的一次同定义技术修复额度将unused旧矩检查分离，未改算法或容差。不要把修复前7通过1ERROR变成8通过，也不要把微型通过当作BGE通过。
3. 原生stdout已完成四群一阶，在hvp_0的create_graph=True求导OOM。结果已保存后退出出现terminate，包装134；零成功元更新。核对是否支持“越过旧失败点，但当前单群二阶资源仍未通过”的限定结论，哪些内存根因仍未知。
4. 本轮明确停止，不要求再设计另一优化/算法、不强制四对照或重训基线。若发现当前实现具体错误，给最小修复及影响；合理建议不自动升级为继续尝试授权。原创新与正式准入边界保持。

请交付中文主报告、机器裁决、实际执行说明及必要证据ZIP。可用小型独立数学验证，不冒称执行BGE/CUDA或连接服务器。原始上一轮输入包、全文及主审为直接背景，未变基础体系不递归重审。包内无正式文本/标签/缓存、私有身份、模型权重或凭据。
