# 表达变化合成数据生成结果

日期：2026-09-10。按用户确认的[重设计方案](SELLER_ALIAS_CONTINUAL_REDESIGN.zh.md)，新数据已一次生成并通过全量独立核验；没有依据模型成绩重选数据。它是独立新研究，旧数据和有效负结果继续保留。

运行目录：[20260910_150500](../reports/seller_alias_continual/20260910/expression_generation/20260910_150500/)。生成及全量验证耗时8.174188秒，退出0；17个数据文件29,195,853字节，含日志18文件29,221,691字节，不含外层执行回执。执行回执登记的18文件均再次核对大小和SHA-256一致。manifest SHA-256为`3f14f1ed47bffabaec90640b9e212f5eab2d702efcf9304a310f22d2a0165f6b`；validation为`70b0cf6016f37a1e7190c3a9f569efaa464c33b68b74a7f0fb014c25d590069c`。

| 项目 | 数量 |
|---|---:|
| 三域账号群 | 360 |
| 合成控制者 | 4,320 |
| 账号 | 10,080 |
| 商品记录 | 36,368 |
| 完整账号对 | 136,080 |
| 同控正例／异控负例 | 7,200／128,880 |

每域train／valid／test分别60／20／40群，28账号／群；商品记录按分区合计18,181／5,922／12,265。生成阶段获准构造、核对本套新合成归属和二值关系；这不授权访问旧封存数据，也不等同于以后训练／valid评价权限已消费。

公共覆盖概率按标题／组织／服务表达三轴为A(.1,.6,.3)、B(.3,.1,.6)、C(.6,.3,.1)。train实际落入公共子集比例分别为A(.3393,.6940,.4810)、B(.4518,.3250,.7042)、C(.7095,.4649,.3298)。落入子集包含自然抽中，不能当作覆盖硬币发生比例。正文独立解码及各域／分区同控、异控一致率完整保存在[validation.json](../reports/seller_alias_continual/20260910/expression_generation/20260910_150500/data/validation.json)。题材总边际保持预定循环；ID隔离、配对、正文结构、数量和预算检查通过。此次精确商品正文／账号文本集合重复检测为0；这不证明不存在共享生成规律或真实域外泛化。

生成前Windows16项定向合同通过、0跳过（3.380秒）；外审后新增理论常数不匹配拒绝用例单独1项通过、0跳过（2.201秒），没有称17项整套重复执行。实际网页审查模型`gpt-5-6-thinking`，自述GPT-5.6 Sol，系用户“暂时用网页端目前选中的模型”例外；完整回复已读，见[原文](../reports/seller_alias_continual/20260910/expression_design/review_reply.json)及[处置](../reports/seller_alias_continual/20260910/expression_design/review.json)。没有确认生成科学阻断；两个未来训练细节已在[首轮实现](SELLER_ALIAS_EXPRESSION_PILOT.zh.md)落实。

后续只读新train／valid公开正文，用现有LaBSE tokenizer实际检查24,103条记录：均不超过256 token，含特殊token最大96；未加载模型权重或读取监督。详见[token长度记录](../reports/seller_alias_continual/20260910/expression_implementation/token_lengths.json)。这项检查不替代实际GPU梯度、状态和完整分数重载验证。

本结果只证明固定种子新数据符合已批准的生成合同；尚不证明各域可学、正常续训遗忘、缓解方法有效或真实中文暗网具有相同变化频率。没有连接Linux、训练模型或使用GPU。
