# 固定预算联合延长诊断实现

2026-09-10。用户已明确采用[并行计划](SELLER_ALIAS_PARALLEL_PLAN.zh.md)第一路：固定1080更新、train/valid各一次、test/owners不读、一张空闲GPU/一个CPU计算线程、预计45—75分钟、3小时/12GiB上限。本地实现和核验后交网页GPT 6 Pro外审，之后单独汇报Linux启动并暂停。当前没有连接Linux、加载科研模型或解析本阶段正式监督。

## 研究问题与原结果

原表达变化实验已出现受控AP/MAP遗忘与新域学习，但三轮联合在A/C低于单域超过原点估计下限.02。诊断只检验原模型、数据、初始化、损失及学习率下，预定六轮联合是否展示兼顾解，以及静态训练风险与valid排序如何变化。不能保证唯一原因，不改原三轮失败，不因六轮仍失败自动延至九轮，不比较缓解方法排名。[原结果](SELLER_ALIAS_EXPRESSION_RESULT.zh.md)及[GPT 6 Pro原审](../reports/seller_alias_continual/20260910/expression_evaluation/20260910_171540/review_reply.json)保留。

## 实际实现与复用边界

- [入口](../scripts/step28_continual_joint.py)、[政策](../schema/step28_continual_joint_policy.json)、[合同测试](../tests/test_step28_continual_joint_contracts.py)、[Linux启动器](../scripts/run_step28_joint_linux_20260910.sh)均为本阶段新增，原8份冻结来源不修改。新政策钉住原运行manifest和原valid评价JSON的SHA-256；原完整结果门继续核对旧来源、输入、日程、模型记录和分数。
- 模型可见输入、商品重数、成对头、BCE/AdamW、梯度裁剪、train/eval切换沿用已审读的纯文本核心。训练为180群混排六轮；前三轮540次与旧日程逐项核对，后三轮使用同一打乱流生成的后缀。dropout全局步索引从0到1079，Adam从540连续到541，两个窗口首更新均检查编码器/头梯度与实际参数变化。
- 前540次不是加载推理模型后新建Adam续训，而是从同一LaBSE和头初始化重建。第540步的模型、完整Adam与原60群valid分数必须全部精确匹配才进入后三轮。完整状态摘要包括元数据，因此重放使用原point/config元数据；新阶段政策及12GiB资源限制单独写入新manifest并由新Budget执行，不能把元数据中的原22GiB记录误读为本阶段实际授权。
- 三轮/六轮联合各对180个train群与60个valid群评分；三个旧单域各对自己的60个train群及全部60个valid群评分。共840个模型×群组合，每点实际保存重载后全部再评分一遍，共1680次群前向。三个单域的模型状态与valid分数也与原记录精确核对。旧单域推理文件没有Adam，重载记录中的“完整状态”在这些点评价只包含模型，不宣称有单域优化器重放。
- Linux在本阶段一次train解析后，对固定eval模型计算逐群`mean(logaddexp(0, logit) - label * logit)`和AP；前者命名为`static_bce_with_logits`，是稳定的BCE-with-logits，不是评价函数截断概率后的`log_loss`。在线epoch BCE单独保留，不混称静态风险。保存逐群两列数值和群顺序，Windows不再解析train。
- Windows在新完整结果门通过后一次解析valid，计算原22项指标；原四个重放点还必须与已保存的原逐群22指标一致。六轮与三轮、六轮与单域、三轮与单域均同群配对、域等权、5000次bootstrap，保留原种子20260910、95%条件区间和logit阈值0。兼顾判断仍要求每域AP不低于对应单域AP−.02，不能用B域改善抵消A/C。
- 峰值同时存在一份完整训练状态约5.66GB与最终推理权重约1.89GB，其他小型结果另计，资源检查采用12GiB上限。完整重载通过后只清理本次work中的精确`.pt`文件，保留一个六轮推理权重、全部分数、逐群train指标及必要证据；旧模型和原结果不删除。

## 本地核验与尚未验证项

已回读原expression实现外审全文及最新结果外审全文；旧实现审查关于评价常量绑定的缺口已在原合同关闭，不重开。主执行者直接审读`expression_run`、`population`、数据对齐/日程、`population_run.point/retain_inference/Budget`与评价指标函数。

已运行：

```powershell
python -X utf8 -B scripts/step28_continual_joint.py --help
python -X utf8 -B -m unittest discover -s tests -p test_step28_continual_joint_contracts.py -v
python -X utf8 -B -m unittest discover -s tests -p test_step28_continual_expression_run_contracts.py -v
```

新增9项、受影响原合同10项均通过，零跳过。新增检查含独立六轮日程、全局dropout/Adam连续计数的编排夹具、不同Adam/完整分数/域映射/资源/监督反例、手算BCE与AP、原完整门失败时valid解析不可达，以及手工合成CSV的一次完整评价；后者5模型×60群的300个AP与sklearn一致。原10项包含660个夹具AP参考和公开输入边界。夹具测试只证明编排/统计，不冒充真实Torch梯度或LaBSE更新。

另外只读核对了正式公开train/valid输入、原完整结果门、原8份来源哈希、原评价哈希与实际前540步日程；没有打开正式标签CSV、heldout正文或owners。Python使用内存编译检查语法，不写pyc；Linux启动器通过本地Git Bash的`bash -n`检查，未执行启动器。

当前无本地Torch实测、无新GPU运行。沿用不变核心的既有真实LaBSE证据；本阶段的模型/完整Adam精确重现与第541次真实更新检查仍须由获准Linux运行兑现，缺失则不得宣称诊断成立或继续评价。当前网页MCP返回`Transport closed`，新的GPT 6 Pro实现审查尚未提交成功，不能称外审通过。审查材料准备后保留必要清单，连接恢复再提交，不启动训练。

独立只读子agent已审查新实现及直接依赖，未提出确认的科研/复现缺陷。主执行者对其每项结论再对照源码、自己的实际测试与旧manifest核验：540/1080连续状态、train域对应、完整门和计数说明一致；旧联合完整状态5,656,101,955字节与推理权重1,886,935,167字节合计7,543,037,122字节。保留其两项范围提醒：train稳定BCE与valid截断概率log_loss不在极端值下严格相等；Budget统计新run目录，最终磁盘总占用另计外层启动日志，不把run峰值冒充整个启动目录峰值。两项均为解释/汇报边界，不改变实验或追加测试。主执行者未确认需修复的代码错误，尚未实测范围不变。审查子agent已完成并调用停止接口；工具没有关闭会话接口，不声称会话已关闭。

本地记录及待送审背景见[核验记录](../reports/seller_alias_continual/20260910/joint_implementation/verification.json)、[网页连接状态](../reports/seller_alias_continual/20260910/joint_implementation/review.json)和[完整审查请求](../reports/seller_alias_continual/20260910/joint_implementation/review_prompt.txt)。

## 第二路与子agent复核

[方法接口准备](SELLER_ALIAS_MITIGATION_IMPLEMENTATION.zh.md)已完成，并由主执行者全文及对应源码复核。确认旧完整群Memory可配置但超字节拒绝、稀疏历史必须显式边索引、共享端点不产生三元交互、阶段内容固定与随机状态推进要区分。第二路尚未实现、训练或验证新方法；256KiB、378边、1620步ER等仍是候选，不占用第一路监督或预算。
