# 本次修订结果外审：完整上下文入口

对象：20261006_135937，B＋Q＋0.1R固定候选的已完成真实结果。当前时间2026-10-06晚，训练14:05:45—17:55:53结束。不是再次审查“是否允许启动”。请按时间、对象、范围解释旧材料；旧未启动／未授权／待审快照不覆盖现行合同，冻结事实不改。

## 阅读顺序与物理位置

1. 当前问题、用户坚持的核心及纪律：docs/RESEARCH_DISCIPLINE.zh.md、AI_RESEARCH_HANDOFF.zh.md、RESEARCH_DISCUSSION.zh.md、SELLER_ALIAS_RELATION_REVISION.zh.md。研究有限历史原文记忆下新群体学习及旧分布未见卖家的排序／基础识别；只含屏蔽标题／描述，当前为中文合成主题与表达混合三域。28账号群、378对、20正358负、每查询27候选且有正例；各域48拟合/12校准/20开发valid。不是跨语言、真实市场、纯表达因果或同卖家别名追踪；test/owners/Audit/private_custody未开放。
2. 本次冻结合同：relation_revision_result/source/docs/SELLER_ALIAS_RELATION_REVISION_PILOT.zh.md（前面均为reports/seller_alias_continual/20261006/），固定B＋Q＋0.1R、d32、epsilon=.001、s0三个顺序、2592更新/九端点；24小时/24GiB。valid在完整盲门后一次解析，28套完整保存后查看。六项继续条件对LOGIT0.1，失败不自动追加训练或调参。
3. 本次实际来源：relation_revision_result/source/的31文件及authorization.json、job/execution.json；不要拿当前工作区旧runner同名文件或审查夹具代替正式源码。通用状态名relation仍保留，真正九端点是relation_revision_stage1/2/3，正式调用独立绑定revision.update。
4. 直接执行：relation_revision_result/job/、job.wrapper.txt、job.console.txt、inventory.json、admission/。模型9份与Memory本体留Linux，只提供逐项实查大小/SHA；没有附件正文、标签、BGE或权重。初始raw、九点三角色分数、28套矩阵和counts均包含。comparison参考的linux_root为历史服务器路径，本包实际对应reference.local_root；没有修改冻结JSON来让路径看似相同。
5. 主要对手：reports/seller_alias_continual/20261004/logit_low_result/job/evaluation下仅本比较所需54矩阵/counts与两份collected，按记录大小/SHA复制核对；不要求collected中本次未依赖的其他历史臂。附LOGIT完成、盲门与访问资格。其权重依法已删除，保存矩阵可复用，无基线重训。
6. 独立核验及主执行解释：relation_revision_result/verification/、result_verify.py、analysis/、result_analyze.py和docs/SELLER_ALIAS_RELATION_REVISION_RESULT.zh.md。独立核验为既有Linux单CPU禁GPU，外层1.12秒、RSS66124KiB；诊断绘图1.10秒、RSS74736KiB。修改仅端点/日志适配及Q＋0.1R组合检查，不重算标签；程序主体不导入训练模块。旧结果报告仅用于描述性历史对照，不新增旧新差值CI或改变验收。

## 已关闭审查的完整原件

history/relation_revision_pilot_external_evidence.zip是209文件原审查证据包，SHA c42e6e0232927ef7c4bc8decc5e9dbebc4889a036cb008ceb88e86a0c1346a6c；208载荷此前已全部核对。它包含完整审查原文、实际网页核验代码和输出，以及input/relation_revision_pilot_review.zip（5427862B、SHA992990d9c1b5f9c5053b6fc675509506ff4d66c5da22247d8c156e818ac2de84）。该input再含history/relation_revision_review.zip（5257725B、SHA480371343ea0be50e4f8fcc614ec1aee5b35b20593151b7fbb3932e75669a9b5），内有305文件的设计背景、旧正式结果、实现来源、前轮审查和K1上下文。按需要解开定位原件，不要求重审所有旧数学。

原生与接入主审report/submission、正式启动环境另直接附上。旧关系目标记忆结果外审及主审也直接附上，旧结果0/6、旧权重清理事实维持。早期risk固定点负结果、group-meta资源不可行关闭、LOGIT0.1当前强基线及用户否决以LOGIT加约束替代核心，不因本次新负结果而改写。

已关闭的实际变化：当前B代替平方当前项，历史新增0.1R，保留Q、32维头、无Dropout、六群Algorithm R和统计迁移。K1用不同历史标签及独立字面总式，在clip前全参数梯度上识别漏R、断R、错权、错标签四反例。新BGE/GPU手写一步、2592次Tiny实际接入、扰动后恢复及候选/基线首域映射已核验；原生三探针不等于全模型独立梯度比较。此轮结果审查复用这些结论，只在新证据出现矛盾时追查直接实现。

## 本次重点与不要混淆的判断

独立保存矩阵核验支持六条件0/6，O/N/Z MAP/AP差值区间均负；主执行还发现首阶段没有改善、额外遗忘区间正、固定0.5当前域第二阶段全漏检。Q在阶段2第一步约13.94—16.18，旧版1.56—2.05；B得到的解与保留平方Q不协调是有代码/日志支撑的解释线索，不是已测得的分量梯度夹角或Q唯一因果贡献。第一步尚未发生第二阶段优化；第一次固化不存在旧统计跨阶段运输。请独立审查这条推断是否过强，以及所有图表／概率误差／群宏与合并counts措辞。

真实原生准入、正式有效性、对具体对手效果、原验收、机制／创新分别裁决。三顺序不是三种子，60实际群条件bootstrap不覆盖重训、开发选择或真实市场外推。旧新版差为描述性历史结果，不能偷换成新前瞻主对手；G局部正均值与N/Z绝对不足并存；N/Z误报降低与漏检增加并存。报告不得把有效负结果泛化为整条核心无效。

本次用户明确要求“把结果同步回来仔细分析”；现行合同包括结果外审，历次明确要求完整上下文。遵循本地Playwright、既有会话、可见6/Pro；后台身份未独立认证。附件只含已授权代码、合成任务的既有小结果与审查证据，无密码、原始正文、真值标签、模型或Memory本体。九权重的此次清理未授权，仍保留；不要求主执行者擅自追加实验、消融、阈值选择或读取真值来满足当前审查。
