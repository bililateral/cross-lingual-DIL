# 调参后H消融实现与核验

母体与科学方案固定为trial3及11月6日审后八配置，范围见[scope](scope.zh.md)。原调参只证明最终排序有限改善，不证明所有组件有效；此次消融正是条件组件对照。原结果报告和旧失败门不修改，用户改写的汇总MD不作证据。

## 实际接线

- `step28_h_ablation.py`：显式current/history角色；ER在无teacher时仍乘alpha；KD历史只走自然分组旧评分保持，不调用历史监督或重组（包括日志）；SEQ不建Memory。NoRegroup用完整B0；MeanPool完整块FP64 mean后一次输出；StandardRank使用原rank_used覆盖四个监督位置；AccountPool先mean记录向量、再同一个匹配头，直接autograd，保存自己的378自然账号教师。Memory全局6群、出生目标不刷新、完整序列化≤1MiB、随机根继承母体。
- `step28_h_ablation_run.py`：正式入口先检查关闭门；准备63身份、源清单、预算/输入/核验绑定；逐阶段864更新、评分恢复、memory类型和实际日志验收。Full仅复用trial3九条正式复验，整批母体/结果处置和27端点身份不全时停止，不自动回退/补训。
- `step28_h_ablation_control.py`：一个固定63条清单，3槽、失败槽整批退休，健康槽继续领取独立剩余任务；全部完整并释放才进入生成/确认/分析。复用history冻结容量与整槽信用，新增实际72GiB可用RAM门，纯poll/resource sleep不计控制费，主动检查仍计费；同桶30秒有限收尾预提继承，不清零失败。
- `step28_h_ablation_evaluate.py`：72条/216端点、432已学域格/17280群前向，三种角色共享raw；全部盲分完整才一次读新pairs，正例数分组在同次保存。全22指标、N/A与合并混淆指标分列，实际域分层5000次共同群索引，唯一Full−ER O/F共同门及初学/新域点保护；其他比较与交互探索性。各seed先平均三顺序后算sample SD。Brier/log-loss的F/F_first方向按低值更优反转，MAP公式不变。

对旧共用文件仅两项兼容接线：三阶段runner计数优先读取新Spec.uses_history、旧spec维持原判定；history dispatch增加默认保持旧slot_admission的可注入admit参数，新控制器实际传入72GiB门。旧正式workspace冻结源码没有改动。

## CPU证据与边界

Linux py310，手写群/TinyEncoder，PyTorch单线程、CUDA_VISIBLE_DEVICES为空；共用的CPU子进程调度例只运行sleep/退出，不做GPU计算。CPU01三项实际通过：独立标量/梯度覆盖15个current/history组合；八配置各两次真实更新与独立AdamW参考、encoder/head变化、保存恢复精确重现；自身教师/六群抽样/出生不刷新/类型；关闭门、RAM/槽信用和真实7个CPU子进程失败槽退休续领。

CPU01评价入口因启动包装遗漏CUBLAS_WORKSPACE_CONFIG的既有数值配置而中止，总15秒和原错误保留；不是科研负结果。仅修包装后CPU02补该入口：216次实际TinyEncoder恢复、17280个手写群前向、1296份22指标/query矩阵，盲完整后一次手写pairs开放；Full−ER相同模型的共同门正确不通过，缺臂被拒，三角色/九比较/交互/5000配对统计及324个正例分组格实际运行，98秒。CPU03只核统计新增低值损失方向、实际域映射与N/A传播，2秒。累计CPU115/1200秒。详细原始回执及实际源码身份保存在cpu01/02/03；最终资格组合这些适用证据，不把最初FAIL改写成PASS。

生成器沿用既有实现及冻结机制，本轮调用0。没有正式文本、真实关系/owners、旧dev标签、新确认生成、BGE载模、GPU核验、正式训练或自动监听。CPU不测三路GPU峰值或完整训练吞吐；用户已取消重复GPU资格，见[gpu_reuse_decision](gpu_reuse_decision.json)。准备门维持关闭，未来正式运行仍须完整身份/资源/授权。

## 正式成本

Full九条/27端点复用；63新增共54432更新、85536梯度群呈现、189新端点。3-worker共享RTX5090 32GB，每槽allocator9.25GiB、整进程/准入10.25GiB；available RAM≥72GiB，总RSS≤64GiB；新增磁盘cap288GiB，初始free≥352GiB，小证据≤2GiB。无需新增硬件。

以已完成HPO三路平均3.35369886累计worker小时/轨迹代理，训练约211.28累计小时，理想满三槽约70.43小时；建议训练累计210—250小时，正式计算按4—5天安排。新分支及本轮确认吞吐尚未实测，资源等待/实施/网页审查另计。硬上限394累计小时＝训练378+确认12+辅助4；每条6小时含60秒关停余量，不跨桶借用、不自动追加/重试。本次实际准备/CPU费用由预算回执承接；原HPO/英文费用不转入本轮。
