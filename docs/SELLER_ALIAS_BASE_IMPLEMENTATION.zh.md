# 基础识别模型实现与核验

## 9月18日用户要求的实质复核

用户要求重新确认实际实现是三个基础识别模型的训练，不能用配置、测试和审查记录互相自洽代替科学逻辑。主执行者因此在12:59将自己的新监听PID102561临时SIGSTOP，并再次确认进程为T状态、旧PID45825不存在、两实验launch/job均不存在。下面12:14放行的记录保留为此前状态；本次补充审查完成前未恢复监听；现已完成下述核对并于13:33恢复同一进程。

**复查发现并修复了真实运行缺陷，原“实现已核验”的证据范围不够。**原入口只查旧版`pooling_mode_*`布尔字段，但服务器SentenceTransformers 5.6.0实际原生模块使用`pooling_mode='cls'/'mean'`。真实LaBSE在`load_model`中被误拒绝，进程6.67秒退出1，早于任何正式标签和参数更新；首次失败、服务器实际模块源码和根因记录均保留在[native_audit](../reports/seller_alias_continual/20260918/base_implementation/native_audit/initial_failure.json)。此前微型编码器绕过实际加载器，19项及7项独立检查没有覆盖此缺口。它是执行兼容性缺陷，不是原生池化前向错误，也不使旧有效实验作废。

最小修复为`pooling_modes`同时读取当前字符串／序列表示与旧布尔字段；实际模型仍须单一预定CLS或mean、维度及Dense模块一致。没有修改实际池化、监督目标、模型、数据、日程、学习率、阈值、预算或主要门。仅入口和其合同测试两份生产来源变化，其余七份不变。原始R1/R2修复保留。

### 从批准目的追踪到实际计算

| 用户要求与疑点 | 亲自追踪到的实际实现 | 能成立的结论与限制 |
|---|---|---|
| 训练三个基础识别模型 | `run`遍历LaBSE／E5／BGE三臂；每次`train_arm`重新调用`load_model`与`make_optimizer`，读取各自通用预训练归档并新建分类头 | 三次独立任务微调；没有继承上一个臂的参数或旧任务checkpoint，不是三模型集成或从零预训练语言模型 |
| 识别同控制者账号关系 | `attach_labels`按账号对身份与固定上三角顺序对齐二值同控监督；`record_texts`仅拼接标题／描述和E5前缀 | 不是预测经营域或账号ID。群、卖家、商品、域ID仅装配／划分／评价，不进入模型输入 |
| 真正更新编码器 | 商品前向不使用`encode`／`detach`／`no_grad`，全部商品归一化后按账号均值再归一化；对称头使用绝对差与乘积；完整378对mean BCE一次反传／Adam更新 | 编码微批4不是每4条商品更新一次；单群是一次更新单位。实际原生CPU核验已覆盖每个编码器block、embedding和head |
| 六轮相同拟合预算 | 原train每域48拟合／12校准；144拟合群每轮完整打乱，三个域联合供数；六轮864更新／模型，合计2592；432→433不重建优化器 | 这一步是为后续持续学习挑选有基本能力的模型配置，当前没有ABC逐域持续训练、ER缓存、教师或蒸馏项 |
| 校准不能混入学习 | `update`仅接收`groups['fit']`日程；校准分数在eval模式下生成，标签只参与统计和六轮阈值；valid只有盲评分 | 36校准群不参与梯度／早停。完整门后单独读取valid；test/owners不读 |
| 低误报下的召回 | 普通未加权BCE拟合，再用训练校准集确定全局阈值，最终在valid三个域分别检查FP≤7、TP≥200 | 可以检验所选配置是否达标；BCE没有直接优化FPR约束，不能保证达标。全部判异控已有94.709%的accuracy但Recall=0，因此不能以accuracy或训练loss下降宣称成功 |
| 更强模型及持续学习起点 | 三种完整编码器配置按匹配协议比较，六轮固定主终点，绝对识别门优先于AP差 | 参数量／预训练任务／分词不同，不称严格等计算。未来持续学习不能直接用见过所有域的联合权重作为首域起点 |

### 三个真实模型的CPU更新证据

使用既有Linux py310、单CPU线程、nice=10、GPU不可见、20分钟上限，调用生产`load_model`、`update`与`score`，每个实际模型只在一个手写28账号、56文本、378对／20正例群上更新一次。没有读取正式文本或标签，没有保留更新后的权重；既有模型归档在加载前逐文件哈希核对。审查脚本为[step28_base_model_audit.py](../scripts/step28_base_model_audit.py)，不是正式训练入口，也不计入2592次正式更新。

| 实际配置 | 编码器block数 | 编码器＋头参数数 | 实际探针全部有梯度且权重变化 | 原生池化独立计算最大差 | 手算BCE差 | 实测秒数 |
|---|---:|---:|---:|---:|---:|---:|
| LaBSE／BertModel | 12 | 471,714,305 | 19项 | 0 | 1.25e-8 | 41.94 |
| multilingual-E5-large／XLMRobertaModel | 24 | 560,152,833 | 29项 | 0 | 1.30e-8 | 71.41 |
| BGE-M3／XLMRobertaModel | 24 | 568,017,153 | 29项 | 0 | 4.67e-9 | 64.84 |

探针覆盖每个block的query权重、词嵌入和分类头；LaBSE另覆盖原生Dense。优化器无重复且覆盖全部模型参数，全部参数`requires_grad=True`。三个模型原生Transformer中未被句向量路径使用的pooler weight/bias没有梯度，这是未调用的支路，不冒称每个存储参数都实际更新。其余参数梯度均存在；本次没有对每个非探针参数逐元素证明变化，也不要求未出现词条行都有梯度。

CPU整轮3分05.31秒，退出0，最大RSS13,313,076KiB；正式GPU作业预算不因此被宣称已验证。新增模式读取合同后20项通过0跳过、2.288秒。三份原生结果、日志与九份生产来源两端SHA全部一致，见[native_return_sync.json](../reports/seller_alias_continual/20260918/base_implementation/native_return_sync.json)。这比先前微型模型证据补充了实际加载、原生前向和真实backbone梯度链；仍不替代CUDA/bf16、正式数据长度、六轮学习及保存恢复、8小时／32GiB或识别门的真实结果。

13:08补交的网页GPT6 Pro外审已完成：实际DOM回复ID`261f2b58-2ae2-448b-9730-a1b893ae1310`、`model_slug=gpt-6-pro`，完整原文及263341字节证据包保存，52成员大小／SHA逐项核对一致。主执行者亲自读完最终回复及独立检查源码；原生API因果、旧入口重建、计算链及R1/R2意见与实际源码吻合。外审20合同＋7独立检查、0跳过是其环境结果，不声称本地再跑；本地20合同及三真实模型证据独立记录。外审没有三真实模型结果，故给出有条件恢复意见，未代替主执行者核实。三份真实报告、各自模型归档和最终九份来源已逐项核对，R3闭合。见[补充外审原文](../reports/seller_alias_continual/20260918/base_implementation/native_review_reply.json)、[逐项处置与真实条件闭合](../reports/seller_alias_continual/20260918/base_implementation/native_review.json)。13:33:10恢复同一PID102561，13:34:05状态刷新且GPU仍只有3999MiB，两正式实验未启动。原批准回执完整保留，仅更新活动监听读取的当前来源；没有新建监听或重置启动记录。恢复回传5文件SHA一致，见[恢复实查](../reports/seller_alias_continual/20260918/base_queue/20260918_113800/native_resume_verification.json)。下方为此次复核前的历史过程。

当前实现外审与主执行者修正已完成。实际网页GPT6 Pro回复`1b22993e-4201-4d19-bf7a-88146b3de7ab`完整原文和971815字节证据ZIP已保存，包内130文件2650794字节逐项大小／SHA核对一致。原报告为有条件放行：无确证科研阻断，但有两项复现缺陷；主执行者独立确认并闭合，不把原报告改称零缺陷，也不声称修正后又进行了一轮网页外审。

- R1：完整门曾信任陈旧模型回执。已在valid解析前重新核对六份实际模型大小／SHA；Linux手工删除和等长修改反例均在mock标签入口前拒绝。
- R2：补齐pooled FPR／Recall／Precision条件95%区间。复用原5000次域内群抽样，先合计三域混淆计数再计算比率；不取域precision平均、不增加随机抽样。与提交版函数直接比较，原点值、各域区间及达标判据完全不变。

最终Linux py310合同19通过0跳过，2.173秒；另外亲自执行7项适用外审独立用例，1.259秒、0跳过，涵盖完整378边梯度、显式AdamW递推和新对象恢复。164项独立数值检查符合各自容差，最大误差2.98e-8来自CPU float32计算；新增六个pooled区间端点由独立频数加权和线性分位数核对。外审原8801项数值检查与本地主审164项分开记录，不称全部亲自重跑。原test05／07用于展示修正前缺陷，未在修正后照搬其预期失败行为；由新合同及独立区间核对覆盖修复。

最终入口SHA-256为`7e409f10f291920bfd88487b972e8ac7e868f38983cdce99ec27edfc4415a4a3`。仅入口与合同测试变化，七份其它直接来源（含政策及五份历史依赖）保持不变；九份来源两端核对一致。原提交包、16项及R1阶段18项证据保持原字节。见[外审处置](../reports/seller_alias_continual/20260918/base_implementation/review.json)、[主执行者实际核验](../reports/seller_alias_continual/20260918/base_implementation/20260918_111900/primary_verification.json)、[最终测试](../reports/seller_alias_continual/20260918/base_implementation/20260918_111900/review_fix_tests.log)。

审查及必要修复验证完成后才于12:14启动新监听PID102561，原PID45825已停止；GPU仍仅余3999MiB，两实验尚未启动，正式标签／更新均0。见[GPU队列](SELLER_ALIAS_GPU_QUEUE.zh.md)。真实预训练模型的原生模块、token长度、CUDA、2592更新、资源预算与识别目标仍须正式入口验证；本轮小型CPU夹具不替代这些实际证据。下方为原方案与修正前历史核验，旧等待、恢复和测试数量按其当时范围理解。
2026-09-18。[用户确认方案](SELLER_ALIAS_BASE_MODEL.zh.md)已落实为三模型匹配的离线训练、训练内校准及完整门后评价。主要目标仍是三个域各自FPR≤0.1%、Recall≥50%，没有改成只看AP或相对LaBSE增量。本记录不是达标结果，也不是新的持续学习方法结果。

用户最新明确脚本在既有Linux py310运行，不在Windows下载依赖或运行科研脚本。本次已据此连接、复制必要源码，并完成单CPU线程手工夹具检查；GPU不可用不妨碍此类检查。后续正式三模型训练仍待另行汇报恢复，既有权重敏感性等待不变。

## 需求到实际计算的核对

| 已确认要求 | 实际实现与证据 | 结论边界 |
|---|---|---|
| 自动判断同控，逐域低误报下召回 | `calibrate`只使用36群校准标签；三个域各允许4/4296误报，取统一阈值；`automatic_report`逐域以整数混淆计数判断，不能平均抵消失败 | valid每域须FP≤7、TP≥200；仅开发集点值筛选 |
| 三编码器匹配比较 | 三模型共用144群、六轮日程、未加权378边mean BCE、AdamW与账号聚合；原通用权重重新初始化训练 | 原LaBSE见过校准群，不能充当匹配参照；不声称等参数或等FLOPs |
| 参数学习与阈值校准隔离 | `partition`只按公开群ID、域和固定SHA规则划分48/12；`schedule`只接收拟合群；`train_arm`的优化器循环只取该日程 | 本次只对手工群划分，正式数据尚未处理 |
| 完整文本、无身份特征 | 仅title＋换行＋description；E5两端均加`query: `，BGE与LaBSE不加；每商品归一化、全部商品均值、账号归一化；absdiff＋product对称头 | 身份和域字段仅装配／对齐／评价，不传入模型；不静默截断 |
| 正确原生模型定义 | LaBSE 768维CLS＋Dense，E5 1024维mean，BGE-M3 1024维CLS dense通道；实际模块、维度、归档哈希及256 token限制在监督前检查 | 本次未加载真实预训练权重，原生模块和CUDA行为仍未实测 |
| 编码器与头共同训练 | 每次显式train、完整BCE反传、统一clip、AdamW；窗口首步分别检查两个模块梯度有限非零且参数改变；Adam步数1…864连续 | 两项小型真实CPU用例验证计算／梯度／两步状态恢复，不冒称已验证2592次GPU训练 |
| 固定训练量，不事后选轮数 | 144群×6轮×3模型＝2592更新；三轮与六轮固定保存，六轮主要比较；第二窗口不重置Adam或打乱流 | 不自动早停、重训、改阈值或挑赢家 |
| 保存和实际恢复 | 每个模型两个评分点，拟合144／校准36／valid60群均在完整model＋Adam保存恢复前后评分并要求逐值相等；六个推理模型另行保存、实际加载与哈希核验 | 正式两遍评分共2880次群前向，另加3次监督前盲前向；本次未执行这些前向 |
| valid完整门后一次读取 | 来源、政策、公开输入、划分、2592更新、六点全部分数、全局阈值绑定、六模型回执、外层exit0／耗时／预算均通过后，才调用一次valid parser | 测试使用手工真值与mock的正式读取接口；缺一个模型时接口未调用；正式监督0次 |
| 正确评价与不确定性 | 22指标完整保留，AP与梯形PR-AUC分列，含MRR/MAP及Recall/NDCG@1/3/5/10；校准阈值下计数、FPR/Recall/Precision逐域报告；对LaBSE做同群配对差及5000次域内群bootstrap | 固定logit0指标为辅助；条件区间不含训练随机性／校准重抽，不构成部署误报率保证 |
| 8小时／32GiB | 启动器timeout8h；Budget覆盖job目录；保存前预留，完整checkpoint与推理模型共存时强制采样 | 仍是离散观测，不称精确磁盘峰值；外层时间、日志与晚到回执另保留 |

## 亲自审读与修正

主执行者从用户目标出发审读[新入口](../scripts/step28_base_model.py)、[政策](../schema/step28_base_model_policy.json)、[启动器](../scripts/run_step28_base_model_linux_20260918.sh)、[测试](../tests/test_step28_base_model_contracts.py)，追踪五份直接旧依赖中的公开输入、二值监督对齐、群日程、模型头与聚合、AdamW、实际恢复、预算及全部指标。未沿用旧入口的LaBSE专用评分，因为它不处理E5前缀和新模型配置；复用底层纯函数，新评分路径覆盖训练与保存恢复两端。

开发中明确处理了float32分数与float64阈值比较：若直接在float32数组上比较相邻float64阈值，边界可能被舍入而纳入并列误报。现在先把logit提升为float64，并用人工并列反例核验。又将校准分数读取写成显式角色，不依赖前一循环变量；补齐阈值／模型／分数对应、校准混淆计数及外层时间核对。最后给原生编码器构造前也设置初始化种子，避免加载器初始化未用参数时的偶然随机状态造成前置检查与正式初始化不一致；头初始化仍独立设置。该最后修正在Linux复核后16项仍全部通过。

已完整阅读三份相关历史外审原文：

- [表达变化实现原文](../reports/seller_alias_continual/20260910/expression_implementation/review_reply.json)：历史身份按原记录，不追认为本轮GPT6 Pro。确认其阈值／bootstrap常量绑定缺口已由旧入口修复；新政策进一步绑定完整文件SHA。
- [联合延长实现原文](../reports/seller_alias_continual/20260910/joint_implementation/review_reply.json)：实际记录GPT6 Pro；承接完整Adam、静态风险、公开输入与完整门的审读范围适用于复用函数，不能替代本轮新增三模型／校准实现审查。
- [联合延长结果原文](../reports/seller_alias_continual/20260910/joint_evaluation/20260910_200019/review_reply.json)：原非阻断摘要笔误已闭合；离散磁盘采样限制保留，本轮共存时强制检查但不冒称连续高水位；低绝对召回与重复valid开发边界继续成立。

本轮未重跑上述历史实验、未重读旧标签；旧有效负结果保持。没有使用子agent。

## Linux实际检查

位置：`/home/yongpeng/cross-lingual/reports/seller_alias_continual/20260918/base_implementation/20260918_111900`。

环境为原`/home/yongpeng/miniconda3/envs/py310/bin/python`，Python3.10.19、Torch2.9.1+cu130、NumPy2.2.6、sentence-transformers5.6.0、transformers4.46.3。未安装或改动服务器环境。`CUDA_VISIBLE_DEVICES=''`且OMP/MKL/OpenBLAS均为1线程。11:15只读资源检查可用内存226723MiB、磁盘约952GiB；GPU仍被其他用户占用，仅余3999MiB，未使用GPU或干预他人进程。

执行命令为以下CPU检查；全部数据由测试手工构造，未运行`train`、`evaluate`正式动作：

```bash
cd /home/yongpeng/cross-lingual
source /home/yongpeng/miniconda3/etc/profile.d/conda.sh
conda activate py310
export CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
bash -n scripts/run_step28_base_model_linux_20260918.sh
python -B scripts/step28_base_model.py --help
python -B -m unittest discover \
  -s tests \
  -p test_step28_base_model_contracts.py \
  -v
```

最终16通过、0失败、0跳过，unittest计时2.350秒；两项为真实CPU Torch。外层用时2.88秒、最大RSS780588KiB（约762.3MiB），不能称正式训练耗时。第一次检查亦16通过0跳过，1.833秒；随后仅初始化种子前置修正触发复验，不重复扩大历史全套。测试中一次把仅供断言的requires_grad损失转为标量产生UserWarning，不影响反传、比较或结果；该提示不是科研失败。测试名称中的“完整运行”指手工回执夹具，不是三模型真实训练。

目前正式train／valid／test／owners读取均0，预训练模型加载0，正式更新0。Linux本次只复制4份新来源、复用5份原来源，初始9文件154104字节均两端大小/SHA一致；最后单文件种子修正后9份最终来源再按实际回执核对，五份原依赖另与9月16日前审SHA一致。原CPU输入ZIP留作首次检查的来源证据，不冒充最终源代码快照。检查日志及回执已原样回传；6026字节ZIP的SHA-256为`f23bdd52f46873c138fecce4c3d4f1a66f2b109798114e45a25b6b17e29b788f`，各成员逐项大小/SHA通过。SSH已退出，没有本轮后台检查进程。

直接证据：[Linux核验](../reports/seller_alias_continual/20260918/base_implementation/20260918_111900/verification.json)、[最终测试日志](../reports/seller_alias_continual/20260918/base_implementation/20260918_111900/final_tests.log)、[资源日志](../reports/seller_alias_continual/20260918/base_implementation/20260918_111900/final_resources.log)、[回传核对](../reports/seller_alias_continual/20260918/base_implementation/return_sync.json)、[历史来源核对](../reports/seller_alias_continual/20260918/base_implementation/historical_sources.json)。政策SHA为`3344572d6f48b4b10aabfe0db536a8052c75cac0c5cde4e1ac5c2a62c68ec5ab`，最终入口SHA为`31f549b527d61b93e1bc667a1b8d8c3b76954336c2c3ba63d699e03f47f1a294`。

## 外审准备与剩余范围

2026-09-18 11:35用户明确要求正式提交，已将原37文件／247507字节ZIP提交至[网页审查会话](https://chatgpt.com/c/6aacb169-4858-83e9-b336-b621dbf968cc)，界面为6 Pro且停止按钮／“Pro思考中”已观察到。提交消息ID为`e64b526f-21c0-4b81-b099-546f033de9e5`，见[提交回执](../reports/seller_alias_continual/20260918/base_implementation/review_submission.json)。当前等待回复，回复实际model_slug尚未取得；不写作外审通过。以下“尚未提交”为准备包冻结时的历史状态，包及清单原字节保持，实时提交消息已说明状态变化。

本轮按“实现与外审准备”整理[审查请求](../reports/seller_alias_continual/20260918/base_implementation/review_request.md)、[外审包](../reports/seller_alias_continual/20260918/base_implementation/review_package.zip)与逐文件清单。尚未向网页提交，不声称GPT6 Pro已审查或模型已达标。审查应独立判断该匹配比较是否能回答自动识别能力问题，重点检查训练内隔离、原生编码器处理、阈值边界、完整门和统计解释。通过审查也不等于真实CUDA训练已通过。

正式运行前仍须在已汇报并恢复的阶段确认资源、真实三模型来源／模块／分词／盲前向；真实模型的首个梯度更新、Adam连续性、各点评分与恢复由正式入口实际验证。没有必要现在追加未批准的模型试训或读取正式真值。任何失败保留证据，不自动换模型、重训或开放test。

Windows临时CPU环境清理已完成。最初误在Windows安装了CPU Torch，用户纠正后停止使用，确认本地相关运行进程为0。原生PowerShell删除被自动审批以`blocked by policy`拒绝，当时目录有20965文件、598953051字节。用户随后手动删除；11:28执行者独立核对`C:/Users/35734/AppData/Local/Temp/seller_alias_base_cpu_20260918`已不存在，见[完成回执](../reports/seller_alias_continual/20260918/base_implementation/windows_cleanup_completion.json)。原拒绝保留在[清理历史](../reports/seller_alias_continual/20260918/base_implementation/windows_cleanup.json)，不将用户删除记为执行者命令成功，也不虚构实测磁盘空间增量。
