# 关系目标记忆修订的外审处置

2026-10-06，Asia/Shanghai。网页外审与本轮主审已收尾，结论是保留B+Q+0.1R及累计统计／缓存迁移核心；未发现已证实的生产计算阻断，唯一必要的K1核验缺口已在新增授权内关闭。这里接收的是定义、实现和增量核验证据，不是新原生准入、正式训练许可、超越LOGIT0.1或创新成立。

## 原件与审查范围

12:36:07实见既有会话回答完成。已回收[网页全文](external_page.json)、[完整报告](external/unpacked/SELLER_ALIAS_RELATION_REVISION.external_review.zh.md)及原证据ZIP；原ZIP为5,356,128字节，SHA256为`5bd17ab891e634e435da12f16a2b994fa3d2df5bdfbd869d16446c005edb426c`，47文件、manifest覆盖46载荷，逐文件大小／SHA全部匹配。早期下载超时未记作成功，最终以实际下载和哈希为准。见[回收核对](recovery.json)。未重新上传或提交整包。

输入仍是12:05提交的5,257,725字节修订包，SHA256为`480371343ea0be50e4f8fcc614ec1aee5b35b20593151b7fbb3932e75669a9b5`。网页请求及可见选择器为6／Pro，后台身份未独立认证。网页执行的是Python3.12.14、NumPy2.3.5的手写公式与有限差分、身份和控制流检查；没有Torch，未重跑项目五测或BGE/GPU。原网页代码和输出完整保留，不冒充Linux证据。

## 主审独立判断与处置

| 意见 | 核查与处理 | 当前结论 |
|---|---|---|
| K1-a共同总项 | 原联合参照和优化核确实都调用history_objective的total；单独R非零梯度不证明其参与总项 | 接受缺口。独立组装B+Q+字面0.1R，不读取生产总项／权重；R按显式边序和每查询正负组合另算 |
| K1-b相同标签 | 原handmade_group只改文本／UID，标签保持同一簇分区 | 接受缺口。历史簇归属作循环排列，保持合法四三元簇／八二元簇、20正边，并断言与当前标签不同 |
| K1-c首次Adam参数 | 首步Adam可弱化梯度缩放的可见差异；原测试缺少clip前梯度判据 | 接受缺口。薄包装捕获真实优化核全参数梯度，比较独立联合期望，并转发真实clip/step，检查各恰一次；保留日志和参数比较 |
| D1精度 | 原Q把Y/w转double；新R在FP32上点积与softplus | 补明共享历史前向与活图，不是同字节p；不改精度实现 |
| D2统计夹具 | count1/stage0/空缓存用于内核依赖性，不能当两份可达完整历史 | 文档和测试注释明确“二次统计内核夹具”；不新增历史构造实验 |
| O1监督器异常清理 | 原成功五测未触发监控自身OSError，不能据此判定运行无效 | 不补跑旧五测；本次独立的窄监督入口用finally回收child，不扩展通用监督平台 |
| 后续原生和正式接入 | 新目标无BGE/GPU一步，新正式runner尚未接入 | 保留为后续条件，不冒充本次已完成；不移植旧预算 |

生产`history_objective`当前明确为Q+0.1R，optimization_step用history.labels，B反传后历史Q/R一次联合反传，全参数统一clip和一次AdamW；因此K1是证据盲点，不是已经发现生产漏R。新测试由真实核调用，四个错误仅以临时mock覆盖历史total或标签构造，生产文件没有改动。旧Q和B单项公式证据按未变范围复用，不为独立聚合重写全部算法。

## 获准增量核验的实际证据

用户对单次Linux py310、空闲单CPU、2分钟、RSS2GiB、新证据4MiB、仅受影响手写用例和四种故障检查、无BGE/GPU/正式数据、失败停止不自动重试的方案明确答复：“授权这次增量CPU核验”。这不是沿用旧一次核验余额。独立目录为`/home/yongpeng/cross-lingual/reports/documentation/20261006/relation_revision/k1`，18份来源核对，CPU0运行前0%使用、可用内存约263.86GB；未操作其他用户或其他项目进程。

实际入口（在上述独立目录）为：

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1 \
/usr/bin/time -f 'wall_seconds=%e\nexit_code=%x\nmax_rss_kib=%M' -o wrapper_time.txt \
timeout --signal=TERM --kill-after=5s 115s taskset -c 0 \
/home/yongpeng/miniconda3/envs/py310/bin/python -B verify.py
```

监督内部110秒／2GiB／3MiB停止，外层115秒TERM加5秒KILL。只运行受影响用例一次，不重新执行两阶段576步。日志实际为1用例通过，四个故障均由preclip gradient断言拒绝；它们是预期故障检查通过，不是四次科研失败或自动重试。

| 记录 | 实际值 |
|---|---:|
| 用例主体 | 0.748秒 |
| 监督墙钟 | 2.310244420秒 |
| 外层墙钟／退出码 | 2.36秒／0 |
| 进程树RSS采样峰值 | 707,657,728字节 |
| 外层time最大RSS | 784,692KiB |
| 失败／错误／跳过／重试 | 0／0／0／0 |
| 四种预期故障 | 漏R、断R、0.1误写0.2、错用当前标签，全部识别 |

全参数梯度比较容差atol2e-7／rtol3e-5；参数比较沿用原atol2e-6／rtol2e-4。两种RSS口径不同，均低于上限。来源运行前后匹配，四份回收载荷匹配，见[k1清单](k1/manifest.json)、[原日志](k1/evidence/unittest.log)、[监督结果](k1/evidence/result.json)、[回收](k1/recovery.json)。原五测仍按原来源记5项通过，不把本次1项加四个故障伪报为重新全量通过。

## 科学意义与剩余边界

主审从原historical_loss核对K、v与二次式，并手工推导：令a=2(Hv-b)/N，t=K^{-T}a，g=X^Tt/378，r=∇pR，则历史Y梯度为w(g+0.1r)^T，w梯度为Y(g+0.1r)+epsilon*t。评分w的直接岭通道没有消失。新K1验证真实自动微分合梯度，网页有限差分仅作其手写公式补充。

主审阅读外审反例代码和保存输出：原投影方向Xδp=0而R上升，说明这条有害方向获得新增历史压力；另一聚合取舍例中Q不变且R下降，仍有一条正确比较翻转。后者没有证明MAP整体下降，也不证明固定BGE可以精确实现该特征。不能将Q的低秩核原样认定为R的核；完整合法群的R比较图连通时，精确分数空间线性不变方向仅共同平移，但参数可达性、聚合取舍和缓存外泛化仍不由此保证。

对正负分数±d/2且X=Y，平方识别是(d-2)^2/4、平方排序是(d-2)^2，故Q=1.25(d-2)^2；R=softplus(-d)。因此d=6时Q的导数为10，0.1R导数约-0.00024726，新增R确实不消除固定间隔回拉。这个解析反例说明风险，不证明当前固定0.1必然无效，也不授权调参。

创新判断保留具体差异而不越过证据：主审核对[RankNet式1—3](https://www.microsoft.com/en-us/research/wp-content/uploads/2005/08/icml_ranking.pdf)、[DER++式6与算法2](https://proceedings.neurips.cc/paper/2020/file/b704ea2c39778f07c617f6b7ce480e9e-Paper.pdf)、[FROMP式8](https://proceedings.neurips.cc/paper/2020/file/2f3bbb9730639e9ea48f309d9a79ff01-Paper.pdf)。R单项和三项结构不能单独主张创新；Q的累计平方监督来源、运输及直接w依赖不等于DER++教师logit MSE，也不能因此自动确立方法贡献。未来需要证明累计信息超出缓存监督的价值，当前不预训所有消融。

本轮不再新增网页审查轮次：原裁决已明确最小补强后推进，生产定义未改，主审按实际受限证据关闭K1。新原生准入、正式数据编排与运行合同尚未建立；原有效负结果0/6、权重删除事实及旧关闭路线均保持。同步和Git交付另核实际状态。
