# 新目标原生与正式接入核验记录

2026-10-06，Asia/Shanghai。两项获准手写核验通过；正式接入外审待结束，主审最终关闭及正式启动尚未完成。以下是实际执行与主执行者初步核查，不提前签发NO_OPEN_BLOCKERS。

当前新核心为B+Q+0.1R，保持累计平方统计与少量缓存迁移。生产核心SHA为a161986cecf82b887909405d5b4312bfbf64a5bf32409aa5f3d46d55d7db9771，旧正式runner为213119e1205ed7cc765cc44f8a6fb8c3822198da25a28ca99a7d81b29a0bc094，均未改。新runner实际装配revision.update，独立模块内的POLICY、point_name、sources、validate_gate供继承函数解析；正式执行仍是train→真实checkpoint→blind_gate→collect→finalize→complete。

主执行者已核对实际更新接线、来源闭包、端点映射及六继续条件。当前总目标由真实新核计算并反传，历史Memory.draw给同一群的标签和参考；阶段末H/b/c/N仍累计旧平方代理，不能称新B或R充分统计。评价映射把候选新端点与LOGIT0.1末端／共享首域保存指标按域和群对应，尚未读取真实基线或正式标签。本次CPU人工基线仅支持接线核验，不支持效果判断。旧编排通用relation状态名／比较臂名保留，实际端点和policy/source身份区分新作业。

| 核验 | 实际执行与结果 | 资源 |
|---|---|---|
| Linux CPU接入 | 13:14:28开始；2项通过、0失败/错误/跳过。新核2592当前／1728历史更新，九端点、28指标套、扰动后完整恢复、盲门与来源gate | 外层64.35秒；采样RSS768090112B；time最大RSS863004KiB |
| Linux BGE/GPU原生 | 13:17:53开始；448×256当前和不同标签历史，参考活特征最大差0，Q/R/合项三探针梯度均有限非零，clip/step各1、Adam step1、三个探针均实际改变 | 外层35.20秒；reserved18589155328B，allocated18316646400B；采样RSS2820386816B；time最大RSS2840876KiB |

两项分别独立600秒预算，各仅一次，外层均exit0，无自动重试。GPU输入和统计是手写内核夹具，三次诊断VJP计入预算，无正式数据、无保存权重。全参数组合梯度区分性复用已关闭K1，三探针原生不替代K1。FP32参数396、梯度394、Adam moments788是实际检查计数；不宣称每个模型参数都有梯度。前向钩子784包含诊断重算，不能当正式一步固定次数。

本次[原证据](evidence/reports/native/result.json)、[CPU日志](evidence/reports/cpu/unittest.txt)、包装日志和预查已回传；原ZIP10916字节，SHA e17eb3073b24af5b203f940828a2813c684e7cc741c5b74fe744dc747f18e917，9载荷核对一致。CPU记录31正式来源与Windows逐项一致，隔离部署33文件的额外直接手写依赖保持记录。两项证据合计小于11KiB压缩；原载荷见recovery_manifest，不以压缩大小替代运行产物预算。

独立Linux目录：/home/yongpeng/cross-lingual/reports/documentation/20261006/relation_revision_pilot/workspace。只建立既有BGE模型链接，未建立正式数据或基线链接；没有正式训练作业／监听。正式运行要等待本轮接入审查结束并处置必要意见；用户已授权范围无需再按子命令确认。

新正式合同一次24小时／24GiB，预计4—6小时仅为启动前估计，尚无开始或完成时刻。正式效果、六项条件及创新仍未知。完整审查上下文见[context](context.zh.md)，实际提交状态另见submission.json。任何审查修订只补受影响证据，不自动扩大预算或重跑。


13:25:22实际提交，13:25:28实见正在回应；可见6／Pro，后台身份未独立认证。提交ZIP5427862字节、SHA992990d9c1b5f9c5053b6fc675509506ff4d66c5da22247d8c156e818ac2de84，根59载荷逐项核对，内含305文件原历史ZIP。仅上传／提交一次；本报告在打包后记录实际执行，不冒称包含于已上传包。
