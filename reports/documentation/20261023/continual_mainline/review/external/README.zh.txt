持续学习抗遗忘主线静态外审交付包
日期：2026-10-23

审查对象
  continual_mainline_plan_review.zip
  233582字节，35成员，input_manifest.json列出34载荷。
  SHA256：17d1bfd3214d6b9f164274527925ef2d14774967edd0ae2d22947e620097f487
  实收包及全部34载荷大小/哈希核对一致。

主要处置
  不把六臂、36次新增搜索、54路径或783.5h作为原抗遗忘主线最低投入。
  保留三顺序/三seed时，唯一H_pos−S主比较可用两臂18路径、零新增HPO。
  若本轮还重证无记忆系统的学习与遗忘场景，首选SEQ_ref/S/H_pos三臂27路径、零新增HPO；SEQ只是诊断，主对手仍为完整S。
  LOGIT的加入和是否单独12次搜索随目标主张决定；ER、S_pos、原H、R0及全矩阵诊断不自动增加。
  用户新增并行问题已纳入主报告第十一节、F-11/F-12及机械文件时间口径，不需另找补充报告。

文件清单
  review_report.zh.txt       完整中文报告；逐项必要性/复用/删除代价、统计/公平/权限、预算、精简方案、并行审查。
  findings.json             12项结构化发现；来源与行号、影响、最小建议、状态、必要性和阻断范围。
  reading_scope.json        35成员的实际阅读主体/范围、身份与未覆盖项；哈希核对不冒充全文语义审读。
  input_identity_check.json 实收ZIP及34载荷的可复核身份记录。
  mechanical_checks.json    分桶加总、P3旧字段代入、前向/更新/权重数、精简公式及计时口径。
  design_comparison.csv     七个设计版本的机械计数与累计拟额度；UTF-8 BOM。
  verify_mechanical.py      仅用Python标准库读取原ZIP和做普通算术的复核脚本。
  delivery_manifest.json    本交付其余八文件的字节数和SHA256；不递归列自身。
  README.zh.txt             本说明。

阅读建议
  先读主报告第一节及第十节，比较愿意支持的具体主张；并行问题看第十一节。
  证据简称与完整包内路径在第十三节；结构化证据采用原输入相对路径和从1开始的行号，JSON字段另给pointer。

分类与阻断口径
  SCIENTIFIC_BLOCKER：本轮未发现必须新增实验或扩权才能修复的已证实科学阻断；过强机制/外推/非劣主张仍不成立。
  REPRODUCIBILITY_DEFECT：F-07一项轻微预算职责歧义，未发生实际双计，需明确bootstrap唯一归属。
  OUT_OF_SCOPE_OVERDESIGN：F-01与F-03，针对把非主线必要工作一律列为核心。
  合理后续建议：其余9项包含主张边界与未来所选实现分支的必要条件；不代表可以一概跳过，也不代表全部变成新增研究。
  blocks_current_static_proposal=false不等于批准执行或原预算。future_execution_condition和execution_requires_condition_if_this_branch_is_selected分别说明所选分支实施前的条件。

预算和并行
  表中6h/10h单run限额、783.5h原总额、207.5h/297.5h等精简总额示例均为累计任务拟额度，非并行墙钟或GPU设备小时。
  精简总额示例暂保留原27.5h非训练桶，仅用于规模比较；不声称这些桶已被新资格验证，也不自动成为预算申请。
  P3和前向耗时仅代入已开放旧测量，不是本轮资源实测。
  独立完整路径/独立study可并行；路径内stage链和同study ask→完成→tell→下一ask必须保留。
  多卡优先一卡一路径；同卡并发需要适用的已有或未来获批最小吞吐/资源证据。
  当前服务器资源及并行工期均未核实；不以总小时数除并发度承诺工期。

机械复核用法
  python3 verify_mechanical.py /path/to/continual_mainline_plan_review.zip /path/to/empty_review_check_directory
  脚本只读取提供的ZIP、JSON与文本字段并写入三个复核产物；不导入包内项目模块。
  这条命令不是训练、推理、资格、数据生成或bootstrap入口。本次已执行该文档/算术复核。
  输入清单自身的完整性由整个ZIP的SHA256绑定；交付清单自身由整个交付ZIP的SHA256绑定。

身份与范围
  当前会话向实例提供模型标识GPT-6 Astra Pro；未独立核验浏览器模型选择器或后台路由。
  当前主实例最终裁决，三个同会话分审作定向支持；阅读范围按实例区分。
  没有科研训练/推理/重采样、没有生成确认数据、没有连接项目服务器或读取未提供标签/owners/private_custody/原市场/权重。
  未重开旧Optuna结果审查，未重跑旧历史，未重新检索包外文献。

权限
  用户已批准本具体输入包上传和外审，包内旧待上传文字按形成时历史状态处理。
  本报告及附件不授予实施、预算、数据开放或额外上传许可。
  后续由主执行者独立处置，再由用户确认具体实施范围。
