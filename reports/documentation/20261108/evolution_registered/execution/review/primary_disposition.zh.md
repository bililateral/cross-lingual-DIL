# Registered实现增量：主执行者独立处置

2026-11-08。实际外审会话为https://chatgpt.com/c/6af028e4-5978-83ea-976d-ea56ecf60c26；原输入193091字节、74载荷、SHA256 97a3027f31321871dc1fb75c9ad01c89bf6d8452b843a7a57499aac2755eba41。外审原包发现0项已证实科研阻断、2项正式入口前须修的复现缺陷、3项非阻断建议。原包和原来源不追改，修后源码没有冒称另获网页复审。

完整外审原文和结构化发现保存在external/；页面给出的7成员返回包为36947字节、SHA256 2608afaee88f44eb7f0a72a9451448f771d3e99de2e7539115d3ba0d636eb9cd，实际回收核对见recovery.json。本处置以完整报告和直接源码为依据，不以审查进行中的“暂未发现”代替最终两项缺陷结论。

用户明确要求修复R01/R02，随后明确：“通过了就直接开始训练吧，外审没发现别的问题。”本记录闭合两处已确认实现错误；六条、双路14GiB、5累计小时和数据/96条语言权限沿原授权，不扩大M/C、数据或预算。真实数据语言和资源前提通过后直接执行。

## 两项必修的独立核对与关闭

| 项目 | 独立核对及最小修改 | 必要证据及结论 |
|---|---|---|
| EVREG-I-R01 | 新worker的train/score分支都在载模前调用gpu_admission，但原函数只有CUDA存在和allocator设置。现直接导入并调用已审step30 configure_gpu，保留CUDA存在检查，恢复torch线程/interop各2、确定性算法、matmul/cudnn TF32关闭和13GiB allocator；不重复新写一套配置 | Linux CPU04调用真实gpu_admission/configure_gpu，仅替换CUDA硬件查询和分配器调用；从确定性关闭、TF32开启状态实际变为所需值，allocator参数13/32正确；无CUDA仍拒绝，CUDA未初始化。源码调用链确认train/score在factory/载模前进入。CLOSED |
| EVREG-I-R02 | 原campaign catch可能在新dev qrels已读或指标已写后触发，却固定声明effects_open=false。删除该无法保证字段，保留原异常、automatic_retry=false、automatic_next_phase=false；既有access尝试原件继续保存，不把尝试存在解释为标签全部读完或人已看结果 | CPU04通过实际campaign异常处理分别注入开标前故障和手写qrels访问/读取后超时；两者异常传播、无错误effects_open字段，后者访问原件存在而前者不存在。子任务使用替身，不是真实分析或训练。CLOSED |

CPU04原始外层2.1762145168613642秒，计3秒，exit0。累计CPU20/180秒；辅助582/3600秒＝先前549＋本轮部署/收尾保守30＋CPU3，训练/GPU仍0。CPU子账包含在辅助内，不重复相加；实际后续数据和运行费用继续承接Linux可变preparation_budget.json，campaign再按既有Ledger计30秒尾部。

证据：../cpu04/receipt.json、../cpu04_lifecycle.json、../cpu04_sources.json、../cpu04_budget.json及两条failure/access原件。回传ZIP5134字节、SHA256 d1107509ba3045abae7f1e7c6a867a0a66b4c56ed60b4c6dd5b8c076ef533fd1，10载荷与清单逐项核对；../repair_sources.json的14份实际生产源与Windows一致。新control为12013字节、SHA256 75456cde5e71cc27272a5c914296d17f4b8a30bda004c638513880e2089f53bc；旧被审版b2ce445a…仍可由Git d721ece8及原审查包精确恢复。

CPU04证明受影响的配置和异常留证行为，不声称真实CUDA确定性、14GiB双路工程实测或Evolution效果已验证。原CPU02实际Tiny训练域/更新、CPU03实际矩形与一次开标证据按未改行为复用；本轮不重跑整套、不增加GPU探针。

## 非阻断建议与原结论边界

- N01：最终报告将从已保存的完整十指标格表列出冻结差值、均值和逐格代价。现有保存包含全部指标，不改训练、评分、判据或新增“价值PASS”，不在此增加独立遥测。
- N02：正常单一campaign的新目录、精确目标、restore身份检查和SHA链已有约束。本轮不加入额外重接回执加固；正式结果验收按现有源/端点链核对，不据恶意替换构造假设增加实现前提。
- N03：固定96条逐样本判定文件按原scope留Linux私有目录，实际生成后保存其来源身份；不向Windows或外审输出正文，不重新抽样，不为此再改一次语言规则。

已独立对照新数据选择/4/2/2、显式E域训练调用、H四logit及V逐记录单位化公式、90格枚举、完整开标门、原P判据和预算实际入口；外审对主体路径的结论与代码及已有证据相符。没有把CPU03合成训练元数据当正式8640更新，也没有把旧17/16GiB GPU核验改称当前14/13GiB双路实测。

两项必修在已审范围内、无新增科学选择，按科研纪律第6节由主审定向闭合，不新开网页轮次。正式实现前提已闭合；实际数据构建、一次固定语言门、源/骨干身份、当前资源及中文暂停状态仍须执行核对。满足这些已有前提后，直接按既有授权启动有限六条并在完整后评分；任何P结论均不自动接M/C。
