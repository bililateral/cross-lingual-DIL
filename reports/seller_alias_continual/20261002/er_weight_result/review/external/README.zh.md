# 2026-10-02 ER权重结果外审证据

先读根目录REVIEW.zh.md。original/er_weight_result_review.zip是收到的原ZIP原字节副本，不是重新压缩后的替代输入。原包365成员，7,194,455字节，SHA256为5de5711fbe6d20107e8607e8c4aec85a2ac47293e845af9c8c6ece1358631db0。

本证据包含网页实际执行的独立保存结果审计、实际生产纯函数体的确定性手工参考、日程／缓存ID参考、报告数值核对和全部失败修订。项目Linux证据仅来自原ZIP；没有从网页连接项目服务器。未加载原生模型或读取正式文本、标签、owners和缓存正文。

## 文件布局

- REVIEW.zh.md：完整中文结果外审，含逐项需求／证据／偏差／最小处置／未验证范围和分类。
- original/：收到的原ZIP。
- source/：独立审计、手工参考、补充审计、失败版本及可移植复现入口。
- logs/：各次包装进程的argv、环境、UTC起止、退出码、完整stdout和stderr。
- results/independent_v1/：主独立draws、5000次配对差分样本、端点、比较、92条件、误差分项、校准／训练诊断。
- results/supplemental_v2/：18训练阶段的ID日程参考、派生dropout种子、335报告数值、792最终实际域差值。
- results/hand_reference_v3/：最终可移植版手工AP／检索／完整损失／连续Adam核对；v1失败、v2修订仍保留。
- results/all_endpoint_summaries.csv：4方法×4角色×7端点×22指标，2464行，全精度均值／三顺序／条件区间。
- results/all_comparison_summaries.csv：四比较primary与raw参照端点，共792行。
- inspection/：带行号原源码派生副本、失败修订说明、探索错误原文。
- FILE_MANIFEST.json、SHA256SUMS.txt：所有其他证据载荷的大小与SHA；两清单互不包含自身或彼此。

## 离线复现入口（不是新实验要求）

Linux现有Python3.10或更高版本，NumPy；手工夹具另用现有Torch CPU。原网页实际为Python3.13.5、NumPy2.3.5、Torch2.10.0+cpu。脚本不联网、不安装依赖、不执行正式训练，若环境缺包会原样失败并保留stderr。

在解压后的证据根目录，选择一个不存在的新输出目录：

```bash
python source/reproduce_saved_review.py --output /path/to/new-offline-review
```

默认使用当前允许的最小CPU号；可以指定`--cpu`为实际可用CPU。无需CPU47。脚本将解压original/里的ZIP到新目录，四次运行的stdout/stderr分别保存；原来源不变。`results/supplied_audit_cpu0`作为历史兼容目录名保留，实际CPU必须看execution.json，不能从目录名推断。

失败v1和探索性重现并不由此成功入口自动重放，以免将预期失败混为成功流程；完整原失败日志和源码已在包中。source/build_review.py保留本网页排版脚本的原工作目录，复算入口不依赖它，也不要求复现排版。

原始job路径及项目审计命令见REVIEW第1、9、12节。SHA清单证明本次交付字节，不证明未提供的Linux原生文件已经由网页检查。
