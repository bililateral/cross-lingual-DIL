# 遗漏 Git 记录分范围补交

2026-10-01，依据用户“把之前遗漏的git也按范围补上”及“继续”的明确指示，补齐本地版本记录。原HEAD为 `79709f492920ce250efead88c9bed703a7851496`；在原分支继续追加提交，不改写旧提交、不回填实验日期。

## 完成范围

已完成25项分范围提交，共核对 9292 条路径变更：7551 个普通文件、22 个LFS指针、1719 条既有删除。每份新增／修改文件的实际大小和SHA-256，与暂存Git对象或LFS指针逐项一致；删除记录均对应当前已不存在的旧路径。当前提交链、每次提交路径与最终索引对象全部核对一致。本清单和逐项回执随后单独作第26项归档提交。

| 范围 | 提交 | 路径数 | 内容 |
|---|---|---:|---|
| 00_boundaries | `a2a41dba8dc8` | 2 | Preserve research bytes and exclude custody payloads |
| 01_names | `de7940f45caf` | 3124 | Record historical filename migration |
| 02_custody | `2877edde63e2` | 389 | Record Linux custody and completed cleanup |
| 03_style | `da979d0c7338` | 118 | Archive closed style transfer evidence |
| 04_foundation | `0b0cc4160411` | 36 | Record seller alias feasibility studies |
| 05_initial | `b2f00fa91350` | 358 | Archive initial limited history experiments |
| 06_population | `c4d1396d3035` | 158 | Record synthetic population arrival study |
| 07_expression | `6946e679bd59` | 99 | Record masked expression dataset study |
| 08_cause | `769792e34d00` | 109 | Record alias signal diagnostic study |
| 09_cause_train | `d79af471338f` | 101 | Record train only cause probe study |
| 10_joint | `803a8e921b3e` | 78 | Record joint training reference study |
| 11_replay | `27028e4182a3` | 98 | Record limited replay experiments |
| 12_probe | `f0fb83781f8d` | 83 | Record replay implementation probes |
| 13_distillation | `778b25463ed4` | 111 | Record historical score distillation study |
| 14_sensitivity | `f627d821f08d` | 110 | Record loss sensitivity study |
| 15_base | `bf79f4620810` | 243 | Record base model comparison and evaluation recovery |
| 16_chinese | `ecc1fc1e0d5e` | 282 | Record Chinese BGE base comparison |
| 17_pooling | `dc5a83bb3d01` | 229 | Record negative pooling experiment |
| 18_ranking | `dbe8da57bc92` | 342 | Record hard ranking improvement study |
| 19_calibration | `2d1d5748bec9` | 704 | Record probability calibration study |
| 20_test | `e83e2237e7a4` | 694 | Record independent base model acceptance |
| 21_literature | `6db0f8e63646` | 20 | Record continual learning literature review |
| 22_continual | `35750111a7b3` | 1381 | Record approved BGE continual pilot implementation |
| 23_result | `f8ed5511d699` | 412 | Record audited continual pilot results |
| 24_docs | `fc95852be50b` | 11 | Reconcile current research documentation |

## 保留边界与验证

- `.gitattributes` 禁止科研文档、代码、政策、测试与结果自动转换换行；原有LFS过滤规则保留。已有冻结文件的空格／换行风格不为提交而改写，相关 `diff --check` 返回记录在各回执。
- 新增忽略规则排除 123 个此前未跟踪的载荷／临时项，原清单大小合计 828,852,044 字节。包括正式文本、标签记忆、身份盐、权重和浏览器日志；这些文件没有因本次Git操作被物理删除。既有历史对象没有被清理。外审小型手工载荷可能同样被后缀规则排除，原外审证据ZIP保留，可恢复原字节。
- 对拟提交的4911份非保管文本进行凭据格式扫描，发现0项。未打印或存储口令、私钥值；扫描不等于正式数据访问。
- 本轮18份冻结科学来源再次与原Linux运行记录匹配。没有新训练、模型加载或标签解析；没有为归档重跑已关闭实验或全量测试。
- 迁移范围一次 `git add` 在完成389条暂存后因旧忽略目录提示退出，核对完整暂存集合后继续。独立test范围一次字节核对遇到262字符Windows路径，改用扩展路径读取同一原文件。两项机械故障、实物核对与恢复分别见恢复回执；实验与证据字节未改。
- 此次只追加Windows本地Git提交；未推送远端、未清理远端LFS、未在Linux安装或使用Git。此前科研结果及当前文档已经由SFTP回传／同步并核对，见本轮结果目录中的同步回执。Git维护记录属于本地版本管理。

## 当前科研状态

训练及唯一valid评价完成，结果已回传并独立核查。结果网页外审已发送；用户刚告知已结束，实际回复及附件将在本次补交完成后亲自审读。此处没有提前把本地核查冒称为外审通过。后续实质审查处置另作独立提交。

完整依据：[原始盘点](before_status.json)、[分组清单](plan.json)、[逐提交回执](commits)、[链与对象核对](verification.json)、[凭据扫描](credential_scan.json)、[忽略路径暂存恢复](staging_recovery.json)、[长路径核对恢复](path_recovery.json)。
