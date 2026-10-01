# 中文暗网马甲识别：研究缺口候选

日期：2026-09-07。状态：文献支持的待检验研究假设；未确定新主线、冻结协议或运行实验。

后续处置：用户已要求不再纠结马甲群／误合并，继续寻找其他方向。因此本页候选甲退出推进顺序，原文与原审查保留历史，不按本页排序启动。后续文字／图片身份线索匹配及经营变更轨迹候选见 [新的选题候选](SELLER_ALIAS_TOPICS.zh.md)；其他旧候选也未因本轮自动获准实施。

用户最新要求只保留三个选题条件：与中文暗网马甲识别相关、有一定研究缺口、有创新空间。不再以实施容易、只能复用现有特征或低计算成本作为选题的首要限制。科研真实性、旧实验结论、Audit A/B 封存和 Linux 使用前汇报暂停的边界不变。真实中文金标签的获取不是本轮前提；也不能将合成证据写成真实市场效果。

前轮“多记录成对证据选择”退回强基线候选。新增核对 [DeepMatcher（SIGMOD 2018）](https://pages.cs.wisc.edu/~anhai/papers1/deepmatcher-tr.pdf) 的 §4.3–4.4 后确认，以对方文本为上下文的对齐、比较和聚合已有直接先例；相对 Step23 不同不等于相对学界新颖。前轮选题与原外审保存在 [文献记录](SELLER_ALIAS_PAPERS.zh.md)，不覆盖旧审查。

## 目录与阅读范围

本轮重新读取项目 `CCF推荐目录.pdf`（封面为 2026 年）和 `北大中文核心期刊目录.pdf`（2023 版）。CCF 的 KDD/SIGIR/SIGMOD/VLDB、ACL/ICML/ICLR、TKDE 等目录位置沿用前轮已核验记录；补读网络安全的 TDSC/TIFS 类目。北大核心第 28 页列有《计算机学报》《软件学报》《计算机研究与发展》《中文信息学报》《计算机工程》。名录收录不意味着所有收录刊物均属顶刊，Findings 也不冒充 ACL 主会。PDF 未修改，未保留下载论文或转文本缓存。

以下以出版方、会议和作者公开原文为依据。部分目录外直接近邻也纳入排重；不能因为不在目标投稿目录就忽略它们。检索未命中不作为“无人研究”的证据。

| 论文与原始来源 | 实际阅读范围及对选题的约束 |
|---|---|
| [Adversarial Matching of Dark Net Market Vendor Accounts](https://www.contrib.andrew.cmu.edu/~nicolasc/publications/Tai-KDD19.pdf)，KDD 2019，CCF A | 回读问题、§5.2–5.3 和 §6.2：RF 成对距离、single/complete/average/minimax 层次聚类、聚类后精确率召回率、冒充与规避均已研究。不是尚无人做马甲群或防冒充。 |
| [VendorLink](https://aclanthology.org/2023.acl-long.481/)，ACL 2023 主会，CCF A | 本轮核对任务与摘要，沿用前轮对表示、开放集和局限的阅读。用广告写作模式关联卖家已存在；开放集本身不新。 |
| [Conformalized Link Prediction on Graph Neural Networks](https://jiank2.github.io/files/papers/zhao24conformalized.pdf)，KDD 2024，CCF A，DOI 10.1145/3637528.3672061 | 阅读 §2–3，特别是定理 3.2：在规定交换性条件下保证单条边预测区间的边际覆盖，并研究集合效率。没有把边际覆盖写成最终马甲群的错误率保证；本地未重做全部定理证明。 |
| [Conformal Prediction Sets with Limited False Positives](https://proceedings.mlr.press/v162/fisch22a/fisch22a.pdf)，ICML 2022，CCF A | 阅读 §3–4：可变集合、零/多个正确答案、FP 数量的期望/高概率约束、集合打分与校准搜索，及附录相关证明段。通用集合风险控制已存在；直接套到账号集合不构成新理论。FP 数量、FDR、区间覆盖须分开。 |
| [Conformal link prediction for false discovery rate control](https://link.springer.com/article/10.1007/s11749-024-00934-w)，TEST，2024 在线发表 | 出版方摘要与 §1.2，作为目录外直接近邻：一般图链接的 FDR 控制已有方法，不能宣称本轮首次提出。未阅读全文/重推证明。 |
| [Conformal network link prediction with false discovery rate control under unstructured missingness](https://www.tandfonline.com/doi/full/10.1080/10618600.2026.2719794)，2026 接受作者版 | 核对出版方摘要，并补读[作者稿假设 1.1](https://arxiv.org/html/2507.07025v2)：有权 graphon 交换性、多重划分、行级及聚合边级 FDR，已有处理依赖和未知缺失模式的工作；但假设缺失模式与网络独立，不能扩为任意信息性缺失。该论文不是本轮顶会依据；未核验全部证明。 |
| [Identifying Users behind Shared Accounts in Online Streaming Services](https://jyunyu.csie.org/docs/pubs/sigir2018paper.pdf)，SIGIR 2018，CCF A | 阅读 §3 的 UI-Past/UI-New 定义、§4 图/会话表示和亲和传播聚类、§5 评价段：同一账号内识别多个用户及未知用户数已有研究。其目标不是跨卖家账号推断共同控制者。 |
| [Parallel Split-Join Networks for Shared Account Cross-domain Sequential Recommendations](https://irlab.science.uva.nl/wp-content/papercite-data/pdf/sun-2022-parallel.pdf)，TKDE，DOI 10.1109/TKDE.2021.3130927，CCF A | 阅读作者接受稿摘要、§I 的角色定义和 split/join 框架。原稿明确潜在角色不必对应真实个人；混合分解本身不能证明恢复了写作者或卖家控制者。使用作者稿，不混淆 DOI 年、稿件下载年与最终卷期。 |
| [一种基于最大公共子图的社交网络对齐方法](https://jos.org.cn/html/2019/7/5831.htm)，《软件学报》2019，30(7):2175–2187，北大核心 | 完整 HTML 可得；阅读摘要、问题与方法框架。结构对齐和邻域配对有先例；不能把结构相似或公共子图直接当卖家同控。未独立核验所有公式与实验。 |
| 《面向共享账户序列推荐的提示增强图注意力网络》，赵中英等，《软件学报》，DOI 10.13328/j.cnki.jos.007626 | 官方期刊平台[优先出版列表](https://rjxb.alljournals.cn/jos/home?id=20210909102755001&name=%E4%B8%BB%E9%A1%B5)显示 2026-04-22，阅读其完整摘要：动态用户数和潜在偏好解耦已有研究。文章正文链接失败，未声称读完正文。 |
| [ChineseBERT: Chinese Pretraining Enhanced by Glyph and Pinyin Information](https://aclanthology.org/2021.acl-long.161.pdf)，ACL 2021 主会，CCF A | 核对原文模型目标及字形/拼音表示：加入形音信息本身已非创新，必须作为中文变体方向的直接参照。未审计其代码。 |
| [Tokenization is Sensitive to Language Variation](https://aclanthology.org/2025.findings-acl.572.pdf)，Findings of ACL 2025 | 阅读任务定义、§6、表 3 和结论相关段。变体鲁棒性与作者风格敏感性之间的区别已有明确研究；不能声称首次发现这个张力。此为 Findings，非 ACL 主会 A 类论文。 |
| [Unraveling Interwoven Roles of Large Language Models in Authorship Privacy: Obfuscation, Mimicking, and Verification](https://aclanthology.org/2025.emnlp-main.753.pdf)，EMNLP 2025 主会，本地目录 B | 阅读问题、交互任务和实验/结论相关段，未审计实现。改写、模仿、验证的相互作用已有直接研究；一般“抗 LLM 改写”不列为新候选。 |

最终补查与外审近邻核对如下，不能以“目录外”排除直接先例：

| 论文与原始来源 | 实际阅读范围及对选题的约束 |
|---|---|
| [Flexible Models for Microclustering with Application to Entity Resolution](https://jwmi.github.io/publications/flexible-models-for-microclustering.pdf)，NeurIPS 2016，CCF A | 阅读问题、§2–3 的微聚类与划分先验、§4 的评价定义。小身份群及群大小分布早已有直接建模，实验也报告 FDR；不能把考虑群大小或报告合并错误说成新意。未将该文的经验 FDR 评价说成频率学错误率保证。 |
| [Multifile Partitioning for Record Linkage and Duplicate Detection](https://arxiv.org/abs/2110.03839)，JASA，2022 在线发表、118(543):1786–1795 | 期刊官方卷期及作者预印本相互核对；读问题、§5.3 的贝叶斯划分决策与拒决设定。整体划分损失、误并/漏并代价和保留未决部分都有先例。未审计采样器或全部附录；作为目录外直接近邻。 |
| [In-context Clustering-based Entity Resolution with Large Language Models: A Design Space Exploration](https://arxiv.org/pdf/2506.02509)，SIGMOD 2026，CCF A | 外审指出后独立核对[会议官方录用名单](https://2026.sigmod.org/sigmod_papers.shtml)，题名与作者匹配；补读作者稿问题及 §5.2–5.3。LLM-CER 已有整组聚类、误聚类检查和层次合并，不能把这些本身当新意。纠正初稿“录用尚未核实”的状态；未重跑其代码。 |
| [Authorship Analysis on Dark Marketplace Forums](https://www.researchgate.net/publication/279848313_Authorship_Analysis_on_Dark_Marketplace_Forums)，EISIC 2015，DOI 10.1109/EISIC.2015.47 | 阅读作者本人上传的原文摘要、任务及 §VI；早已提到马甲与多人共享账号，不能称首次发现此现象。原文研究作者/马甲分类，没有因此解决写作来源与卖家控制者的联合消歧。仅作目录外排重，不将页面推荐论文的摘要误接到本论文。 |
| [Idiosyncratic but not Arbitrary: Learning Idiolects in Online Registers Reveals Distinctive yet Consistent Individual Styles](https://aclanthology.org/2021.emnlp-main.25.pdf)，EMNLP 2021 主会，本地目录 B | 阅读 §1–3 与 §6.2：个人/群体语言变异、孪生作者表示和 tokenizer 敏感性已有研究，不把个人选择习惯本身称为新发现。未审计代码或所有附录。 |
| [IDIOLEX: Unified and Continuous Representations for Idiolectal and Stylistic Variation](https://arxiv.org/html/2604.04704v1)，2026-04-06 作者预印本 | 外审指出后独立读取 §2–3：同文档/同作者/同方言/跨方言的层次邻近监督、语言特征辅助目标，已联合学习个人与社区变异；不能将一般解耦重新包装成贡献。未确认正式会议身份，未审计代码。尚未据此认定已解决中文歧义规范形式下的条件化变体选择与同控匹配。 |

外审另指出两篇近邻，本地已独立核对原文：

- [Learn then Test: Calibrating Predictive Algorithms to Achieve Risk Control](https://arxiv.org/pdf/2110.01052)：作者原稿 §1.1、§2 的定义允许一般输出空间与非单调风险，但校准单位要求独立同分布等条件；正式发表于 *The Annals of Applied Statistics* 2025，19(2):1641–1662，DOI 10.1214/24-AOAS1998，卷期由[作者单位出版记录](https://www.gsb.stanford.edu/faculty-research/publications/learn-then-test-calibrating-predictive-algorithms-achieve-risk)核对。未重推全部证明；作为目录外直接近邻。
- [Towards Improving Code Stylometry Analysis in Underground Forums](https://petsymposium.org/popets/2022/popets-2022-0007.pdf)，PoPETs 2022(1):126–147：读原文摘要与 §2，明确提到同一开发者使用不同账号、一个账号由不同用户运营及单样本多作者。不能称首次发现作者/账号不一一对应，也不据其代码作者识别推断已解决中文卖家共同控制关系。该文不是本轮中文现象的实测依据。

此前中文身份共指、复制处理、风格/内容分离、解释和精排的近邻仍有效，详见前轮文献记录。另发现 `You May Not Be Your Username: Cross-Market Vendor Alias Attribution with Automatic Multi-Signal Evaluation` 的 ResearchGate 条目，但本轮未定位可靠出版身份和原文，列作未解决的检索线索，不据其摘要拼接内容判断方法或新颖性。

## 候选甲：控制最终误合并的马甲群发现

研究问题：**面对大量中文卖家账号，如何恢复同一控制者的账号群，并让“群合并引入的错误身份关系”成为模型的直接优化目标？**

一个确切的结构现象：两个各含 10 个账号、内部同控的不同卖家群，一条错误桥边若触发整群合并，就新增 100 条错误同控关系。因此，检测出的边错误少，不自动意味着传递闭包后的错误少。这是手算例子，不是现有数据实测；也不是批评 Tai 的 minimax 等同于 single linkage。

现有马甲群聚类、图边置信度、一般集合风险控制都已存在。待研究的具体增量是：将拟合并群的全部新增账号关系作为联合决策单位，学习合并收益和错误代价，在规定风险预算下选择群合并/拆分；明确区分“最终闭包内错误账号对比例”和“混入异控账号的群比例”，只选其中一种作主要目标。群大小、共享账号引起的依赖和自适应合并历史要进入问题定义，不能直接拿边级阈值保证替代。

能争取的贡献是**风险约束下的结构化身份恢复方法**及其成立条件；不是给旧分数套置信区间。必须比较 Tai 各类链接、相关聚类、贝叶斯划分决策、普通概率校准，以及“现成集合风险方法直接校准最终划分”的强对照。后一个对照尤其重要：Fisch 等的框架本就允许集合函数，不能把“整体看集合”当独占创新。NeurIPS 微聚类、JASA 划分损失与 LLM-CER 还排除了“考虑群大小”“加误并代价”“用大模型整体聚类/纠错”这些宽泛贡献。

进一步限定：记最终划分内预测同控对数为 `R`，其中错误对数为 `V`，则 FP 数量是 `V`，FDP 是 `V / max(R, 1)`，FDR 是 FDP 的期望。若讨论群污染，分母取最终至少含两个账号的预测群数 `K`，分子取其中混入不同控制者的群数，约定 `K=0` 时比例为 0；同时评价正确关系恢复，不能让全单例凭零错误取胜。这些定义只澄清概念，本轮不选择或冻结主终点。

Fisch 的任意集合打分不意味着其定理直接控制任意风险。Learn then Test 可将预先固定的整套群恢复程序作为预测映射，在独立同分布的市场集合等校准单位上控制相应风险；不要求集合内各条边独立。把合成世界用作统计抽样单位也不等于允许把私有世界标识输入模型。因此“群内依赖”“非嵌套集合”“最终划分”都不足以单独建立创新，现成整体校准必须是强对照。

缺口判断：可争取的具体贡献收窄为**在相同最终风险预算下，通过新的结构求解恢复更多正确身份关系**。目前尚未建立优于“相关聚类加现成最终划分校准”的算子或保证，不能宣称方法已成立。任何保证都依赖明确采样等条件，不能承诺合成校准自动覆盖任意真实市场。属于结构与统计方法候选，不是旧背景删失/联系方式频率问题，也不重启旧四臂迁移。

## 候选乙：写作者与控制者分离的马甲识别

研究问题：**当卖家账号可能由多个人撰写内容、不同卖家也可能共用代写或发布服务时，如何识别账号是否同控，而不把“出现同一写作者”直接当成同一卖家？**

以假设场景说明：同一卖家两个账号由不同员工发文，整体文风差异大；两个不同卖家聘用同一代写，局部文风反而很像。这里始终判断当前账号是否由同一控制者控制，不研究账号转让/接管，也不把任务改成单纯识别作者或运营人员。

拟研究方法需要分开建模两个关系：公开记录所体现的写作来源集合重叠，以及账号层的共同控制。“来源集合有交集”可以不传递；“同一具体写作者”本身仍有传递性，不能混淆。共同控制在固定控制者定义下应当形成一致的身份划分。研究目标是恢复来源重叠与控制关系，不声称找出了具体员工。用公开记录估计多种写作成分，并结合独立的公开账号关联信息，学习哪些来源重叠支持同控、哪些只是共享服务；必须区分联合消歧的收益与关联通道单独带来的收益。训练监督仍可用完整同控标签；不要求人为把它藏成弱监督，也不允许把合成器的作者/控制者隐变量作为模型输入。

已有 SIGIR/TKDE/《软件学报》工作解决了共享账户的潜在用户或偏好分解，EISIC 2015 也已提到暗网多人共享账号，所以“首次发现共享账号”“每个账号多个向量”“自动选混合成分数”“加混合专家”均不是增量。可争取的贡献在于**来源关系与控制关系的联合消歧**，使模型同时处理“同控异文风”和“异控同来源”两个方向的混淆，而不是只提高账号内作者聚类分数。

缺口判断：从所读近邻看，目标差异比普通片段交互更明显；可辨识性风险也更高。若两个控制者安排出了完全相同的全部公开记录，任何算法都不能强行区分；混合成分也不自动等于真实员工。现有根没有验证“多人写作—第三方服务—账号控制”的独立机制，不能直接将旧数据宣称支持该场景。未来是否采用新数据条件需另定，不是本轮建数据授权，也不以取得真实中文标签为前提。

## 候选丙：保留个人变体选择习惯的中文马甲匹配

研究问题：**中文卖家使用谐音、形近字、拆字或中英数字混写时，怎样兼顾同义表达的可比较性，以及个人对变体的选择习惯？**

问题的关键不是简单纠错。公共黑话的相同使用可能只是同一圈子；同一卖家也可能换用另一套替代写法。完全保留字面差异可能漏配，全部规范化又可能抹掉可用习惯。该现象作为动机是假设，尚未测得在现有中文原文中的流行程度和身份效力。

更具体的待检验方法：联合保留原文、候选规范形式和两者间的变换关系；对“在相同语义/上下文条件下选择哪种变体”建模，同时估计社区/平台共有替换习惯。账号匹配利用条件化后的个人偏好，规范形式不确定时保留多种解释。学习目标需要同时区分同控不同变体、异控共享变体，避免只奖励把所有变体压成同一向量。

ChineseBERT 已有形音融合，Findings ACL 2025 已讨论变体鲁棒性与风格敏感性；EMNLP 2021 与 IDIOLEX 已覆盖一般个人/群体语言表示。“加入拼音/字形”“换 tokenizer”“原文与清洗文拼接”“联合学习个人和社区风格”均不能单独作为新意。可争取的增量进一步限定为**对歧义规范形式及上下文条件化的变体选择建模，并用其改进同控匹配**：不是只学句子总体风格相似度，而是比较同一表达机会下的选择倾向，并保留无法确定规范形式的情况。需超过原文字符模型、ChineseBERT、规范化文本、原文加变换特征及适配的个人语言表示等参照；不能仅靠自己合成的固定替换字典制造胜出。

缺口判断：中文特征联系最直接，可能形成适度的方法增量，但近邻覆盖和实证信号的不确定性大于候选甲。若个人身份与社区完全绑定，二者效应无法可靠分离；个人写作偏好也不自动证明控制者相同。它不同于已失败的英文初始化比较和 W/N 去词义字符参照；旧负结果不直接否定本候选，也不因本候选恢复那些旧实验。

## 本轮取舍与边界

补查完成后的本地优先顺序为 **乙、甲、丙**。乙直接针对写作来源与卖家控制关系的错位，任务差异更明显；甲的错误对象清楚，但通用结构决策与风险校准已有较强覆盖；丙最贴近中文表达，却需进一步超过已有个人/社区风格模型。这个判断不是按实现成本排序，也不是外审替本地决定主线。

三者都是待检验命题，不承诺投稿等级、效果或“全球首次”。乙、丙可以用完整合成监督检验明确机制，但不能给每个控制者固定写作者组合或替换字典后，将生成规则中的答案当作通用身份识别贡献。现有根未因此获得新场景资格；也不要求把取得真实中文标签设为研究前提。不以理论证明完成、先验确保成功作为提出简单基线的门槛；也不把简单基线可试写成已有足够论文贡献。

没有使用 Linux、同步文件、读取新标签或启动实验；既有主张及失败终点保持关闭。本轮实际网页 GPT 6 Pro 定点审查已完成，原答复与独立判断保存在 [审查记录](../reports/seller_alias_papers/20260907/review.json)。外审同样建议乙、甲、丙，认为没有阻断候选留存的缺陷，但尚不足以声称创新成立；新增近邻及统计/关系边界已由本地原文核对后纳入。审查是建议，不是新颖性证明或新实验授权；没有对未变代码或旧结果追加审查。
