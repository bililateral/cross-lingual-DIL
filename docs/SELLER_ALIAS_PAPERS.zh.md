# 项目刊会目录与既有文献核对

目录和文献核对日期：2026-09-06。2026-10-04按用户清理指令删除本页未采用的候选推荐与后续实验设想，仅保留仍被既有研究计划引用的目录来源、页码、散列及文献阅读范围。当前主线的文献定位见[持续学习文献](SELLER_ALIAS_LITERATURE.zh.md)。原始选题外审见[原记录](../reports/seller_alias_papers/20260906/review.json)，不改写原件，也不据历史建议启动实验。

## 目录核对

实际读取项目根目录的两份 PDF，用既有 pdftotext 在内存中抽取，没有安装依赖或保存论文下载副本：

- [CCF推荐目录.pdf](../CCF推荐目录.pdf)：封面为 2026 年目录，共 72 页；PDF 第 32 页列 TOIS、TKDE、VLDBJ 等 A 类期刊，第 35 页列 SIGMOD、SIGKDD、ICDE、SIGIR、VLDB 等 A 类会议，第 57 页列 ACL、ICML、NeurIPS、AAAI、ICLR 等 A 类会议。SHA-256：`e3abf1844d602d4b2302b3dfad0779ac7de89a03b1c7fd15737030ff59b965fb`。
- [北大中文核心期刊目录.pdf](../北大中文核心期刊目录.pdf)：封面为 2023 版名录，共 32 页；PDF 第 28 页登记《计算机学报》《软件学报》《计算机研究与发展》《中文信息学报》《计算机工程》等。SHA-256：`90f24bde8122ddce114e6b83588809deb970a868098beab423d87dea87a1a89c`。这是用户提供的名录副本，名录资格不表示其中每刊同属顶刊。

以下以目录内 A 类会议／期刊和相关中文重要期刊为主要依据；补充近邻另标层级。论文年份按出版记录核对，不使用搜索引擎的抓取日期代替出版日期，也不把 Findings、workshop 或预印本算作主会 A 类论文。

## 本轮实际阅读与结论

| 论文与来源 | 本轮阅读范围 | 对选题的直接约束 |
| --- | --- | --- |
| [Adversarial Matching of Dark Net Market Vendor Accounts](https://www.contrib.andrew.cmu.edu/~nicolasc/publications/Tai-KDD19.pdf)，KDD 2019，A | 原论文任务、公开特征、随机森林与第 5.3 节层次聚类；结合本项目此前已读标签／误差部分 | 卖家成对打分后聚类已存在；加聚类或“同一身份有传递性”不构成新意。 |
| [VendorLink](https://aclanthology.org/2023.acl-long.481/)，ACL 2023，A | 原 PDF 的模型／开放集广告相似度与第 9 节限制 | 文本表示、开放集和低资源迁移已有工作；主要英文广告、同名标签假设、主题和解释限制均有披露。不能把中文替换或对比学习本身称为首创。 |
| [Deep Entity Matching with Pre-Trained Language Models（Ditto）](https://www.vldb.org/pvldb/vol14/p50-li.pdf)，PVLDB 14(1):50–60／VLDB 2021，A | 第 2.2–2.3 节成对输入，第 3 节领域标记、TF-IDF 摘要、数据增强 | 有开源实现；把两个对象一起编码已是强基线。输入摘要和字段标记也不是新贡献。出版卷期有 2020／会议年 2021 两种常见引用，以原文卷期和 DOI 为准。 |
| [预训练语言模型实体匹配的可解释性](https://www.jos.org.cn/html/2023/3/6794.htm)，《软件学报》2023，34(3):1087–1108 | 出版方完整 HTML 已取回；阅读模型背景、属性序反事实／关联说明、近邻增强与结论。公式尚未独立推导 | 字段顺序、交互及低置信度近邻增强已有研究；只给注意力热图或加 kNN 不新。网页工具一度 403，后通过普通公开 HTTP 请求读到原文，不是只依据搜索摘要。 |
| [PINE: Extracting Correlated Token Pairs for Explainable Entity Matching](https://link.springer.com/article/10.1007/s00778-026-00995-3)，The VLDB Journal 2026，35:42，A | 出版方 HTML 第 3 节近邻、第 4.1–4.3 节方法；2026-07-22 发表，也列入 VLDB 2026 期刊报告 | 两阶段 LIME、跨记录词对、差异／空缺词对及贡献解释已有最新工作。不能宣称首次成对词证据解释；解释模型决定不等于证明真实同控。 |
| [基于因子图的不一致记录对消歧方法](https://crad.ict.ac.cn/cn/article/pdf/preview/10.7544/issn1000-1239.2020.20180691.pdf)，《计算机研究与发展》2020，57(1):175–187 | 出版方 PDF 首页／摘要与任务；正文复杂公式未完整审计 | 已有多匹配器冲突协调和因子图方案。不能把普通多模型一致性融合当作空白，也不据这篇摘要立刻实现。 |
| [Counterfactual Augmentation for Robust Authorship Representation Learning](https://doi.org/10.1145/3626772.3657956)，SIGIR 2024，A；[作者存档](https://par.nsf.gov/servlets/purl/10581084) | 出版记录、存档 PDF 的训练目标、实验与消融段；部分页面不可文本抽取，未声称全文精读 | 内容／风格反事实、对比学习与不变正则已有作者表示研究。旧迁移失败不否决所有相关方法，但这一宽泛方向不优先重启。 |
| [ThriftLLM: On Cost-Effective Selection of Large Language Models for Classification Queries](https://www.vldb.org/pvldb/vol18/p4410-huang.pdf)，PVLDB 2025，18(11):4410–4423，A | 原 PDF 的问题定义与方法概览；组合优化证明未复核 | 成本约束模型选择已有成熟近邻；可借其研究问题，不照搬多大模型集成或宣称首个预算控制。 |
| [暗网网页用户身份信息聚合方法](https://www.ecice06.com/CN/10.19678/j.issn.1000-3428.0066805)，《计算机工程》2023，49(11):187–194,210；目录内中文核心，非本表“顶刊”依据 | 出版方摘要、图题和任务说明；未声称阅读全文 | 身份信息及上下文的共指聚合已有中文暗网研究。它的任务／数据与卖家同控标签不能互换；不预设其数据能够获取。 |

本轮检索还排除了角色类别识别、网站流量指纹、知识图谱常规对齐等同名异任务论文。没有找到并读完足以作为本轮方法依据的《计算机学报》直接同任务论文，故不为凑齐三大中文刊而虚列。
