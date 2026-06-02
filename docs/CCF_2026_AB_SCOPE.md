# CCF 2026 A/B Venue Scope

Source PDF: `中国计算机学会推荐国际学术会议和期刊目录第七版（2026年3月更新）.pdf`

Local extracted text:

- `ccf_2026_catalog_text.txt`: layout-preserving extraction, best for section boundaries and counts.
- `ccf_2026_catalog_raw.txt`: raw extraction, best for reading wrapped venue names.

## Catalog Coverage

The PDF covers CCF-recommended international journals and conferences across ten areas:

1. 计算机体系结构/并行与分布计算/存储系统
2. 计算机网络
3. 网络与信息安全
4. 软件工程/系统软件/程序设计语言
5. 数据库/数据挖掘/内容检索
6. 计算机科学理论
7. 计算机图形学与多媒体
8. 人工智能
9. 人机交互与普适计算
10. 交叉/综合/新兴

Parsed A/B scope count from the layout extraction:

- CCF-A/B journals: 150
- CCF-A/B conferences: 190
- Total A/B venues: 340

## Project-Relevant Priority Scope

For this darknet seller sockpuppet / cross-lingual identity verification project, the most relevant CCF-A/B venues are not evenly distributed across all ten areas. Literature search should prioritize the following areas.

### Network and Information Security

CCF-A journals:

- TDSC: IEEE Transactions on Dependable and Secure Computing
- TIFS: IEEE Transactions on Information Forensics and Security
- Journal of Cryptology

CCF-B journals:

- TOPS: ACM Transactions on Privacy and Security
- Computers & Security
- Designs, Codes and Cryptography
- JCS: Journal of Computer Security
- Cybersecurity

CCF-A conferences:

- CCS: ACM Conference on Computer and Communications Security
- S&P: IEEE Symposium on Security and Privacy
- USENIX Security
- NDSS: Network and Distributed System Security Symposium
- CRYPTO
- EUROCRYPT

CCF-B conferences:

- ACSAC
- ASIACRYPT
- ESORICS
- FSE
- CSFW / CSF
- SRDS
- CHES
- DSN
- RAID
- PKC
- TCC

Use this area first for papers on darknet markets, cybercrime, account linkage, online underground economy, deception, privacy/security abuse, and adversarial identity infrastructure.

### Database, Data Mining, and Information Retrieval

CCF-A journals:

- TODS
- TOIS
- TKDE
- VLDBJ

CCF-B journals:

- TKDD
- TWEB
- DKE
- DMKD
- IPM
- Information Sciences
- Information Systems
- JASIST
- JWS
- KAIS
- DSE

CCF-A conferences:

- SIGMOD
- SIGKDD
- ICDE
- SIGIR
- VLDB

CCF-B conferences:

- CIKM
- WSDM
- PODS
- DASFAA
- ECML-PKDD
- ISWC
- ICDM
- ICDT
- EDBT
- CIDR
- SDM
- RecSys
- WISE

Use this area for graph mining, entity resolution, author/user linkage, information retrieval over markets/forums, text-pair modeling, imbalanced pair classification, and ranking/evaluation methods.

### Artificial Intelligence

CCF-A journals:

- AI: Artificial Intelligence
- TPAMI
- IJCV
- JMLR

CCF-B journals:

- Computational Linguistics
- CVIU
- DKE
- TAC
- TASLP
- IEEE Transactions on Cybernetics
- TEC
- TFS
- TNNLS
- JAIR
- Journal of Automated Reasoning
- Machine Learning
- Neural Computation
- Neural Networks
- Pattern Recognition
- TACL

CCF-A conferences:

- AAAI
- NeurIPS
- ACL
- CVPR
- ICCV
- ICML
- ICLR

CCF-B conferences:

- COLT
- EMNLP
- ECAI
- ECCV
- ICRA
- ICAPS
- ICCBR
- COLING
- KR
- UAI
- AAMAS
- PPSN
- NAACL
- IJCAI

Use this area for cross-lingual representation learning, metric learning, text embeddings, graph neural networks, imbalanced learning, calibration, uncertainty, and NLP-based authorship or user verification.

### Human-Computer Interaction, Ubiquitous Computing, and Social Computing

CCF-A journals:

- TOCHI
- IJHCS

CCF-B journals:

- CSCW
- HCI
- IEEE Transactions on Human-Machine Systems
- IWC
- IJHCI
- UMUAI
- TSMC
- CCF TPCI

CCF-A conferences:

- CSCW
- CHI
- UbiComp
- UIST

CCF-B conferences:

- GROUP
- IUI
- ISS
- ECSCW
- PERCOM
- MobileHCI
- ICWSM

Use this area for social computing, online community abuse, user behavior modeling, platform governance, and human-centered security/privacy work.

### Cross / Comprehensive / Emerging

CCF-A journals:

- JACM
- Proceedings of the IEEE
- Science China Information Sciences
- Bioinformatics

CCF-B journals:

- TASAE
- TGARS
- TITS
- TMI
- TR
- TCBB
- JCST
- JAMIA
- PLOS Computational Biology
- The Computer Journal
- WWW: World Wide Web
- FCS
- BCRA

CCF-A conferences:

- WWW
- RTSS

CCF-B conferences:

- CogSci
- BIBM
- EMSOFT
- ISMB
- RECOMB
- MICCAI
- WINE

Use WWW especially for web-scale abuse, social networks, graph/entity linkage, misinformation, and online-market studies.

### Computer Networks

CCF-A conferences:

- SIGCOMM
- MobiCom
- INFOCOM
- NSDI

CCF-B conferences:

- SenSys
- CoNEXT
- SECON
- IPSN
- MobiSys
- ICNP
- MobiHoc
- NOSSDAV
- IWQoS
- IMC

Use this area selectively for measurement, anonymity/network infrastructure, traffic abuse, or marketplace infrastructure papers.

## Search Discipline for This Project

When searching related work for the paper:

1. Prefer CCF-A first, then CCF-B.
2. Treat security, data mining/IR, AI/NLP, social computing/HCI, and WWW as the primary pool.
3. Do not use CCF-C unless the user explicitly asks for broader coverage or a topic has no A/B evidence.
4. For every candidate paper, record venue, year, CCF area, CCF rank, task relevance, and whether it directly supports the project's method or only supplies background.
