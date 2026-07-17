# Step24 内容去耦的跨语言作者风格表征实验

更新时间：2026-07-17

状态：已预注册并完成代码实现，等待 Windows 下载冻结模型后同步至 Linux 数值执行。

## 1. 为什么需要 Step24

当前项目已经排除了几条看似能解决中文正例稀缺、实际没有增加独立身份信息的路线：

- Step21 对已有中文 positive 做 identifier-redacted 文本变换，但相对等权复制控制的 AP 差为 `-0.001179`，component bootstrap 置信区间跨零；它是无可测表示收益的零结果。
- Step22 从同一 seller 的商品库存构造 item-disjoint pseudo views，但其增益低于等权复制；它强化的是商品主题而不是跨账号身份。
- Step23 在真实商品上增加 item-to-item 分布统计，主模型 AP 为 `0.458483`，显著低于 matched aggregate 的 `0.593218`；复杂分布特征提高了模板/主题负例的高分尾部。
- Step15-v7 clean 在 200 条 internal development test 上 AP 为 `0.463904`，而旧 v6 约为 `0.594897`。旧 v6 不能恢复为论文主线，因为其特征可能含 identifier shortcut，且选择协议不如 v7 严格。

这些结果共同说明：

> 当前模型主要看到商品内容、主题、模板和聚合结构。局部合成或增加同源商品视图并没有产生新的身份事实，也没有可靠地学到跨主题稳定的作者风格。

Step24 不再合成项目标签或文本，也不继续增加高维 pair feature。它引入在大规模外部作者数据或受控风格数据上预训练的冻结多语言作者表征，把“内容语义相似”和“写作风格相似”拆成两个正交视角，再用只有三项输入的强正则 LR/L2 检验风格视角是否提供增量信息。

## 2. 文献依据

### 2.1 主要方法依据：EMNLP 2025 多语言作者表征

[Leveraging Multilingual Training for Authorship Representation: Enhancing Generalization across Languages and Domains](https://aclanthology.org/2025.emnlp-main.1766/) 是 EMNLP 2025 主会论文。作者在 36 种语言、13 个域、超过 450 万作者上训练统一作者表征，并提出：

1. Probabilistic Content Masking：训练时随机遮蔽低频内容 token，鼓励模型依赖功能词和作者习惯，而不是主题词。
2. Language-Aware Batching：同语言样本进入对比学习 batch，避免语言差异成为过于容易的负例。

论文公开模型 [Blablablab/multilingual-style-representation](https://huggingface.co/Blablablab/multilingual-style-representation)，模型基于 XLM-RoBERTa-large，输出 `1024` 维归一化文档向量，训练数据和评估明确覆盖中文。Step24 固定到 Hugging Face commit `b0147bbf450424fe72c8525fcc02e2e39e3a4024`。

Step24 不复现其 450 万作者训练，也不在当前小数据上微调 0.6B 模型。它只使用公开冻结表示，测试一个更窄且可证伪的问题：

> 已在大规模真实作者数据上学得的多语言作者表示，能否为中文地下市场 seller-pair 提供超出 E5 商品语义的身份排序信号？

该论文明确指出一个限制：训练数据假设每个作者只写一种语言，未证明同一作者跨语言文本可以直接对齐。本项目当前 Step24 不是直接比较 EN 文本与 ZH 文本；它仍然是 English seller-pair source supervision 迁移到 Chinese seller-pair target verification，因此不把论文没有验证的 cross-language same-author claim 当作前提。但地下市场域迁移仍然必须由本项目实验证明。

### 2.2 内容去耦控制：Findings ACL 2025 mStyleDistance

[mStyleDistance: Multilingual Style Embeddings and their Evaluation](https://aclanthology.org/2025.findings-acl.869.pdf) 使用受控平行改写在九种非英语语言上训练内容无关的风格表示，明确包含中文。它公开了 MIT 许可模型 [StyleDistance/mstyledistance](https://huggingface.co/StyleDistance/mstyledistance)，Step24 固定到 commit `d66ed25e48225a503b21a65bc804caf06c886f96`。

该模型的目标是让相同风格在不同内容下保持接近，因此适合作为独立的 content-independence control。但其作者验证结果总体弱于 EMNLP 2025 的直接作者表征，所以 Step24 不把它单独指定为主方法，而把它与主要作者表示一起纳入固定三特征融合。

### 2.3 评估协议依据：TACL 2024 与 EMNLP 2022

[Addressing Topic Leakage in Cross-Topic Evaluation for Authorship Verification](https://aclanthology.org/2024.tacl-1.75/) 指出，作者验证即使采用 cross-topic split，仍可能残留 topic leakage，导致模型排序和结论不稳定。Step24 因此必须单独报告 `template_clone_not_controller` 和 `semantic_topic_not_controller` 的高分尾部，而不能只看 aggregate AP。

[Rethinking the Authorship Verification Experimental Setups](https://aclanthology.org/2022.emnlp-main.380/) 发现作者验证模型会利用 topic 和 named-entity shortcut，并证明删除 named entities 可以改善 DarkReddit 跨语料泛化。Step24 沿用项目已修复的 identifier redaction，不允许 seller alias、联系方式、PGP、URL、candidate-rule count 或未清洗 profile embedding 重新进入 clean scorer。

### 2.4 中文外部诊断依据

[CCTAA: A Reproducible Corpus for Chinese Authorship Attribution Research](https://aclanthology.org/2022.lrec-1.633/) 提供可复现的中文跨主题作者归属语料。CCTAA 可以在未来作为外部中文 style sanity check，但其作者不是地下市场 seller，不能并入 Step5，也不能当成本项目中文马甲 positive。

## 3. Step24 不是哪类实验

Step24 不是：

- 生成中文马甲标签；
- 用公开作者数据冒充暗网 seller-pair；
- 在当前 `valid/test` 上挑模型；
- LoRA 或大模型微调；
- Step21 文本增强的改名版本；
- 把作者风格 cosine 直接解释为同一控制者真值；
- 恢复可能包含联系方式的 v6 profile embedding。

两种新模型参数始终冻结。项目内新增的独立真实中文身份关系数量仍为零。

## 4. 当前数据边界

Step24 只读取 canonical `train`：

| Pool | Positive | Negative | Total |
|---|---:|---:|---:|
| English source train | 116 | 285 | 401 |
| Chinese target train | 229 | 344 | 573 |

中文 `229` 个 train positive 中：

- `213` 是 `silver_train_only`；
- `16` 是 canonical non-silver；
- evidence-type 总数为 `57` direct identifier、`29` component anchor、`143` style/structural soft。

因此所有 `573` 行上的 grouped OOF 只能作为 silver-supported internal development evidence。Step24 必须同时报告：

- canonical non-silver；
- direct/component positive 加全部 negative；
- soft positive 加全部 negative；
- silver-only sensitivity；
- template-clone negative；
- semantic-topic negative。

Step24-v1 不编码 canonical `valid/test` seller，不读取其标签，不读取其历史模型分数。

## 5. 输入文本与防泄漏

### 5.1 完全复用 v7 clean text

三种编码器必须看到完全相同的 seller 文本。Step24 重新执行 v7 redaction 并以 corpus SHA-256 验证逐 seller 文本与冻结 v7 E5 输入一致。

允许字段：

- `category_concat_top`
- `signature_title_concat`
- `title_concat_top`
- `signature_description_concat`
- `description_concat_top`

禁止字段：

- seller raw alias；
- normalized alias；
- market / seller ID；
- contact sections；
- structured identity snapshot；
- 未清洗 `profile_text`。

Step3 occurrence literals 与高精度 PGP、email、Telegram、QQ、微信、URL、钱包等规则继续参与 redaction。模型输入中不添加 `[IDENTIFIER_REMOVED]` 标记，以免“被删除次数”本身成为 identity shortcut。

### 5.2 Train-only 编码

Step24-v1 只为 canonical train pair 涉及的 seller 生成新风格 embedding。输出 metadata 必须记录：

- `encoded_split=train`；
- `valid_test_seller_encoded_count=0`；
- `locally_finetuned=false`；
- 模型目录内容指纹；
- clean text corpus hash；
- matrix hash。

## 6. 三个固定 pair features

对 pair `(seller_left, seller_right)` 只计算：

1. `identifier_redacted_e5_cosine`
2. `pcm_multilingual_authorship_cosine`
3. `mstyledistance_cosine`

明确禁止：

- identifier features；
- `candidate_rule_count_raw` 或其变体；
- 64d 随机投影；
- item-to-item distribution features；
- test-fitted IDF/OOV 统计；
- 未清洗 reranker；
- 当前 valid/test 的任何统计量。

这是刻意的低维设计。当前有效监督只有 `401 + 573` 行，其中中文高质量 positive 更少；三维模型可以回答视角是否有增量信息，而不会把新表征变成另一个高维过拟合实验。

## 7. 固定对照矩阵

### 7.1 Raw controls

- raw identifier-redacted E5 cosine；
- raw PCM multilingual authorship cosine；
- raw mStyleDistance cosine。

Raw control 不训练，只检验单一冻结表示的排序能力。

### 7.2 LR/L2 controls

固定三个模型，不做候选搜索：

| Model | Features | Purpose |
|---|---|---|
| `e5_lr_l2_control` | E5 cosine | 相同训练协议下的 semantic-only control |
| `style_only_lr_l2_control` | 两个 style cosine | 风格视角是否可独立工作 |
| `semantic_style_lr_l2_primary` | E5 + 两个 style cosine | 预注册主方法 |

LR 使用固定 `L2=10.0`、标准化、无 class balancing，并沿用 `domain × evidence_type × confidence × component` factorized weight。没有超参数搜索、阈值选择或模型 seed 选择。

## 8. 两种训练协议

### 8.1 Source-only transfer

只用 `401` 条 English train 拟合三个 LR/L2，直接给全部 Chinese train 打分。

该协议回答：

> 不使用中文标签时，外部预训练作者风格是否改善 English-to-Chinese pair-verification transfer？

### 8.2 Target grouped OOF adaptation

Chinese train 按 Step16I 重新计算的 seller connected component 做五折：

1. 每折保留全部 English train；
2. 加入其余四折 Chinese train；
3. 预测未见 Chinese seller components；
4. imputation、standardization、weights 与 LR 系数全部在折内拟合；
5. 每个 Chinese pair 恰好获得一次 OOF score；
6. 同一 seller component 不得跨折。

该协议回答：

> 在不让同一控制组件泄漏到训练和预测两侧的条件下，style view 是否为目标域适配提供增量排序信息？

## 9. 晋级门槛

Step24-v1 必须同时满足：

1. target OOF primary AP 比 E5-only LR/L2 至少高 `0.03`；
2. component-grouped paired bootstrap 的 AP delta 95% CI lower bound 不低于 `0`；
3. source-only primary AP 至少高 `0.02`，且其 component-grouped paired bootstrap 95% CI lower bound 不低于 `0`；
4. source-only 的 non-silver AP delta 不低于 `-0.01`，direct/component positive 加全部 negative 的 AP 不下降；
5. target OOF canonical non-silver AP delta 不低于 `-0.01`；
6. target OOF direct/component positive 加全部 negative 的 AP 不下降；
7. direct/component positive mean score 下降不超过 `0.03`；
8. template/topic negatives 的 mean、q95、top-decile mean 增幅分别不超过 `0.02`；
9. 任何 valid/test 标签、分数和 pair feature 都没有参与选择。

全部通过只代表可以进入下一道验证，不代表论文结论成立。

## 10. 晋级后的唯一流程

若 `promotion_eligible=true`：

1. 冻结模型目录 hash、Step24 policy、pair feature schema、LR 系数与输出 manifest；
2. 在已有 canonical valid 上只检查一次，不调模型；
3. 构建并冻结真实、模型配置冻结后收集的 Step20 prospective holdout；
4. 所有方法在 Step20 上统一评估一次；
5. 通过统计审计后才进入 Step11/17 explicit allow-list graph validation。

若 `promotion_eligible=false`：

- 冻结 Step24 为负结果；
- 不更换 L2、不增加交互项、不尝试更多 style encoder；
- 停止追求当前数据上的模型性能论文；
- 将论文主线转为跨语言地下市场身份链接的数据集、证据型概念漂移、topic/identifier shortcut 与受控负结果分析。

## 11. 模型许可与可复现性

`mStyleDistance` 模型卡标注 MIT。EMNLP 2025 作者表征模型在预注册时的 Hugging Face 页面没有显示 license tag。项目因此：

- 可以记录其上游 repo、论文、目录 hash 和推理配置；
- 不把模型权重提交 Git 或随论文数据重新分发；
- 投稿前必须再次核对上游许可或向作者确认；
- 若许可不允许研究复现，只能把该模型降级为不可分发外部 control，并以 MIT 的 mStyleDistance 作为可复现 control。

## 12. 为什么这是一条实际出路

该路线不声称解决真实 positive 稀缺。它解决的是另一个当前已经被实验证明存在的问题：现有表示把同主题、同模板和同商品结构误当成身份相似。

与 Step21-23 的区别是：

- Step21-23 只重新排列本项目已有的同一批内容，独立身份信息没有增加；
- Step24 的主要作者表示在外部 450 万作者上学习，新增的是外部学得的作者习惯归纳偏置；
- 项目标签仍只负责一个低维线性检验，不负责从几十个高质量 positive 中训练大模型；
- 预训练模型是否适用于中文地下市场由严格 OOF 和未来 Step20 判定，不靠论文名或理论预设。

因此 Step24 是当前成本、数据规模、泄漏纪律和研究主线之间最合理的下一项可证伪实验。

## 13. Linux 入口

先在 Windows 项目根目录安装 `huggingface_hub` 并运行固定版本下载器：

```powershell
python -m pip install -U huggingface_hub
python scripts\step24_download_models_windows.py
```

下载器会验证远端 commit、保存完整本地快照，并在每个模型目录写入 `step24_model_provenance.json`。模型权重不提交 Git。把代码、policy、测试以及下面两个目录同步到 Linux：

```text
models/step24/authorship/multilingual_style_representation/
models/step24/authorship/mstyledistance/
```

随后在 Linux 项目根目录执行：

```bash
bash scripts/run_step24_content_independent_authorship_linux_20260717.sh
```

数值运行完成后必须同步整个目录：

```text
reports/step24_content_independent_authorship/v1_20260717/
```

以 `step24_sync_manifest.json` 校验完整性。不能只同步 evaluation summary。
