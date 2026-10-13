# C_q针对性静态外审回收包

日期：2026-10-13。

**推荐：可以进入下一步合同准备。** 核心运算静态成立，已有可检验的具体任务化差别；尚未建立独立最终方法贡献、实际收益、原生资格或执行授权。F05—F08的解释／配对补正直接并入后续合同，不要求先重开全项目审查或扩跑实验。

## 文件导航

| 文件／目录 | 内容 |
|---|---|
| `review.zh.md` | 完整中文审查正文：结论、独立证明／反例、10项发现、最小修改、控制与贡献边界。 |
| `minimal_edits.zh.md` | F05—F08可直接并入下一合同的建议文字；没有修改输入。 |
| `literature.zh.md` | 十项实际取得的一手论文、方法对应、具体阅读及未读范围。 |
| `audit/input_verification.json` | 原ZIP身份、成员及输入清单核验。 |
| `audit/final_input_integrity.json` | 打包前再次核对原ZIP与解压副本不变。 |
| `audit/reading_scope.zh.md`、`audit/reading_scope.json` | 逐文件实际阅读范围、未读补集、大小及SHA。 |
| `audit/access_and_limits.zh.md` | 附件解析限制、公开文献范围、模型路由不可核验、执行边界。 |
| `repro/` | 人工数学核对完整源码、输入／假设、环境、输出、stdout/stderr、66项梯度明细、317种分配及失败／修正记录。 |
| `payload_manifest.json`、`FILE_SIZES_SHA256.tsv` | 回收载荷逐文件大小和SHA；两份清单自身不递归列入自身。 |

外部随附的`C_q_static_review_20261013_receipt.json`及`C_q_static_review_20261013_files.tsv`列出ZIP**全部成员**的大小和SHA，包括两份内部清单。外部JSON还列出最终ZIP的大小和SHA，因此不存在要求一个文件包含自身最终哈希的递归矛盾。

## 人工数学重放

原始记录已经完整保留。需要自行重放时，先复制`repro/`到新的工作目录，使用Python及NumPy运行：

```text
python run_checks.py
```

该命令会重写复制目录内的记录文件，故不要在希望保持原回收字节的目录中运行。实际本轮环境是Python 3.13.5、NumPy 2.3.5；没有torch或项目依赖。脚本不是项目C_q实现，不可据它启动项目训练或宣称项目Linux py310资格通过。

## 输入身份

实收`review_input(20261013-052157).zip`：185957字节，24成员，23载荷加根清单。

SHA-256：`d2cbd78c62ca9f448c4de3fe617e8f74c824c57c12b3e27d8750f9cdd2343531`。

本包不重复附原输入代码、模型、Memory、正式文本／关系标签、私有资产或论文全文。完整S和旧C的关闭结果未追改。
