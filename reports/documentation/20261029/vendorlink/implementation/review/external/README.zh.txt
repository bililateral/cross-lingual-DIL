VendorLink 独立科研外审交付说明

先读 report.zh.txt。findings.json 是结构化裁决；input_verification.json 给出输入ZIP与27载荷逐项大小/SHA；reading_scope.tsv 给出全部28成员实际阅读范围；cpu_evidence_crosswalk.json 核对四轮原件与当前源码。

microchecks/data_metric_results.json 是当前源码的手写盲载荷／同分指标微核对；microchecks/static_numeric_results.json 是AST、独立标量公式与静态控制流核对。相应脚本均不加载模型、不跑训练或现有测试套件，不访问网络／服务器。它们不是项目Linux原生证据。

如需重现本报告有限微核对，在另行指定的隔离环境中对原包解压目录运行：
python -B microchecks/data_metric_microcheck.py /absolute/path/to/extracted_input
python -B microchecks/static_numeric_checks.py /absolute/path/to/input.zip /absolute/path/to/extracted_input /absolute/path/to/output_directory
第二个脚本在独立输出目录写出审查核对JSON/TSV，并建立microchecks子目录。第一项使用NumPy；没有Torch／Transformers依赖。此说明不授予任何项目科研执行权限。

OUTPUT_MANIFEST.json 列明本交付所有其他文件的大小/SHA；清单自身不作自引用hash。ZIP外部大小/SHA在交付消息中列明。原送审ZIP、正式数据、模型、Memory或项目源码副本未重打包进交付物。

裁决：审查已完成；需定向修复及来源补齐；正式训练未授权。
