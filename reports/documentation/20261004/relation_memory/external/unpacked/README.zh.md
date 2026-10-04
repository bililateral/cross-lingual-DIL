# 关系目标记忆：网页独立数学审查证据

本包对应2026-10-04独立设计外审。所有数值例子是网页工作区手写构造；没有使用项目Linux服务器、BGE、正式数据、正式权重、owners文件或私有资产。脚本中的分组标签按附件公开的4个三账号组和8个两账号组生成，不来自任何实际控制者文件。

## 内容与复算

- upload/SELLER_ALIAS_RELATION_MEMORY.zh.md：用户提供审查输入原件。
- audit_math/independent_math.py：NumPy独立定义核验及完整群反例。
- audit_math/independent_math_results.json：实际结构化结果。
- audit_math/run_stdout.json：原始运行输出。
- audit_functional/counterexample.py：Python Fraction精确有理数三列反例。
- audit_functional/counterexample_result.json：精确反例结果。
- manifest.json：包内来源字节和SHA-256。

在解压目录、已有NumPy的Python环境中执行：

    python audit_math/independent_math.py
    python audit_functional/counterexample.py

本轮环境Python 3.12.14，NumPy 2.3.5。无需下载或加载模型，不需网络。

脚本是独立参考，不是候选项目实现。其中显式逆仅用于独立核对式(7)的参考表达式；候选解的两种表达仍使用线性方程求解。小网格只用于展示数学反例可能使阶段代理上升或下降，不是候选超参数搜索。完整群边界结论使用明确写出的两组构造。

本包不认证自动微分实现、原生资源、正式效果或创新性。当前审查裁决与全部限定见单独的外审Markdown。哈希用于文件身份，不代替科学正确性判断。
