# 审查中的机械失败和表述补正

一次补查CPU证据链时，审查者错误假定`source.tar.gz`有名为source_manifest.json的成员，执行`t.extractfile('source_manifest.json')`得到：

```
Traceback (most recent call last):
  File "<stdin>", line 15, in <module>
  File "/usr/lib/python3.13/tarfile.py", line 2494, in extractfile
    tarinfo = self.getmember(member)
  File "/usr/lib/python3.13/tarfile.py", line 2058, in getmember
    raise KeyError("filename %r not found" % name)
KeyError: "filename 'source_manifest.json' not found"
```

该次inline工具命令退出1，未写出计划的evidence_chain.json；先前input_integrity.json已有30个成员逐项对比并全部匹配。这里保留的是从实际工具响应摘录的异常，不冒充单独捕获的完整原始stdout/stderr文件。

随后列出真实tar成员，确认第30项为`reports/documentation/20261013/query_replay_implementation/authorization.json`；来源清单为tar旁的独立文件。审查时先前“29来源加清单”的口头更新已明确纠正。没有缺失应有的29来源，也没有项目来源散列不匹配。

更正后的机械脚本为`check_evidence_chain.py`，实际结果和stdout/stderr/exit_code完整保留；源清单29项与CPU结果列表相同、30个tar成员与当前文件相同、state中10个回传文件身份相同。这是包装核对修正，不是重跑科研、资格或正式训练。

另一次外审手写脚本变量遮蔽失败独立记于`repro/failures_and_corrections.zh.md`，未混作项目CPU失败。未隐去失败后宣称“全部单次通过”。
