"""Render the already-released aggregates; no access to score matrices."""
import json
import math
from pathlib import Path


def show(value, places=3):
    return "未定义" if value is None else f"{value:.{places}f}"


def percent(value):
    return "未定义" if value is None else f"{100 * value:.1f}%"


def line(values):
    return "|" + "|".join(str(value) for value in values) + "|"


def key(row):
    return f"{row['domain']} {row['start']}→{row['end']}"


def main():
    work = Path(__file__).resolve().parent
    batch = json.loads((work / "analyze/aggregates.json").read_text())
    rows, pairs = batch["transitions"], batch["paired"]
    assert batch["status"] == "COMPLETE" and len(rows) == len(pairs) == 36
    output = ["# 全候选开发分数漂移逐格结果", "",
              "A、B、C 依次为 AlphaBay、Dream、Agora。三个方法分别为同配置 ER、确定性 Full（det）和确定性 β/4 Full（quarter）。每个市场各用其完整固定候选池。百分数为相应查询组的共同平移均方除以绝对 MSE。", "",
              "## 正确账号名次和分数变化", "",
              "|市场与端点|seed|方法|保持第一|跌出第一|升到第一|两端非第一|保持组平移占比|跌出组平移占比|保持组相对 MSE|跌出组相对 MSE|跌出后的平均名次|",
              "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        groups = r["groups"]
        kept, lost = groups["kept_first"], groups["lost_first"]
        assert sum(g["n"] for name, g in groups.items() if name != "all") == groups["all"]["n"]
        for g in groups.values():
            assert g["fell"] + g["unchanged"] + g["rose"] == g["n"]
            if g["n"]:
                assert math.isclose(g["abs_mse"], g["shift_mse"] + g["relative_mse"], abs_tol=1e-12, rel_tol=1e-11)
        output.append(line([key(r), r["seed"], r["method"], kept["n"], lost["n"],
                            groups["gained_first"]["n"], groups["neither_first"]["n"],
                            percent(kept["shift_share"]), percent(lost["shift_share"]),
                            show(kept["relative_mse"]), show(lost["relative_mse"]), show(lost["rank_after"], 2)]))
    output += ["", "## 分差与跌落名次数的描述性 Spearman 相关", "",
               "起点第一组包含保持和跌出第一的所有查询。跌出子组单列，以显示只看少数已跌出的查询会怎样改变相关。并列取平均秩，常量或不足两行时填未定义。", "",
               "|市场与端点|seed|方法|起点第一数|绝对 ρ|相对 ρ|跌出数|跌出子组绝对 ρ|跌出子组相对 ρ|",
               "|---|---:|---|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        a, b = (r["rank_loss_association"][k] for k in ("initially_first", "lost_first"))
        output.append(line([key(r), r["seed"], r["method"], a["n"], show(a["rho_absolute"]), show(a["rho_relative"]),
                            b["n"], show(b["rho_absolute"]), show(b["rho_relative"])]))
    output += ["", "## 两种方法均在起点答对的同一批查询", "",
               "各行的左、右方法共用同一分母。只有跨过共同起点的查询才进入该行。相同查询可能出现在不同端点和 seed 的比较中。", "",
               "|市场与端点|seed|左方法|右方法|共同起点正确|左跌出|右跌出|两者都跌出|",
               "|---|---:|---|---|---:|---:|---:|---:|"]
    for p in pairs:
        assert 0 <= p["both_lost"] <= min(p["left_lost"], p["right_lost"])
        assert p["left_lost"] + p["right_lost"] - p["both_lost"] <= p["common_start_correct"]
        output.append(line([key(p), p["seed"], p["left"], p["right"], p["common_start_correct"],
                            p["left_lost"], p["right_lost"], p["both_lost"]]))
    output += ["", "各端点平均名次和所有分组的误差、升/平/跌查询数均可从同批 aggregates.json 查看。", ""]
    with (work / "analyze/tables.zh.md").open("x", encoding="utf-8") as file:
        file.write("\n".join(output))
    rho = [r["rank_loss_association"]["initially_first"]["rho_relative"] for r in rows]
    quarters = [r for r in rows if r["method"] == "quarter" and r["start"] == "R11" and r["end"] == "R31"]
    print(json.dumps(dict(complete_method_cells=len(rows), complete_pairs=len(pairs),
                          rho_relative_initially_first_range=[min(rho), max(rho)],
                          quarter_A_to_C_stable_shift_percent=[100*r["groups"]["kept_first"]["shift_share"] for r in quarters],
                          quarter_A_to_C_lost_shift_percent=[100*r["groups"]["lost_first"]["shift_share"] for r in quarters],
                          no_new_matrix_reads=True)))


if __name__ == "__main__":
    main()
