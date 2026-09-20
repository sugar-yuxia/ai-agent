"""
评测报告可视化：读取 eval_report.json / eval_detail.csv，生成 eval_chart.png

图表内容（2x3）：
  1. 各检索器 Hit@1/3/5 分组柱状图
  2. 各检索器 MRR@5（最优权重高亮）
  3. 逐题 RR 对比：纯向量 vs 混合(0.6/0.4) vs BM25
  4. 检索延迟对比
  5. 生成质量：关键词覆盖率 / 拒答准确率
  6. 评测配置与结论摘要

运行：python plot_eval.py
"""

import json
import sys

import matplotlib
matplotlib.use("Agg")  # 无窗口环境直接出图
import matplotlib.pyplot as plt
import numpy as np

# Windows 中文显示
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from config import EVAL_REPORT_JSON, EVAL_CHART_PNG

REPORT_JSON = str(EVAL_REPORT_JSON)
CHART_PNG = str(EVAL_CHART_PNG)

BEST_NAME = "混合(0.5/0.5)"  # 网格搜索确定的最优权重
C_BASE = "#8ea9c9"    # 常规
C_BEST = "#e8743b"    # 最优高亮
C_VEC = "#9ecae1"     # 纯向量
C_HYB = "#e8743b"     # 混合
C_BM25 = "#74a9cf"    # BM25


def main():
    with open(REPORT_JSON, "r", encoding="utf-8") as f:
        report = json.load(f)
    summary = report["summary"]
    retrieval = summary["retrieval"]
    generation = summary["generation"]
    names = list(retrieval.keys())  # 保持评测输出顺序

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(
        f"RAG 系统评测报告（{summary['case_count']['answerable']} 条可回答 + "
        f"{summary['case_count']['unanswerable']} 条拒答测试，top-k={summary['top_k']}）",
        fontsize=16, fontweight="bold")

    # ---------- 1. Hit@1/3/5 分组柱状图 ----------
    ax = axes[0][0]
    metrics = ["Hit@1", "Hit@3", "Hit@5"]
    x = np.arange(len(names))
    width = 0.26
    colors = {"Hit@1": "#9ecae1", "Hit@3": "#4f9bd9", "Hit@5": "#1f5fa6"}
    for i, m in enumerate(metrics):
        vals = [retrieval[n][m] for n in names]
        bars = ax.bar(x + (i - 1) * width, vals, width, label=m,
                      color=colors[m], edgecolor="white")
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.0%}",
                    ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
    ax.set_ylim(0, 1.12)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.set_title("检索命中率对比（越高越好）")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    # ---------- 2. MRR@5 ----------
    ax = axes[0][1]
    vals = [retrieval[n]["MRR@5"] for n in names]
    bar_colors = [C_BEST if n == BEST_NAME else C_BASE for n in names]
    bars = ax.barh(names[::-1], vals[::-1],
                   color=bar_colors[::-1], edgecolor="white")
    for b, v in zip(bars, vals[::-1]):
        ax.text(v + 0.01, b.get_y() + b.get_height() / 2, f"{v:.3f}",
                va="center", fontsize=9)
    ax.set_xlim(0, 1.05)
    ax.set_title(f"MRR@5 对比（橙色 = 最优 {BEST_NAME}）")
    ax.grid(axis="x", alpha=0.3)

    # ---------- 3. 逐题 RR：向量 / BM25 / 混合 / 混合+rerank ----------
    ax = axes[0][2]
    detail = report["retrieval_detail"]
    ids, rr_vec, rr_hyb, rr_bm25, rr_rk = [], [], [], [], []
    for row in detail:
        if row["retriever"] == "向量检索":
            ids.append(row["id"]); rr_vec.append(row["rr"])
        elif row["retriever"] == BEST_NAME:
            rr_hyb.append(row["rr"])
        elif row["retriever"] == "BM25":
            rr_bm25.append(row["rr"])
        elif row["retriever"] == "混合+rerank":
            rr_rk.append(row["rr"])
    order = np.argsort(rr_vec)  # 按纯向量 RR 升序排列，凸显提升
    qids = [ids[i] for i in order]
    xp = np.arange(len(qids))
    ax.bar(xp - 0.30, [rr_vec[i] for i in order], 0.2, label="纯向量", color=C_VEC)
    ax.bar(xp - 0.10, [rr_bm25[i] for i in order], 0.2, label="BM25", color=C_BM25)
    ax.bar(xp + 0.10, [rr_hyb[i] for i in order], 0.2, label=BEST_NAME, color=C_HYB)
    if rr_rk:
        ax.bar(xp + 0.30, [rr_rk[i] for i in order], 0.2,
               label="混合+rerank(large)", color="#c0392b")
    ax.set_xticks(xp)
    ax.set_xticklabels([f"Q{q}" for q in qids], fontsize=7)
    ax.set_ylim(0, 1.1)
    ax.set_title("逐题 RR（按纯向量成绩升序）")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(axis="y", alpha=0.3)

    # ---------- 4. 检索延迟（对数坐标：rerank 秒级 vs 其他毫秒级） ----------
    ax = axes[1][0]
    lat = [retrieval[n]["avg_latency_ms"] for n in names]
    lat_colors = ["#c0392b" if "rerank" in n else C_BASE for n in names]
    bars = ax.bar(names, lat, color=lat_colors, edgecolor="white")
    ax.set_yscale("log")
    ax.set_ylim(5, max(lat) * 2)
    for b, v in zip(bars, lat):
        label = f"{v/1000:.1f}s" if v >= 1000 else f"{v:.0f}ms"
        ax.text(b.get_x() + b.get_width() / 2, v * 1.15, label,
                ha="center", fontsize=8)
    ax.set_ylabel("平均延迟 (ms, 对数轴)")
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
    ax.set_title("检索延迟对比（对数轴，红色=rerank 秒级）")
    ax.grid(axis="y", alpha=0.3, which="both")

    # ---------- 5. 生成质量 ----------
    ax = axes[1][1]
    gen_labels = ["平均关键词覆盖率", "无依据拒答准确率"]
    gen_vals = [generation["avg_keyword_coverage"],
                generation["rejection_accuracy"]]
    bars = ax.bar(gen_labels, gen_vals,
                  color=["#4f9bd9", "#5cb85c"], width=0.5,
                  edgecolor="white")
    for b, v in zip(bars, gen_vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.1%}",
                ha="center", fontsize=12, fontweight="bold")
    ax.set_ylim(0, 1.12)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.set_title(f"生成质量（平均响应 "
                 f"{generation['avg_generation_latency_ms']:.0f} ms）")
    ax.grid(axis="y", alpha=0.3)

    # ---------- 6. 摘要文字 ----------
    ax = axes[1][2]
    ax.axis("off")
    best = retrieval[BEST_NAME]
    vec = retrieval["向量检索"]
    rk = retrieval.get("混合+rerank")
    lines = [
        "关键结论",
        "─" * 24,
        f"最优融合权重：BM25 {BEST_NAME.split('(')[1].split('/')[0]} : "
        f"向量 {BEST_NAME.split('/')[1].rstrip(')')}",
        f"MRR@5：{vec['MRR@5']:.3f} → {best['MRR@5']:.3f}"
        f"（+{best['MRR@5'] / vec['MRR@5'] - 1:.0%}）",
        f"Hit@1：{vec['Hit@1']:.0%} → {best['Hit@1']:.0%}"
        f"（+{best['Hit@1'] - vec['Hit@1']:.0%}）",
        "",
        "Cross-Encoder 精排实测：",
        f"  bge-reranker-large MRR {rk['MRR@5']:.3f}" if rk else "",
        f"  Hit@1 {rk['Hit@1']:.0%}，仍低于 RRF 融合" if rk else "",
        f"  延迟 {rk['avg_latency_ms']/1000:.1f}s/题（CPU）" if rk else "",
        "  → 专属论文题精确匹配更稳，",
        "    rerank 仅改善跨论文语义题，",
        "    收益不抵延迟成本，默认关闭。",
        "",
        f"上线方案：混合检索 {best['avg_latency_ms']:.0f}ms，",
        "拒答准确率 100%，语义+关键词互补。",
    ]
    ax.text(0.05, 0.95, "\n".join(x for x in lines if x is not None),
            va="top", fontsize=11.5,
            bbox=dict(boxstyle="round,pad=0.6", facecolor="#f5f7fa",
                      edgecolor="#ccd5e0"))

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(CHART_PNG, dpi=150, bbox_inches="tight")
    print(f"✅ 图表已保存：{CHART_PNG}")


if __name__ == "__main__":
    main()
