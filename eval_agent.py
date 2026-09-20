# -*- coding: utf-8 -*-
"""
单跳 RAG vs 调研 Agent 对照评测（调研型多论文问题）。

指标：
- 来源召回率 source recall：期望论文出现在「检索命中文档并集」中的比例
  · 单跳模式：一次检索 top-4
  · Agent 模式：多步 retrieve 命中文档去重并集（评测自主多步探索的价值）
- 关键词覆盖率：最终答案命中期望论文名关键词的比例
- 成本：耗时、检索次数/步数、证据规模

运行：python eval_agent.py
输出：agent_eval_report.json
"""

import json
import sys
import time

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from config import RESEARCH_QUESTIONS_FILE, AGENT_EVAL_REPORT
from rag_core import RAGEngine
from research_agent import ResearchAgent
from eval_rag import keyword_coverage

QUESTIONS_FILE = str(RESEARCH_QUESTIONS_FILE)
REPORT_FILE = str(AGENT_EVAL_REPORT)


def source_hit(names: list[str], expected: list[str]) -> dict[str, bool]:
    """每个期望来源是否被任一命中文档文件名包含（大小写不敏感）。"""
    blob = " ".join(names).lower()
    return {e: (e.lower() in blob) for e in expected}


def single_hop(engine: RAGEngine, q: str):
    t0 = time.perf_counter()
    answer, sources = engine.ask(q)  # 上线配置：一次检索 top-4
    elapsed = (time.perf_counter() - t0) * 1000
    names = [s["source"] for s in sources]
    return answer, names, elapsed


def agent_mode(agent: ResearchAgent, q: str):
    result = agent.run(q)
    # 命中文档并集（跨所有 retrieve 动作，去重）
    names = sorted({m["source"] for m in result["all_retrieved"]})
    return result["answer"], names, result["stats"]["elapsed_ms"], result


def main():
    cases = json.load(open(QUESTIONS_FILE, encoding="utf-8"))
    print("⏳ 初始化引擎...", flush=True)
    engine = RAGEngine()
    agent = ResearchAgent(engine)

    rows = []
    sh_recall, ag_recall, sh_cov, ag_cov = [], [], [], []
    sh_lat, ag_lat, ag_steps, ag_calls, ag_chars = [], [], [], [], []

    for c in cases:
        q, expected = c["question"], c["expected_sources"]
        print(f"\n=== {c['id']} {q}", flush=True)

        # ---- 单跳 ----
        sh_answer, sh_names, sh_ms = single_hop(engine, q)
        sh_hits = source_hit(sh_names, expected)
        sh_r = sum(sh_hits.values()) / len(expected)
        # 论文名出现即视为关键词覆盖（每篇一个关键词）
        sh_k = sum(1 for e in expected if e.lower() in sh_answer.lower()) / len(expected)
        sh_recall.append(sh_r); sh_cov.append(sh_k); sh_lat.append(sh_ms)

        # ---- Agent ----
        ag_answer, ag_names, ag_ms, detail = agent_mode(agent, q)
        ag_hits = source_hit(ag_names, expected)
        ag_r = sum(ag_hits.values()) / len(expected)
        ag_k = sum(1 for e in expected if e.lower() in ag_answer.lower()) / len(expected)
        ag_recall.append(ag_r); ag_cov.append(ag_k); ag_lat.append(ag_ms)
        ag_steps.append(detail["stats"]["steps"])
        ag_calls.append(detail["stats"]["retrieve_calls"])
        ag_chars.append(detail["stats"]["evidence_chars"])

        print(f"  单跳 recall={sh_r:.0%} 命中{[e for e,v in sh_hits.items() if v]}",
              flush=True)
        print(f"  Agent recall={ag_r:.0%} 命中{[e for e,v in ag_hits.items() if v]}"
              f" | {detail['stats']['retrieve_calls']}次检索/"
              f"{detail['stats']['steps']}步/{ag_ms/1000:.1f}s", flush=True)
        print(f"  Agent queries: {detail['stats']['queries']}", flush=True)

        rows.append({
            "id": c["id"], "question": q, "expected": expected,
            "singlehop_recall": sh_r, "agent_recall": ag_r,
            "singlehop_hit": sh_hits, "agent_hit": ag_hits,
            "singlehop_answer_coverage": sh_k,
            "agent_answer_coverage": ag_k,
            "singlehop_latency_ms": round(sh_ms),
            "agent_latency_ms": round(ag_ms),
            "agent_steps": detail["stats"]["steps"],
            "agent_retrieve_calls": detail["stats"]["retrieve_calls"],
            "agent_evidence_chars": detail["stats"]["evidence_chars"],
            "agent_queries": detail["stats"]["queries"],
        })

    n = len(cases)
    summary = {
        "n_questions": n,
        "singlehop": {
            "avg_source_recall": round(sum(sh_recall) / n, 4),
            "avg_answer_keyword_coverage": round(sum(sh_cov) / n, 4),
            "avg_latency_ms": round(sum(sh_lat) / n),
        },
        "agent": {
            "avg_source_recall": round(sum(ag_recall) / n, 4),
            "avg_answer_keyword_coverage": round(sum(ag_cov) / n, 4),
            "avg_latency_ms": round(sum(ag_lat) / n),
            "avg_steps": round(sum(ag_steps) / n, 2),
            "avg_retrieve_calls": round(sum(ag_calls) / n, 2),
            "avg_evidence_chars": round(sum(ag_chars) / n),
        },
    }

    print("\n" + "=" * 64)
    print(f"{'指标':<22}{'单跳 RAG':>14}{'调研 Agent':>14}")
    print("-" * 64)
    print(f"{'期望来源召回率':<20}{summary['singlehop']['avg_source_recall']:>13.1%}"
          f"{summary['agent']['avg_source_recall']:>14.1%}")
    print(f"{'答案论文名覆盖':<20}{summary['singlehop']['avg_answer_keyword_coverage']:>13.1%}"
          f"{summary['agent']['avg_answer_keyword_coverage']:>14.1%}")
    print(f"{'平均耗时':<20}{summary['singlehop']['avg_latency_ms']/1000:>12.1f}s"
          f"{summary['agent']['avg_latency_ms']/1000:>13.1f}s")
    print(f"{'平均检索次数':<20}{1:>14}"
          f"{summary['agent']['avg_retrieve_calls']:>14.2f}")
    print("=" * 64)

    json.dump({"summary": summary, "detail": rows},
              open(REPORT_FILE, "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print(f"✅ 报告已保存：{REPORT_FILE}")


if __name__ == "__main__":
    main()
