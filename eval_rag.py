"""
RAG 效果评测脚本

指标：
  检索层（三种检索器对比，无 LLM 调用）：
    - Hit@k   ：top-k 召回结果中是否包含期望来源文件，k=1/3/5
    - MRR@5   ：期望来源首次出现位置的倒数，越靠前分数越高
  生成层（仅混合检索，调用 DeepSeek）：
    - 关键词覆盖率：答案是否覆盖 expected_keywords（"|" 分隔的同义词组，命中其一即可）
    - 拒答准确率  ：资料中无答案的问题是否被正确拒答

运行：
  python eval_rag.py
产物：
  eval_report.json  汇总指标（可直接引用）
  eval_detail.csv   每条问题 × 每种检索器的明细（可做图）
"""

import env_setup  # noqa: F401  环境引导，必须最先导入

import csv
import io
import json
import sys
import time
import os

# Windows 控制台 UTF-8，避免中文表格乱码
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever

from config import EVAL_QUESTIONS_FILE, EVAL_REPORT_JSON, EVAL_DETAIL_CSV
from rag_core import RAGEngine

TOP_K = 5
QUESTIONS_FILE = str(EVAL_QUESTIONS_FILE)
REPORT_JSON = str(EVAL_REPORT_JSON)
DETAIL_CSV = str(EVAL_DETAIL_CSV)

# 拒答识别词：回答中出现任一即判定模型认为"资料中无答案"
REJECT_PATTERNS = ["无法回答", "没有提供", "未提及", "不能回答", "无法从",
                   "没有相关", "未涉及", "资料中没有"]


def build_retrievers(engine: RAGEngine) -> dict:
    """构建统一 top_k 的 BM25 / 向量 / 混合（含权重网格）/ 混合+rerank 检索器。"""
    bm25 = BM25Retriever.from_documents(engine.chunks)
    bm25.k = TOP_K
    vector = engine.vector_db.as_retriever(search_kwargs={"k": TOP_K})
    retrievers = {
        "BM25": bm25,
        "向量检索": vector,
        "混合(0.4/0.6)": EnsembleRetriever(
            retrievers=[bm25, vector], weights=[0.4, 0.6]),
    }
    # 融合权重网格：在评测集上网格搜索最优 (BM25, 向量) 权重
    for w_bm25, w_vec in [(0.3, 0.7), (0.5, 0.5), (0.6, 0.4), (0.7, 0.3)]:
        retrievers[f"混合({w_bm25:.1f}/{w_vec:.1f})"] = EnsembleRetriever(
            retrievers=[bm25, vector], weights=[w_bm25, w_vec])
    # 两阶段：混合召回（fetch_k 扩大到 TOP_K*2）→ bge-reranker 精排取 TOP_K
    # 用 engine自带的 reranker 和候选检索器；返回的 docs 直接是精排后 top-N
    hybrid_recall = EnsembleRetriever(
        retrievers=[bm25, vector], weights=[0.5, 0.5])

    class RerankWrapper:
        """适配 LCEL 调用风格：invoke(q) → list[Document]，内部走 rerank。"""
        def invoke(self, q: str):
            cands = hybrid_recall.invoke(q)
            seen, unique = set(), []
            for d in cands:
                k = hash(d.page_content)
                if k not in seen:
                    seen.add(k)
                    unique.append(d)
            return engine.reranker.rank(q, unique, top_n=5)  # top_n 给满 TOP_K

    retrievers["混合+rerank"] = RerankWrapper()
    return retrievers


def source_names(docs) -> list[str]:
    return [os.path.basename(d.metadata.get("source", "")) for d in docs]


def first_hit_rank(names: list[str], expected: list[str]) -> int | None:
    """期望来源首次命中的 1-based 排名；expected 含 "*" 表示召回任意非空结果即可。"""
    if "*" in expected:
        return 1 if names else None
    for i, name in enumerate(names, 1):
        if any(token in name for token in expected):
            return i
    return None


def hit_at_k(rank: int | None, k: int) -> int:
    return 1 if rank is not None and rank <= k else 0


def keyword_coverage(answer: str, keyword_groups: list[str]) -> float:
    """同义词组（| 分隔）命中比例：每组任一别名出现即算该组命中。"""
    if not keyword_groups:
        return 0.0
    hit = 0
    for group in keyword_groups:
        aliases = [a.strip().lower() for a in group.split("|") if a.strip()]
        if any(a in answer.lower() for a in aliases):
            hit += 1
    return hit / len(keyword_groups)


def is_rejected(answer: str) -> bool:
    return any(p in answer for p in REJECT_PATTERNS)


def main():
    # --retrieval-only：只跑检索评测与权重网格搜索，不调用 LLM（快速调参用）
    retrieval_only = "--retrieval-only" in sys.argv

    with open(QUESTIONS_FILE, "r", encoding="utf-8") as f:
        cases = json.load(f)

    print("⏳ 初始化 RAG 引擎（使用本地缓存索引）...")
    # 评测时显式启用 reranker，用于对比两阶段 vs 单阶段
    engine = RAGEngine(use_reranker=True)
    retrievers = build_retrievers(engine)

    answerable = [c for c in cases if c.get("answerable", True)]
    unanswerable = [c for c in cases if not c.get("answerable", True)]

    # ---------- 检索层评测 ----------
    detail_rows = []
    # 每种检索器的累计指标
    agg = {name: {"hit1": 0, "hit3": 0, "hit5": 0, "rr": 0.0,
                  "latency": 0.0, "n": 0}
           for name in retrievers}

    print(f"\n🔍 检索评测：{len(answerable)} 条可回答问题 × {len(retrievers)} 种检索器")
    for case in answerable:
        q = case["question"]
        for name, retriever in retrievers.items():
            t0 = time.perf_counter()
            docs = retriever.invoke(q)
            latency = (time.perf_counter() - t0) * 1000
            names = source_names(docs)
            rank = first_hit_rank(names, case["expected_sources"])

            h1, h3, h5 = hit_at_k(rank, 1), hit_at_k(rank, 3), hit_at_k(rank, 5)
            rr = 1.0 / rank if rank is not None else 0.0

            a = agg[name]
            a["hit1"] += h1; a["hit3"] += h3; a["hit5"] += h5
            a["rr"] += rr; a["latency"] += latency; a["n"] += 1

            detail_rows.append({
                "id": case["id"], "question": q, "retriever": name,
                "hit@1": h1, "hit@3": h3, "hit@5": h5,
                "rr": round(rr, 4), "latency_ms": round(latency, 1),
                "top5_sources": " | ".join(names[:5]),
            })
        print(f"  ✓ Q{case['id']}: {q[:30]}")

    # ---------- 生成层评测（仅混合检索，可跳过） ----------
    gen_rows, coverages, gen_latencies = [], [], []
    rejected = 0
    if retrieval_only:
        print("\n⏭️  --retrieval-only 模式：跳过生成层评测（不调用 LLM）")
    else:
        print(f"\n💬 生成评测：{len(answerable)} 条关键词覆盖 + "
              f"{len(unanswerable)} 条拒答（调用 DeepSeek）")
        for case in answerable:
            t0 = time.perf_counter()
            answer, _ = engine.ask(case["question"])
            latency = (time.perf_counter() - t0) * 1000
            coverage = keyword_coverage(answer, case.get("expected_keywords", []))
            coverages.append(coverage)
            gen_latencies.append(latency)
            gen_rows.append({"id": case["id"], "question": case["question"],
                             "keyword_coverage": round(coverage, 3),
                             "latency_ms": round(latency, 1),
                             "answer_preview": answer[:80].replace("\n", " ")})
            print(f"  ✓ Q{case['id']}: 关键词覆盖率 {coverage:.0%}")

        for case in unanswerable:
            answer, _ = engine.ask(case["question"])
            ok = is_rejected(answer)
            rejected += int(ok)
            gen_rows.append({"id": case["id"], "question": case["question"],
                             "rejected_correctly": int(ok),
                             "answer_preview": answer[:80].replace("\n", " ")})
            print(f"  ✓ Q{case['id']}: 拒答 {'正确' if ok else '失败（模型硬答了）'}")

    # ---------- 汇总 ----------
    n = len(answerable)
    retrieval_summary = {}
    for name, a in agg.items():
        retrieval_summary[name] = {
            "Hit@1": round(a["hit1"] / n, 4),
            "Hit@3": round(a["hit3"] / n, 4),
            "Hit@5": round(a["hit5"] / n, 4),
            "MRR@5": round(a["rr"] / n, 4),
            "avg_latency_ms": round(a["latency"] / a["n"], 1),
        }
    generation_summary = {
        "avg_keyword_coverage": round(sum(coverages) / len(coverages), 4)
        if coverages else None,
        "rejection_accuracy": round(rejected / len(unanswerable), 4)
        if unanswerable else None,
        "avg_generation_latency_ms": round(sum(gen_latencies) / len(gen_latencies), 1)
        if gen_latencies else None,
    }
    summary = {
        "case_count": {"answerable": n, "unanswerable": len(unanswerable)},
        "top_k": TOP_K,
        "retrieval": retrieval_summary,
        "generation": generation_summary,
    }

    # ---------- 输出报告 ----------
    with open(REPORT_JSON, "w", encoding="utf-8") as f:
        json.dump({"summary": summary,
                   "retrieval_detail": detail_rows,
                   "generation_detail": gen_rows},
                  f, ensure_ascii=False, indent=2)

    with open(DETAIL_CSV, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["id", "question", "retriever",
                           "hit@1", "hit@3", "hit@5", "rr",
                           "latency_ms", "top5_sources"])
        writer.writeheader()
        writer.writerows(detail_rows)

    # 控制台对比表
    print("\n" + "=" * 70)
    print("📊 检索效果对比（越高越好）")
    print("=" * 70)
    print(f"{'检索器':<16}{'Hit@1':>9}{'Hit@3':>9}{'Hit@5':>9}"
          f"{'MRR@5':>9}{'均延迟(ms)':>12}")
    for name, m in retrieval_summary.items():
        print(f"{name:<16}{m['Hit@1']:>9.1%}{m['Hit@3']:>9.1%}"
              f"{m['Hit@5']:>9.1%}{m['MRR@5']:>9.3f}"
              f"{m['avg_latency_ms']:>12.1f}")

    # 权重选型建议：混合变体中按 MRR@5（并列时看 Hit@1）选最优
    hybrid_variants = {k: v for k, v in retrieval_summary.items()
                       if k.startswith("混合(")}
    if hybrid_variants:
        best = max(hybrid_variants.items(),
                   key=lambda kv: (kv[1]["MRR@5"], kv[1]["Hit@1"]))
        print(f"\n🏆 最优融合权重：{best[0]}（MRR@5={best[1]['MRR@5']:.3f}，"
              f"Hit@1={best[1]['Hit@1']:.1%}）→ 建议同步到 rag_core.py")

    if not retrieval_only:
        print("\n📊 生成效果")
        print("-" * 70)
        print(f"平均关键词覆盖率 : {generation_summary['avg_keyword_coverage']:.1%}")
        print(f"无依据拒答准确率 : {generation_summary['rejection_accuracy']:.1%}")
        print(f"平均生成延迟     : "
              f"{generation_summary['avg_generation_latency_ms']} ms")
    print("=" * 70)
    print(f"\n💾 报告已保存：{REPORT_JSON}")
    print(f"💾 明细已保存：{DETAIL_CSV}")


if __name__ == "__main__":
    main()
