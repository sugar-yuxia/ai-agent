"""
Agent 治理层单元测试（v2）。

patch _decide 返回预设 (thought, tool_name, args)，绕过 LLM 层，
专注测试治理逻辑：重复拦截 / 配额 / 熔断器 / 去重 / trace 格式。
"""

import sys
import json
import tempfile
import urllib.error
from pathlib import Path
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from research_agent import ResearchAgent  # noqa: E402


def _make_engine(retrieve_docs=None):
    engine = MagicMock()
    engine.final_top_k = 3
    engine.retrieve.return_value = retrieve_docs or []
    engine.llm = MagicMock()  # synthesize 可能会调 finish 后的综合
    return engine


def _make_doc(source, content, page=None):
    from langchain_core.documents import Document
    meta = {"source": source}
    if page is not None:
        meta["page"] = page - 1
    return Document(page_content=content, metadata=meta)


def _run_with_decisions(agent, decisions, final_docs=None):
    """
    用预设决策序列跑一轮 agent。
    decisions: [(name, args), ...]  —— 每个元素是 _decide 应返回的 (thought, name, args)
    最后必须含 finish，否则 agent 会因 MAX_STEPS 强制合成。
    """
    iter_dec = iter(decisions)

    def _decide_side(*args, **kwargs):
        try:
            name, args = next(iter_dec)
            return f"thought for {name}", name, args
        except StopIteration:
            # 兜底：finish
            return "thought fallback", "finish", {}

    with patch.object(ResearchAgent, "_decide", side_effect=_decide_side):
        # synthesize 也会调 LLM，mock 掉避免网络调用
        with patch.object(ResearchAgent, "_synthesize",
                          return_value="合成报告"):
            final = agent.run("测试问题")
    return final


class TestQuotaBudget:
    """retrieve 配额 4 → 第 5 次被拦截。"""

    def test_retrieve_quota_enforced(self):
        """retrieve 配额 4 → 用满后 agent 应该自动 finish，不再尝试更多。"""
        docs = [_make_doc("a.pdf", f"证据{i}") for i in range(6)]
        agent = ResearchAgent(
            _make_engine(docs), trace_dir=tempfile.mkdtemp(),
            tools=["retrieve"])
        # 给出 6 次 retrieve 决策 + finish，但配额只有 4
        # agent 应该在第 5 步开始时因 no_tool_left 强制 finish
        decisions = (
            [("retrieve", {"query": f"q{i}"}) for i in range(6)]
            + [("finish", {})]
        )
        final = _run_with_decisions(agent, decisions)

        retrieve_steps = [s for s in final["trace"] if s["action"] == "retrieve"]
        # 恰好 4 次成功执行，没有被拦截的（配额耗尽在预算检查阶段就收了）
        assert len(retrieve_steps) == 4
        assert all(not s["rejected"] for s in retrieve_steps)
        assert final["stats"]["tool_calls"]["retrieve"] == 4


class TestQueryDedup:
    """相同/近义 query → 被重复拦截；不同 query → 通过。"""

    def test_identical_query_blocked(self):
        docs = [_make_doc("a.pdf", "内容")]
        agent = ResearchAgent(
            _make_engine(docs), trace_dir=tempfile.mkdtemp(),
            tools=["retrieve"])
        decisions = [
            ("retrieve", {"query": "WiFi CSI 人体姿态估计"}),
            ("retrieve", {"query": "WiFi CSI 人体姿态估计"}),  # 重复
            ("finish", {}),
        ]
        final = _run_with_decisions(agent, decisions)

        blocked = [s for s in final["trace"]
                   if s["action"] == "retrieve" and s["rejected"]]
        assert len(blocked) == 1
        assert "重复" in blocked[0]["obs_summary"] or \
               "相似" in blocked[0]["obs_summary"]

    def test_different_queries_pass(self):
        docs = [_make_doc("a.pdf", "内容")]
        agent = ResearchAgent(
            _make_engine(docs), trace_dir=tempfile.mkdtemp(),
            tools=["retrieve"])
        decisions = [
            ("retrieve", {"query": "WiFi CSI 人体姿态估计"}),
            ("retrieve", {"query": "轻量级边缘部署方案"}),  # 不同
            ("finish", {}),
        ]
        final = _run_with_decisions(agent, decisions)

        blocked = [s for s in final["trace"]
                   if s["action"] == "retrieve" and s["rejected"]]
        assert len(blocked) == 0


class TestCircuitBreaker:
    """arxiv_search 连续失败 2 次 → 第 3 次被熔断拦截。"""

    def test_arxiv_circuit_breaker(self):
        # _tool_arxiv 抛 URLError 模拟网络不通
        with patch.object(ResearchAgent, "_tool_arxiv",
                          side_effect=urllib.error.URLError("unreachable")):
            agent = ResearchAgent(
                _make_engine(), trace_dir=tempfile.mkdtemp(),
                tools=["retrieve", "arxiv_search"])
            # 3 次 arxiv（前 2 次触发熔断）→ finish
            decisions = [
                ("arxiv_search", {"query": "q1"}),
                ("arxiv_search", {"query": "q2"}),
                ("arxiv_search", {"query": "q3"}),  # 应该被熔断拦截
                ("finish", {}),
            ]
            final = _run_with_decisions(agent, decisions)

        arxiv_steps = [s for s in final["trace"]
                       if s["action"] == "arxiv_search"]
        assert len(arxiv_steps) == 3
        # 前 2 次执行了（但 failed=True 所以没计配额），第 3 次被熔断拦截
        # 注：前 2 次 rejected=False（执行了但工具失败），第 3 次 rejected=True
        executed = [s for s in arxiv_steps if not s["rejected"]]
        stopped = [s for s in arxiv_steps if s["rejected"]]
        assert len(executed) == 2
        assert len(stopped) == 1
        assert "熔断" in stopped[0]["obs_summary"] or \
               "停用" in stopped[0]["obs_summary"]


class TestEvidenceDedup:
    """内容相同的文档 → 只加 1 条证据。"""

    def test_duplicate_content_not_added(self):
        docs = [_make_doc("a.pdf", "相同证据") for _ in range(3)]
        agent = ResearchAgent(
            _make_engine(docs), trace_dir=tempfile.mkdtemp(),
            tools=["retrieve"])
        decisions = [
            ("retrieve", {"query": "q1"}),
            ("finish", {}),
        ]
        final = _run_with_decisions(agent, decisions)

        assert len(final["evidence"]) == 1
        assert final["evidence"][0]["content"] == "相同证据"


class TestTraceFormat:
    """trace JSONL 格式校验。"""

    def test_trace_jsonl_valid(self):
        docs = [_make_doc("a.pdf", "测试", page=1)]
        agent = ResearchAgent(
            _make_engine(docs), trace_dir=tempfile.mkdtemp(),
            tools=["retrieve"])
        decisions = [
            ("retrieve", {"query": "q1"}),
            ("finish", {}),
        ]
        final = _run_with_decisions(agent, decisions)

        trace_files = list(
            Path(final.get("_trace_dir", "")).parent.parent.glob("agent_trace_*.jsonl")
        ) if False else list(Path(agent.trace_dir).glob("agent_trace_*.jsonl"))
        assert len(trace_files) >= 1
        path = trace_files[0]

        events = [json.loads(l) for l in path.read_text(encoding="utf-8")
                  .strip().split("\n")]
        assert events[0]["event"] == "run_start"
        assert events[-1]["event"] == "run_end"
        step_events = [e for e in events if e["event"] == "step"]
        assert len(step_events) >= 1
        assert "action" in step_events[0]
        assert "args" in step_events[0]
        assert "latency_ms" in step_events[0]
        assert "stats" in events[-1]


# 需要 urllib.error 供 TestCircuitBreaker 使用
import urllib.error  # noqa: E402  (放在文件末尾避免导入顺序警告)
