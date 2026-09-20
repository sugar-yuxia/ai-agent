"""
文献调研 Agent v2：在 RAGEngine 之上加一层受治理的多工具 ReAct 决策循环。

设计要点（对应 Agent 应用开发岗位的 harness 能力面）：
1. 任务规划：LLM 把调研问题拆成 2-4 个子问题，形成待办队列
2. Agent 循环：思考 → 工具调用 → 观察 → 再决策，模型自主决定用哪个工具、何时收尾
3. 多工具 + Function Calling：
   - retrieve      ：本地文献库混合检索（BM25+向量 RRF）
   - scholar_search  ：Semantic Scholar 公开 API，补充本地库之外的公开论文（联网能力）
   - python_exec   ：子进程执行 Python 代码，做统计/计算（可验证计算能力）
   - finish        ：收尾信号，最终报告始终基于证据原文合成（防幻觉）
   工具 schema 经 bind_tools 下发，模型走原生 tool_calls 协议，不再靠正则解析 JSON
4. 工具治理：
   - 每类工具独立调用配额；步骤预算；证据字符（近似 token）预算
   - 检索类工具按 query 相似度拦截重复调用（per-tool 历史）；python 按代码哈希去重
   - 网络/执行失败以错误观察回灌且不扣配额，让模型自行调整
   - 预算耗尽强制 finish（tool_choice 强制），杜绝死循环
5. 结构化记忆：证据按内容哈希去重累积，跨工具共享同一编号体系
6. 可观测：每步结构化 trace（思考/动作/参数/命中/耗时），run_stream 产出事件流，
   run() 同步落 JSONL

对外接口：
    agent = ResearchAgent(engine)                    # 默认启用全部工具
    agent = ResearchAgent(engine, tools=["retrieve"])  # 仅本地检索（评测用，结果可复现）
    final = agent.run(question)                      # 同步，返回 dict（含 answer/trace/...）
    for ev in agent.run_stream(question): ...        # 流式事件，供 Web 展示思考过程
"""

from __future__ import annotations

import json
import re
import os
import sys
import time
import difflib
import hashlib
import subprocess
import urllib.parse
import urllib.request
import urllib.error
from dataclasses import dataclass, field, asdict
from datetime import datetime

from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.tools import tool

from config import RUNS_DIR

# ---------------- 工具 schema（仅用于 bind_tools 下发，实体执行在 Agent 内） ----------------


class RetrieveArgs(BaseModel):
    """本地文献库检索参数"""
    query: str = Field(..., description="具体的中文检索问题，面向本地文献库")


class ScholarArgs(BaseModel):
    """Semantic Scholar 公开论文检索参数"""
    query: str = Field(..., description="学术检索词，建议用英文关键词以获得更好召回")


class PythonExecArgs(BaseModel):
    """Python 代码执行参数"""
    code: str = Field(..., description="要执行的 Python 代码，用 print() 输出需要的结果")


class FinishArgs(BaseModel):
    """收尾信号"""
    answer: str = Field("", description="可留空：系统会基于全部证据原文生成最终报告")


@tool("retrieve", args_schema=RetrieveArgs)
def _retrieve_schema(query: str) -> str:  # pragma: no cover - 仅提供 schema
    """在本地学术文献库（31 篇 WiFi-CSI 感知论文）中检索相关段落。"""
    raise NotImplementedError


@tool("scholar_search", args_schema=ScholarArgs)
def _scholar_schema(query: str) -> str:  # pragma: no cover - 仅提供 schema
    """在 Semantic Scholar 公开库中搜索论文（标题+摘要），用于补充本地文献库没有的内容。"""
    raise NotImplementedError


@tool("python_exec", args_schema=PythonExecArgs)
def _python_schema(code: str) -> str:  # pragma: no cover - 仅提供 schema
    """执行一段 Python 代码并返回 stdout，用于数字计算、数据整理等可验证任务。"""
    raise NotImplementedError


@tool("finish", args_schema=FinishArgs)
def _finish_schema(answer: str) -> str:  # pragma: no cover - 仅提供 schema
    """证据已足够时收尾。最终报告由系统基于证据原文生成。"""
    raise NotImplementedError

# ---------------- Prompt ----------------

PLAN_PROMPT = """你是学术文献调研助手。把下面的调研问题拆解为 2-4 个互不重叠、可通过文献检索回答的中文子问题。
只输出 JSON：{{"sub_questions": ["子问题1", "子问题2", ...]}}，不要任何其他内容。"""

ACTION_PROMPT = """你正在通过调用工具完成学术文献调研。当前状态如下：

【待调研子问题】
{todo}

【已掌握的证据条目】（[编号] 用于最终引用）
{evidence_index}

【前序动作与观察】（避免重复已被拦截/已执行的动作）
{scratchpad}

【可用工具与剩余配额】
{tools_text}

规则：
1. 先在 content 中用一两句话简述思考，然后调用恰好一个工具
2. retrieve 查本地文献库；scholar_search 补充本地库没有的公开论文（建议英文关键词）；
   python_exec 做计算/统计；证据已足够时调用 finish
3. 每类工具有独立配额，耗尽后改用其他工具或 finish
4. 检索类 query 要具体、彼此不同，覆盖不同子问题；不要同义改写重复检索
5. 根据【已掌握的证据条目】自行判断覆盖情况；证据已能回答调研问题时立即 finish，
   绝不要为了用完配额而发起"研究现状/综述"之类的泛化检索
6. 若某方向检索不到目标内容，不要同义改写，应换用论文名、专有名词或方法关键词
   从全新角度检索一次，仍无结果则转向其他子问题
7. 不要编造证据以外的论文名、数据或结论"""

FORCE_FINISH_PROMPT = """已达到步数/工具调用预算上限，必须立即调用 finish 工具。

【待调研子问题】
{todo}

【已掌握的证据条目】（[编号] 用于最终引用）
{evidence_index}

content 中简述证据覆盖情况即可，finish 的 answer 参数留空，
系统会基于全部证据原文生成最终综述。"""

SYNTH_PROMPT = """你是严谨的学术文献调研助手。请基于【证据】撰写一份结构化中文调研报告，回答调研问题。

要求：
1. 事实只能来自证据，每个关键结论后用 [编号] 标注来源（如 [1][3]）
2. 按主题/方法分点组织，适当对比不同工作；不要逐条罗列
3. 证据未覆盖的方面，单列「资料不足之处」如实说明
4. 不要编造证据中没有的论文名、数据或结论
5. 报告末尾给出「参考文献」列表：编号 - 来源（本地文件注明页码）

【调研问题】
{question}

【调研子问题】
{sub_questions}

【证据】
{evidence}
只输出报告正文。"""

# ---------------- 数据结构 ----------------


@dataclass
class Evidence:
    idx: int
    source: str          # 本地文件名 / Scholar:<id> / python_exec
    page: int | None
    content: str
    content_hash: str


@dataclass
class TraceStep:
    step: int
    thought: str
    action: str
    args: dict
    obs_summary: str
    retrieved: list[dict] = field(default_factory=list)
    rejected: bool = False
    latency_ms: int = 0


# ---------------- Agent ----------------


class ResearchAgent:
    """受预算与治理约束的多工具 ReAct 文献调研 Agent（function calling 版）。"""

    # ---- 预算（harness 的核心：给不可靠的模型套可靠边界）----
    MAX_STEPS = 10             # 总循环步数上限（含被治理拦截的尝试）
    TOOL_QUOTAS = {            # 每类工具独立配额（被拦截的尝试不计数）
        "retrieve": 4,
        "scholar_search": 3,
        "python_exec": 2,
    }
    RETRIEVE_TOP_K = 5         # 本地检索单次返回块数
    ARXIV_MAX_RESULTS = 4      # arXiv 单次返回条数
    ARXIV_TIMEOUT = 6          # arXiv 请求超时（秒；网络不通时快速失败）
    MAX_EVIDENCE_CHARS = 12000  # 证据累积字符上限（近似 token 预算）
    SIMILAR_THRESHOLD = 0.85   # query 相似度阈值，超过则判为重复调用
    MAX_PARSE_RETRY = 2        # 单步工具调用解析失败的自纠次数
    FAIL_THRESHOLD = 2         # 熔断阈值：同一工具连续失败 N 次后本轮停用
    PY_TIMEOUT = 15            # python_exec 超时（秒）
    PY_OUTPUT_CAP = 1500       # python_exec 输出截断长度

    ALL_TOOLS = ("retrieve", "scholar_search", "python_exec")

    def __init__(self, engine, trace_dir: str = str(RUNS_DIR),
                 tools: list[str] | None = None):
        self.engine = engine
        self.llm = engine.llm
        self.trace_dir = trace_dir
        os.makedirs(trace_dir, exist_ok=True)
        # 启用的工具（finish 恒可用）；评测时传 ["retrieve"] 保证结果可复现
        self.enabled_tools = [t for t in self.ALL_TOOLS
                              if tools is None or t in tools]

    # ---------- 工具实体 ----------

    def _tool_retrieve(self, query: str) -> list[dict]:
        """本地检索：临时调大 final_top_k 取更多候选。"""
        saved = self.engine.final_top_k
        self.engine.final_top_k = self.RETRIEVE_TOP_K
        try:
            docs = self.engine.retrieve(query)
        finally:
            self.engine.final_top_k = saved
        return [{
            "source": os.path.basename(d.metadata.get("source", "")),
            "page": (d.metadata.get("page", 0) or 0) + 1
                    if d.metadata.get("page") is not None else None,
            "content": d.page_content,
        } for d in docs]

    def _tool_scholar(self, query: str) -> list[dict]:
        """Semantic Scholar 公开 API 检索（JSON）。网络失败由调用方捕获为错误观察。
        国内可直连，无需代理；免费额度 1 req/s，对 demo 够用。
        可选：设环境变量 S2_API_KEY 提升额度（https://www.semanticscholar.org/product/api）。"""
        url = ("https://api.semanticscholar.org/graph/v1/paper/search?"
               f"query={urllib.parse.quote(query)}"
               "&limit=4"
               "&fields=title,abstract,year,authors,externalIds,url")
        headers = {"User-Agent": "research-agent-demo/2.0"}
        api_key = os.environ.get("S2_API_KEY")
        if api_key:
            headers["x-api-key"] = api_key
        # 免费额度有限，429 时退避重试一次
        for attempt in range(2):
            try:
                req = urllib.request.Request(url, headers=headers)
                with urllib.request.urlopen(req, timeout=self.ARXIV_TIMEOUT) as r:
                    data = json.loads(r.read())
                break
            except urllib.error.HTTPError as e:
                if e.code == 429 and attempt == 0:
                    time.sleep(2)
                    continue
                raise
        out = []
        for p in data.get("data", []) or []:
            title = (p.get("title") or "").strip()
            abstract = (p.get("abstract") or "").strip()
            pid = p.get("externalIds", {}).get("ArXiv") or p.get("paperId", "")[:8]
            year = p.get("year") or ""
            authors = ", ".join(
                a.get("name", "") for a in (p.get("authors") or [])[:3])
            out.append({
                "source": f"Scholar:{pid}",
                "page": None,
                "content": f"{title} ({year}, {authors})\n{abstract}"[:900],
            })
        return out

    def _tool_python(self, code: str) -> dict:
        """子进程执行 Python（演示级沙箱：仅超时与输出截断；生产需容器级隔离）。"""
        try:
            r = subprocess.run(
                [sys.executable, "-c", code], capture_output=True,
                text=True, encoding="utf-8", errors="replace",
                timeout=self.PY_TIMEOUT)
        except subprocess.TimeoutExpired:
            return {"ok": False,
                    "text": f"执行超时（>{self.PY_TIMEOUT}s）"}
        err = (r.stderr or "").strip()
        if r.returncode != 0:
            return {"ok": False,
                    "text": f"退出码 {r.returncode}\n{err[-600:]}"}
        out = (r.stdout or "").strip() or "（无标准输出）"
        return {"ok": True, "text": out[:self.PY_OUTPUT_CAP]}

    # ---------- 治理：重复调用拦截 ----------

    @staticmethod
    def _norm_query(q: str) -> str:
        return re.sub(r"[\s?？。.,，、的了吗呢吧！!]+", "", q).lower()

    def _is_duplicate_query(self, query: str, history: list[str]) -> bool:
        """治理：与历史 query 高度相似则拦截，防止模型重复调用同一工具。"""
        nq = self._norm_query(query)
        if not nq:
            return True
        for h in history:
            if difflib.SequenceMatcher(None, nq, self._norm_query(h)).ratio() \
                    >= self.SIMILAR_THRESHOLD:
                return True
        return False

    # ---------- LLM 交互 ----------

    def _plan(self, question: str) -> list[str]:
        """规划：拆子问题。失败时退化为单元素（原问题），不阻断主流程。"""
        try:
            raw = (self.llm.invoke([
                SystemMessage(content=PLAN_PROMPT),
                HumanMessage(content=question),
            ]).content)
            text = raw if isinstance(raw, str) else "".join(map(str, raw))
            match = re.search(r"\{.*\}", text.strip(), re.S)
            plan = json.loads(match.group(0)).get("sub_questions", []) \
                if match else []
            plan = [str(q).strip() for q in plan if str(q).strip()]
            return plan[:4] or [question]
        except Exception:
            return [question]

    def _scratchpad_text(self, trace: list[TraceStep]) -> str:
        """把最近几步动作/观察压缩成文本回灌模型（ReAct 的 observation）。"""
        if not trace:
            return "（暂无，这是第一步）"
        lines = []
        for t in trace[-6:]:
            q = t.args.get("query", "") or t.args.get("code", "")
            lines.append(
                f"第{t.step}步 {t.action}"
                + (f"（{q[:50]}）" if q else "")
                + f" → {t.obs_summary}")
        return "\n".join(lines)

    def _tools_text(self, active: list[str], quotas_left: dict[str, int]) -> str:
        desc = {
            "retrieve": "本地文献库检索，参数 {{\"query\": \"中文检索问题\"}}",
            "scholar_search": "Semantic Scholar 公开论文检索，参数 {{\"query\": \"检索词（建议英文）\"}}",
            "python_exec": "执行 Python 代码，参数 {{\"code\": \"代码\"}}",
            "finish": "证据已足够时收尾，参数 {{\"answer\": \"可留空\"}}",
        }
        lines = [f"- {t}：{desc[t]}（剩余配额 {quotas_left.get(t, '∞')}）"
                 for t in active]
        lines.append("- finish：" + desc["finish"])
        return "\n".join(lines)

    def _decide(self, todo: list[str], evidence_index: str, scratchpad: str,
                active: list[str], quotas_left: dict[str, int],
                force_finish: bool):
        """决策一步：走原生 tool_calls 协议。

        返回 (thought, tool_name, args)。模型未调用工具时抛 ValueError 触发自纠。
        """
        if force_finish:
            prompt = FORCE_FINISH_PROMPT.format(
                todo="\n".join(f"- {t}" for t in todo) or "（无）",
                evidence_index=evidence_index or "（暂无）")
            msgs = [SystemMessage(content=prompt),
                    HumanMessage(content="请立即调用 finish 工具。")]
            try:
                # tool_choice 强制指定 finish，比纯提示词约束更可靠
                resp = self.llm.bind_tools([_finish_schema]).invoke(
                    msgs, tool_choice="finish")
            except Exception:
                resp = self.llm.bind_tools([_finish_schema]).invoke(msgs)
        else:
            prompt = ACTION_PROMPT.format(
                todo="\n".join(f"- {t}" for t in todo) or "（无）",
                evidence_index=evidence_index or "（暂无）",
                scratchpad=scratchpad,
                tools_text=self._tools_text(active, quotas_left))
            msgs = [SystemMessage(content=prompt),
                    HumanMessage(content="请思考并调用工具。")]
            resp = self.llm.bind_tools(self._tool_specs(active)).invoke(msgs)

        content = resp.content
        if not isinstance(content, str):
            content = "".join(map(str, content))
        thought = content.strip()[:200]
        calls = getattr(resp, "tool_calls", None) or []
        if not calls:
            raise ValueError("模型未调用任何工具")
        call = calls[0]  # 治理：每步只采纳第一个动作
        return thought, str(call["name"]), dict(call.get("args") or {})

    def _tool_specs(self, active: list[str]) -> list:
        """启用工具的 schema 列表（finish 恒可用）。"""
        spec_map = {
            "retrieve": _retrieve_schema,
            "scholar_search": _scholar_schema,
            "python_exec": _python_schema,
        }
        return [spec_map[t] for t in active] + [_finish_schema]

    # ---------- 主循环（生成器，事件驱动） ----------

    def run_stream(self, question: str):
        """事件流：plan / thought / observation / final。

        final 事件载荷为完整结果 dict（answer/evidence/trace/stats）。
        """
        t_start = time.perf_counter()
        sub_questions = self._plan(question)
        yield {"type": "plan", "sub_questions": sub_questions}

        todo = list(sub_questions)
        active: list[str] = list(self.enabled_tools)  # 本轮可用工具（可被熔断移除）
        disabled_reasons: dict[str, str] = {}
        consecutive_fail: dict[str, int] = {t: 0 for t in active}
        evidence: list[Evidence] = []
        seen_hashes: set[str] = set()
        # per-tool 去重历史：检索类存 query，python 存代码哈希
        query_history: dict[str, list[str]] = {t: [] for t in active}
        code_hashes: set[str] = set()
        tool_counts: dict[str, int] = {t: 0 for t in active}
        all_retrieved_meta: list[dict] = []  # 供评测：所有检索动作的命中文档
        trace: list[TraceStep] = []

        def evidence_index_text() -> str:
            # 预览取 140 字：足以让模型从综述/知识图谱类证据中发现具体论文名，
            # 进而以该名为关键词发起精确检索（探索-利用闭环）
            return "\n".join(
                f"[{e.idx}] {e.source}"
                + (f"（第{e.page}页）" if e.page else "")
                + f": {e.content[:140].replace(chr(10), ' ')}…"
                for e in evidence)

        def quotas_left() -> dict[str, int]:
            return {t: self.TOOL_QUOTAS[t] - tool_counts[t]
                    for t in self.enabled_tools}

        force_finish = False
        answer = None
        for step in range(1, self.MAX_STEPS + 1):
            # ---- 预算检查：证据超限 / 可用工具耗尽（配额用完或被熔断）→ 强制收尾 ----
            total_chars = sum(len(e.content) for e in evidence)
            no_tool_left = all(
                (t not in active) or tool_counts[t] >= self.TOOL_QUOTAS[t]
                for t in self.enabled_tools)
            if total_chars >= self.MAX_EVIDENCE_CHARS or no_tool_left:
                force_finish = True

            # ---- 决策（含 tool_calls 解析 + 自纠重试）----
            thought, name, args, raw_err = "", None, {}, None
            for attempt in range(self.MAX_PARSE_RETRY + 1):
                try:
                    thought, name, args = self._decide(
                        todo, evidence_index_text(),
                        self._scratchpad_text(trace), active,
                        quotas_left(), force_finish)
                    break
                except Exception as exc:  # 未调用工具/schema 错误 → 重试
                    raw_err = str(exc)
                    if attempt == self.MAX_PARSE_RETRY:
                        break
            if name is None:
                force_finish = True  # 解析持续失败 → 强制基于已有证据收尾
                thought, name, args = f"动作解析失败（{raw_err}），强制收尾", "finish", {}

            yield {"type": "thought", "step": step, "thought": thought,
                   "action": name, "force_finish": force_finish}

            # ---- finish：仅作收尾信号，最终报告基于证据原文合成 ----
            if name == "finish" or force_finish:
                answer = self._synthesize(question, sub_questions, evidence)
                break

            # ---- 未知/停用工具治理 ----
            if name not in self.enabled_tools or name not in active:
                reason = disabled_reasons.get(name, "")
                obs = (f"错误：未知、未启用或已停用的工具 '{name}'"
                       f"{('（' + reason + '）') if reason else ''}，"
                       f"可用：{', '.join(active)}、finish。")
                trace.append(TraceStep(
                    step, thought, str(name), args, obs,
                    rejected=True, latency_ms=0))
                yield {"type": "observation", "step": step, "text": obs,
                       "rejected": True}
                continue

            # ---- 配额治理 ----
            if tool_counts[name] >= self.TOOL_QUOTAS[name]:
                obs = (f"'{name}' 配额已用完（{self.TOOL_QUOTAS[name]} 次），"
                       f"请改用其他工具或 finish。")
                trace.append(TraceStep(
                    step, thought, name, args, obs, rejected=True))
                yield {"type": "observation", "step": step, "text": obs,
                       "rejected": True}
                continue

            # ---- 参数校验 ----
            query = str(args.get("query", "")).strip()
            code = str(args.get("code", "")).strip()
            if name in ("retrieve", "scholar_search") and not query:
                obs = f"错误：{name} 缺少非空 query 参数。"
                trace.append(TraceStep(
                    step, thought, name, args, obs, rejected=True))
                yield {"type": "observation", "step": step, "text": obs,
                       "rejected": True}
                continue
            if name == "python_exec" and not code:
                obs = "错误：python_exec 缺少非空 code 参数。"
                trace.append(TraceStep(
                    step, thought, name, args, obs, rejected=True))
                yield {"type": "observation", "step": step, "text": obs,
                       "rejected": True}
                continue

            # ---- 重复调用拦截（检索类按相似度，python 按代码哈希）----
            if name in ("retrieve", "scholar_search"):
                if self._is_duplicate_query(query, query_history[name]):
                    pending = "；".join(todo[:3]) if todo else "（无）"
                    obs = (f"已对「{query}」做过高度相似检索，禁止同义改写重试。"
                           f"请换用论文名/专有名词检索其他方向，待覆盖子问题："
                           f"{pending}；若均已覆盖请直接 finish。"
                           f"（本次不计配额）")
                    trace.append(TraceStep(
                        step, thought, name, {**args, "query": query}, obs,
                        rejected=True))
                    yield {"type": "observation", "step": step, "text": obs,
                           "rejected": True}
                    continue
            elif name == "python_exec":
                h = hashlib.md5(code.encode("utf-8")).hexdigest()
                if h in code_hashes:
                    obs = "错误：完全相同的代码已执行过，禁止重复执行。（本次不计配额）"
                    trace.append(TraceStep(
                        step, thought, name, {**args, "code": code}, obs,
                        rejected=True))
                    yield {"type": "observation", "step": step, "text": obs,
                           "rejected": True}
                    continue
                code_hashes.add(h)

            # ---- 执行工具 ----
            t0 = time.perf_counter()
            failed = False
            if name == "retrieve":
                items = self._tool_retrieve(query)
                obs = (f"命中 {len(items)} 块"
                       + (f"，首块来自 {items[0]['source']}" if items else ""))
            elif name == "scholar_search":
                try:
                    items = self._tool_scholar(query)
                    obs = (f"arXiv 命中 {len(items)} 篇："
                           + "；".join(i["source"] for i in items[:3])
                           + ("…" if len(items) > 3 else ""))
                    if not items:
                        obs = "arXiv 未命中任何论文，请换关键词或改用其他工具。"
                except Exception as exc:
                    items, failed = [], True
                    obs = (f"arXiv 检索失败：{type(exc).__name__}。"
                           f"该方向暂不可用，请改用 retrieve 或转向其他子问题。"
                           f"（本次不计配额）")
            else:  # python_exec
                r = self._tool_python(code)
                failed = not r["ok"]
                items = [{"source": "python_exec", "page": None,
                          "content": r["text"]}]
                obs = ("执行成功，输出 " + r["text"][:80].replace("\n", " ")
                       if r["ok"] else f"执行失败：{r['text'][:150]}")

            latency = int((time.perf_counter() - t0) * 1000)

            # ---- 记账（失败的网络/执行调用不计配额）----
            if not failed:
                tool_counts[name] += 1
                consecutive_fail[name] = 0
                if name in ("retrieve", "scholar_search"):
                    query_history[name].append(query)
            else:
                # 熔断器：连续失败达阈值 → 本轮停用该工具，避免模型反复撞墙
                consecutive_fail[name] = consecutive_fail.get(name, 0) + 1
                if consecutive_fail[name] >= self.FAIL_THRESHOLD and name in active:
                    active.remove(name)
                    disabled_reasons[name] = "连续失败熔断"
                    obs += (f" 已连续失败 {consecutive_fail[name]} 次，"
                            f"工具 '{name}' 本轮停用。")
            meta = [{"source": i["source"], "page": i["page"]} for i in items]
            all_retrieved_meta.extend(meta)

            # ---- 结构化记忆：证据按内容哈希去重累积 ----
            new_added = []
            for it in items:
                h = hashlib.md5(it["content"].encode("utf-8")).hexdigest()
                if h in seen_hashes:
                    continue
                seen_hashes.add(h)
                ev = Evidence(
                    idx=len(evidence) + 1,
                    source=it["source"],
                    page=it["page"],
                    content=it["content"],
                    content_hash=h)
                evidence.append(ev)
                new_added.append(ev)

            trace.append(TraceStep(
                step, thought, name, args, obs, retrieved=meta,
                latency_ms=latency))
            yield {"type": "observation", "step": step, "text": obs,
                   "query": query or None, "retrieved": meta,
                   "new_evidence": [e.idx for e in new_added],
                   "tool": name, "latency_ms": latency,
                   "quota_charged": not failed}

            # 配额恰好打满 → 提示模型下一步收尾
            if not failed and tool_counts[name] >= self.TOOL_QUOTAS[name]:
                if all((t not in active) or tool_counts[t] >= self.TOOL_QUOTAS[t]
                       for t in self.enabled_tools):
                    force_finish = True
        else:
            # 达到 MAX_STEPS 仍未 finish：基于已收集证据强制合成
            answer = self._synthesize(question, sub_questions, evidence)

        elapsed_ms = int((time.perf_counter() - t_start) * 1000)
        result = {
            "answer": answer,
            "sub_questions": sub_questions,
            "evidence": [asdict(e) for e in evidence],
            "trace": [asdict(t) for t in trace],
            "all_retrieved": all_retrieved_meta,
            "stats": {
                "steps": len(trace),
                "retrieve_calls": tool_counts.get("retrieve", 0),
                "tool_calls": dict(tool_counts),
                "evidence_count": len(evidence),
                "evidence_chars": sum(len(e.content) for e in evidence),
                "elapsed_ms": elapsed_ms,
                "queries": {t: qs for t, qs in query_history.items() if qs},
            },
        }
        self._write_trace(question, result)
        yield {"type": "final", **result}

    def _synthesize(self, question: str, sub_questions: list[str],
                    evidence: list[Evidence]) -> str:
        """最终综合：用全部证据原文生成带 [编号] 引用的结构化报告。"""
        if not evidence:
            return "根据现有资料无法回答该问题。"
        context = "\n\n".join(
            f"[{e.idx}]（{e.source}"
            + (f" 第{e.page}页" if e.page else "")
            + f"）\n{e.content}"
            for e in evidence)
        prompt = SYNTH_PROMPT.format(
            question=question,
            sub_questions="\n".join(f"- {s}" for s in sub_questions),
            evidence=context)
        try:
            return (self.llm.invoke([HumanMessage(content=prompt)]).content).strip()
        except Exception:
            return "根据现有资料无法回答该问题。"

    def _write_trace(self, question: str, result: dict) -> str:
        """全链路 trace 落 JSONL，供运行审计与回溯。"""
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(self.trace_dir, f"agent_trace_{ts}.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            f.write(json.dumps({"event": "run_start", "question": question,
                                "sub_questions": result["sub_questions"]},
                               ensure_ascii=False) + "\n")
            for t in result["trace"]:
                f.write(json.dumps({"event": "step", **t},
                                   ensure_ascii=False) + "\n")
            f.write(json.dumps({"event": "run_end", "stats": result["stats"]},
                               ensure_ascii=False) + "\n")
        return path

    def run(self, question: str) -> dict:
        """同步执行，返回 final 结果 dict。"""
        final = None
        for ev in self.run_stream(question):
            if ev["type"] == "final":
                final = ev
        return final
