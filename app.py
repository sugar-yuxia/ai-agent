"""
Web 服务入口：
- POST /api/ask     ：JSON 问答接口（mode=chat 单跳问答 / mode=research 调研 Agent）
- GET  /            ：自定义 Gradio 对话界面（快速问答 / 深度调研两种模式，
                       流式输出 + 思考过程 + 参考来源面板）

启动：python app.py            （使用缓存索引）
      python app.py --rebuild  （文档更新后重建索引）
"""

import env_setup  # noqa: F401,F402  必须在 gradio/uvicorn 之前导入以完成环境引导

import time
from html import escape as _esc

import uvicorn
import gradio as gr
from fastapi import FastAPI
from pydantic import BaseModel, Field

from rag_core import RAGEngine
from research_agent import ResearchAgent

# 服务启动时初始化一次 RAG 引擎，全局复用
engine = RAGEngine()
# 深度调研 Agent：多工具（本地检索/arXiv/Python 执行）+ function calling
agent = ResearchAgent(engine)

# ========== FastAPI JSON 接口 ==========

app = FastAPI(title="学术论文 RAG 问答服务", version="1.0")


class ChatMessage(BaseModel):
    role: str = Field(..., pattern="^(user|assistant)$")
    content: str


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, description="用户最新问题")
    history: list[ChatMessage] = Field(
        default_factory=list, description="历史对话（不含最新问题），用于多轮指代消解")
    mode: str = Field("chat", pattern="^(chat|research)$",
                      description="chat=单跳问答；research=多工具调研 Agent（较慢，10-60s）")


class SourceItem(BaseModel):
    source: str
    page: int | None = None
    snippet: str


class AskResponse(BaseModel):
    answer: str
    sources: list[SourceItem]


@app.post("/api/ask", response_model=AskResponse,
          summary="知识库问答（chat 单跳 / research 调研 Agent）")
def ask_api(req: AskRequest):
    history = [m.model_dump() for m in req.history]
    if req.mode == "research":
        result = agent.run(req.question)
        sources = [{
            "source": e["source"],
            "page": e["page"],
            "snippet": e["content"][:120].replace("\n", " ").strip(),
        } for e in result["evidence"]]
        return AskResponse(answer=result["answer"], sources=sources)
    answer, sources = engine.ask(req.question, history)
    return AskResponse(answer=answer, sources=sources)


# ========== Gradio 对话界面（自定义 Blocks 布局） ==========

HEADER_HTML = """
<div class="hero">
  <div class="hero-title">📚 学术论文智能问答系统</div>
  <div class="hero-sub">31 篇 WiFi-CSI 感知论文 · BM25 + 向量混合检索 · DeepSeek 生成 · 答案附来源溯源</div>
  <div class="hero-chips">
    <span class="chip">🔍 混合检索 39ms</span>
    <span class="chip">🛡️ 无依据拒答 100%</span>
    <span class="chip">📄 页码级溯源</span>
    <span class="chip">⚡ 流式输出</span>
  </div>
</div>
"""

FOOTER_HTML = """
<div class="footer">LangChain · FAISS · BM25 · BGE Embedding · FastAPI + Gradio ｜
评价体系：36 题评测集 · MRR@5 0.793 · Hit@1 73.3%</div>
"""

CUSTOM_CSS = """
/* 整体背景与字体 */
.gradio-container {
    background: linear-gradient(180deg, #f4f6fb 0%, #eef1f8 100%);
    font-family: "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif;
}
/* 顶部横幅 */
.hero {
    background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 55%, #a855f7 100%);
    border-radius: 16px;
    padding: 26px 28px 20px;
    color: #fff;
    margin-bottom: 10px;
    box-shadow: 0 8px 24px rgba(79, 70, 229, .25);
}
.hero-title { font-size: 26px; font-weight: 700; letter-spacing: .5px; }
.hero-sub  { font-size: 14px; opacity: .92; margin: 8px 0 12px; }
.hero-chips { display: flex; gap: 8px; flex-wrap: wrap; }
.chip {
    background: rgba(255, 255, 255, .18);
    border: 1px solid rgba(255, 255, 255, .35);
    border-radius: 999px;
    padding: 3px 12px;
    font-size: 12px;
    backdrop-filter: blur(4px);
}
/* 聊天气泡 */
.message-row .bubble {
    border-radius: 14px !important;
    box-shadow: 0 1px 3px rgba(0, 0, 0, .06);
}
/* 参考来源面板 */
.sources-panel {
    background: #fff;
    border: 1px solid #e5e9f2;
    border-radius: 12px;
    padding: 14px 16px;
    min-height: 200px;
}
.sources-panel h3 { margin: 0 0 10px; font-size: 15px; }
.src-item {
    background: #f7f8fc;
    border-left: 3px solid #6366f1;
    border-radius: 8px;
    padding: 8px 10px;
    margin-bottom: 8px;
    font-size: 13px;
    line-height: 1.55;
}
.src-name { font-weight: 600; color: #374151; }
.src-page { color: #6366f1; font-weight: 600; }
.src-snippet { color: #6b7280; }
/* 状态条 */
.status-bar { font-size: 12.5px; color: #8b92a5; padding: 2px 4px; }
/* 底部 */
.footer {
    text-align: center; color: #9aa2b1; font-size: 12.5px;
    padding: 14px 0 4px;
}
/* 示例按钮 */
.example-btn { border-radius: 999px !important; font-size: 13px !important; }
/* 模式切换 */
.mode-radio { justify-content: center; margin-bottom: 2px; }
.mode-radio label { font-size: 13.5px; }
/* 调研过程条目 */
.proc-item {
    background: #f5f3ff;
    border-left: 3px solid #a855f7;
    border-radius: 8px;
    padding: 8px 10px;
    margin-bottom: 8px;
    font-size: 12.5px;
    line-height: 1.6;
    color: #4b5563;
    word-break: break-word;
}
.proc-lat { color: #a855f7; font-size: 11px; font-weight: 600; }
/* 调研过程滚动区（限高，内部滚动，参考来源始终可见） */
.proc-scroll {
    max-height: 52vh;
    overflow-y: auto;
    padding-right: 6px;
    margin-top: 6px;
}
.proc-scroll::-webkit-scrollbar { width: 6px; }
.proc-scroll::-webkit-scrollbar-thumb {
    background: #d8b4fe; border-radius: 3px;
}
.proc-scroll::-webkit-scrollbar-track { background: #f5f3ff; border-radius: 3px; }
"""


def sources_panel_html(sources: list[dict]) -> str:
    """来源渲染为卡片列表（左色条 + 文件名 + 页码 + 片段）。"""
    if not sources:
        return '<div class="sources-panel">暂无来源</div>'
    items = []
    for s in sources:
        page = f'<span class="src-page"> · 第 {s["page"]} 页</span>' if s["page"] else ""
        snippet = (s["snippet"] or "")[:80]
        items.append(
            f'<div class="src-item">'
            f'<div class="src-name">📄 {s["source"]}{page}</div>'
            f'<div class="src-snippet">{snippet}…</div></div>')
    return (f'<div class="sources-panel"><h3>📎 参考来源（{len(sources)}）</h3>'
            + "".join(items) + "</div>")


def right_panel_html(proc: list[str], sources: list[dict]) -> str:
    """右侧面板：调研过程（Agent 思考链）+ 参考来源两段拼接。"""
    parts = []
    if proc:
        parts.append('<div class="sources-panel"><h3>🧠 调研过程</h3>'
                     '<div class="proc-scroll">'
                     + "".join(proc) + "</div></div>")
    parts.append(sources_panel_html(sources))
    return "".join(parts)


def chat_fn(message: str, history: list, mode: str = "chat"):
    """路由：快速问答（单跳 RAG）/ 深度调研（多工具 Agent）。"""
    if mode and "深度调研" in mode:
        yield from research_fn(message, history)
    else:
        yield from quick_ask_fn(message, history)


def quick_ask_fn(message: str, history: list):
    """快速问答：三路输出（对话历史 / 状态条 / 来源面板），支持多轮指代改写。"""
    if not message or not message.strip():
        yield history, "", sources_panel_html([]), ""
        return

    # Gradio 传入的 history 只含历史轮次，不含当前问题
    new_history = list(history) + [{"role": "user", "content": message}]
    yield new_history, "🧠 正在结合对话历史理解问题…", sources_panel_html([]), ""

    # 1) 查询改写：把"它/这个方法"等指代补全为独立检索问题
    t0 = time.perf_counter()
    rewritten = engine.rewrite_question(message, history)
    rewrite_ms = (time.perf_counter() - t0) * 1000
    if rewritten.strip() != message.strip():
        yield new_history, f"🧠 改写检索问题：「{rewritten}」", sources_panel_html([]), ""

    # 2) 用改写后的问题检索
    t1 = time.perf_counter()
    docs = engine.retrieve(rewritten)
    retrieve_ms = (time.perf_counter() - t1) * 1000
    sources = engine.format_sources(docs)
    status = (f"✅ 改写 {rewrite_ms:.0f}ms · 检索 {retrieve_ms:.0f}ms"
              f" · {len(docs)} 块 · 生成中…")
    yield new_history, status, sources_panel_html(sources), ""

    # 3) 携带历史流式生成（LLM 自行用历史理解指代，用上下文作答）
    partial = ""
    for token in engine.stream_answer(message, docs, history):
        partial += token
        yield new_history + [{"role": "assistant", "content": partial}], \
            status, sources_panel_html(sources), ""

    done = (f"✅ 回答完成（改写 {rewrite_ms:.0f}ms · 检索 {retrieve_ms:.0f}ms"
            f" · {len(docs)} 块上下文）")
    if rewritten.strip() != message.strip():
        done += f"\n🔎 实际检索问题：{rewritten}"
    yield new_history + [{"role": "assistant", "content": partial}], \
        done, sources_panel_html(sources), ""


def research_fn(message: str, history: list):
    """深度调研：流式展示 拆解 → 思考 → 工具观察 → 报告 的完整事件链。"""
    if not message or not message.strip():
        yield history, "", right_panel_html([], []), ""
        return

    new_history = list(history) + [{"role": "user", "content": message}]
    yield new_history, "📋 正在拆解调研问题…", right_panel_html([], []), ""

    proc: list[str] = []  # 思考过程 HTML 条目
    for ev in agent.run_stream(message):
        if ev["type"] == "plan":
            items = "<br>".join(
                f"{i}. {_esc(q)}"
                for i, q in enumerate(ev["sub_questions"], 1))
            proc.append(f'<div class="proc-item"><b>📋 调研规划</b><br>{items}</div>')
            yield new_history, (f"📋 已拆解 {len(ev['sub_questions'])} 个子问题，"
                                "开始多工具检索…"), right_panel_html(proc, []), ""
        elif ev["type"] == "thought":
            if ev["thought"]:
                proc.append(f'<div class="proc-item"><b>🤔 第 {ev["step"]} 步思考</b>'
                            f'<br>{_esc(ev["thought"])}</div>')
            yield new_history, f"🤔 第 {ev['step']} 步：{ev['action']}…", \
                right_panel_html(proc, []), ""
        elif ev["type"] == "observation":
            charged = "" if ev.get("quota_charged", True) else "（未计配额）"
            proc.append(f'<div class="proc-item"><b>🛠 {_esc(ev.get("tool", ""))} 观察</b>'
                        f'<br>{_esc(ev["text"])}{_esc(charged)}'
                        + (f' <span class="proc-lat">{ev["latency_ms"]}ms</span>'
                           if ev.get("latency_ms") else "") + "</div>")
            yield new_history, f"🛠 第 {ev['step']} 步执行完成，继续…", \
                right_panel_html(proc, []), ""
        elif ev["type"] == "final":
            srcs = [{"source": e["source"], "page": e["page"],
                     "snippet": e["content"][:120].replace("\n", " ").strip()}
                    for e in ev["evidence"]]
            tools_str = " · ".join(f"{k}×{v}" for k, v in
                                   ev["stats"]["tool_calls"].items() if v) or "无调用"
            st = (f"✅ 调研完成（{ev['stats']['steps']} 步 · {tools_str} · "
                  f"{ev['stats']['evidence_count']} 条证据 · "
                  f"{ev['stats']['elapsed_ms'] / 1000:.1f}s）")
            # 报告分块"流式"上屏
            ans = ev["answer"] or ""
            for i in range(0, max(len(ans), 1), 24):
                yield new_history + [{"role": "assistant", "content": ans[:i + 24]}], \
                    st, right_panel_html(proc, srcs), ""
            yield new_history + [{"role": "assistant", "content": ans}], st, \
                right_panel_html(proc, srcs), ""
            return


EXAMPLES = [
    "WiFi CSI 人体姿态估计有哪些主流方法？",
    "AdaPose 解决了什么问题？",
    "C-MambaPose 使用了什么框架结构？",
    "ESPARGOS 数据集的特点是什么？",
]

with gr.Blocks(title="学术论文智能问答") as demo:
    gr.HTML(HEADER_HTML)

    with gr.Row():
        # ------- 左侧：对话区 -------
        with gr.Column(scale=7):
            mode_radio = gr.Radio(
                choices=["💬 快速问答", "🔍 深度调研"],
                value="💬 快速问答", show_label=False, container=False,
                elem_classes=["mode-radio"])
            chatbot = gr.Chatbot(height=500, label=None, show_label=False)
            with gr.Row():
                msg = gr.Textbox(placeholder="输入你的问题，回车发送…",
                                 scale=9, show_label=False, container=False,
                                 autofocus=True)
                send = gr.Button("发送 🚀", variant="primary", scale=1)
            status = gr.Markdown("", elem_classes=["status-bar"])
            gr.HTML("<div style='height:2px'></div>")
            with gr.Row():
                example_btns = [gr.Button(e, size="sm", elem_classes=["example-btn"])
                                for e in EXAMPLES]

        # ------- 右侧：来源面板 -------
        with gr.Column(scale=3):
            sources_md = gr.HTML(sources_panel_html([]))

    gr.HTML(FOOTER_HTML)

    # ---------- 事件绑定 ----------
    # 清空输入框靠 generator 内部 yield 空字符串（msg 加在 outputs 末尾）。
    # .then() 在 Gradio 6 里要等 generator 完全结束才执行，对长流式无效。
    submit_triggers = [msg.submit, send.click]
    for trigger in submit_triggers:
        trigger(
            fn=chat_fn,
            inputs=[msg, chatbot, mode_radio],
            outputs=[chatbot, status, sources_md, msg],
        )

    for btn, text in zip(example_btns, EXAMPLES):
        # 闭包捕获 text；点击仅填入输入框，由用户手动点发送触发问答
        btn.click(fn=lambda t=text: t, outputs=[msg])

# Gradio 挂载到根路径；显式注册的 /api/ask 优先于根路径的 catch-all
# Gradio 6：theme/css 从 Blocks 构造器移到挂载/启动参数
app = gr.mount_gradio_app(
    app, demo, path="/",
    theme=gr.themes.Soft(primary_hue="indigo", neutral_hue="slate"),
    css=CUSTOM_CSS)


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=7860)
