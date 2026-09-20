"""
RAG 核心引擎：文档加载 → 中文语义切分 → 混合检索（BM25 + 向量）→ Cross-Encoder 精排 → 流式生成
- FAISS 索引与切分结果本地持久化，二次启动免重建
- BM25 解决纯向量检索对专业术语（模型名、缩写）召回不准的问题
- 两阶段检索：混合召回扩大候选池 → bge-reranker Cross-Encoder 精排取 top-k
"""

import env_setup  # noqa: F401  环境引导，必须最先导入，在 huggingface 系库之前设置环境变量

import os
import sys
import pickle

from config import DOCS_DIR, INDEX_DIR, EMBEDDING_MODEL, RERANKER_MODEL
from langchain_deepseek import ChatDeepSeek
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import (
    DirectoryLoader, TextLoader, PyPDFLoader, Docx2txtLoader,
)
from langchain_community.retrievers import BM25Retriever
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_classic.retrievers import EnsembleRetriever

# 多轮对话的问答 Prompt：注入历史消息 + 检索上下文
# 显式约束"无依据则拒答"，抑制幻觉
RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """你是一个严谨的知识库问答助手。请严格基于下面提供的上下文回答问题。

要求：
1. 答案必须来自上下文，不要编造上下文中没有的信息
2. 如果上下文与问题无关或信息不足，请直接回答"根据现有资料无法回答该问题"
3. 使用中文回答，条理清晰
4. 可以结合对话历史理解用户的指代（如"它""这个方法"），但事实依据只能来自上下文

上下文：
{context}"""),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{question}"),
])

# 查询改写 Prompt：把带指代/省略的追问改写成独立、完整的检索问题
CONTEXTUALIZE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """根据对话历史，把用户的最新问题改写为一个独立、完整、可直接用于文献检索的中文问题。
要求：
- 补全"它/这个/该方法/上面提到的"等指代所指的具体名词
- 只输出改写后的问题本身，不要任何解释、前缀或回答
- 如果最新问题本身已经完整独立，原样返回"""),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{question}"),
])


class BgeReranker:
    """基于 BGE-reranker-v2-m3 的 Cross-Encoder 精排器。

    与双塔 embedding 不同：Cross-Encoder 把 (query, doc) 拼成一句话输入
    Transformer，输出单一相关度分数。精度更高但更慢，适合对召回结果二次精排。
    """

    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3"):
        from sentence_transformers import CrossEncoder
        self.model = CrossEncoder(model_name)
        self.model.model.eval()  # 推理模式，关闭 dropout

    def rank(self, query: str, docs: list[Document],
             top_n: int | None = None) -> list[Document]:
        """对 docs 按与 query 的相关度排序，返回 top_n（默认全部）。"""
        if not docs:
            return []
        pairs = [(query, d.page_content) for d in docs]
        scores = self.model.predict(pairs).tolist()
        ranked = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
        result = [d for _, d in ranked]
        return result[:top_n] if top_n else result


class RAGEngine:
    """封装文档解析、混合检索、Cross-Encoder 精排、LLM 生成的完整 RAG 管线。"""

    def __init__(
        self,
        docs_dir: str = str(DOCS_DIR),
        index_dir: str = str(INDEX_DIR),
        embedding_model: str = EMBEDDING_MODEL,
        # 本地路径加载 bge-reranker-large（2.1GB，curl 直链下载，离线可用）
        reranker_model: str = RERANKER_MODEL,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        fetch_k: int = 8,       # 两阶段：先召回 8 候选
        rerank_top_n: int = 2,  # 精排后取 top-2（精排已筛过，少而精）
        final_top_k: int = 4,   # 无精排时直接给 LLM 的 RRF 前 N 块（多给上下文）
        # reranker 开关：large 模型经评测确认有效后可在 app 中开启
        use_reranker: bool = False,
        # 融合权重由 eval_rag.py 在 36 题评测集上网格搜索确定：
        # 语料扩至 31 篇后 0.5/0.5 最优，MRR@5 0.793（BM25 单路 0.744 / 向量单路 0.584）
        bm25_weight: float = 0.5,
        vector_weight: float = 0.5,
    ):
        self.docs_dir = docs_dir
        self.index_dir = index_dir
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.fetch_k = fetch_k
        self.rerank_top_n = rerank_top_n
        self.final_top_k = final_top_k
        self.use_reranker = use_reranker

        os.makedirs(index_dir, exist_ok=True)
        self.chunks_cache = os.path.join(index_dir, "chunks.pkl")

        print("⏳ 初始化 Embedding 模型...")
        self.embedding = HuggingFaceEmbeddings(model_name=embedding_model)
        self.llm = ChatDeepSeek(model="deepseek-chat", temperature=0)
        self.answer_chain = RAG_PROMPT | self.llm | StrOutputParser()
        # 查询改写链（多轮指代消解），仅在存在历史时调用
        self.rewrite_chain = CONTEXTUALIZE_PROMPT | self.llm | StrOutputParser()

        # Reranker 默认关闭：初始化成本较高，开启时才加载
        self.reranker = None
        if use_reranker:
            print("⏳ 初始化 Reranker（Cross-Encoder）...")
            self.reranker = BgeReranker(reranker_model)
            print("✅ Reranker 就绪")

        rebuild = "--rebuild" in sys.argv
        self._init_retriever(rebuild, bm25_weight, vector_weight)

    # ---------- 构建 / 加载 ----------

    def _load_raw_documents(self) -> list[Document]:
        """按扩展名分别加载，避免二进制文件（docx/pdf）被当纯文本读取。"""
        docs: list[Document] = []
        docs += DirectoryLoader(
            self.docs_dir, glob="**/*.txt", loader_cls=TextLoader,
            loader_kwargs={"autodetect_encoding": True}).load()
        docs += DirectoryLoader(
            self.docs_dir, glob="**/*.pdf", loader_cls=PyPDFLoader).load()
        docs += DirectoryLoader(
            self.docs_dir, glob="**/*.docx", loader_cls=Docx2txtLoader).load()
        return docs

    def _split(self, docs: list[Document]) -> list[Document]:
        """中文友好分隔：段落 → 句子 → 逗号，尽量不切断语义。"""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""],
        )
        return splitter.split_documents(docs)

    def _init_retriever(self, rebuild: bool, bm25_weight: float, vector_weight: float):
        cache_valid = (
            not rebuild
            and os.path.exists(self.chunks_cache)
            and os.path.exists(os.path.join(self.index_dir, "index.faiss"))
        )

        if cache_valid:
            # 缓存命中：加载切分块与 FAISS 索引，跳过耗时的 PDF 解析与向量化
            with open(self.chunks_cache, "rb") as f:
                self.chunks = pickle.load(f)
            vector_db = FAISS.load_local(
                self.index_dir, self.embedding,
                allow_dangerous_deserialization=True,  # 仅加载本地自建索引
            )
            print(f"✅ 已从缓存加载 {len(self.chunks)} 个文本块")
        else:
            print("⏳ 加载文档...")
            docs = self._load_raw_documents()
            print(f"✅ 已加载 {len(docs)} 个文档")
            self.chunks = self._split(docs)
            print(f"✅ 切分完成，共 {len(self.chunks)} 个块")

            print("⏳ 构建向量索引...")
            vector_db = FAISS.from_documents(self.chunks, self.embedding)
            vector_db.save_local(self.index_dir)
            with open(self.chunks_cache, "wb") as f:
                pickle.dump(self.chunks, f)
            print(f"✅ 向量索引已保存到 {self.index_dir}")

        # 暴露给评测/外部使用：向量库与两路单路检索器
        self.vector_db = vector_db
        # 两阶段检索的召回器：BM25（关键词）+ FAISS（语义），RRF 融合
        # 注意 fetch_k 提升到 8，为后续 reranker 留足候选池
        self.bm25_retriever = BM25Retriever.from_documents(self.chunks)
        self.bm25_retriever.k = self.fetch_k
        self.vector_retriever = vector_db.as_retriever(
            search_kwargs={"k": self.fetch_k})
        self.candidate_retriever = EnsembleRetriever(
            retrievers=[self.bm25_retriever, self.vector_retriever],
            weights=[bm25_weight, vector_weight],
        )

        # 完整检索器（外部用 self.retriever.invoke(question) 调用）
        self.retriever = self.candidate_retriever  # 占位；retrieve() 内做完整两阶段
        print("✅ 两阶段检索器（BM25+向量召回 → bge-reranker 精排）就绪")

    def _dedup(self, docs: list[Document]) -> list[Document]:
        """RRF 融合后会出现重复块，按内容去重避免 reranker 浪费算力。"""
        seen = set()
        result = []
        for d in docs:
            key = hash(d.page_content)
            if key not in seen:
                seen.add(key)
                result.append(d)
        return result

    def retrieve(self, question: str) -> list[Document]:
        """检索：混合召回 → 去重 →（可选）bge-reranker 精排取 top-N。

        use_reranker=False 时直接返回 RRF 融合排序的前 N 块（轻量上线路径）；
        True 时用 Cross-Encoder 对候选池二次精排。
        """
        candidates = self._dedup(self.candidate_retriever.invoke(question))
        if self.use_reranker and self.reranker is not None:
            return self.reranker.rank(question, candidates,
                                      top_n=self.rerank_top_n)
        return candidates[:self.final_top_k]

    @staticmethod
    def format_sources(docs: list[Document]) -> list[dict]:
        """提取去重后的来源信息（文件名、页码），用于回答溯源。"""
        seen = set()
        sources = []
        for d in docs:
            path = d.metadata.get("source", "未知来源")
            name = os.path.basename(path)
            page = d.metadata.get("page")  # PDF 页码从 0 开始
            key = (name, page)
            if key in seen:
                continue
            seen.add(key)
            sources.append({
                "source": name,
                "page": (page + 1) if isinstance(page, int) else None,
                "snippet": d.page_content[:120].replace("\n", " ").strip(),
            })
        return sources

    def _build_context(self, docs: list[Document]) -> str:
        return "\n\n".join(d.page_content for d in docs)

    # ---------- 多轮对话：历史消息转换与查询改写 ----------

    @staticmethod
    def _to_lc_messages(history: list[dict] | None):
        """Gradio messages 格式 [{role, content}] → LangChain 消息对象。

        只保留最近 6 轮（12 条），避免历史过长稀释上下文、增加 token。
        """
        if not history:
            return []
        msgs = []
        for m in history[-12:]:
            role, content = m.get("role"), m.get("content", "")
            if not content:
                continue
            msgs.append(HumanMessage(content=content) if role == "user"
                        else AIMessage(content=content))
        return msgs

    def rewrite_question(self, question: str,
                         history: list[dict] | None) -> str:
        """用 LLM 把带指代的追问改写为独立检索问题；无历史或失败时原样返回。"""
        if not history:
            return question
        try:
            rewritten = self.rewrite_chain.invoke({
                "history": self._to_lc_messages(history),
                "question": question,
            }).strip()
            # 兜底：改写结果异常（为空/过长）时退回原问题
            return rewritten if rewritten and len(rewritten) <= len(question) * 4 \
                else question
        except Exception:
            return question  # 改写失败不应阻断主流程

    # ---------- 查询 ----------

    def ask(self, question: str,
            history: list[dict] | None = None) -> tuple[str, list[dict]]:
        """非流式问答（支持多轮历史）：返回（答案，来源列表）。"""
        rewritten = self.rewrite_question(question, history)
        docs = self.retrieve(rewritten)
        answer = self.answer_chain.invoke({
            "context": self._build_context(docs),
            "history": self._to_lc_messages(history),
            "question": question,
        })
        return answer, self.format_sources(docs)

    def stream_answer(self, question: str, docs: list[Document],
                      history: list[dict] | None = None):
        """基于给定检索结果流式生成答案（注入历史），token 生成器。"""
        yield from self.answer_chain.stream({
            "context": self._build_context(docs),
            "history": self._to_lc_messages(history),
            "question": question,
        })
