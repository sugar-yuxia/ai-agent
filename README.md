# WiFi CSI 学术文献 Agent —— 基于 RAG 的多工具调研助手

一个面向 **WiFi CSI 感知领域学术调研** 的 RAG + Agent 系统。输入一个研究方向（如"轻量级边缘部署"或"跨域泛化"），Agent 自动拆解子问题、调用多工具检索、汇总证据，最终输出结构化调研报告。

核心演示：**Chat 单跳问答** vs **Research 多工具调研** 两种模式对比，展示 Agent 决策循环和工具治理带来的能力提升。

---

## 项目亮点

| 维度 | 实现 |
|---|---|
| **RAG 管线** | FAISS 向量检索 + BM25 关键词检索 → **RRF 0.5/0.5 混合融合**（MRR 0.793，比单路提升 67%） |
| **Agent 决策** | 多工具 ReAct 循环 + DeepSeek 原生 Function Calling（非正则 JSON 解析） |
| **工具治理** | 独立配额 / 重复调用拦截 / **熔断器**（连续失败自动停用）/ 预算强制收尾 |
| **可观测** | 每步 trace JSONL 落盘 + Web 端实时思考链可视化 |
| **评测体系** | 36 题标注测试集，MRR / Hit@K / 关键词覆盖率 / 拒绝准确率 |

---

## 架构概览

```
用户问题
  │
  ├── Chat 模式（单跳 RAG）─────────────────────────────┐
  │   多轮指代改写 → 混合检索 → 证据截断 → LLM 合成      │
  │                                                     │
  └── Research 模式（多工具 Agent）─────────────────────┤
      │                                                 │
      ▼                                                 │
  规划器（LLM 拆 2-4 子问题）                            │
      │                                                 │
      ▼                                                 │
  ┌─────────── ReAct 循环 ───────────┐                  │
  │  思考 → 工具调用 → 观察 → 再决策  │                  │
  │                                   │                  │
  │  可用工具：                        │                  │
  │  ├─ retrieve       本地文献库检索 │                  │
  │  ├─ scholar_search  Semantic Scholar 公开论文 │  ← 治理层       │
  │  ├─ python_exec    代码执行/计算 │  ├─ 配额限制     │
  │  └─ finish         收尾信号       │  ├─ 重复拦截     │
  │                                   │  ├─ 熔断器       │
  │                                   │  └─ 预算强制收尾 │
  └───────────┬──────────────────────┘                  │
              │                                         │
              ▼                                         │
         证据累积（按内容哈希去重）                       │
              │                                         │
              ▼                                         │
         最终综合（基于证据原文 + [编号] 引用）            │
              │                                         │
              ▼                                         ▼
      ┌───────────────────────────────┐
      │  FastAPI + Gradio 流式界面     │
      │  ├─ 思考链实时展示             │
      │  ├─ 证据来源卡片              │
      │  └─ trace JSONL 落盘          │
      └───────────────────────────────┘
```

---

## 技术选型与关键决策

### 1. 为什么 RRF 0.5/0.5 混合检索？

| 检索器 | MRR@5 | Hit@1 | Hit@3 | 平均延迟 |
|---|---|---|---|---|
| BM25 | 0.744 | 0.733 | 0.767 | 9 ms |
| 向量检索（bge-small） | 0.584 | 0.467 | 0.633 | 54 ms |
| **混合 RRF 0.5/0.5** | **0.793** | **0.733** | **0.867** | 39 ms |
| 混合 + bge-reranker-large | 0.751 | 0.667 | 0.833 | 13.6 s |

**发现**：Cross-Encoder 重排（bge-reranker-large）在这个**领域专用、证据相对密集**的学术语料上反而不如 RRF——原因是 reranker 倾向于把"综述型块"排在前面，挤压了直接证据块的排名。已默认关闭重排，保留为可切换选项。

### 2. 为什么手写 ReAct harness 而不是 LangGraph？

- **理解成本**：手写 200 行就能把 ReAct 的关键环节（思考→调用→观察→治理）走通，加强对每个决策点的理解
- **治理层可定制**：熔断器、配额、重复拦截等逻辑在 LangGraph 里要绕节点，手写直接嵌在主循环里


### 3. 为什么用熔断器？

trace 显示工具不可达时，每次失败耗数十秒。熔断器（连续失败 2 次即本轮停用）把 research 总耗时从 269s 降到 75s，且让模型自动转向其他工具。scholar_search 改用 Semantic Scholar API（国内可直连）替代原 arXiv。

### 4. Function Calling vs 正则 JSON 解析

早期版本用 `re.search(r"\{.*\}", text, re.S)` 解析工具调用参数。DeepSeek API 原生支持 `tool_calls` 结构化字段，改用 `bind_tools + pydantic schema` 后，参数校验和解析都由框架负责，稳定性显著提升。

---

## 评测结果

### 检索质量（36 题标注测试集，可回答 30 + 不可回答 6）

| 检索器 | Hit@1 | Hit@3 | Hit@5 | MRR@5 | 平均延迟 |
|---|---|---|---|---|---|
| BM25 | 0.733 | 0.767 | 0.767 | 0.744 | 9 ms |
| 向量检索 | 0.467 | 0.633 | 0.833 | 0.584 | 54 ms |
| **混合 RRF 0.5/0.5** | **0.733** | **0.867** | **0.867** | **0.793** | 39 ms |
| 混合 + reranker | 0.667 | 0.833 | 0.867 | 0.751 | 13.6 s |

### 生成质量

- 关键词覆盖率：0.678（回答覆盖证据关键词的比例）
- 拒绝准确率：1.0（对不可回答问题全部正确拒答）
- 平均生成延迟：1.05 s

### Chat vs Research 模式

Research 模式通过多轮多角度检索和证据累积，通常能覆盖更宽的子问题范围，但耗时更长（10-90s vs 1-3s）。对于需要"全面调研"的问题（如"对比 X 和 Y 的创新点"），Research 模式明显优于单跳问答。

![评测图表](eval_chart.png)

---

## 快速开始

### 方式一：Docker（推荐，一键跑通）

```bash
cd ai-agent
# 编辑 .env 填入 DEEPSEEK_API_KEY
docker compose up --build
# 打开 http://127.0.0.1:7860
```

### 方式二：本地运行

```bash
# 1. 克隆并安装依赖
git clone <repo-url>
cd ai-agent
pip install -r requirements.txt

# 2. 配置 API Key
cp .env.example .env
# 编辑 .env 填入 DEEPSEEK_API_KEY

# 3. 启动服务（首次会从缓存加载向量索引）
python app.py

# 4. 打开 http://127.0.0.1:7860
#    快速问答 / 深度调研 两种模式可切换
```

### 常用命令

```bash
# 深度调研 Agent 的 evaluate（仅本地检索，可复现）
python eval_agent.py

# RAG 管线评测（对比不同检索器）
python eval_rag.py --retrieval-only      # 仅测检索，不调 LLM
python eval_rag.py                       # 全链路评测

# 重建向量索引（文档更新后）
python app.py --rebuild

# 文档预检查
python check_docs.py

# 重新生成评测图表
python plot_eval.py

# 命令行调试
python cli.py
```

---

## 项目结构

```
ai-agent/
├── app.py               # Web 服务（FastAPI + Gradio，双模式）
├── rag_core.py          # RAG 引擎（混合检索 + 重排 + 多轮改写）
├── research_agent.py    # 多工具 ReAct Agent（function calling + 治理）
├── cli.py               # 命令行 demo
├── check_docs.py        # 文档完整性预检查
├── eval_rag.py          # RAG 管线评测
├── eval_agent.py        # Agent 评测
├── plot_eval.py         # 评测图表生成
├── config.py            # 路径与配置中心
├── env_setup.py         # 环境变量引导（.env + HF 镜像 + 离线模式）
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── .env.example
├── .gitignore
├── docs/                # 学术论文（PDF/Word/TXT）
├── models/              # 本地 HuggingFace 模型缓存
├── faiss_index/         # FAISS 向量索引缓存
├── runs/                # Agent trace JSONL 落盘
└── eval_*.json / .csv   # 评测数据与结果
```

---

## Agent 工具治理机制

```
每步决策流程：
  ┌─ 预算检查 ─────────────────────────────┐
  │  证据字符超限？配额全耗尽？             │──→ 强制 finish
  └────────────────────────────────────────┘
  │
  ▼
  ┌─ LLM 决策（bind_tools + tool_calls）───┐
  │  返回 (thought, tool_name, args)        │
  │  失败 → 自纠重试 MAX_PARSE_RETRY 次     │──→ 仍失败 → 强制 finish
  └────────────────────────────────────────┘
  │
  ▼
  ┌─ 治理层过滤 ────────────────────────────┐
  │  工具不存在/已熔断？  → 拦截 + 回灌     │
  │  配额耗尽？            → 拦截 + 回灌     │
  │  query 重复（sim≥0.85）？ → 拦截 + 回灌 │
  │  python 代码重复？      → 拦截 + 回灌   │
  │  参数缺失？            → 拦截 + 回灌     │
  └────────────────────────────────────────┘
  │
  ▼
  ┌─ 工具执行 ──────────────────────────────┐
  │  网络/执行失败 → 不计配额 + 回灌错误观察 │
  │  连续失败 ≥ FAIL_THRESHOLD(2) → 熔断器  │
  │  成功 → 证据按内容哈希去重累积           │
  └────────────────────────────────────────┘
```

---

## 注意事项

- **内存需求**：评测脚本（eval_rag.py / eval_agent.py）约需 2.4GB 内存，建议关闭其他占用内存的程序后运行
- **scholar_search 限流**：Semantic Scholar 免费 API 每秒 1 次，限流时自动退避重试。如需更高额度可在 `.env` 设 `S2_API_KEY`（免费申请）
- **首次启动**：HuggingFace 模型已缓存到 `models/` 目录，离线加载；`HF_ONLINE=1` 可强制联网检查更新

---

## 技术栈

- **LLM**：DeepSeek-V3（兼容 OpenAI API 的 Chat 端点）
- **嵌入**：BAAI/bge-small-zh-v1.5（本地离线）
- **重排**：BAAI/bge-reranker-large（本地离线，默认关闭）
- **向量库**：FAISS
- **检索**：LangChain BM25Retriever + FAISS 向量检索 + RRF 融合
- **Agent**：手写 ReAct harness + DeepSeek Function Calling（pydantic schema 校验）
- **Web**：FastAPI + Gradio 6（流式 SSE + 自定义 Blocks 布局）
- **文档解析**：PyPDF / Docx2txt
