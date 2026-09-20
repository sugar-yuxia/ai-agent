"""
项目路径与配置中心。

所有跨模块共享的路径都从这里取，避免硬编码绝对路径导致跨机不可用。
路径解析基于本文件所在目录，clone 到任何位置都能直接运行。

环境变量优先级：系统 env > 项目根目录 .env > 内置默认（相对路径）。
"""

from __future__ import annotations

import os
from pathlib import Path

# 项目根目录：本文件位于根，向上取一层即可
PROJECT_ROOT = Path(__file__).resolve().parent

# ---- 运行时数据目录 ----
DOCS_DIR = PROJECT_ROOT / "docs"
INDEX_DIR = PROJECT_ROOT / "faiss_index"
MODELS_DIR = PROJECT_ROOT / "models"
RUNS_DIR = PROJECT_ROOT / "runs"

# ---- 评测产物 ----
EVAL_QUESTIONS_FILE = PROJECT_ROOT / "eval_questions.json"
EVAL_REPORT_JSON = PROJECT_ROOT / "eval_report.json"
EVAL_DETAIL_CSV = PROJECT_ROOT / "eval_detail.csv"
RESEARCH_QUESTIONS_FILE = PROJECT_ROOT / "research_questions.json"
AGENT_EVAL_REPORT = PROJECT_ROOT / "agent_eval_report.json"
EVAL_CHART_PNG = PROJECT_ROOT / "eval_chart.png"

# ---- 模型默认路径/名（可被环境变量覆盖）----
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5")
RERANKER_MODEL = os.environ.get(
    "RERANKER_MODEL", str(MODELS_DIR / "bge-reranker-large"))

# ---- 确保运行时目录存在（首次启动自动创建）----
for _d in (DOCS_DIR, INDEX_DIR, MODELS_DIR, RUNS_DIR):
    _d.mkdir(parents=True, exist_ok=True)
