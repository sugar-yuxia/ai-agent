"""
环境变量引导模块。
必须在 import gradio / huggingface_hub 等库之前最先导入：
huggingface_hub 在导入瞬间就读取并缓存这些环境变量，之后再设置无效。

用法（app.py / rag_core.py 第一行）：
    import env_setup  # noqa: F401

Key 管理：
- 优先从项目根目录 .env 文件加载（若安装了 python-dotenv）
- DEEPSEEK_API_KEY 必须由环境变量或 .env 提供，缺失时启动即报错
  （不再写任何兜底默认 key，避免泄露风险）
"""

import os
import sys
from pathlib import Path

# 从同目录 .env 加载（若 python-dotenv 可用；否则只依赖系统环境变量）
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent / ".env")
except ImportError:
    pass

# 国内 HuggingFace 镜像
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
# 模型已缓存到本地时强制离线，跳过对 huggingface.co 的版本校验（直连会超时）
# 首次下载新模型时设环境变量 HF_ONLINE=1 临时打开联网，下载完再删除该变量
if os.environ.get("HF_ONLINE") == "1":
    os.environ.pop("HF_HUB_OFFLINE", None)
else:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

# DeepSeek 配置：必须由环境变量或 .env 提供
_deepseek_key = os.environ.get("DEEPSEEK_API_KEY")
if not _deepseek_key:
    sys.stderr.write(
        "\n❌ 未检测到 DEEPSEEK_API_KEY。\n"
        "请在环境变量或项目根目录 .env 文件中设置，示例：\n"
        "  DEEPSEEK_API_KEY=sk-xxxxxxxx\n"
        "（可复制 .env.example 为 .env 后修改）\n\n")
    raise RuntimeError("DEEPSEEK_API_KEY 未配置")

os.environ.setdefault("OPENAI_API_KEY", _deepseek_key)
os.environ.setdefault("OPENAI_BASE_URL", "https://api.deepseek.com/v1")
