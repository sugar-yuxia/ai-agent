# 基础镜像：python 3.11 + slim 足够，HuggingFace 模型从本地卷加载
FROM python:3.11-slim

WORKDIR /app

# 系统依赖：docx2txt 需要 unzip，pypdf 纯 Python 不需要额外依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
        unzip curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# 先装依赖（利用 Docker 缓存：requirements.txt 不变则复用层）
COPY requirements.txt .
# --no-cache-dir 减小镜像体积；timeout 防国内网络抖动
RUN pip install --no-cache-dir --default-timeout=120 -r requirements.txt

# 应用代码（最后拷贝，代码变动不影响依赖层缓存）
COPY . .

# 环境变量：HF 镜像 + 离线模式 + DeepSeek API Key
# DEEPSEEK_API_KEY 从宿主机 .env 文件传入（docker-compose.yml 配置）
ENV HF_ENDPOINT=https://hf-mirror.com \
    HF_HUB_OFFLINE=1 \
    PYTHONUNBUFFERED=1

EXPOSE 7860

# uvicorn 托管 FastAPI + Gradio 挂载
# host 0.0.0.0 让容器外能访问
CMD ["python", "app.py"]
