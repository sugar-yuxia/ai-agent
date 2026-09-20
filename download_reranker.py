"""
独立下载 bge-reranker-large 到本地 HF 缓存。
- 纯 ASCII 输出，避免 Windows GBK 控制台 UnicodeEncodeError
- snapshot_download 支持断点续传，只拉取 CrossEncoder 必需文件
- 环境变量在 env_setup 之前单独设置（HF_ONLINE=1 时关闭离线模式）

用法：
  # PowerShell:
  $env:HF_ONLINE = "1"; python download_reranker.py
"""

import os
import sys
import time

# 强制 stdout UTF-8，双重保险
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ.pop("HF_HUB_OFFLINE", None)          # 下载必须联网
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "60"     # 单次请求超时 60s
os.environ["PYTHONIOENCODING"] = "utf-8"

MODEL_ID = "BAAI/bge-reranker-large"

# CrossEncoder 加载只需要这些；排除 onnx/flax/tf/pytorch_model.bin(用 safetensors)
ALLOW = [
    "config.json",
    "model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.txt",
    "special_tokens_map.json",
]


def cache_size_mb() -> float:
    root = os.path.expanduser(r"~\.cache\huggingface\hub")
    total = 0
    for dirpath, _, files in os.walk(root):
        if "bge-reranker-large" not in dirpath:
            continue
        for f in files:
            try:
                total += os.path.getsize(os.path.join(dirpath, f))
            except OSError:
                pass
    return total / (1024 * 1024)


def main():
    from huggingface_hub import snapshot_download

    print(f"[INFO] target model : {MODEL_ID}", flush=True)
    print(f"[INFO] mirror       : {os.environ['HF_ENDPOINT']}", flush=True)
    print(f"[INFO] cache size before: {cache_size_mb():.1f} MB", flush=True)
    print("[INFO] downloading (resumable; this model is ~1.3 GB)...",
          flush=True)

    t0 = time.time()
    last = [-1.0]

    def progress(downloaded, total):
        # huggingface_hub 回调；约每 10MB 打印一行
        if total > 0:
            pct = downloaded / total * 100
            if pct - last[0] >= 5 or downloaded == total:
                last[0] = pct
                print(f"[PROGRESS] {pct:5.1f}%  "
                      f"{downloaded/1e6:7.1f}/{total/1e6:.1f} MB", flush=True)

    path = snapshot_download(
        repo_id=MODEL_ID,
        allow_patterns=ALLOW,
        max_workers=2,          # 低并发，降低连接重置概率
    )
    dt = time.time() - t0
    print(f"[OK] snapshot path  : {path}", flush=True)
    print(f"[OK] cache size after : {cache_size_mb():.1f} MB", flush=True)
    print(f"[OK] elapsed        : {dt:.0f} s", flush=True)

    # 立即做一次最小加载验证（不触发联网）
    os.environ["HF_HUB_OFFLINE"] = "1"
    print("[INFO] verifying CrossEncoder load (offline)...", flush=True)
    from sentence_transformers import CrossEncoder
    model = CrossEncoder(MODEL_ID)
    score = model.predict([("WiFi CSI pose estimation",
                            "This paper studies human pose from WiFi CSI.")])[0]
    print(f"[OK] CrossEncoder loaded, sanity score = {float(score):.4f}",
          flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[ERROR] {type(e).__name__}: {e}", flush=True)
        sys.exit(1)
