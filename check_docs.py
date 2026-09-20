"""
文档质量预检：在重建索引前扫描 docs/ 下所有 PDF/DOCX，识别会污染索引的坏文档。

检查项：
  PDF  ：页数、提取总字符数、平均字符/页、空页/近空页数量、疑似扫描件
  DOCX ：提取字符数、有效段落数
判定：
  - 平均字符/页 < 100 且存在大量空页 → 疑似扫描版（无文本层），标 ❌
  - 总字符数过少 → 提取失败/异常，标 ⚠️
  - 正常 → ✅

运行：python check_docs.py
"""

import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from pypdf import PdfReader
import docx2txt

from config import DOCS_DIR

# 判定阈值
MIN_CHARS_PER_PAGE = 100     # 低于此值且空页占比高视为扫描件
NEAR_EMPTY_PAGE_CHARS = 20   # 单页字符数低于此值记为近空页


def check_pdf(path: str) -> dict:
    name = os.path.basename(path)
    try:
        reader = PdfReader(path)
        n_pages = len(reader.pages)
        page_lens = []
        for page in reader.pages:
            text = page.extract_text() or ""
            page_lens.append(len(text.strip()))
        total = sum(page_lens)
        empty = sum(1 for l in page_lens if l < NEAR_EMPTY_PAGE_CHARS)
        avg = total / max(n_pages, 1)
        if avg < MIN_CHARS_PER_PAGE and empty / max(n_pages, 1) > 0.5:
            status = "❌ 疑似扫描件（无文本层）"
        elif total < 500:
            status = "⚠️ 提取文本过少"
        elif empty / max(n_pages, 1) > 0.3:
            status = "⚠️ 空页占比偏高"
        else:
            status = "✅"
        return {"name": name, "type": "PDF", "pages": n_pages,
                "chars": total, "avg_per_page": round(avg),
                "empty_pages": empty, "status": status}
    except Exception as e:
        return {"name": name, "type": "PDF", "pages": "-",
                "chars": 0, "avg_per_page": 0, "empty_pages": "-",
                "status": f"❌ 读取失败：{type(e).__name__}"}


def check_docx(path: str) -> dict:
    name = os.path.basename(path)
    try:
        text = docx2txt.process(path) or ""
        n_para = len([p for p in text.split("\n") if p.strip()])
        status = "✅" if len(text) >= 500 else "⚠️ 提取文本过少"
        return {"name": name, "type": "DOCX", "pages": "-",
                "chars": len(text), "avg_per_page": "-",
                "empty_pages": "-", "status": status}
    except Exception as e:
        return {"name": name, "type": "DOCX", "pages": "-",
                "chars": 0, "avg_per_page": "-", "empty_pages": "-",
                "status": f"❌ 读取失败：{type(e).__name__}"}


def main():
    files = sorted(
        f for f in os.listdir(DOCS_DIR)
        if f.lower().endswith((".pdf", ".docx")))
    print(f"🔍 扫描 {DOCS_DIR}，共 {len(files)} 个文件\n")

    results = []
    for f in files:
        path = os.path.join(DOCS_DIR, f)
        r = check_pdf(path) if f.lower().endswith(".pdf") else check_docx(path)
        results.append(r)
        print(f"[{r['status']}] {f[:60]}")

    # 汇总表
    print("\n" + "=" * 100)
    print(f"{'文件名':<58}{'类型':<6}{'页数':>5}{'字符数':>9}{'均字/页':>8}{'空页':>5}  状态")
    print("-" * 100)
    for r in results:
        print(f"{r['name'][:56]:<58}{r['type']:<6}{str(r['pages']):>5}"
              f"{r['chars']:>9}{str(r['avg_per_page']):>8}"
              f"{str(r['empty_pages']):>5}  {r['status']}")

    bad = [r for r in results if r["status"].startswith("❌")]
    warn = [r for r in results if r["status"].startswith("⚠️")]
    total_chars = sum(r["chars"] for r in results)
    print("=" * 100)
    print(f"合计 {len(results)} 个文件 | 总提取字符 {total_chars:,} | "
          f"✅ {len(results)-len(bad)-len(warn)} | ⚠️ {len(warn)} | ❌ {len(bad)}")
    if bad:
        print("\n❌ 以下文件建议从 docs/ 移出（或 OCR 后再用）：")
        for r in bad:
            print("   -", r["name"])
    if warn:
        print("\n⚠️ 以下文件可保留，但会产生少量低质量块，建议抽查：")
        for r in warn:
            print("   -", r["name"])


if __name__ == "__main__":
    main()
