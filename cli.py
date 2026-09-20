"""
交互式命令行问答 demo（开发期调试用）。

正式启动服务请运行：python app.py
"""

import env_setup  # noqa: F401  环境引导，必须最先导入

from rag_core import RAGEngine


def main():
    engine = RAGEngine()
    print("\n✅ RAG 引擎就绪，输入问题开始问答（q 退出）\n")
    while True:
        question = input("❓ 请输入问题：").strip()
        if question.lower() in ("q", "quit", "exit"):
            print("👋 已退出")
            break
        if not question:
            continue
        print("⏳ 思考中...")
        answer, sources = engine.ask(question)
        print("\n===== 回答 =====")
        print(answer)
        if sources:
            print("\n📎 来源：")
            for s in sources:
                page = f" · 第 {s['page']} 页" if s["page"] else ""
                print(f"  - {s['source']}{page}")
        print()


if __name__ == "__main__":
    main()
