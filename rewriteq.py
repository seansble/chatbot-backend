# rewriteq_debug.py
#!/usr/bin/env python
import sys
import json
from pathlib import Path
import numpy as np
from dotenv import load_dotenv  # 추가!
import os


load_dotenv()  # ← 이거 추가!

# 확인
print(f"QDRANT_URL: {os.getenv('QDRANT_URL')}")
print(f"QDRANT_API_KEY: {'설정됨' if os.getenv('QDRANT_API_KEY') else '없음'}")

sys.path.append(str(Path(__file__).parent / "backend"))

from backend.rag.embedder import BGEEmbedder
from backend.rag.vectorstore import QdrantVectorStore


def index_to_qdrant():
    print("🚀 Indexing documents to Qdrant")
    print("=" * 50)

    # 1. JSON 로드
    knowledge_path = Path(__file__).parent / "backend" / "data" / "knowledge_kiwi.json"
    with open(knowledge_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    knowledge_items = data["knowledge"]
    print(f"📊 Loaded {len(knowledge_items)} documents from JSON")

    # 카테고리별 확인
    categories = {}
    for item in knowledge_items:
        cat = item.get("category", "unknown")
        categories[cat] = categories.get(cat, 0) + 1

    print("\n📂 카테고리별 문서 수:")
    for cat, count in sorted(categories.items()):
        print(f"  - {cat}: {count}개")

    # 2. 벡터스토어 초기화
    print("\n🔧 Initializing Qdrant...")
    vector_store = QdrantVectorStore()

    # 기존 컬렉션 삭제
    try:
        vector_store.client.delete_collection("unemployment_rag")
        print("🗑️ 기존 컬렉션 삭제")
    except:
        pass

    # 3. 새 컬렉션 생성
    vector_store.create_collection(dimension=1024)

    # 4. 문서 준비
    print("\n📝 Preparing documents...")
    documents = []
    for item in knowledge_items:
        doc = {
            "text": item["text"],
            "parent_text": item.get("parent_text", item["text"]),
            "doc_key": item.get("doc_key"),
            "metadata": {
                "category": item.get("category", ""),
                "keywords": item.get("keywords", []),
                "priority": item.get("priority", 5),
                "question": item.get("question", ""),
                "answer": item.get("answer", ""),
            },
            "source": "knowledge_faq",
        }
        documents.append(doc)

    print(f"✅ {len(documents)}개 문서 준비 완료")

    # 5. 임베딩 생성
    print("\n🧮 Creating embeddings...")
    embedder = BGEEmbedder()
    texts = [doc["text"] for doc in documents]
    embeddings = embedder.embed_documents(texts)
    print(f"✅ {len(embeddings)}개 임베딩 생성 완료")

    # 6. Qdrant에 저장
    print("\n💾 Saving to Qdrant...")
    vector_store.add_documents(documents, np.array(embeddings))

    # 7. 검증
    print("\n🔍 Verifying upload...")
    from qdrant_client import QdrantClient
    import os

    client = QdrantClient(
        url=os.getenv("QDRANT_URL"), api_key=os.getenv("QDRANT_API_KEY")
    )

    info = client.get_collection("unemployment_rag")
    print(f"✅ Qdrant에 {info.points_count}개 문서 저장됨")

    if info.points_count != len(documents):
        print(f"⚠️ 경고: {len(documents) - info.points_count}개 문서 누락!")


if __name__ == "__main__":
    index_to_qdrant()
