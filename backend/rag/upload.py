# backend/rag/upload.py
#!/usr/bin/env python
"""
knowledge_kiwi.json을 Qdrant에 업로드
Usage: python upload.py
"""

import json
import sys
import hashlib
from pathlib import Path
import numpy as np
import os

sys.path.append(str(Path(__file__).parent.parent))
from rag.vectorstore import QdrantVectorStore
from rag.embedder import BGEEmbedder


def upload_to_qdrant():
    """knowledge_kiwi.json을 Qdrant에 업로드"""

    # 실제 임베딩 사용하려면 환경변수 설정
    os.environ["USE_REAL_EMBEDDING"] = "true"

    # 1. knowledge_kiwi.json 로드
    knowledge_path = Path(__file__).parent.parent / "data" / "knowledge_kiwi.json"

    if not knowledge_path.exists():
        print(f"❌ {knowledge_path} not found!")
        return

    print(f"📂 Loading {knowledge_path}")
    with open(knowledge_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # knowledge 리스트 추출
    knowledge = data.get("knowledge", [])
    print(f"📊 Found {len(knowledge)} items")

    # 2. BGE-M3 임베더와 벡터스토어 초기화
    print("🔧 Initializing BGE-M3 embedder and Qdrant...")
    embedder = BGEEmbedder()
    vector_store = QdrantVectorStore()

    # 3. 컬렉션 생성 (BGE-M3 차원: 1024)
    vector_store.create_collection(dimension=1024)

    # 4. 배치로 처리
    batch_size = 10

    for i in range(0, len(knowledge), batch_size):
        batch = knowledge[i : i + batch_size]
        print(
            f"🔄 Processing batch {i//batch_size + 1}/{(len(knowledge)-1)//batch_size + 1}"
        )

        # 텍스트 리스트
        texts = [item["text"] for item in batch]

        # embed_documents 메서드 사용!
        embeddings = embedder.embed_documents(texts)

        # Qdrant에 추가할 문서 형식
        documents = []
        for item in batch:
            # doc_key가 없으면 생성
            doc_key = item.get("doc_key")
            if not doc_key:
                doc_key = hashlib.sha1(item["text"].encode("utf-8")).hexdigest()

            doc = {
                "text": item["text"],
                "doc_key": doc_key,  # 추가
                "metadata": {
                    "id": item["id"],
                    "doc_key": doc_key,  # 메타데이터에도 추가
                    "category": item.get("category", ""),
                    "keywords": item.get("keywords", []),
                    "priority": item.get("priority", 5),
                },
                "source": "knowledge_faq",
            }
            documents.append(doc)

        # Qdrant에 업로드
        vector_store.add_documents(documents, embeddings)
        print(f"  ✅ Uploaded {len(documents)} documents with doc_keys")

    print("\n✨ All documents uploaded to Qdrant!")

    # 5. BM25 인덱스 재구축
    print("🔄 Rebuilding BM25 index with Kiwi tokens...")
    vector_store.rebuild_bm25_index()

    print("🎉 Complete! Knowledge base is ready for hybrid search!")


if __name__ == "__main__":
    upload_to_qdrant()
