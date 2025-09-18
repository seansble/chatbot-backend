# clear_and_reindex.py
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / "backend"))

from backend.rag.vectorstore import QdrantVectorStore

# 1. 컬렉션 삭제
vector_store = QdrantVectorStore()
try:
    vector_store.client.delete_collection("unemployment_rag")
    print("✅ 기존 컬렉션 삭제")
except:
    pass

# 2. 다시 rewriteq.py 실행
print("이제 python rewriteq.py 실행하세요")