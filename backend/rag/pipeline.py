# backend/rag/pipeline.py (Railway에서 실행)
import os
import json
import uuid
from typing import List, Dict
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import numpy as np
from tqdm import tqdm
from kiwipiepy import Kiwi


class RAGPipeline:
    def __init__(self):
        print("🚀 RAG Pipeline 초기화 중...")

        # 1. Kiwipiepy 초기화 (선택사항)
        print("🇰🇷 Kiwi 한국어 처리기 로딩...")
        self.kiwi = Kiwi()

        # 2. BGE-M3 임베딩 모델 (embedder.py와 동일!)
        print("🤖 BGE-M3 임베딩 모델 로딩...")
        self.embedder = SentenceTransformer("BAAI/bge-m3")

        # 3. Qdrant 연결 (config.py에서 가져오기)
        print("☁️ Qdrant Cloud 연결중...")
        try:
            # Railway 환경에서는 config에서
            from config import QDRANT_URL, QDRANT_API_KEY
        except ImportError:
            # 로컬 테스트시 환경변수
            QDRANT_URL = os.getenv("QDRANT_URL")
            QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

        self.qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

        print("✅ 모든 컴포넌트 준비 완료!\n")

    def parse_documents(self, file_paths: List[str]) -> List[Dict]:
        """문서 파싱 - 심플 버전"""
        all_texts = []

        for file_path in file_paths:
            print(f"📖 파싱 중: {file_path}")

            if file_path.endswith(".json"):
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    for item in data.get("faqs", []):
                        text = f"질문: {item['q']}\n답변: {item['a']}"

                        all_texts.append(
                            {
                                "text": text,
                                "metadata": {
                                    "source": "knowledge.json",
                                    "category": item.get("category", ""),
                                    "keywords": item.get("keywords", []),
                                },
                            }
                        )

        print(f"✅ {len(all_texts)}개 문서 파싱 완료")
        return all_texts

    def chunk_texts(self, documents: List[Dict], max_size: int = 500) -> List[Dict]:
        """한국어 기반 청킹"""
        chunks = []

        print("✂️ 한국어 청킹 시작...")
        for doc in tqdm(documents, desc="청킹"):
            text = doc["text"]

            # FAQ는 이미 구조화되어 있으므로 그대로 유지
            if "질문:" in text and "답변:" in text:
                chunks.append(doc)
                continue

            # 긴 텍스트는 문장 단위로 분할
            if len(text) > max_size:
                sentences = self.kiwi.split_into_sents(text)
                current_chunk = ""
                current_size = 0

                for sent in sentences:
                    sent_text = sent.text
                    if current_size + len(sent_text) > max_size:
                        if current_chunk:
                            chunks.append(
                                {"text": current_chunk, "metadata": doc["metadata"]}
                            )
                        current_chunk = sent_text
                        current_size = len(sent_text)
                    else:
                        current_chunk += " " + sent_text
                        current_size += len(sent_text)

                if current_chunk:
                    chunks.append({"text": current_chunk, "metadata": doc["metadata"]})
            else:
                chunks.append(doc)

        print(f"✅ {len(chunks)}개 청크 생성 완료")
        return chunks

    def embed_chunks(self, chunks: List[Dict]) -> np.ndarray:
        """임베딩 생성"""
        print("🧮 임베딩 생성 중...")
        texts = [chunk["text"] for chunk in chunks]

        # 배치 처리
        embeddings = self.embedder.encode(
            texts, normalize_embeddings=True, show_progress_bar=True, batch_size=32
        )

        print(f"✅ {embeddings.shape} 임베딩 생성 완료")
        return embeddings

    def upload_to_qdrant(self, chunks: List[Dict], embeddings: np.ndarray):
        """Qdrant Cloud 업로드"""
        print("☁️ Qdrant Cloud 업로드 중...")

        collection_name = "unemployment_rag"

        # 컬렉션 존재 확인
        try:
            collections = self.qdrant.get_collections()
            exists = any(c.name == collection_name for c in collections.collections)

            if exists:
                print("⚠️ 기존 컬렉션 발견")
                # 기존 데이터 확인
                count = self.qdrant.count(collection_name=collection_name)
                print(f"  현재 {count.count}개 벡터 존재")

                # 삭제 여부 확인
                response = input("기존 컬렉션을 삭제하고 새로 만들까요? (y/n): ")
                if response.lower() == "y":
                    self.qdrant.delete_collection(collection_name)
                    print("기존 컬렉션 삭제")
                else:
                    print("기존 컬렉션 유지, 업로드 중단")
                    return
        except:
            pass

        # 새 컬렉션 생성 (1024차원)
        self.qdrant.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
        )
        print("새 컬렉션 생성 (1024차원)")

        # 포인트 생성
        points = []
        for i, chunk in enumerate(chunks):
            points.append(
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=embeddings[i].tolist(),
                    payload={"text": chunk["text"], **chunk["metadata"]},
                )
            )

        # 업로드
        self.qdrant.upsert(collection_name=collection_name, points=points)

        print(f"✅ {len(points)}개 벡터 업로드 완료!")

    def run(self, file_paths: List[str] = None):
        """전체 파이프라인 실행"""
        print("=" * 50)
        print("🚀 RAG 파이프라인 시작")
        print("=" * 50)

        # 기본 파일 경로
        if file_paths is None:
            file_paths = ["backend/data/knowledge.json"]

        # 1. 파싱
        documents = self.parse_documents(file_paths)

        # 2. 청킹
        chunks = self.chunk_texts(documents)

        # 3. 임베딩
        embeddings = self.embed_chunks(chunks)

        # 4. 업로드
        self.upload_to_qdrant(chunks, embeddings)

        print("\n" + "=" * 50)
        print("🎉 파이프라인 완료!")
        print(f"📊 최종 통계:")
        print(f"  - 문서: {len(documents)}개")
        print(f"  - 청크: {len(chunks)}개")
        print(f"  - 벡터: {embeddings.shape}")
        print("=" * 50)


# Railway에서 직접 실행 가능
if __name__ == "__main__":
    import sys

    # 경로 설정
    if "backend" not in sys.path:
        sys.path.insert(0, "backend")

    pipeline = RAGPipeline()

    # Railway 환경에서 knowledge.json 경로
    files = ["backend/data/knowledge.json"]

    pipeline.run(files)
