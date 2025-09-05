from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import uuid
from typing import List, Dict
import numpy as np
import logging
from rank_bm25 import BM25Okapi
import re
import os

logger = logging.getLogger(__name__)


class QdrantVectorStore:
    def __init__(self, collection_name: str = "unemployment_rag"):
        """Qdrant 클라이언트 초기화"""
        # 환경변수 체크
        qdrant_url = os.getenv("QDRANT_URL")
        qdrant_api_key = os.getenv("QDRANT_API_KEY")

        if qdrant_url and qdrant_api_key:
            # Railway/Production - Qdrant Cloud
            self.client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
            logger.info("✅ Qdrant Cloud 모드 연결 성공")
        else:
            # 로컬 개발 - Docker Qdrant
            self.client = QdrantClient(host="localhost", port=6333)
            logger.info("✅ Qdrant Docker 모드 연결 성공")

        self.collection_name = collection_name

        # BM25를 위한 문서 저장
        self.documents = []
        self.bm25 = None
        self._load_all_documents()  # 초기화시 문서 로드

    def tokenize_korean(self, text: str) -> List[str]:
        """간단한 한글 토크나이징 (Java 불필요)"""
        # 소문자 변환
        text = text.lower()

        # 특수문자를 공백으로 치환
        text = re.sub(r"[^\w\s가-힣]", " ", text)

        # 공백으로 분리
        tokens = text.split()

        # n-gram 생성 (부분 문자열 추가)
        extended_tokens = []
        for token in tokens:
            extended_tokens.append(token)

            # 한글 단어면 부분 문자열 추가
            if re.match(r"^[가-힣]+$", token) and len(token) >= 3:
                # "반복수급" → ["반복수급", "반복", "수급"]
                for i in range(2, len(token)):
                    extended_tokens.append(token[:i])  # 앞부분
                    extended_tokens.append(token[i:])  # 뒷부분

                # 2글자씩 분리도 추가
                for i in range(0, len(token) - 1):
                    extended_tokens.append(token[i : i + 2])

        # 중복 제거
        return list(set(extended_tokens))

    def _load_all_documents(self):
        """모든 문서 로드 및 BM25 초기화"""
        try:
            # Qdrant에서 모든 문서 가져오기
            result = self.client.scroll(
                collection_name=self.collection_name,
                limit=100,
                with_payload=True,
                with_vectors=False,
            )

            self.documents = []
            for point in result[0]:
                doc = {
                    "id": point.id,
                    "text": point.payload["text"],
                    "metadata": point.payload.get("metadata", {}),
                    "source": point.payload.get("source", ""),
                }
                self.documents.append(doc)

            # BM25 초기화 - 커스텀 토크나이저 사용
            if self.documents:
                logger.info("BM25 토크나이징 중...")
                tokenized_docs = [
                    self.tokenize_korean(doc["text"]) for doc in self.documents
                ]
                self.bm25 = BM25Okapi(tokenized_docs)
                logger.info(
                    f"✅ BM25 커스텀 한글 토크나이저로 초기화: {len(self.documents)}개 문서"
                )
        except Exception as e:
            logger.warning(f"BM25 초기화 실패: {e}")
            self.documents = []
            self.bm25 = None

    def create_collection(self, dimension: int):
        """컬렉션 생성"""
        try:
            # 기존 컬렉션 확인
            collections = self.client.get_collections().collections
            if any(c.name == self.collection_name for c in collections):
                logger.info(f"컬렉션 '{self.collection_name}' 이미 존재")
                return

            # 새 컬렉션 생성
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=dimension, distance=Distance.COSINE),
            )
            logger.info(f"✅ 컬렉션 '{self.collection_name}' 생성 완료")
        except Exception as e:
            logger.error(f"컬렉션 생성 오류: {e}")

    def add_documents(self, documents: List[Dict], embeddings: np.ndarray):
        """문서 추가"""
        points = []

        for i, doc in enumerate(documents):
            point_id = str(uuid.uuid4())
            point = PointStruct(
                id=point_id,
                vector=embeddings[i].tolist(),
                payload={
                    "text": doc["text"],
                    "metadata": doc.get("metadata", {}),
                    "source": doc.get("source", "unknown"),
                },
            )
            points.append(point)

        self.client.upsert(collection_name=self.collection_name, points=points)
        logger.info(f"✅ {len(points)}개 문서 인덱싱 완료")

        # BM25 업데이트
        self._load_all_documents()

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Dict]:
        """벡터 검색"""
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding.tolist(),
            limit=top_k,
        )

        return [
            {
                "text": hit.payload["text"],
                "metadata": hit.payload.get("metadata", {}),
                "score": hit.score,
            }
            for hit in results
        ]

    def bm25_search(self, query: str, top_k: int = 10) -> List[Dict]:
        """BM25 키워드 검색 - 커스텀 토크나이저 사용"""
        if not self.bm25 or not self.documents:
            logger.warning("BM25 사용 불가 - 문서가 없습니다")
            return []

        # 쿼리도 같은 토크나이저 사용
        tokenized_query = self.tokenize_korean(query)
        logger.info(f"BM25 쿼리 토큰: {tokenized_query[:10]}")  # 디버깅용

        scores = self.bm25.get_scores(tokenized_query)

        # 점수와 문서 인덱스 쌍
        doc_scores = [(score, idx) for idx, score in enumerate(scores)]
        doc_scores.sort(reverse=True, key=lambda x: x[0])

        results = []
        for score, idx in doc_scores[:top_k]:
            if score > 0:
                doc = self.documents[idx].copy()
                doc["bm25_score"] = float(score)
                doc["score"] = float(score)  # 호환성을 위해
                results.append(doc)

        logger.info(f"BM25 검색: {len(results)}개 문서 발견")
        return results
