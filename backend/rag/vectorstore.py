# backend/rag/vectorstore.py
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import uuid
from typing import List, Dict
import numpy as np
import logging
from rank_bm25 import BM25Okapi
import os
import json
import hashlib
from pathlib import Path
from .tokenizer import KiwiTokenizer

logger = logging.getLogger(__name__)


class QdrantVectorStore:
    def __init__(self, collection_name: str = "unemployment_rag"):
        """Qdrant 클라이언트 초기화"""
        # 환경변수 체크
        qdrant_url = os.getenv("QDRANT_URL")
        qdrant_api_key = os.getenv("QDRANT_API_KEY")

        logger.info(
            f"Environment check - URL: {qdrant_url[:30] if qdrant_url else 'None'}"
        )
        logger.info(f"Environment check - Key: {'Set' if qdrant_api_key else 'None'}")

        if qdrant_url and qdrant_api_key:
            # Railway/Production - Qdrant Cloud
            try:
                self.client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
                logger.info("✅ Qdrant Cloud 모드 연결 시도")
            except Exception as e:
                logger.error(f"Qdrant Cloud 연결 실패: {e}")
                # 폴백으로 로컬 시도
                self.client = QdrantClient(host="localhost", port=6333)
        else:
            # 로컬 개발 - Docker Qdrant
            self.client = QdrantClient(host="localhost", port=6333)
            logger.info("✅ Qdrant Docker 모드 연결 성공")

        self.collection_name = collection_name

        # Kiwi 토크나이저 초기화
        self.tokenizer = KiwiTokenizer()

        # BM25를 위한 문서 저장
        self.documents = []
        self.tokenized_docs = []
        self.doc_keys = []
        self.bm25 = None

        # BM25 초기화 시도
        try:
            # self._load_all_documents()  # 서버 시작 시 로드 비활성화
            self.documents = []
            self.tokenized_docs = []
            self.doc_keys = []
            self.bm25 = None
            logger.info("초기 문서 로드 건너뜀 - 검색 시 로드됨")
        except Exception as e:
            logger.warning(f"BM25 초기화 실패 (정상): {e}")
            self.documents = []
            self.tokenized_docs = []
            self.doc_keys = []
            self.bm25 = None

    def _load_all_documents(self):
        """모든 문서 로드 및 BM25 초기화"""
        try:
            # 먼저 knowledge_kiwi.json이 있는지 확인
            knowledge_path = (
                Path(__file__).parent.parent / "data" / "knowledge_kiwi.json"
            )

            if knowledge_path.exists():
                # knowledge_kiwi.json에서 로드
                logger.info(f"📂 Loading from {knowledge_path}")
                with open(knowledge_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                self.documents = []
                self.tokenized_docs = []
                self.doc_keys = []

                # knowledge_kiwi.json 구조: {"knowledge": [...], "metadata": {...}}
                if isinstance(data, dict) and "knowledge" in data:
                    knowledge_items = data["knowledge"]
                else:
                    knowledge_items = data if isinstance(data, list) else []

                for item in knowledge_items:
                    # 각 item이 dict인지 확인
                    if not isinstance(item, dict):
                        logger.warning(f"Skipping non-dict item: {type(item)}")
                        continue

                    doc_key = item.get("doc_key")
                    if not doc_key:
                        doc_key = hashlib.sha1(item["text"].encode("utf-8")).hexdigest()

                    doc = {
                        "id": item.get("id", str(uuid.uuid4())),
                        "doc_key": doc_key,
                        "text": item.get("text", ""),
                        "metadata": {
                            "category": item.get("category", ""),
                            "keywords": item.get("keywords", []),
                            "priority": item.get("priority", 5),
                            "question": item.get("question", ""),
                            "answer": item.get("answer", ""),
                        },
                        "source": item.get("source", "knowledge_faq"),
                    }
                    self.documents.append(doc)
                    self.doc_keys.append(doc_key)

                    # 이미 토큰화된 데이터 사용
                    if "tokens" in item and isinstance(item["tokens"], list):
                        tokens = item["tokens"]
                    else:
                        # 없으면 실시간 토큰화
                        tokens = self.tokenizer.tokenize(doc["text"])

                    # 빈 토큰 처리
                    if not tokens:
                        tokens = ["[EMPTY]"]

                    self.tokenized_docs.append(tokens)

                logger.info(
                    f"✅ knowledge_kiwi.json에서 {len(self.documents)}개 문서 로드"
                )

            else:
                # Qdrant에서 모든 문서 가져오기 (기존 방식)
                logger.info("knowledge_kiwi.json 없음 - Qdrant에서 로드")
                result = self.client.scroll(
                    collection_name=self.collection_name,
                    limit=1000,
                    with_payload=True,
                    with_vectors=False,
                )

                self.documents = []
                self.tokenized_docs = []
                self.doc_keys = []

                for point in result[0]:
                    doc_key = point.payload.get("doc_key")
                    if not doc_key:
                        doc_key = hashlib.sha1(
                            point.payload.get("text", "").encode("utf-8")
                        ).hexdigest()

                    doc = {
                        "id": point.id,
                        "doc_key": doc_key,
                        "text": point.payload.get("text", ""),
                        "metadata": point.payload.get("metadata", {}),
                        "source": point.payload.get("source", ""),
                    }
                    self.documents.append(doc)
                    self.doc_keys.append(doc_key)

                    # Kiwi로 토큰화
                    tokens = self.tokenizer.tokenize(doc["text"])

                    if not isinstance(tokens, list):
                        tokens = []
                    if not tokens:
                        tokens = ["[EMPTY]"]

                    self.tokenized_docs.append(tokens)

            # BM25 초기화
            if self.tokenized_docs and all(
                isinstance(doc, list) for doc in self.tokenized_docs
            ):
                self.bm25 = BM25Okapi(self.tokenized_docs)
                logger.info(
                    f"✅ BM25 Kiwi 토크나이저로 초기화: {len(self.documents)}개 문서"
                )

                # 첫 3개 문서의 토큰 샘플 출력 (디버깅용)
                for i in range(min(3, len(self.tokenized_docs))):
                    logger.debug(
                        f"문서 {i+1} 토큰 샘플: {self.tokenized_docs[i][:10]}..."
                    )

            else:
                logger.error("BM25 초기화 실패: 토큰 형식 오류")
                self.bm25 = None

        except Exception as e:
            logger.error(f"BM25 초기화 실패: {e}")
            import traceback

            logger.error(traceback.format_exc())
            self.documents = []
            self.tokenized_docs = []
            self.doc_keys = []
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

            # doc_key 추가
            doc_key = doc.get("doc_key")
            if not doc_key:
                doc_key = hashlib.sha1(doc["text"].encode("utf-8")).hexdigest()

            point = PointStruct(
                id=point_id,
                vector=embeddings[i].tolist(),
                payload={
                    "text": doc["text"],
                    "doc_key": doc_key,
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
                "doc_key": hit.payload.get("doc_key"),
                "metadata": hit.payload.get("metadata", {}),
                "score": hit.score,
            }
            for hit in results
        ]

    def bm25_search(self, query: str, top_k: int = 10) -> List[Dict]:
        """BM25 키워드 검색 - Kiwi 토크나이저 사용"""
        if not self.bm25 or not self.documents:
            logger.warning("BM25 사용 불가 - 문서가 없거나 초기화 실패")
            return []

        try:
            # 1. 기본 토큰화
            query_tokens = self.tokenizer.tokenize(query)

            # 2. 쿼리만 동의어 확장
            expanded_query = self.tokenizer.expand_query(query_tokens)

            # 타입 체크 - 반드시 list[str]
            if not isinstance(expanded_query, list):
                logger.error(f"BM25 쿼리 타입 오류: {type(expanded_query)}")
                return []

            # 빈 쿼리 처리
            if not expanded_query:
                expanded_query = ["[QUERY]"]

            # 문자열만 허용
            expanded_query = [
                str(token) for token in expanded_query if isinstance(token, str)
            ]

            logger.info(f"BM25 쿼리 토큰: {expanded_query[:10]}")

            # 3. BM25 검색
            scores = self.bm25.get_scores(expanded_query)

            # 점수와 문서 인덱스 쌍
            doc_scores = [(score, idx) for idx, score in enumerate(scores)]
            doc_scores.sort(reverse=True, key=lambda x: x[0])

            results = []
            for score, idx in doc_scores[:top_k]:
                if score > 0:
                    doc = self.documents[idx].copy()
                    doc["bm25_score"] = float(score)
                    doc["score"] = float(score)
                    results.append(doc)

            logger.info(f"BM25 검색: {len(results)}개 문서 발견")

            # 상위 3개 결과 로깅 (디버깅용)
            for i, result in enumerate(results[:3]):
                doc_key = result.get("doc_key")
                doc_key_display = doc_key[:8] + "..." if doc_key else "N/A"

                logger.debug(
                    f"BM25 결과 {i+1} - doc_key: {doc_key_display}, "
                    f"score: {result.get('score', 0):.3f}"
                )

            return results

        except Exception as e:
            logger.error(f"BM25 검색 중 에러: {e}")
            return []

    def rebuild_bm25_index(self):
        """BM25 인덱스 재구축 (수동 호출용)"""
        logger.info("🔄 BM25 인덱스 재구축 시작...")
        self._load_all_documents()
        if self.bm25:
            logger.info("✅ BM25 인덱스 재구축 완료")
        else:
            logger.error("❌ BM25 인덱스 재구축 실패")

    def get_query_tags(self, query: str) -> set:
        """질의에서 관련 태그 추출"""
        TAG_MAP = {
            "육아휴직": {"육아휴직", "복직", "경력단절", "출산"},
            "임금체불": {"임금체불", "체불확인서", "노동부"},
            "권고사직": {"권고사직", "구조조정", "해고"},
            "실업급여": {"실업급여", "수급자격", "고용보험"},
            "계약만료": {"계약만료", "계약직", "비자발적"},
            "자진퇴사": {"자진퇴사", "자발적", "정당한사유"},
            "구직활동": {"구직활동", "실업인정", "워크넷"},
        }

        tags = set()
        for keyword, related_tags in TAG_MAP.items():
            if keyword in query:
                tags |= related_tags
        return tags

    def hybrid_search(
        self, query: str, query_embedding: np.ndarray, top_k: int = 10
    ) -> List[Dict]:
        """하이브리드 검색 - RRF 기반 결합 (doc_key 사용)"""

        # BM25 첫 검색 시 자동 로드
        if self.bm25 is None or not self.documents:
            logger.info("BM25 미초기화 - 첫 검색 시 로드 중...")
            try:
                self._load_all_documents()
                if self.bm25:
                    logger.info(f"✅ BM25 로드 성공: {len(self.documents)}개 문서")
                else:
                    logger.warning("BM25 로드 실패 - 벡터 검색만 사용")
            except Exception as e:
                logger.error(f"BM25 로드 중 에러: {e}")

        # 1. 각각 검색 (더 많이 수집)
        bm25_results = self.bm25_search(query, top_k=50) if self.bm25 else []
        vector_results = self.search(query_embedding, top_k=100)

        # 2. doc_key 기반 RRF 스코어 계산
        k = 10

        # doc_key -> (rank, doc) 매핑
        bm25_hits = {}
        for i, doc in enumerate(bm25_results):
            doc_key = doc.get("doc_key")
            if not doc_key:
                doc_key = hashlib.sha1(doc["text"].encode()).hexdigest()
            bm25_hits[doc_key] = (i + 1, doc)

        vector_hits = {}
        for i, doc in enumerate(vector_results):
            doc_key = doc.get("doc_key")
            if not doc_key:
                doc_key = hashlib.sha1(doc["text"].encode()).hexdigest()
            vector_hits[doc_key] = (i + 1, doc)

        # RRF 스코어 계산
        def rrf(rank, k=10):
            return 1 / (k + rank)

        bm25_rrf = {key: rrf(rank) for key, (rank, _) in bm25_hits.items()}
        vector_rrf = {key: rrf(rank) for key, (rank, _) in vector_hits.items()}

        # 3. 쿼리 타입 판단
        query_tokens = self.tokenizer.tokenize(query)
        token_count = len(query_tokens)

        has_numbers = any(
            token.replace("년", "").replace("월", "").replace("일", "").isdigit()
            for token in query_tokens
        )

        if token_count <= 4 and has_numbers:
            alpha = 0.5
        elif token_count > 10:
            alpha = 0.7
        else:
            alpha = 0.6

        logger.info(f"하이브리드 가중치 - 토큰수: {token_count}, alpha: {alpha}")

        # 4. doc_key 기반 스코어 결합
        all_keys = set(bm25_rrf.keys()) | set(vector_rrf.keys())
        combined = {}
        all_docs = {}

        for doc_key in all_keys:
            # 스코어 결합
            bm25_score = bm25_rrf.get(doc_key, 0)
            vector_score = vector_rrf.get(doc_key, 0)

            score = alpha * vector_score + (1 - alpha) * bm25_score

            # 양측 교집합 보너스 (먼저 적용)
            if doc_key in bm25_rrf and doc_key in vector_rrf:
                score += 0.02

            combined[doc_key] = score

            # 문서 정보 저장 (한쪽에서만 가져오기)
            if doc_key in bm25_hits:
                all_docs[doc_key] = bm25_hits[doc_key][1]
            elif doc_key in vector_hits:
                all_docs[doc_key] = vector_hits[doc_key][1]

        # 5. 질의 적합도 기반 보너스 계산
        query_tags = self.get_query_tags(query)
        important_phrases = {
            "육아휴직",
            "임금체불",
            "권고사직",
            "계약만료",
            "실업급여",
            "자진퇴사",
            "구직활동",
        }
        query_phrases = {p for p in important_phrases if p in query}

        for doc_key in combined.keys():
            doc = all_docs[doc_key]
            metadata = doc.get("metadata", {})
            doc_keywords = set(metadata.get("keywords", []))

            bonus = 0.0

            # 1) 태그 교집합 보너스 (메인)
            overlap = len(query_tags & doc_keywords)
            if overlap >= 2:
                bonus += 0.06
                logger.debug(f"태그 교집합 {overlap}개: {query_tags & doc_keywords}")
            elif overlap == 1:
                bonus += 0.04

            # 2) 프레이즈 정확 매칭
            if any(phrase in doc["text"] for phrase in query_phrases):
                bonus += 0.03
                logger.debug(f"프레이즈 매칭: {query_phrases}")

            # 3) 전역 priority (최소화)
            priority = metadata.get("priority", 5)
            if priority >= 9:
                bonus += 0.02
            elif priority >= 7:
                bonus += 0.01

            # 총합 제한
            bonus = min(bonus, 0.09)
            combined[doc_key] += bonus

            if bonus > 0:
                logger.debug(
                    f"보너스 적용 - 문서: {doc['text'][:30]}, "
                    f"태그교집합: {overlap}, 프레이즈: {bool(query_phrases & {p for p in query_phrases if p in doc['text']})}, "
                    f"priority: {priority}, 총보너스: {bonus:.3f}"
                )

        # 6. 정렬
        sorted_results = sorted(combined.items(), key=lambda x: x[1], reverse=True)

        # 7. 결과 생성
        results = []
        seen_keys = set()

        # 상위 결과 추가
        for doc_key, score in sorted_results[:top_k]:
            doc = all_docs[doc_key].copy()
            doc["hybrid_score"] = score
            doc["bm25_rrf"] = bm25_rrf.get(doc_key, 0)
            doc["vector_rrf"] = vector_rrf.get(doc_key, 0)
            results.append(doc)
            seen_keys.add(doc_key)

        # 8. 교차 보장 - BM25 1위와 Vector 1위 강제 포함
        if bm25_results and len(results) >= 5:
            bm25_top_key = list(bm25_hits.keys())[0] if bm25_hits else None
            if bm25_top_key and bm25_top_key not in seen_keys:
                doc = bm25_hits[bm25_top_key][1].copy()
                doc["hybrid_score"] = combined.get(bm25_top_key, 0)
                doc["bm25_rrf"] = bm25_rrf.get(bm25_top_key, 0)
                doc["vector_rrf"] = 0
                results.insert(4, doc)

        if vector_results and len(results) >= 6:
            vector_top_key = list(vector_hits.keys())[0] if vector_hits else None
            if vector_top_key and vector_top_key not in seen_keys:
                doc = vector_hits[vector_top_key][1].copy()
                doc["hybrid_score"] = combined.get(vector_top_key, 0)
                doc["bm25_rrf"] = 0
                doc["vector_rrf"] = vector_rrf.get(vector_top_key, 0)
                results.insert(5, doc)

        # 최종 top_k 유지
        results = results[:top_k]

        logger.info(f"하이브리드 검색 완료: {len(results)}개 결과")

        # 디버그 로깅
        for i, result in enumerate(results[:3]):
            doc_key = result.get("doc_key")
            doc_key_display = doc_key[:8] + "..." if doc_key else "N/A"

            logger.debug(
                f"하이브리드 결과 {i+1} - doc_key: {doc_key_display}, "
                f"hybrid: {result.get('hybrid_score', 0):.3f}, "
                f"bm25_rrf: {result.get('bm25_rrf', 0):.3f}, "
                f"vector_rrf: {result.get('vector_rrf', 0):.3f}"
            )

        return results
