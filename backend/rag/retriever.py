# backend/rag/retriever.py
from typing import List, Dict, Optional
import numpy as np
import logging
from .embedder import BGEEmbedder
from .vectorstore import QdrantVectorStore

logger = logging.getLogger(__name__)


class RAGRetriever:
    def __init__(self, use_reranker: bool = False, use_hybrid: bool = True):
        self.embedder = BGEEmbedder()
        self.vector_store = QdrantVectorStore()
        self.use_reranker = use_reranker
        self.use_hybrid = use_hybrid

        if use_hybrid:
            logger.info("✅ 하이브리드 검색 활성화 (BM25 + Vector)")

        if use_reranker:
            from .reranker import BGEReranker

            self.reranker = BGEReranker()
            logger.info("✅ BGE Reranker 활성화")

    def normalize_scores(self, scores: List[float]) -> np.ndarray:
        """점수 정규화 (0-1 범위)"""
        if not scores:
            return np.array([])

        scores = np.array(scores)
        min_score = scores.min()
        max_score = scores.max()

        if max_score - min_score > 0:
            return (scores - min_score) / (max_score - min_score)
        return np.ones_like(scores) * 0.5

    def hybrid_search(self, query: str, top_k: int = 10) -> List[Dict]:
        """하이브리드 검색 (BM25 + Vector)"""
        # 1. BM25 키워드 검색
        bm25_results = self.vector_store.bm25_search(query, top_k=20)

        # 2. 벡터 검색
        query_embedding = self.embedder.embed_query(query)
        vector_results = self.vector_store.search(query_embedding, top_k=20)

        # 3. 결과 병합 (텍스트 기준으로 중복 제거)
        combined_docs = {}

        # BM25 결과 처리
        for doc in bm25_results:
            doc_key = doc["text"][:100]  # 텍스트 앞부분을 키로 사용
            combined_docs[doc_key] = {
                "text": doc["text"],
                "metadata": doc.get("metadata", {}),
                "bm25_score": doc.get("bm25_score", 0),
                "vector_score": 0,
                "source": doc.get("source", ""),
            }

        # 벡터 결과 처리
        for doc in vector_results:
            doc_key = doc["text"][:100]
            if doc_key in combined_docs:
                combined_docs[doc_key]["vector_score"] = doc["score"]
            else:
                combined_docs[doc_key] = {
                    "text": doc["text"],
                    "metadata": doc.get("metadata", {}),
                    "bm25_score": 0,
                    "vector_score": doc["score"],
                    "source": doc.get("source", ""),
                }

        # 4. 점수 정규화 및 결합
        final_results = []

        # 모든 점수 수집 (정규화를 위해)
        all_bm25_scores = [
            doc["bm25_score"] for doc in combined_docs.values() if doc["bm25_score"] > 0
        ]
        all_vector_scores = [
            doc["vector_score"]
            for doc in combined_docs.values()
            if doc["vector_score"] > 0
        ]

        # 정규화
        norm_bm25 = (
            self.normalize_scores(all_bm25_scores) if all_bm25_scores else np.array([])
        )
        norm_vector = (
            self.normalize_scores(all_vector_scores)
            if all_vector_scores
            else np.array([])
        )

        # 최종 점수 계산
        bm25_idx = 0
        vector_idx = 0

        for doc_key, doc in combined_docs.items():
            final_score = 0

            # BM25 점수 정규화 적용
            if doc["bm25_score"] > 0 and len(norm_bm25) > 0:
                doc["norm_bm25"] = float(norm_bm25[bm25_idx])
                bm25_idx += 1
                final_score += doc["norm_bm25"] * 0.3  # BM25 가중치 30%

            # 벡터 점수 정규화 적용
            if doc["vector_score"] > 0 and len(norm_vector) > 0:
                doc["norm_vector"] = float(norm_vector[vector_idx])
                vector_idx += 1
                final_score += doc["norm_vector"] * 0.7  # 벡터 가중치 70%

            # 메타데이터 부스팅
            priority = doc["metadata"].get("priority", 0)
            if priority > 8:
                final_score += 0.1

            doc["score"] = final_score
            final_results.append(doc)

        # 5. 정렬 및 반환
        final_results.sort(key=lambda x: x["score"], reverse=True)

        logger.info(
            f"하이브리드 검색 완료: BM25 {len(bm25_results)}개 + Vector {len(vector_results)}개 → {len(final_results[:top_k])}개 선택"
        )

        return final_results[:top_k]

    def retrieve(self, query: str, top_k: int = 5, rerank_top_k: int = 3) -> List[Dict]:
        """질문에 대한 관련 문서 검색"""
        logger.info(f"🔍 검색 중: {query[:50]}...")

        # 하이브리드 검색 사용 여부
        if self.use_hybrid:
            results = self.hybrid_search(
                query, top_k=10 if self.use_reranker else top_k
            )
        else:
            # 기존 벡터 검색만
            query_embedding = self.embedder.embed_query(query)
            search_top_k = 10 if self.use_reranker else top_k
            results = self.vector_store.search(query_embedding, top_k=search_top_k)

        # 리랭킹 적용
        if self.use_reranker and results:
            from .reranker import BGEReranker

            results = self.reranker.rerank(query, results, top_k=rerank_top_k)
            logger.info(f"✅ BGE 리랭킹 적용: Top {rerank_top_k}")
        elif not self.use_hybrid:
            # 하이브리드 검색을 사용하지 않을 때만 정렬
            results = sorted(results, key=lambda x: x["score"], reverse=True)[:top_k]
        else:
            # 하이브리드 검색은 이미 정렬됨
            results = results[:top_k]

        logger.info(f"✅ {len(results)}개 문서 최종 선택")

        return results

    def format_context(self, results: List[Dict]) -> str:
        """검색 결과를 컨텍스트로 포맷팅"""
        context = "=== 관련 정보 ===\n\n"

        for i, result in enumerate(results, 1):
            # 리랭킹된 경우 final_score, 하이브리드면 score 사용
            score = result.get("final_score", result.get("score", 0))
            context += f"[정보 {i}] (관련도: {score:.2f})\n"
            context += f"카테고리: {result['metadata'].get('category', 'N/A')}\n"
            context += f"{result['text']}\n"
            context += "-" * 50 + "\n"

        return context
