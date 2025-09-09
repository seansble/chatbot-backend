# backend/rag/retriever.py
from typing import List, Dict, Optional
import numpy as np
import logging
from .embedder import BGEEmbedder
from .vectorstore import QdrantVectorStore

logger = logging.getLogger(__name__)


class RAGRetriever:
    def __init__(self, use_reranker: bool = False, use_hybrid: bool = True):
        """RAG 검색기 초기화"""
        self.embedder = BGEEmbedder()
        self.vector_store = QdrantVectorStore()
        self.use_reranker = use_reranker
        self.use_hybrid = use_hybrid

        logger.info(
            f"✅ RAG Retriever 초기화 - Hybrid: {use_hybrid}, Reranker: {use_reranker}"
        )

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        """우리 RAG 시스템으로 검색"""
        logger.info(f"🔍 검색 중: {query[:50]}...")

        # 우리 하이브리드 검색 호출
        query_embedding = self.embedder.embed_query(query)
        results = self.vector_store.hybrid_search(query, query_embedding, top_k=top_k)

        # workflow.py가 기대하는 포맷으로 변환
        formatted_results = []
        for result in results:
            formatted_results.append(
                {
                    "text": result["text"],
                    "parent_text": result.get(
                        "parent_text", result["text"]
                    ),  # Parent 추가
                    "score": result.get("hybrid_score", result.get("score", 0)),
                    "metadata": result.get("metadata", {}),
                    "source": result.get("source", "unknown"),
                    "bm25_score": result.get("bm25_rrf", 0),
                    "vector_score": result.get("vector_rrf", 0),
                }
            )

        logger.info(f"✅ {len(formatted_results)}개 문서 검색 완료")

        return formatted_results

    def format_context(self, results: List[Dict]) -> str:
        """검색 결과를 컨텍스트로 포맷팅"""
        context = "=== 관련 정보 ===\n\n"

        for i, result in enumerate(results, 1):
            score = result.get("score", 0)
            context += f"[정보 {i}] (관련도: {score:.2f})\n"
            context += (
                f"카테고리: {result.get('metadata', {}).get('category', 'N/A')}\n"
            )
            context += f"{result['text']}\n"
            context += "-" * 50 + "\n"

        return context
