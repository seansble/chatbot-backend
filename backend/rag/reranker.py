"""BGE 리랭킹 모듈 - CrossEncoder 버전"""

from sentence_transformers import CrossEncoder
import numpy as np
import logging
from typing import List, Dict

logger = logging.getLogger(__name__)


class BGEReranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-base"):
        """BGE 리랭킹 모델 초기화"""
        logger.info(f"Loading BGE reranker model: {model_name}")
        logger.info("첫 실행시 모델 다운로드로 2-3분 걸릴 수 있습니다...")

        # CrossEncoder로 BGE reranker 로드
        self.model = CrossEncoder(model_name, max_length=512)
        logger.info("✅ BGE Reranker 모델 로드 완료!")

    def rerank(self, query: str, documents: List[Dict], top_k: int = 3) -> List[Dict]:
        """문서 리랭킹"""
        if not documents:
            return []

        # 쿼리-문서 쌍 생성
        pairs = [[query, doc["text"]] for doc in documents]

        # 리랭킹 점수 계산
        scores = self.model.predict(pairs)

        # 점수 정규화 (min-max scaling)
        scores = np.array(scores)
        min_score = scores.min()
        max_score = scores.max()

        if max_score > min_score:
            normalized_scores = (scores - min_score) / (max_score - min_score)
        else:
            normalized_scores = np.ones_like(scores) * 0.5

        # 결과 업데이트
        for i, doc in enumerate(documents):
            doc["original_score"] = doc.get("score", 0)
            doc["rerank_score"] = float(normalized_scores[i])

            # BGE 패밀리는 서로 잘 맞으므로 리랭킹 비중 높임
            doc["final_score"] = (
                doc["original_score"] * 0.25 + doc["rerank_score"] * 0.75
            )

        # 정렬
        reranked = sorted(documents, key=lambda x: x["final_score"], reverse=True)

        logger.info(f"리랭킹 완료: Top {min(top_k, len(reranked))} 선택")

        return reranked[:top_k]
