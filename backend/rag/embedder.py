# backend/rag/embedder.py
import numpy as np
from typing import List
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BGEEmbedder:
    _instance = None
    _model = None

    def __new__(cls):
        """싱글톤 패턴으로 모델 한 번만 로드"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """BGE-M3 임베더 - 로컬에서도 실제 모델 사용"""
        self.dimension = 1024

        # 이미 모델이 로드되어 있으면 스킵
        if BGEEmbedder._model is not None:
            self.model = BGEEmbedder._model
            return

        # 환경변수 체크 제거 - 항상 실제 모델 로드 시도
        try:
            from sentence_transformers import SentenceTransformer

            logger.info("Loading BGE-M3 model...")
            BGEEmbedder._model = SentenceTransformer("BAAI/bge-m3")
            self.model = BGEEmbedder._model
            self.use_dummy = False  # 실제 모델 사용
            logger.info(f"✅ BGE-M3 loaded successfully! 차원: {self.dimension}")

        except Exception as e:
            logger.warning(f"Failed to load BGE-M3 model: {e}")
            logger.info("Falling back to dummy embedder")
            self.model = None
            self.use_dummy = True  # 더미 모드

    def embed_query(self, query: str) -> np.ndarray:
        """질의 임베딩 - BGE-M3 프리픽스 적용"""
        if self.model is None:
            # 더미 임베딩 (시드 고정으로 일관성 유지)
            np.random.seed(hash(query) % 10000)
            return np.random.randn(self.dimension)  # randn으로 변경

        # 실제 BGE-M3 임베딩
        try:
            prefixed_query = f"query: {query}"
            embedding = self.model.encode(
                prefixed_query,
                normalize_embeddings=True,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
            return embedding
        except Exception as e:
            logger.error(f"Error in embed_query: {e}")
            np.random.seed(hash(query) % 10000)
            return np.random.randn(self.dimension)

    def embed_documents(self, texts: List[str]) -> np.ndarray:
        """문서 임베딩 - BGE-M3 프리픽스 적용"""
        if not texts:
            return np.array([])

        if self.model is None:
            logger.debug(f"Generating dummy embeddings for {len(texts)} documents")
            np.random.seed(hash(str(texts)) % 10000)
            return np.random.randn(len(texts), self.dimension)  # randn으로 변경

        try:
            # BGE-M3 규약: passage 프리픽스 필수!
            prefixed_texts = [f"passage: {text}" for text in texts]

            embeddings = self.model.encode(
                prefixed_texts,  # 프리픽스 추가된 문서들
                normalize_embeddings=True,
                show_progress_bar=False,
                convert_to_numpy=True,
                batch_size=32,
            )
            logger.debug(
                f"Generated real document embeddings with prefix: shape {embeddings.shape}"
            )
            return embeddings
        except Exception as e:
            logger.error(f"Error in embed_documents: {e}")
            return np.random.randn(len(texts), self.dimension)  # randn으로 통일
