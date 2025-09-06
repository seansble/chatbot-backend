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
        """BGE-M3 임베더 - 전처리와 동일한 모델 사용"""
        self.dimension = 1024
        
        # 이미 모델이 로드되어 있으면 스킵
        if BGEEmbedder._model is not None:
            self.model = BGEEmbedder._model
            return
        
        # Railway 환경에서만 실제 모델 로드
        if os.getenv("RAILWAY_ENVIRONMENT") or os.getenv("USE_REAL_EMBEDDING"):
            try:
                from sentence_transformers import SentenceTransformer
                
                logger.info("Loading BGE-M3 model...")
                BGEEmbedder._model = SentenceTransformer("BAAI/bge-m3")
                self.model = BGEEmbedder._model
                logger.info(f"✅ BGE-M3 loaded successfully! 차원: {self.dimension}")
                
            except Exception as e:
                logger.warning(f"Failed to load BGE-M3 model: {e}")
                logger.info("Falling back to dummy embedder")
                self.model = None
        else:
            logger.info("Using dummy embedder for local development")
            self.model = None

    def embed_documents(self, texts: List[str]) -> np.ndarray:
        """문서 임베딩"""
        if not texts:
            return np.array([])

        if self.model is None:
            # 더미 임베딩
            logger.debug(f"Generating dummy embeddings for {len(texts)} texts")
            return np.random.random((len(texts), self.dimension))

        # 실제 BGE-M3 임베딩
        try:
            embeddings = self.model.encode(
                texts,
                normalize_embeddings=True,  # 정규화 중요!
                show_progress_bar=False,
                batch_size=32,
                convert_to_numpy=True
            )
            logger.debug(f"Generated real embeddings: shape {embeddings.shape}")
            return embeddings
        except Exception as e:
            logger.error(f"Error in embed_documents: {e}")
            return np.random.random((len(texts), self.dimension))

    def embed_query(self, query: str) -> np.ndarray:
        """질의 임베딩"""
        if self.model is None:
            # 더미 임베딩
            logger.debug("Generating dummy embedding for query")
            return np.random.random(self.dimension)

        # 실제 BGE-M3 임베딩
        try:
            embedding = self.model.encode(
                query,
                normalize_embeddings=True,  # 정규화 중요!
                show_progress_bar=False,
                convert_to_numpy=True
            )
            logger.debug(f"Generated real query embedding: shape {embedding.shape}")
            return embedding
        except Exception as e:
            logger.error(f"Error in embed_query: {e}")
            return np.random.random(self.dimension)