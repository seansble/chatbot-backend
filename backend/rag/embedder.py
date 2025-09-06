# backend/rag/embedder.py
import numpy as np
from typing import List
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BGEEmbedder:
    def __init__(self, model_name: str = "BM-K/KoSimCSE-roberta-base"):
        """KoSimCSE 임베더 - 한국어 특화 경량 모델"""
        self.model_name = model_name
        self.dimension = 1024
        self.model = None
        self.tokenizer = None

        # Railway 환경에서만 실제 모델 로드
        if os.getenv("RAILWAY_ENVIRONMENT") or os.getenv("USE_REAL_EMBEDDING"):
            try:
                from transformers import AutoTokenizer, AutoModel
                import torch

                logger.info(f"Loading KoSimCSE model: {model_name}")
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.model = AutoModel.from_pretrained(model_name)
                self.model.eval()
                logger.info(f"✅ KoSimCSE loaded! 차원: {self.dimension}")
            except Exception as e:
                logger.warning(f"Failed to load model: {e}")
                logger.info("Falling back to dummy embedder")
        else:
            logger.info("Using dummy embedder for local development")

    def embed_documents(self, texts: List[str]) -> np.ndarray:
        """문서 임베딩"""
        if not texts:
            return np.array([])

        if self.model is None:
            # 더미 임베딩
            return np.random.random((len(texts), self.dimension))

        # 실제 임베딩
        import torch

        inputs = self.tokenizer(
            texts, padding=True, truncation=True, return_tensors="pt", max_length=512
        )
        with torch.no_grad():
            outputs = self.model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1)
        return embeddings.numpy()

    def embed_query(self, query: str) -> np.ndarray:
        """질의 임베딩"""
        if self.model is None:
            # 더미 임베딩
            return np.random.random(self.dimension)

        # 실제 임베딩
        embeddings = self.embed_documents([query])
        return embeddings[0]
