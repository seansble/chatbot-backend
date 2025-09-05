import json
import logging
from typing import List, Dict
from .embedder import BGEEmbedder
from .vectorstore import QdrantVectorStore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentIndexer:
    def __init__(self):
        logger.info("DocumentIndexer 초기화 중...")
        self.embedder = BGEEmbedder()
        self.vector_store = QdrantVectorStore()
        # 임베딩 차원에 맞춰 컬렉션 생성
        self.vector_store.create_collection(dimension=self.embedder.dimension)
        
    def index_knowledge_json(self, json_path: str = 'backend/data/knowledge.json'):
        """knowledge.json 인덱싱"""
        logger.info(f"📚 {json_path} 인덱싱 시작...")
        
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except FileNotFoundError:
            logger.error(f"파일을 찾을 수 없음: {json_path}")
            return 0
            
        documents = []
        for faq in data['faqs']:
            # 질문과 답변을 하나의 문서로
            doc_text = f"질문: {faq['q']}\n답변: {faq['a']}"
            
            documents.append({
                'text': doc_text,
                'metadata': {
                    'id': faq['id'],
                    'category': faq['category'],
                    'priority': faq.get('priority', 0),
                    'short_answer': faq.get('a_short', ''),
                    'context_match': faq.get('context_match', [])
                },
                'source': 'knowledge.json'
            })
        
        # 텍스트만 추출
        texts = [doc['text'] for doc in documents]
        
        # 임베딩 생성
        logger.info(f"🔄 {len(texts)}개 문서 임베딩 중...")
        embeddings = self.embedder.embed_documents(texts)
        
        # 벡터스토어에 저장
        self.vector_store.add_documents(documents, embeddings)
        logger.info(f"✅ {len(documents)}개 FAQ 인덱싱 완료!")
        
        return len(documents)