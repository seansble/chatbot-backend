"""LangGraph RAG 워크플로우 - LLM 직접 평가"""

from typing import Dict, List, TypedDict, Optional, Any
from langgraph.graph import StateGraph, END
import logging
import re
import json

logger = logging.getLogger(__name__)


class RAGState(TypedDict):
    """워크플로우 상태"""
    # 입력
    query: str
    
    # 전처리
    processed_query: str
    query_type: str
    
    # 검색
    documents: List[Dict]
    relevance_score: float
    context: str
    
    # 평가
    coverage_details: Dict[str, Any]  # LLM 평가 결과
    coverage_score: float
    missing_parts: List[str]  # 빠진 부분들
    
    # 답변 생성
    answer_method: str
    raw_answer: str
    final_answer: str
    
    # 메타데이터
    confidence: float
    debug_path: List[str]


class SemanticRAGWorkflow:
    def __init__(self, retriever):
        self.retriever = retriever
        self.workflow = self._build_workflow()

    def _build_workflow(self):
        """워크플로우 구성"""
        workflow = StateGraph(RAGState)
        
        # 노드 정의
        workflow.add_node("analyze_query", self.analyze_query)
        workflow.add_node("rag_search", self.rag_search)
        workflow.add_node("llm_evaluate_coverage", self.llm_evaluate_coverage)
        workflow.add_node("generate_from_rag", self.generate_from_rag)
        workflow.add_node("enhance_missing", self.enhance_missing)
        workflow.add_node("regenerate_full", self.regenerate_full)
        workflow.add_node("format_final", self.format_final)
        
        # 엣지 정의
        workflow.set_entry_point("analyze_query")
        workflow.add_edge("analyze_query", "rag_search")
        workflow.add_edge("rag_search", "llm_evaluate_coverage")
        
        # 평가 후 라우팅
        workflow.add_conditional_edges(
            "llm_evaluate_coverage",
            self.route_by_coverage,
            {
                "complete": "generate_from_rag",     # 90% 이상
                "partial": "enhance_missing",        # 50-89%
                "insufficient": "regenerate_full"    # 50% 미만
            }
        )
        
        workflow.add_edge("generate_from_rag", "format_final")
        workflow.add_edge("enhance_missing", "format_final")
        workflow.add_edge("regenerate_full", "format_final")
        workflow.add_edge("format_final", END)
        
        return workflow.compile()

    def analyze_query(self, state: RAGState) -> RAGState:
        """쿼리 전처리"""
        query = state["query"]
        state["debug_path"] = ["analyze_query"]
        
        # 구어체 정규화
        replacements = {
            "때려치": "자진퇴사",
            "짤렸": "해고",
            "얼마나": "얼마",
            "언제부터": "언제",
            "되나요": "가능",
            "받을 수 있": "수급 가능"
        }
        
        processed = query
        for old, new in replacements.items():
            processed = processed.replace(old, new)
        
        state["processed_query"] = processed
        state["query_type"] = self._classify_query_type(processed)
        
        logger.info(f"Query processed: {processed[:50]}...")
        return state

    def rag_search(self, state: RAGState) -> RAGState:
        """RAG 검색"""
        state["debug_path"].append("rag_search")
        query = state["processed_query"]
        
        # 하이브리드 검색
        results = self.retriever.retrieve(query, top_k=5)
        
        state["documents"] = results
        state["relevance_score"] = results[0]["score"] if results else 0.0
        state["context"] = "\n".join([doc["text"] for doc in results[:3]])
        
        logger.info(f"Retrieved {len(results)} documents")
        return state

    def llm_evaluate_coverage(self, state: RAGState) -> RAGState:
        """LLM을 사용한 충족도 평가"""
        state["debug_path"].append("llm_evaluate")
        
        try:
            from openai import OpenAI
            import config
            
            # proxies 파라미터 제거
            client = OpenAI(
                base_url="https://api.together.xyz/v1",
                api_key=config.OPENROUTER_API_KEY
            )
            
            prompt = f"""질문: {state['query']}

RAG 검색 결과:
{state['context']}

이 검색 결과가 질문의 모든 요소에 답변하는지 평가하세요.
확인 사항:
- 질문에서 요구하는 모든 정보가 포함되었는가?
- 숫자, 비율, 기간 등 구체적 정보가 필요한 경우 제공되었는가?
- 조건이나 가능 여부를 묻는 경우 명확히 답변되었는가?

JSON 형식으로 답변:
{{
  "coverage_score": 0.0~1.0,
  "covered_elements": ["답변된 요소들"],
  "missing_elements": ["빠진 요소들"],
  "evaluation": "평가 설명"
}}"""

            completion = client.chat.completions.create(
                model=config.EVAL_MODEL,  # Qwen2.5-7B
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,  # 평가는 일관성 있게
                max_tokens=300
                # response_format 파라미터 제거 (Together AI에서 지원 안함)
            )
            
            # JSON 파싱
            try:
                # 응답에서 JSON 부분만 추출
                response_text = completion.choices[0].message.content
                # JSON 블록 찾기
                json_start = response_text.find('{')
                json_end = response_text.rfind('}') + 1
                if json_start >= 0 and json_end > json_start:
                    json_text = response_text[json_start:json_end]
                    evaluation = json.loads(json_text)
                else:
                    raise ValueError("No JSON found in response")
                    
            except Exception as json_error:
                logger.warning(f"JSON parsing error: {json_error}")
                # 기본값 사용
                evaluation = {
                    "coverage_score": 0.7,
                    "covered_elements": [],
                    "missing_elements": [],
                    "evaluation": "JSON 파싱 실패, 기본값 사용"
                }
            
            state["coverage_score"] = evaluation.get("coverage_score", 0.5)
            state["coverage_details"] = evaluation
            state["missing_parts"] = evaluation.get("missing_elements", [])
            
            logger.info(f"LLM Coverage: {state['coverage_score']:.2f}, Missing: {state['missing_parts']}")
            
        except Exception as e:
            logger.error(f"LLM evaluation failed: {e}")
            # Fallback: 간단한 규칙 기반 평가
            state["coverage_score"] = 0.7 if state["relevance_score"] > 0.5 else 0.3
            state["coverage_details"] = {"evaluation": "LLM 평가 실패, 기본값 사용"}
            state["missing_parts"] = []
        
        return state

    def route_by_coverage(self, state: RAGState) -> str:
        """충족도에 따른 라우팅"""
        score = state["coverage_score"]
        
        if score >= 0.9:
            return "complete"
        elif score >= 0.5:
            return "partial"
        else:
            return "insufficient"

    def generate_from_rag(self, state: RAGState) -> RAGState:
        """RAG 결과로 답변 생성"""
        state["debug_path"].append("generate_from_rag")
        
        try:
            from openai import OpenAI
            import config
            
            # proxies 파라미터 제거
            client = OpenAI(
                base_url="https://api.together.xyz/v1",
                api_key=config.OPENROUTER_API_KEY
            )
            
            prompt = f"""질문: {state['query']}

검색된 정보:
{state['context']}

위 정보를 바탕으로 자연스럽고 완전한 답변을 작성하세요.
간결하고 명확하게 답변하되, 모든 정보를 포함하세요."""

            completion = client.chat.completions.create(
                model=config.EVAL_MODEL,  # Qwen2.5-7B
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=500
            )
            
            state["raw_answer"] = completion.choices[0].message.content
            state["answer_method"] = "rag_complete"
            state["confidence"] = 0.9
            
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            state["raw_answer"] = state["context"][:500]
            state["answer_method"] = "rag_direct"
            state["confidence"] = 0.7
        
        return state

    def enhance_missing(self, state: RAGState) -> RAGState:
        """빠진 부분만 보강"""
        state["debug_path"].append("enhance_missing")
        
        missing_parts = state.get("missing_parts", [])
        
        if not missing_parts:
            # 빠진 게 없으면 RAG 그대로 사용
            return self.generate_from_rag(state)
        
        try:
            from openai import OpenAI
            import config
            
            # proxies 파라미터 제거
            client = OpenAI(
                base_url="https://api.together.xyz/v1",
                api_key=config.OPENROUTER_API_KEY
            )
            
            missing_text = ", ".join(missing_parts)
            
            prompt = f"""질문: {state['query']}

현재 답변:
{state['context']}

다음 정보가 부족합니다: {missing_text}

부족한 정보를 추가하여 완전한 답변을 작성하세요.
실업급여 관련 최신 정책을 반영하여 답변하세요."""

            completion = client.chat.completions.create(
                model=config.MODEL,  # Qwen3-235B or QwQ-32B
                messages=[{"role": "user", "content": prompt}],
                temperature=0.5,
                max_tokens=600
            )
            
            state["raw_answer"] = completion.choices[0].message.content
            state["answer_method"] = "enhanced"
            state["confidence"] = 0.8
            
        except Exception as e:
            logger.error(f"Enhancement failed: {e}")
            return self.generate_from_rag(state)
        
        return state

    def regenerate_full(self, state: RAGState) -> RAGState:
        """전체 재생성 (RAG 불충분)"""
        state["debug_path"].append("regenerate_full")
        
        try:
            from openai import OpenAI
            import config
            
            # proxies 파라미터 제거
            client = OpenAI(
                base_url="https://api.together.xyz/v1",
                api_key=config.OPENROUTER_API_KEY
            )
            
            prompt = f"""질문: {state['query']}

참고 정보:
{state['context']}

위 질문에 대해 완전하고 정확한 답변을 생성하세요.
2025년 최신 실업급여 정책을 반영하여 답변하세요.
숫자, 기간, 조건 등 구체적인 정보를 포함하세요."""

            completion = client.chat.completions.create(
                model=config.MODEL,  # Qwen3-235B or QwQ-32B
                messages=[{"role": "user", "content": prompt}],
                temperature=0.5,
                max_tokens=800
            )
            
            state["raw_answer"] = completion.choices[0].message.content
            state["answer_method"] = "regenerated"
            state["confidence"] = 0.85
            
        except Exception as e:
            logger.error(f"Regeneration failed: {e}")
            state["raw_answer"] = "죄송합니다. 답변 생성 중 오류가 발생했습니다."
            state["answer_method"] = "error"
            state["confidence"] = 0.0
        
        return state

    def format_final(self, state: RAGState) -> RAGState:
        """최종 포맷팅"""
        state["debug_path"].append("format_final")
        
        answer = state.get("raw_answer", "관련 정보를 찾을 수 없습니다.")
        
        # 답변 끝에 메타 정보 추가 (개발용)
        if logger.level == logging.DEBUG:
            debug_info = f"\n\n[디버그: 충족도 {state.get('coverage_score', 0):.1%}, 방법: {state.get('answer_method', 'unknown')}]"
            answer += debug_info
        
        state["final_answer"] = answer
        
        logger.info(f"Workflow complete: {' → '.join(state['debug_path'])}")
        return state

    def _classify_query_type(self, query: str) -> str:
        """질문 유형 분류"""
        if any(word in query for word in ["얼마", "금액", "계산"]):
            return "calculation"
        elif any(word in query for word in ["자격", "조건", "가능"]):
            return "qualification"
        elif any(word in query for word in ["어떻게", "방법", "절차"]):
            return "procedure"
        else:
            return "general"

    def run(self, query: str) -> Dict:
        """워크플로우 실행"""
        initial_state = {
            "query": query,
            "processed_query": "",
            "query_type": "",
            "documents": [],
            "relevance_score": 0.0,
            "context": "",
            "coverage_details": {},
            "coverage_score": 0.0,
            "missing_parts": [],
            "answer_method": "",
            "raw_answer": "",
            "final_answer": "",
            "confidence": 0.0,
            "debug_path": []
        }
        
        result = self.workflow.invoke(initial_state)
        
        return {
            "answer": result.get("final_answer", ""),
            "confidence": result.get("confidence", 0.0),
            "method": result.get("answer_method", "unknown"),
            "coverage": result.get("coverage_score", 0.0),
            "debug": {
                "path": result.get("debug_path", []),
                "coverage_details": result.get("coverage_details", {}),
                "missing_parts": result.get("missing_parts", [])
            }
        }


# 기존 클래스 대체
RAGWorkflow = SemanticRAGWorkflow
ImprovedRAGWorkflow = SemanticRAGWorkflow