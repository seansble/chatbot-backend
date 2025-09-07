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
    coverage_details: Dict[str, Any]
    coverage_score: float
    missing_parts: List[str]

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
                "complete": "generate_from_rag",
                "partial": "enhance_missing",
                "insufficient": "regenerate_full",
            },
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
            "받을 수 있": "수급 가능",
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
        """LLM을 사용한 충족도 평가 - 개선된 프롬프트"""
        state["debug_path"].append("llm_evaluate")

        try:
            from openai import OpenAI
            import config

            client = OpenAI(
                base_url="https://api.together.xyz/v1",
                api_key=config.OPENROUTER_API_KEY,
            )

            # 더 명확한 평가 프롬프트
            prompt = f"""사용자 질문: {state['query']}
RAG 문서: {state['context']}

작업: 질문에서 요구하는 정보가 RAG 문서에 있는지 평가

평가 방법:
1. 사용자가 명시적으로 질문한 내용만 확인
2. RAG 문서에 해당 정보가 있는지 체크
3. 질문하지 않은 내용은 평가하지 않음

JSON 응답:
{{
  "coverage_score": 0.0-1.0,
  "found_elements": ["RAG에서 찾은 정보"],
  "missing_elements": ["질문했지만 RAG에 없는 정보"],
  "evaluation": "간단한 설명"
}}"""

            completion = client.chat.completions.create(
                model=config.EVAL_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=300,
            )

            # JSON 파싱
            try:
                response_text = completion.choices[0].message.content
                json_start = response_text.find("{")
                json_end = response_text.rfind("}") + 1
                if json_start >= 0 and json_end > json_start:
                    json_text = response_text[json_start:json_end]
                    evaluation = json.loads(json_text)
                else:
                    raise ValueError("No JSON found in response")

            except Exception as json_error:
                logger.warning(f"JSON parsing error: {json_error}")
                evaluation = {
                    "coverage_score": 0.7,
                    "found_elements": [],
                    "missing_elements": [],
                    "evaluation": "JSON 파싱 실패, 기본값 사용",
                }

            state["coverage_score"] = evaluation.get("coverage_score", 0.5)
            state["coverage_details"] = evaluation
            state["missing_parts"] = evaluation.get("missing_elements", [])

            logger.info(
                f"LLM Coverage: {state['coverage_score']:.2f}, Missing: {state['missing_parts']}"
            )

        except Exception as e:
            logger.error(f"LLM evaluation failed with error type: {type(e).__name__}")
            logger.error(f"Error message: {str(e)}")

            # Fallback: 간단한 규칙 기반 평가
            state["coverage_score"] = 0.7 if state["relevance_score"] > 0.5 else 0.3
            state["coverage_details"] = {"evaluation": "LLM 평가 실패, 기본값 사용"}
            state["missing_parts"] = []

        return state

    def route_by_coverage(self, state: RAGState) -> str:
        """충족도에 따른 라우팅 - threshold 조정"""
        score = state["coverage_score"]

        if score >= 0.7:  # 0.9 → 0.7로 낮춤 (RAG 더 신뢰)
            return "complete"
        elif score >= 0.4:  # 0.5 → 0.4로 낮춤
            return "partial"
        else:
            return "insufficient"

    def generate_from_rag(self, state: RAGState) -> RAGState:
        """RAG 결과로 답변 생성"""
        state["debug_path"].append("generate_from_rag")

        try:
            from openai import OpenAI
            import config

            client = OpenAI(
                base_url="https://api.together.xyz/v1",
                api_key=config.OPENROUTER_API_KEY,
            )

            # 질문과 컨텍스트를 명확히 구분
            prompt = f"""다음은 실업급여 관련 정보입니다. 사용자의 질문에 대해서만 답변하세요.

사용자 질문: {state['query']}

참고 정보:
{state['context']}

위 참고 정보를 활용하여 사용자의 질문에 대한 답변만 작성하세요.
다른 질문이나 예시는 절대 포함하지 마세요.
간결하고 명확하게 답변하세요."""

            completion = client.chat.completions.create(
                model=config.EVAL_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": "실업급여 전문 상담사입니다. 질문에 대한 답변만 간결하게 제공하세요.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                max_tokens=500,
            )

            state["raw_answer"] = completion.choices[0].message.content
            state["answer_method"] = "rag_complete"
            state["confidence"] = 0.9

        except Exception as e:
            logger.error(f"Generation failed with error: {str(e)}")

            # 폴백: 질문과 가장 관련된 답변만 추출
            if state.get("documents") and len(state["documents"]) > 0:
                answer = self._extract_relevant_answer(
                    state["query"], state["documents"]
                )
                state["raw_answer"] = answer
            else:
                state["raw_answer"] = (
                    "죄송합니다. 관련 정보를 찾을 수 없습니다. 고용센터 1350으로 문의해주세요."
                )

            state["answer_method"] = "rag_direct"
            state["confidence"] = 0.7

        return state

    def enhance_missing(self, state: RAGState) -> RAGState:
        """빠진 부분만 보강 - 개선된 프롬프트"""
        state["debug_path"].append("enhance_missing")

        missing_parts = state.get("missing_parts", [])

        if not missing_parts:
            # 빠진 게 없으면 RAG 그대로 사용
            return self.generate_from_rag(state)

        try:
            from openai import OpenAI
            import config

            client = OpenAI(
                base_url="https://api.together.xyz/v1",
                api_key=config.OPENROUTER_API_KEY,
            )

            missing_text = ", ".join(missing_parts)

            # Qwen3와 동일한 톤 유지하면서 RAG 우선
            prompt = f"""사용자 질문: {state['query']}

RAG 검색 결과 (2025년 최신):
{state['context']}

RAG에서 찾지 못한 정보: {missing_text}

작업 지시:
1. RAG 검색 결과를 기본으로 사용 (정보 충돌시 RAG 우선)
2. {missing_text}에 대한 정보만 추가로 보충
3. 친근하고 이해하기 쉽게 설명
4. 💼, 🎂 같은 이모지 적절히 사용
5. 최종 답변은 800자 이내

답변:"""

            completion = client.chat.completions.create(
                model=config.MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": "당신은 실업급여 전문 상담사입니다. 친근하고 정확하게 답변하세요.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,  # 일관성을 위해 낮게
                max_tokens=1000,
            )  # 충분한 토큰 확보

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

            client = OpenAI(
                base_url="https://api.together.xyz/v1",
                api_key=config.OPENROUTER_API_KEY,
            )

            prompt = f"""질문: {state['query']}

참고 정보 (2025년 기준):
{state['context']}

위 질문에 대해 완전하고 정확한 답변을 생성하세요.
참고 정보가 있다면 우선 활용하고, 부족한 부분은 보충하세요.
2025년 최신 실업급여 정책을 반영하여 답변하세요.
숫자, 기간, 조건 등 구체적인 정보를 포함하세요."""

            completion = client.chat.completions.create(
                model=config.MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": "당신은 실업급여 전문 상담사입니다. 정확하고 최신 정보로 답변하세요.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                max_tokens=1000,
            )

            state["raw_answer"] = completion.choices[0].message.content
            state["answer_method"] = "regenerated"
            state["confidence"] = 0.85

        except Exception as e:
            logger.error(f"Regeneration failed: {e}")
            # 폴백 답변
            if state.get("documents"):
                answer = self._extract_relevant_answer(
                    state["query"], state["documents"]
                )
                state["raw_answer"] = answer
            else:
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

        # coverage_score가 0으로 리셋되는 버그 수정
        if state.get("coverage_score") == 0 and state.get("answer_method") != "error":
            logger.warning("Coverage score was 0, using confidence instead")
            state["coverage_score"] = state.get("confidence", 0.5)

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

    def _extract_relevant_answer(self, query: str, documents: List[Dict]) -> str:
        """문서에서 관련 답변 추출"""
        query_lower = query.lower()

        # 키워드 매칭으로 가장 관련된 답변 찾기
        for doc in documents[:2]:
            text = doc.get("text", "")

            # "질문:" "답변:" 패턴으로 분리
            if "답변:" in text:
                parts = text.split("질문:")
                for part in parts:
                    # 질문의 키워드가 포함된 부분 찾기
                    if any(
                        keyword in query_lower for keyword in ["조건", "자격", "가능"]
                    ):
                        if any(
                            keyword in part.lower()
                            for keyword in ["조건", "자격", "가능"]
                        ):
                            if "답변:" in part:
                                answer = (
                                    part.split("답변:")[1].split("질문:")[0].strip()
                                )
                                return answer

                # 첫 번째 답변 반환 (폴백)
                answer = text.split("답변:")[1].split("질문:")[0].strip()
                return answer

        # 기본 답변
        return """실업급여 수급을 위한 기본 조건:
1. 이직일 이전 18개월간 고용보험 가입기간 180일 이상
2. 비자발적 퇴직 (권고사직, 계약만료 등)
3. 근로 의사와 능력이 있고 적극적 구직활동 중
4. 이직 후 12개월 이내 신청

자세한 상담은 고용센터 1350으로 문의하세요."""

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
            "debug_path": [],
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
                "missing_parts": result.get("missing_parts", []),
            },
        }


# 기존 클래스 대체
RAGWorkflow = SemanticRAGWorkflow
ImprovedRAGWorkflow = SemanticRAGWorkflow
