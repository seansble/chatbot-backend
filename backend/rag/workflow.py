"""LangGraph RAG 워크플로우 - 평가자 기반 조건부 처리"""

from typing import Dict, List, TypedDict, Literal, Optional
from langgraph.graph import StateGraph, END
import logging
import re

logger = logging.getLogger(__name__)


class RAGState(TypedDict):
    """워크플로우 상태 - 평가 필드 추가"""

    # 입력
    query: str  # 원본 질문

    # 전처리 단계
    processed_query: str  # 정제된 질문
    query_type: str  # 질문 유형
    is_blocked: bool  # 차단 여부

    # 검색 단계
    search_strategy: str  # 검색 전략
    documents: List[Dict]  # 검색된 문서들
    relevance_score: float  # RAG 관련도 점수

    # 평가 단계 (새로 추가)
    requirements: List[str]  # 질문의 요구사항 분해
    coverage: Dict[str, bool]  # 요구사항 커버리지
    coverage_score: float  # 커버리지 점수 (0-1)
    needs_enhancement: bool  # LLM 보강 필요 여부

    # 답변 생성 단계
    answer_method: str  # 답변 생성 방법
    context: str  # 구성된 컨텍스트
    raw_answer: str  # 1차 답변
    final_answer: str  # 최종 답변

    # 메타데이터
    confidence: float  # 전체 신뢰도
    iteration: int  # 재시도 횟수
    debug_path: List[str]  # 디버깅 경로


class ImprovedRAGWorkflow:
    def __init__(self, retriever):
        self.retriever = retriever
        self.workflow = self._build_workflow()

        # 질문 유형별 키워드 매핑
        self.query_patterns = {
            "calculation": ["얼마", "금액", "계산", "월급", "일당"],
            "qualification": ["자격", "조건", "가능", "될까", "되나요"],
            "procedure": ["어떻게", "방법", "절차", "신청", "서류"],
            "specific_case": ["임금체불", "권고사직", "계약만료", "육아휴직"],
        }

        # 차단 패턴 (복잡한 질문은 제외)
        self.block_patterns = [
            (r"얼마.*받", "금액_계산_금지"),
            (r"계산.*해", "금액_계산_금지"),
            (r"\d+만원.*받", "금액_계산_금지"),
        ]

        # 명백한 미스매치 패턴
        self.mismatch_patterns = [
            ("반복수급", ["재취업", "촉진"]),
            ("이직확인서", ["권고사직", "자진퇴사"]),
            ("65세", ["임금체불", "프리랜서"]),
            ("몇번째", ["조건", "자격"]),
        ]

    def _build_workflow(self):
        """평가자 중심 워크플로우 구성"""
        workflow = StateGraph(RAGState)

        # === 노드 정의 ===
        workflow.add_node("analyze_query", self.analyze_query)
        workflow.add_node("rag_search", self.rag_search)  # 모든 질문 RAG
        workflow.add_node("decompose_question", self.decompose_question)  # 새 노드
        workflow.add_node("evaluate_coverage", self.evaluate_coverage)  # 새 노드
        workflow.add_node("enhance_partial", self.enhance_partial)  # 새 노드
        workflow.add_node("regenerate_full", self.regenerate_full)  # 새 노드
        workflow.add_node("verify_quality", self.verify_quality)  # 새 노드
        workflow.add_node("format_final", self.format_final)

        # === 엣지 정의 ===

        # 시작: 분석 → RAG 검색
        workflow.set_entry_point("analyze_query")
        workflow.add_edge("analyze_query", "rag_search")

        # RAG 검색 → 질문 분해
        workflow.add_edge("rag_search", "decompose_question")

        # 질문 분해 → 평가
        workflow.add_edge("decompose_question", "evaluate_coverage")

        # 평가 결과에 따른 분기
        workflow.add_conditional_edges(
            "evaluate_coverage",
            self.route_by_coverage,
            {
                "use_rag": "format_final",  # 80% 이상 - RAG 그대로
                "enhance": "enhance_partial",  # 40-79% - 부분 보강
                "regenerate": "regenerate_full",  # 40% 미만 - 재생성
            },
        )

        # 보강/재생성 → 품질 검증
        workflow.add_edge("enhance_partial", "verify_quality")
        workflow.add_edge("regenerate_full", "verify_quality")

        # 품질 검증 후 분기
        workflow.add_conditional_edges(
            "verify_quality",
            self.route_after_verify,
            {
                "pass": "format_final",
                "retry": "regenerate_full",  # 1회만 재시도
                "fallback": "format_final",  # 포기
            },
        )

        # 최종 포맷팅 → 종료
        workflow.add_edge("format_final", END)

        return workflow.compile()

    def analyze_query(self, state: RAGState) -> RAGState:
        """쿼리 분석 - 차단 로직 제거"""
        query = state["query"]
        state["debug_path"] = ["analyze_query"]
        state["iteration"] = 0

        # 전처리: 구어체 → 표준어
        replacements = {
            "때려치": "자진퇴사",
            "짤렸": "해고 권고사직",
            "배민": "배달 라이더 프리랜서",
            "쿠팡": "쿠팡플렉스 프리랜서",
            "얼마": "금액",
            "깎": "감액",
            "못받": "미수급",  # "임금체불"로 변환 안함!
        }

        processed = query.lower()
        for old, new in replacements.items():
            if old in processed:
                processed = processed.replace(old, new)
                logger.info(f"변환: '{old}' → '{new}'")

        # 질문 유형 분류
        query_type = "general"
        for type_name, patterns in self.query_patterns.items():
            if any(p in processed for p in patterns):
                query_type = type_name
                break

        state["processed_query"] = processed
        state["query_type"] = query_type
        state["is_blocked"] = False  # 차단 안함

        logger.info(f"분석 완료: 유형={query_type}")
        return state

    def rag_search(self, state: RAGState) -> RAGState:
        """모든 질문 RAG 검색"""
        state["debug_path"].append("rag_search")
        query = state["processed_query"]

        # 하이브리드 검색 실행
        results = self.retriever.retrieve(query, top_k=5)

        state["documents"] = results
        state["relevance_score"] = results[0]["score"] if results else 0.0
        state["context"] = self._build_context(results)

        logger.info(f"RAG 검색: {len(results)}개, 점수={state['relevance_score']:.2f}")
        return state

    def decompose_question(self, state: RAGState) -> RAGState:
        """질문을 요구사항으로 분해"""
        state["debug_path"].append("decompose_question")
        query = state["query"]

        requirements = []

        # 간단한 규칙 기반 분해
        if "가능" in query or "되나요" in query:
            requirements.append("가능_여부")

        if "언제" in query or "기간" in query:
            requirements.append("시점_기간")

        if "얼마" in query or "금액" in query:
            requirements.append("금액_수치")

        if "어떻게" in query or "방법" in query:
            requirements.append("방법_절차")

        if "조건" in query or "자격" in query:
            requirements.append("조건_자격")

        # 복잡한 질문 감지
        if len(query) > 100 or query.count(",") >= 2:
            requirements.append("복합_상황")

        # 기본 요구사항
        if not requirements:
            requirements = ["일반_정보"]

        state["requirements"] = requirements
        logger.info(f"요구사항 분해: {requirements}")
        return state

    def evaluate_coverage(self, state: RAGState) -> RAGState:
        """RAG 답변의 요구사항 커버리지 평가 - LLM 사용"""
        state["debug_path"].append("evaluate_coverage")

        documents = state["documents"]
        query = state["query"]

        if not documents:
            state["coverage_score"] = 0.0
            state["needs_enhancement"] = True
            return state

        # 상위 2개 문서로 컨텍스트 구성
        context = "\n".join([doc["text"] for doc in documents[:2]])

        try:
            from openai import OpenAI
            import json
            import config

            client = OpenAI(
                base_url="https://api.together.xyz/v1",
                api_key=config.OPENROUTER_API_KEY,
            )

            # 평가 프롬프트
            eval_prompt = f"""질문: {query}
            
    RAG 검색 결과: {context}

    이 검색 결과가 질문에 충분히 답하는지 0-1 점수로 평가하세요.

    JSON 형식으로만 답하세요:
    {{"score": 0.85, "sufficient": true}}"""

            completion = client.chat.completions.create(
                model=config.EVAL_MODEL,  # Qwen2.5-7B 사용
                messages=[{"role": "user", "content": eval_prompt}],
                temperature=0.1,
                max_tokens=100,
            )

            # 응답 파싱
            response = completion.choices[0].message.content
            import re

            json_match = re.search(r"\{.*\}", response, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                state["coverage_score"] = result.get("score", 0.5)
                state["needs_enhancement"] = not result.get("sufficient", False)
            else:
                state["coverage_score"] = 0.5
                state["needs_enhancement"] = True

        except Exception as e:
            logger.error(f"LLM 평가 실패: {e}")
            state["coverage_score"] = 0.5
            state["needs_enhancement"] = True

        logger.info(f"LLM 평가: 점수={state['coverage_score']}")
        return state

    def route_by_coverage(self, state: RAGState) -> str:
        """커버리지에 따른 라우팅"""
        score = state["coverage_score"]

        if score >= 0.8:
            return "use_rag"
        elif score >= 0.4:
            return "enhance"
        else:
            return "regenerate"

    def enhance_partial(self, state: RAGState) -> RAGState:
        """부분 답변 LLM 보강"""
        state["debug_path"].append("enhance_partial")

        try:
            from app import generate_ai_answer

            # 부족한 부분만 보강하는 프롬프트
            missing = [req for req, covered in state["coverage"].items() if not covered]

            enhanced_query = f"""
            질문: {state['query']}
            
            현재 답변: {state['documents'][0]['text'] if state['documents'] else ''}
            
            다음 부분을 보완해서 답변해주세요: {', '.join(missing)}
            """

            answer = generate_ai_answer(enhanced_query, stream=False)
            state["raw_answer"] = answer
            state["confidence"] = 0.7
            state["answer_method"] = "enhanced"

        except Exception as e:
            logger.error(f"보강 실패: {e}")
            state["raw_answer"] = (
                state["documents"][0]["text"] if state["documents"] else ""
            )
            state["confidence"] = 0.5

        return state

    def regenerate_full(self, state: RAGState) -> RAGState:
        """LLM 전체 재생성"""
        state["debug_path"].append("regenerate_full")
        state["iteration"] += 1

        try:
            from app import generate_ai_answer

            # RAG 결과 포함해서 재생성
            context = state.get("context", "")
            full_query = f"""
            질문: {state['query']}
            
            참고 정보:
            {context}
            
            위 정보를 참고하여 정확하고 완전한 답변을 생성하세요.
            """

            answer = generate_ai_answer(full_query, stream=False)
            state["raw_answer"] = answer
            state["confidence"] = 0.9
            state["answer_method"] = "regenerated"

        except Exception as e:
            logger.error(f"재생성 실패: {e}")
            state["raw_answer"] = "답변 생성 중 오류가 발생했습니다."
            state["confidence"] = 0.0

        return state

    def verify_quality(self, state: RAGState) -> RAGState:
        """품질 검증 (간단한 체크만)"""
        state["debug_path"].append("verify_quality")

        answer = state.get("raw_answer", "")

        # 기본적인 품질 체크
        quality_ok = (
            len(answer) > 20 and "오류" not in answer and state["confidence"] > 0.3
        )

        state["quality_verified"] = quality_ok
        logger.info(f"품질 검증: {'통과' if quality_ok else '실패'}")
        return state

    def route_after_verify(self, state: RAGState) -> str:
        """품질 검증 후 라우팅"""
        if state.get("quality_verified", False):
            return "pass"
        elif state["iteration"] < 1:  # 1회만 재시도
            return "retry"
        else:
            return "fallback"

    def format_final(self, state: RAGState) -> RAGState:
        """최종 포맷팅"""
        state["debug_path"].append("format_final")

        # RAG 답변이 있으면 사용, 없으면 raw_answer
        if not state.get("raw_answer") and state["documents"]:
            state["raw_answer"] = state["documents"][0]["text"]

        answer = state.get("raw_answer", "관련 정보를 찾을 수 없습니다.")
        confidence = state.get("confidence", state.get("coverage_score", 0))

        # 신뢰도별 prefix
        if confidence >= 0.8:
            prefix = "✅ "
        elif confidence >= 0.5:
            prefix = "📌 "
        else:
            prefix = "ℹ️ "

        state["final_answer"] = prefix + answer

        logger.info(f"경로: {' → '.join(state['debug_path'])}")
        logger.info(f"최종 신뢰도: {confidence:.2f}")

        return state

    def _build_context(self, documents: List[Dict]) -> str:
        """문서로부터 컨텍스트 구성"""
        if not documents:
            return ""

        context = "관련 정보:\n"
        for i, doc in enumerate(documents[:3], 1):
            context += f"{i}. {doc['text'][:200]}\n"

        return context

    def run(self, query: str) -> Dict:
        """워크플로우 실행"""
        initial_state = {
            "query": query,
            "processed_query": "",
            "query_type": "",
            "is_blocked": False,
            "search_strategy": "",
            "documents": [],
            "relevance_score": 0.0,
            "requirements": [],
            "coverage": {},
            "coverage_score": 0.0,
            "needs_enhancement": False,
            "answer_method": "",
            "context": "",
            "raw_answer": "",
            "final_answer": "",
            "confidence": 0.0,
            "iteration": 0,
            "debug_path": [],
        }

        result = self.workflow.invoke(initial_state)

        return {
            "answer": result.get("final_answer", result.get("raw_answer", "")),
            "documents": result.get("documents", []),
            "context": result.get("context", ""),
            "confidence": result.get("confidence", 0.0),
            "debug_path": result.get("debug_path", []),
            "method": result.get("answer_method", "unknown"),
            "coverage_score": result.get("coverage_score", 0.0),
        }


# 기존 RAGWorkflow를 새 버전으로 교체
RAGWorkflow = ImprovedRAGWorkflow
