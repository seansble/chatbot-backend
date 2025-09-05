# test_complex_llm.py

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "backend"))

from rag.retriever import RAGRetriever
from rag.workflow import ImprovedRAGWorkflow
import time

print("🔥 초복잡 질문 LLM 처리 테스트")
print("=" * 60)

retriever = RAGRetriever(use_hybrid=True)
workflow = ImprovedRAGWorkflow(retriever)

# 정말 복잡한 테스트 질문들
super_complex_questions = [
    """2021년 3월부터 A회사에서 월 350만원 받으며 일하다가 2022년 11월에 임금체불 4개월로 
    자진퇴사해서 실업급여 150일 받았는데, 90일 받고 2023년 3월에 B회사 재취업했다가 
    수습 3개월만에 능력부족으로 해고당하고, 그때는 급여 안받고 프리랜서로 6개월 일하면서 
    특수고용 보험료 냈다가 2024년 1월에 C회사 정규직으로 월 400만원에 입사했는데 
    2024년 12월에 회사가 갑자기 폐업했어요. 지금 2025년 1월인데 신청하면 몇번째 수급이고 
    감액률은 얼마나 되나요? 그리고 2023년에 못받은 60일은 어떻게 되나요?""",
    """작년 8월 권고사직으로 실업급여 받다가 10월에 쿠팡플렉스 시작했는데 일당 5만원 
    넘는 날이 많아서 감액됐고, 올해 1월에 정규직 취업했다가 3개월만에 회사 구조조정으로 
    또 권고사직 당했는데, 이번이 몇번째인지 모르겠고 깎이는게 얼마인지, 
    그리고 쿠팡플렉스는 계속 하고 있는데 이것도 신고해야 하나요?""",
    """65세 되기 2개월 전에 20년 다닌 회사에서 정년퇴직했는데 퇴직금으로 치킨집 차렸다가 
    6개월만에 폐업하고 다시 경비직 취업했는데 3개월만에 건강문제로 퇴사했어요. 
    실업급여 신청 가능한가요? 자영업 기간이랑 65세 넘은 것 때문에 안될까요?""",
]

for i, question in enumerate(super_complex_questions, 1):
    print(f"\n{'='*50}")
    print(f"테스트 {i}: 복잡도 극상")
    print(f"질문 길이: {len(question)}자")
    print(f"질문: {question[:100]}...")
    print("-" * 50)

    start = time.time()

    try:
        result = workflow.run(question)
        elapsed = time.time() - start

        # 디버그 경로 확인
        path = result.get("debug_path", [])
        method = result.get("method", "unknown")

        print(f"\n처리 경로: {' → '.join(path)}")
        print(f"처리 방법: {method}")

        # LLM 직접 처리인지 확인
        if "llm_direct" in path:
            print("✅ LLM이 직접 처리했습니다!")
        else:
            print("❌ RAG로 처리됨 (복잡도 판단 실패)")

        print(f"\n답변 (처음 300자):")
        print(result["answer"][:300])
        print(f"\n신뢰도: {result['confidence']:.2f}")
        print(f"처리 시간: {elapsed:.2f}초")

    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback

        traceback.print_exc()

# LLM 호출 여부 테스트
print("\n" + "=" * 60)
print("📊 테스트 요약")
print("=" * 60)
print("로컬에서 LLM 호출하려면:")
print("1. .env에 TOGETHER_API_KEY 설정 확인")
print("2. app.py의 generate_ai_answer 함수가 작동하는지 확인")
print("3. 복잡도 점수가 5점 이상인지 확인")
