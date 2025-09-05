"""인터랙티브 RAG 테스트"""
import sys
sys.path.append('backend')
from rag.retriever import RAGRetriever
from colorama import init, Fore, Style

init()  # Windows 색상 지원

def test_interactive():
    retriever = RAGRetriever()
    
    print(f"{Fore.CYAN}{'='*60}")
    print("실업급여 RAG 시스템 테스트")
    print(f"{'='*60}{Style.RESET_ALL}")
    
    # 변형 테스트 질문들
    test_cases = [
        ("자진퇴사", "자발적 퇴사", "스스로 그만둠"),
        ("실업급여 얼마", "구직급여 금액", "수급액 계산"),
        ("반복수급", "여러번 받으면", "네번째 실업급여"),
    ]
    
    print(f"\n{Fore.YELLOW}[변형 테스트]{Style.RESET_ALL}")
    for original, *variations in test_cases:
        print(f"\n원본: {original}")
        for var in variations:
            results = retriever.retrieve(var, top_k=1)
            if results:
                print(f"  '{var}' → 점수: {results[0]['score']:.3f}")
                print(f"           매칭: {results[0]['metadata'].get('category')}")
    
    print(f"\n{Fore.GREEN}[자유 질문 모드]{Style.RESET_ALL}")
    while True:
        query = input(f"\n{Fore.CYAN}질문 입력 (q=종료): {Style.RESET_ALL}")
        if query.lower() == 'q':
            break
            
        results = retriever.retrieve(query, top_k=3)
        
        if not results:
            print(f"{Fore.RED}검색 결과 없음{Style.RESET_ALL}")
            continue
            
        print(f"\n{Fore.GREEN}=== 검색 결과 ==={Style.RESET_ALL}")
        for i, result in enumerate(results, 1):
            score = result['score']
            
            # 점수별 색상
            if score > 0.8:
                color = Fore.GREEN
            elif score > 0.7:
                color = Fore.YELLOW
            else:
                color = Fore.RED
                
            print(f"\n{color}[{i}] 관련도: {score:.3f}{Style.RESET_ALL}")
            print(f"카테고리: {result['metadata'].get('category')}")
            
            # 답변 미리보기
            text = result['text']
            if '답변:' in text:
                answer = text.split('답변:')[1].strip()[:100]
                print(f"답변: {answer}...")

if __name__ == "__main__":
    test_interactive()