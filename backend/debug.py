# 디버그용 테스트
def test_extract():
    from kiwipiepy import Kiwi
    kiwi = Kiwi()
    result = kiwi.tokenize("회사에서 3년 6개월간")
    
    tokens = []
    skip_next = False
    
    for i, token in enumerate(result):
        print(f"{i}: {token.form}:{token.tag}, skip_next={skip_next}")
        
        if skip_next:
            print(f"  -> 스킵됨")
            skip_next = False
            continue
            
        if token.tag[0] in ["J", "E", "X", "S"]:
            print(f"  -> 제외됨 (조사/어미)")
            continue
        
        if token.tag == "SN":
            if i + 1 < len(result) and result[i + 1].tag == "NNB":
                combined = token.form + result[i + 1].form
                tokens.append(combined)
                skip_next = True
                print(f"  -> 결합: {combined}")
                continue
                
        if token.tag.startswith("N"):
            tokens.append(token.form)
            print(f"  -> 추가: {token.form}")
    
    print(f"최종: {tokens}")

test_extract()