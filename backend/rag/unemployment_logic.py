# backend/rag/unemployment_logic.py - v5.0 (GPT 권고사항 반영)
"""
실업급여 통합 로직 모듈 v5.0
- 반복수급 최댓값 선택
- 나이 기본값 25세
- 청년/장애 특례 정확히 처리
"""

import re
import logging
import json
from typing import Dict, Any, Optional, Tuple, List, Set
from dataclasses import dataclass
from functools import lru_cache
import time

logger = logging.getLogger(__name__)

# -------------------------------
# Kiwi 필수화 + 헬스체크
# -------------------------------
try:
    from kiwipiepy import Kiwi
except ImportError as e:
    raise ImportError("Kiwi is required in production. Install 'kiwipiepy'.") from e

try:
    _KIWI = Kiwi(num_workers=0)
    _ = _KIWI.tokenize("헬스체크")
except Exception as e:
    raise RuntimeError(f"Kiwi health check failed: {e}")

# -------------------------------
# 한글 숫자 유틸
# -------------------------------
_KO_ONE = {"일": 1, "이": 2, "삼": 3, "사": 4, "오": 5, "육": 6, "칠": 7, "팔": 8, "구": 9}
_KO_TENS_WORD = {"초반": 2, "중반": 5, "후반": 8}

NATIVE_DECADES = {"스물": 20, "서른": 30, "마흔": 40, "쉰": 50, "예순": 60, "일흔": 70, "여든": 80, "아흔": 90}
NATIVE_ONES = {
    "영":0, "공":0,
    "한":1, "하나":1, "일":1,
    "두":2, "둘":2, "이":2,
    "세":3, "셋":3, "삼":3,
    "네":4, "넷":4, "사":4,
    "다섯":5, "오":5,
    "여섯":6, "육":6,
    "일곱":7, "칠":7,
    "여덟":8, "팔":8,
    "아홉":9, "구":9
}
_SUFFIX_RE = re.compile(r"(살|세|인데|입니다|이에요|이예요|입니다만|이지만)$")

def ko_word_to_int(tok: str) -> Optional[int]:
    if not tok:
        return None
    tok = _SUFFIX_RE.sub("", tok.strip()).replace(" ", "")
    if "십" in tok:
        if tok == "십":
            return 10
        parts = tok.split("십", 1)
        tens = NATIVE_ONES.get(parts[0], 1) if parts[0] else 1
        ones = NATIVE_ONES.get(parts[1], 0) if parts[1] else 0
        return tens * 10 + ones
    for dec, base in NATIVE_DECADES.items():
        if tok.startswith(dec):
            rest = tok[len(dec):]
            return base + NATIVE_ONES.get(rest, 0)
    return NATIVE_ONES.get(tok)

def ko_hundreds_phrase_to_int(s: str) -> Optional[int]:
    if not s:
        return None
    m = re.fullmatch(r"([이삼사오육칠팔구])백(?:(?:([일이삼사오육칠팔구])?십)?([일이삼사오육칠팔구])?)?", s)
    if not m:
        return None
    val = _KO_ONE[m.group(1)] * 100
    tens_digit = m.group(2)
    ones_digit = m.group(3)
    if "십" in s:
        val += (_KO_ONE[tens_digit] * 10) if tens_digit else 10
    if ones_digit:
        val += _KO_ONE[ones_digit]
    return val

def ko_compact_number_to_int(text: str) -> Optional[int]:
    if not text:
        return None
    m = re.fullmatch(r"([일이삼사오육칠팔구])천([일이삼사오육칠팔구])?백?", text)
    if m:
        val = _KO_ONE[m.group(1)] * 1000
        if m.group(2):
            val += _KO_ONE[m.group(2)] * 100
        return val
    m = re.fullmatch(r"([일이삼사오육칠팔구])천", text)
    if m:
        return _KO_ONE[m.group(1)] * 1000
    m = re.fullmatch(r"([일이삼사오육칠팔구])백", text)
    if m:
        return _KO_ONE[m.group(1)] * 100
    return None

DISABILITY_RX = re.compile(
    r"(장애|장애인|장애\s*등록|장애\s*등급|장애\s*[1-6]\s*급|"
    r"지체장애|시각장애|청각장애|지적장애|자폐|뇌병변|정신장애|발달장애|"
    r"산재\s*장해|장해\s*등급|장해\s*[1-9]\s*급)"
)

# -------------------------------
# 세그멘테이션
# -------------------------------
@dataclass
class Segment:
    type: str
    text: str
    ordinal: Optional[str] = None
    duration: Optional[str] = None
    salary: Optional[str] = None
    period: Optional[str] = None
    issue: Optional[str] = None
    start_pos: int = 0
    end_pos: int = 0

class KiwiSegmenter:
    ISSUE_RX = re.compile(r"(체불|미지급|못\s*받|폐업|부도|파산|갑질|괴롭힘|맣했|망함)")
    ORD_RX = re.compile(r"(첫|두\s*번째|세\s*번째|네\s*번째|마지막|두번째|세번째|네번째)")

    def __init__(self):
        self.kiwi = _KIWI

    def segment(self, text: str) -> List[Segment]:
        return self._segment_with_kiwi(text)

    def _segment_with_kiwi(self, text: str) -> List[Segment]:
        segments: List[Segment] = []
        tokens = self.kiwi.tokenize(text)
        current: Optional[Segment] = None
        i = 0
        while i < len(tokens):
            tk = tokens[i]
            form, tag = tk.form, tk.tag

            win = text[tk.start:min(tk.start+10, len(text))]
            ord_m = self.ORD_RX.match(win)
            if ord_m:
                if current:
                    segments.append(current)
                current = Segment(type="career", text="", ordinal=ord_m.group(1), start_pos=tk.start)

            if form in ["직장", "회사", "시즌", "근무", "일", "다니"]:
                if not current:
                    current = Segment(type="career", text="", start_pos=tk.start)
                current.text += form + " "

            elif tag.startswith("NR") or tag.startswith("SN"):
                if i+1 < len(tokens):
                    n1 = tokens[i+1]
                    if n1.form in ["년", "개월", "월"]:
                        dur = form + n1.form
                        if current:
                            current.duration = dur
                            current.text += dur + " "
                        i += 1
                    elif i+2 < len(tokens) and n1.form == "개" and tokens[i+2].form == "월":
                        dur = form + "개월"
                        if current:
                            current.duration = dur
                            current.text += dur + " "
                        i += 2

            elif form in ["만", "만원", "백", "천"]:
                if current:
                    back = text[max(0, tk.start-12):tk.start]
                    m = re.search(
                        r"([일이삼사오육칠팔구]백[일이삼사오육칠팔구십]*|[일이삼사오육칠팔구]천[일이삼사오육칠팔구]?백?)$",
                        back
                    )
                    if m:
                        current.salary = m.group(1) + "만"
                        current.text += current.salary + " "

            win2 = text[tk.start:min(tk.start+12, len(text))]
            im = self.ISSUE_RX.search(win2)
            if im:
                if current and current.type == "career":
                    segments.append(current)
                current = Segment(type="issue", text=im.group(0), issue=im.group(0), start_pos=tk.start)

            i += 1

            if current and i == len(tokens):
                current.end_pos = tk.end

        if current:
            segments.append(current)
        return segments or self._segment_with_regex(text)

    def _segment_with_regex(self, text: str) -> List[Segment]:
        segs: List[Segment] = []
        career_pat = (
            r'(첫|두번째|세번째|마지막)?\s*(?:직장|회사|시즌)?[^,\.]*?'
            r'(\d+\s*개월|\d+년|일년반)[^,\.]*?'
            r'(\d+만|[가-힣]+백[가-힣]*만)?'
        )
        for m in re.finditer(career_pat, text):
            segs.append(Segment(
                type="career", text=m.group(0), ordinal=m.group(1),
                duration=m.group(2), salary=m.group(3),
                start_pos=m.start(), end_pos=m.end()
            ))
        issue_pat = r'(마지막\s*\d+개월)?\s*(체불|못\s*받|폐업|갑질|망했|부도|미지급)'
        for m in re.finditer(issue_pat, text):
            overlap = any(s.type=="career" and m.start()>=s.start_pos and m.end()<=s.end_pos for s in segs)
            if not overlap:
                segs.append(Segment(
                    type="issue", text=m.group(0), period=m.group(1), issue=m.group(2),
                    start_pos=m.start(), end_pos=m.end()
                ))
        return sorted(segs, key=lambda x: x.start_pos)

# -------------------------------
# 형태소 보강
# -------------------------------
class MorphBasedExtractor:
    def __init__(self):
        self.employment_keywords = {
            "정규직","계약직","프리랜서","특고","특수고용",
            "일용직","예술인","자영업","자영업자","개인사업",
            "알바","아르바이트","파트타임","시간제","일당","막노동"
        }

    def _filter(self, morphs: List[Dict]) -> List[Dict]:
        sel = []
        for i, m in enumerate(morphs):
            pos = m.get("pos",""); txt = m.get("text","")
            if pos in ["NNG","NNP","NNB","SN","NR","VV","VA","MM","MAG","XSN","XSV"]:
                sel.append({"index":i,"text":txt,"pos":pos})
        return sel

    def extract(self, morphs: List[Dict]) -> Dict[str, Any]:
        res: Dict[str, Any] = {}
        f = self._filter(morphs)
        for i, m in enumerate(f):
            t, p = m["text"], m["pos"]
            nx = f[i+1] if i+1 < len(f) else None
            if p in ["NR","SN"] and nx and nx["text"] in ["세","살"]:
                age = ko_word_to_int(t) if p=="NR" else (int(t) if t.isdigit() else None)
                if age and 15<=age<=100: res["age"] = age
            if p in ["NR","SN"] and nx:
                if nx["text"]=="년":
                    years = ko_word_to_int(t) if p=="NR" else (int(t) if t.isdigit() else None)
                    if years: res["eligible_months"] = res.get("eligible_months",0)+years*12
                elif nx["text"]=="개월" and p=="SN" and t.isdigit():
                    res["eligible_months"] = res.get("eligible_months",0)+int(t)
            if p in ["NR","SN"] and nx and nx["text"] in ["만","만원"]:
                if p=="NR":
                    v = ko_hundreds_phrase_to_int(t)
                    if v: res["monthly_salary"] = v*10_000
                elif t.isdigit():
                    res["monthly_salary"] = int(t)*10_000
            if t in self.employment_keywords:
                res.setdefault("employment_sequence", []).append((m["index"], t))
            if "장애" in t or "장해" in t:
                res["disability"] = True
        if "employment_sequence" in res:
            seq = sorted(res["employment_sequence"])
            res["employment_types"] = [t for _, t in seq]
            res["employment_type"] = seq[-1][1] if seq else None
            res.pop("employment_sequence")
        return res

# -------------------------------
# 정규식 코어 - 반복수급 개선
# -------------------------------
class PrecisionVariableExtractor:
    EMPLOYMENT_PATTERNS = {
        "정규직":"정규직","정직원":"정규직","무기계약":"정규직",
        "계약직":"계약직","기간제":"계약직","임시직":"계약직",
        "프리랜서":"프리랜서","프리":"프리랜서",
        "특고":"특고","특수고용":"특고",
        "일용직":"일용직","일당":"일용직","건설":"일용직","막노동":"일용직",
        "예술인":"예술인","작가":"예술인","배우":"예술인","가수":"예술인",
        "파트타임":"파트타임","아르바이트":"파트타임","알바":"파트타임","시간제":"파트타임",
    }

    RESIGNATION_CATEGORY = {
        "비자발적":[
            "권고사직","정리해고","구조조정","계약만료","계약 만료",
            "해고","짤렸","잘렸","쫓겨났","나가라",
            "회사폐업","폐업","부도","파산","망했","망함","문닫","문 닫",
            "영업종료","영업 종료","폐점","접었","접음","시즌 종료","시즌종료","촬영 종료","촬영종료",
            "계약 끝","끝나서","종료되","종료라",
        ],
        "정당한자발적":[
            "임금체불","월급못받","급여미지급","체불","떼먹","미지급",
            "괴롭힘","갑질","폭언","폭행","성희롱",
            "통근","왕복","편도",
            "질병","부상","우울증","공황장애","번아웃",
            "육아","간병","부모돌봄","가족간병",
        ],
        "자발적":[
            "이직","전직","개인사정","그냥그만","자진퇴사","커리어","연봉인상","더좋은곳",
        ],
    }

    SPECIAL_REASONS = {
        "권고사직":["권고사직","사직권고","나가달라"],
        "계약만료":["계약만료","계약종료","계약 만료","재계약거절","시즌 종료","촬영 종료","계약 끝","단기.*끝"],
        "회사폐업":["회사폐업","폐업","부도","파산","망했","망함","문닫","문 닫"],
        "구조조정":["구조조정","정리해고","감원","인원감축"],
        "임금체불":["임금체불","월급못받","급여미지급","체불","떼먹",r"노동청\s*신고"],
        "직장내괴롭힘":["괴롭힘","갑질","폭언","욕설","폭행","성희롱"],
        "통근곤란":[r"편도\s*[2-9]\s*시간",r"왕복\s*[4-9]\s*시간","통근못","너무멀"],
        "질병/부상":["아파서","병원","수술","우울증","공황장애","부상","산재"],
        "가족돌봄":["육아","출산","임신","간병","가족간병","돌봄"],
    }

    def _normalize(self, q: str) -> str:
        q = re.sub(r"\s+", " ", q.strip())
        repl = {
            "일년반":"18개월", "일 년 반":"18개월", "일년 반":"18개월",
            "반년":"6개월",
        }
        for k, v in repl.items():
            q = q.replace(k, v)
        return q

    def extract_all(self, text: str) -> Dict[str, Any]:
        q = self._normalize(text)
        return {
            "age": self._age(q),
            "disability": True if DISABILITY_RX.search(q) else None,
            "employment_type": self._employment_last(q),
            "resignation_category": self._resign_cat(q),
            "special_reason": self._special_reason(q),
            "eligible_months": self._months(q),
            "monthly_salary": self._salary(q),
            "repetition_count": self._repetition(q),  # 개선된 버전
        }

    def _age(self, q: str) -> Optional[int]:
        # 패턴 1: "23살", "23세", "만 23세"
        m = re.search(r"(?:만\s*)?(\d{2,3})\s*(?:세|살)\b", q)
        if m: return int(m.group(1))
        
        # 패턴 2: "20대 초반/중반/후반"
        m = re.search(r"([2-6]0)대\s*(초반|중반|후반)?", q)
        if m: return int(m.group(1)) + (_KO_TENS_WORD.get(m.group(2), 5) if m.group(2) else 5)
        
        # 패턴 3: "이십삼세"
        m = re.search(r"([일이삼사오육칠팔구]십)([일이삼사오육칠팔구])?\s*세", q)
        if m: return ko_word_to_int((m.group(1) or "") + (m.group(2) or ""))
        
        # 패턴 4: "스물셋", "서른하나"
        m = re.search(r"(스물|서른|마흔|쉰|예순|일흔|여든|아흔)\s*(한|하나|두|둘|세|셋|네|넷|다섯|여섯|일곱|여덟|아홉)?", q)
        if m: return ko_word_to_int((m.group(1) or "") + (m.group(2) or ""))
        
        return None

    def _employment_last(self, q: str) -> Optional[str]:
        hits = [(m.start(), self.EMPLOYMENT_PATTERNS.get(m.group(0), m.group(0)))
                for m in re.finditer(r"(정규직|정직원|계약직|프리랜서|프리|특고|특수고용|일용직|예술인|파트타임|아르바이트|알바|시간제)", q)]
        return hits[-1][1] if hits else None

    def _resign_cat(self, q: str) -> Optional[str]:
        if re.search(r"(체불|못\s*받|노동청|미지급)", q):
            return "정당한자발적"
        for cat, ws in self.RESIGNATION_CATEGORY.items():
            for w in ws:
                if re.search(w, q): return cat
        return None

    def _special_reason(self, q: str) -> Optional[str]:
        for reason, pats in self.SPECIAL_REASONS.items():
            for p in pats:
                if re.search(p, q): return reason
        return None

    def _months(self, q: str) -> Optional[int]:
        chebul_months = 0
        m = re.search(r"마지막\s*(\d+)\s*개월\s*체불", q)
        if m:
            chebul_months = int(m.group(1))

        m = re.search(r"총\s*(\d+)\s*개월", q)
        if m: return int(m.group(1))

        m = re.search(r"작년\s*(\d{1,2})월.*올해\s*(\d{1,2})월\s*(까지)?", q)
        if m:
            s, e = int(m.group(1)), int(m.group(2)); inc = 1 if m.group(3) else 0
            return (12 - s) + e + inc

        m = re.search(r"(\d{4})년\s*(\d{1,2})월\s*(?:부터|~)\s*(\d{4})년\s*(\d{1,2})월\s*(까지)?", q)
        if m:
            sy, sm, ey, em = map(int, m.groups()[:4]); inc = 1 if m.group(5) else 0
            return (ey - sy) * 12 + (em - sm) + inc

        processed: Set[Tuple[int, int]] = set()
        total = 0

        for m in re.finditer(r"([일이삼사오육칠팔구십]+)\s*년\b(?!\s*\d+\s*월)", q):
            if not any(ps <= m.start() <= pe for ps, pe in processed):
                y = ko_word_to_int(m.group(1))
                if y:
                    total += y * 12
                    processed.add((m.start(), m.end()))

        for m in re.finditer(r"(\d+)\s*년\s*(\d+)\s*개월", q):
            if (m.start(), m.end()) not in processed:
                total += int(m.group(1)) * 12 + int(m.group(2))
                processed.add((m.start(), m.end()))

        for m in re.finditer(r"(?<!\d)([1-9]\d?)\s*년\b(?!\s*\d+\s*월)", q):
            if not any(ps <= m.start() <= pe for ps, pe in processed):
                total += int(m.group(1)) * 12
                processed.add((m.start(), m.end()))

        for m in re.finditer(r"(\d+)\s*개월", q):
            if not any(ps <= m.start() <= pe for ps, pe in processed):
                if "체불" not in q[max(0, m.start()-20):m.end()+20]:
                    total += int(m.group(1))
                    processed.add((m.start(), m.end()))

        if chebul_months > 0 and total > chebul_months:
            total -= chebul_months
        return total or None

    def _salary(self, q: str) -> Optional[int]:
        # 시급 처리
        m = re.search(r"시급\s*(만|천)\s*원?", q)
        if m: return (10000 if m.group(1) == "만" else 1000) * 8 * 22

        m = re.search(r"시급\s*(\d+)(?:\s*(천|만))?\s*원?", q)
        if m:
            hourly = int(m.group(1)); unit = m.group(2)
            if unit == "만": hourly *= 10000
            elif unit == "천": hourly *= 1000
            elif hourly < 100: hourly *= 10000
            return hourly * 8 * 22

        # 연봉 처리
        m = re.search(r"연봉\s*([일이삼사오육칠팔구]천[일이삼사오육칠팔구]?백?)\s*만?\s*원?", q)
        if m:
            n = ko_compact_number_to_int(m.group(1)) or 0
            return (n * 10_000) // 12

        m = re.search(r"연봉\s*(\d{3,4})\s*(?:만원|만|)\b", q)
        if m: return (int(m.group(1)) * 10_000) // 12

        # 월급 처리
        cands: List[Tuple[int, int, int]] = []
        for m in re.finditer(r"([이삼사오육칠팔구]백[일이삼사오육칠팔구십]*)\s*만?\s*원?", q):
            v = ko_hundreds_phrase_to_int(m.group(1))
            if v: cands.append((m.start(), m.end(), v * 10_000))
        for m in re.finditer(r"월급\s*(\d{2,4})\s*만", q):
            cands.append((m.start(), m.end(), int(m.group(1)) * 10_000))
        for m in re.finditer(r"(\d{2,4})\s*만\s*원", q):
            cands.append((m.start(), m.end(), int(m.group(1)) * 10_000))
        for m in re.finditer(r"(\d{2,3})\s*만원대\s*(초반|중반|후반)?", q):
            base = int(m.group(1)) * 10_000
            if m.group(2):
                base += {"초반": 200_000, "중반": 500_000, "후반": 800_000}[m.group(2)]
            cands.append((m.start(), m.end(), base))
        if cands:
            s, e, val = max(cands, key=lambda x: x[0])
            around = q[max(0, s-25):e+25]
            if re.search(r"(조금|약간|좀)\s*넘(?:게)?", around):
                val += 100_000
            return val
        return None

    def _repetition(self, q: str) -> Optional[int]:
        """반복수급 횟수 추출 - 최댓값 선택 (GPT 권고 반영)"""
        counts = []
        
        # 패턴 1: "N번째 받으려/수급"
        for m in re.finditer(r"(첫|두|세|네|다섯|여섯)\s*번째\s*(받으려|수급하려|받을려|받|수급)", q):
            num = {"첫":1,"두":2,"세":3,"네":4,"다섯":5,"여섯":6}.get(m.group(1))
            if num:
                counts.append(num)
        
        # 패턴 2: 숫자 "3번째"
        for m in re.finditer(r"(\d)\s*번째\s*(받|수급)", q):
            counts.append(int(m.group(1)))
        
        # 패턴 3: 붙은 형태
        for word, num in {"첫번째":1,"두번째":2,"세번째":3,"네번째":4,"다섯번째":5,"여섯번째":6}.items():
            if word in q:
                counts.append(num)
        
        # 최댓값 반환
        return max(counts) if counts else None

# -------------------------------
# LLM 검증 클래스 (개선된 버전)
# -------------------------------
class LLMVerifier:
    """LLM 기반 변수 검증 및 보정"""
    
    def __init__(self, llm_client=None):
        self.llm = llm_client
        self.cache = {}
        self.enabled = True
        
        # LLM 클라이언트 초기화
        if not self.llm:
            try:
                from openai import OpenAI
                import sys
                from pathlib import Path
                sys.path.append(str(Path(__file__).parent.parent))
                import config
                
                self.llm = OpenAI(
                    base_url=config.API_BASE_URL,  # TOGETHER_API_KEY 사용
                    api_key=config.TOGETHER_API_KEY,
                )
                self.model = config.MODEL
                self.timeout = 5  # 5초 고정
                self.threshold = getattr(config, 'LLM_VERIFICATION_THRESHOLD', 0.5)
            except Exception as e:
                logger.error(f"LLM client initialization failed: {e}")
                self.enabled = False
    
    def verify_and_correct(self, query: str, extracted_vars: Dict, calc_result: Dict = None) -> Dict:
        """추출된 변수 검증 및 수정"""
        
        if not self.enabled:
            return extracted_vars
        
        # 1. LLM 검증이 필요한지 판단 (개선된 게이트)
        if not self._needs_verification(query, extracted_vars, calc_result):
            logger.info("LLM verification skipped - confidence high enough")
            return extracted_vars
        
        # 2. 캐시 확인
        cache_key = self._get_cache_key(query)
        if cache_key in self.cache:
            logger.info("Using cached LLM verification result")
            return self.cache[cache_key]
        
        # 3. LLM 프롬프트 생성
        prompt = self._build_prompt(query, extracted_vars, calc_result)
        
        # 4. LLM 호출 (재시도 금지)
        try:
            start_time = time.time()
            
            messages = [
                {"role": "system", "content": "실업급여 변수 검증 전문가. JSON만 출력."},
                {"role": "user", "content": prompt}
            ]
            
            completion = self.llm.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.2,
                max_tokens=400,
                timeout=self.timeout
            )
            
            response = completion.choices[0].message.content
            elapsed = time.time() - start_time
            logger.info(f"LLM verification completed in {elapsed:.2f}s")
            
            # 5. 응답 파싱 및 검증
            corrected = self._parse_response(response)
            
            # 6. 기존 변수와 병합
            final_vars = self._merge_and_validate(extracted_vars, corrected)
            
            # 7. 캐시 저장
            self.cache[cache_key] = final_vars
            
            return final_vars
            
        except Exception as e:
            logger.error(f"LLM verification failed: {e}")
            return extracted_vars
    
    def _needs_verification(self, query: str, vars: Dict, calc: Dict = None) -> bool:
        """LLM 검증 필요 여부 판단 (청년/장애 특례 반영)"""
        
        # 나이와 장애 여부 확인
        age = vars.get("age", 25)
        disability = vars.get("disability", False)
        is_youth = 18 <= age <= 34
        
        # 최소 개월수 계산
        min_months = 3 if (is_youth or disability) else 6
        
        # 급여가 0원이고 금액 표식이 있을 때만
        if not vars.get("monthly_salary") or vars.get("monthly_salary", 0) == 0:
            if any(word in query for word in ["만원", "백만원", "천만원", "만 원"]):
                logger.info("LLM verification needed: salary is 0 but amount markers exist")
                return True
        
        # 기간이 최소 개월수 미만
        months = vars.get("eligible_months")
        if months is None or months < min_months:
            logger.info(f"LLM verification needed: months {months} < min {min_months}")
            return True
        
        # 신뢰도가 낮음
        confidence = vars.get("confidence", {})
        if isinstance(confidence, dict):
            overall = confidence.get("overall", 0)
            if overall < self.threshold:
                logger.info(f"LLM verification needed: low confidence {overall}")
                return True
        
        # 퇴사 사유 불명
        if not vars.get("resignation_category"):
            logger.info("LLM verification needed: resignation category unknown")
            return True
        
        # 반복수급 언급되었는데 카운트 없음
        if any(word in query for word in ["반복", "번째", "수급"]) and not vars.get("repetition_count"):
            logger.info("LLM verification needed: repetition mentioned but no count")
            return True
        
        # 특수 사유가 있는 경우
        if vars.get("special_reason") in ["임금체불", "직장내괴롭힘", "질병/부상"]:
            logger.info("LLM verification needed: special reason exists")
            return True
        
        return False
    
    def _build_prompt(self, query: str, vars: Dict, calc: Dict = None) -> str:
        """간소화된 프롬프트 생성"""
        
        prompt = f"""실업급여 변수 검증. JSON만 출력.

[원본] {query}

[현재값]
- 나이: {vars.get('age')}
- 급여: {vars.get('monthly_salary', 0)}원
- 기간: {vars.get('eligible_months', 0)}개월
- 퇴사: {vars.get('resignation_category')}
- 특별사유: {vars.get('special_reason')}
- 반복횟수: {vars.get('repetition_count')}

[지시사항]
1. 급여 0원이면 원문에서 "만원", "백만원" 등 찾기
2. "이십일년" → 252개월, "일년반" → 18개월
3. 체불/폐업 → 정당한자발적/비자발적
4. 청년(18-34세)과 장애인은 3개월도 가능

{{
  "age": 숫자 또는 null,
  "monthly_salary": 숫자 또는 null,
  "eligible_months": 숫자 또는 null,
  "resignation_category": "비자발적"|"정당한자발적"|"자발적"|null,
  "special_reason": 문자열 또는 null,
  "repetition_count": 숫자 또는 null,
  "confidence": 0.0-1.0
}}"""
        
        return prompt
    
    def _parse_response(self, response: str) -> Dict:
        """JSON 응답 파싱 및 검증"""
        try:
            # JSON 추출
            json_match = re.search(r'\{[^}]+\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                data = json.loads(json_str)
            else:
                data = json.loads(response)
            
            # 스키마 검증
            validated = {}
            
            # 나이 (15 ~ 100)
            if "age" in data and data["age"] is not None:
                age = int(data["age"])
                if 15 <= age <= 100:
                    validated["age"] = age
            
            # 월급 (0 ~ 10,000,000)
            if "monthly_salary" in data and data["monthly_salary"] is not None:
                salary = int(data["monthly_salary"])
                if 0 <= salary <= 10_000_000:
                    validated["monthly_salary"] = salary
            
            # 개월수 (0 ~ 600)
            if "eligible_months" in data and data["eligible_months"] is not None:
                months = int(data["eligible_months"])
                if 0 <= months <= 600:
                    validated["eligible_months"] = months
            
            # 퇴사 카테고리
            if "resignation_category" in data:
                if data["resignation_category"] in ["비자발적", "정당한자발적", "자발적"]:
                    validated["resignation_category"] = data["resignation_category"]
            
            # 특별 사유
            if "special_reason" in data and data["special_reason"]:
                validated["special_reason"] = str(data["special_reason"])
            
            # 반복 횟수
            if "repetition_count" in data and data["repetition_count"] is not None:
                count = int(data["repetition_count"])
                if 1 <= count <= 10:
                    validated["repetition_count"] = count
            
            # 신뢰도
            if "confidence" in data:
                validated["llm_confidence"] = float(data.get("confidence", 0.5))
            
            return validated
            
        except Exception as e:
            logger.error(f"Failed to parse LLM response: {e}")
            return {}
    
    def _merge_and_validate(self, original: Dict, corrected: Dict) -> Dict:
        """기존 변수와 LLM 수정값 병합"""
        
        # 원본 복사
        result = original.copy()
        
        # LLM 수정값 적용
        for key, value in corrected.items():
            if key == "llm_confidence":
                # LLM 신뢰도는 별도 저장
                result["llm_confidence"] = value
            elif value is not None:
                # 원본과 크게 다르면 로깅
                if key in original and original[key]:
                    orig_val = original[key]
                    if isinstance(orig_val, (int, float)) and isinstance(value, (int, float)):
                        if abs(orig_val - value) / max(orig_val, value, 1) > 0.5:
                            logger.info(f"Large change in {key}: {orig_val} → {value}")
                
                result[key] = value
                result[f"{key}_source"] = "llm"  # 출처 표시
        
        # LLM 검증 플래그
        result["llm_verified"] = True
        
        # 전체 신뢰도 재계산
        if "confidence" not in result:
            result["confidence"] = {}
        
        if "llm_confidence" in result:
            result["confidence"]["overall"] = min(
                result["confidence"].get("overall", 0.5) * 1.2,
                0.95
            )
        
        return result
    
    def _get_cache_key(self, query: str) -> str:
        """캐시 키 생성"""
        import hashlib
        return hashlib.md5(query.encode()).hexdigest()[:16]

# -------------------------------
# 메인 통합 로직 (수정 버전)
# -------------------------------
class UnemploymentLogic:
    DAILY_MAX = 66_000
    DAILY_MIN = 64_192
    YOUTH = (18, 34)  # 청년 범위 18-34세
    REP_PENALTY = {1:1.0, 2:1.0, 3:0.9, 4:0.75, 5:0.6, 6:0.5}  # 반복수급 감액률

    AMBIGUOUS = {
        "얼마전": (3, 3, 0.5), "최근": (3, 3, 0.6), "꽤오래": (18, 18, 0.5),
        "오래전": (36, 36, 0.4), "작년쯤": (15, 15, 0.7),
        "적당히": (2_500_000, 2_500_000, 0.5),
        "괜찮게": (3_500_000, 3_500_000, 0.6),
        "많이": (5_000_000, 5_000_000, 0.5),
        "조금": (2_000_000, 2_000_000, 0.5),
    }

    def __init__(self, llm_client=None):
        self.pve = PrecisionVariableExtractor()
        self.morph = MorphBasedExtractor()
        self.segmenter = KiwiSegmenter()
        self.llm_verifier = LLMVerifier(llm_client)

    def extract_variables_with_llm(self, query: str, query_info: Optional[Dict]=None) -> Dict[str, Any]:
        """변수 추출 + LLM 검증 (통합 파이프라인)"""
        text = self.pve._normalize(query)
        
        # 1. 기존 추출 로직
        if self._is_complex_case(text):
            seg_res = self._extract_with_segments(text)
            if seg_res and seg_res.get("confidence", {}).get("overall", 0) > 0.7:
                extracted = seg_res
            else:
                extracted = self._extract_traditional(text, query_info)
        else:
            extracted = self._extract_traditional(text, query_info)
        
        # 2. LLM 검증 단계 추가 (config에서 제어)
        try:
            import sys
            from pathlib import Path
            sys.path.append(str(Path(__file__).parent.parent))
            import config
            
            if getattr(config, 'LLM_VERIFICATION_ENABLED', False):
                verified = self.llm_verifier.verify_and_correct(query, extracted)
                return verified
        except Exception as e:
            logger.warning(f"LLM verification not available: {e}")
        
        return extracted

    def calculate_benefit_days(self, age: int, months: int, disability: bool = False) -> int:
        """수급 일수 계산 (청년/장애 특례 포함)"""
        # None 방어
        age = int(age or 25)  # 기본값 25세 (청년 특례 가능)
        months = int(months or 0)
        
        if disability and age >= 50:
            return 270
        if age < 30:
            table = [(0,12,120),(12,36,150),(36,60,180),(60,120,210),(120,999,240)]
        elif age < 50:
            table = [(0,12,120),(12,36,180),(36,60,210),(60,120,240),(120,999,270)]
        else:
            table = [(0,12,180),(12,36,210),(36,60,240),(60,120,270),(120,999,270)]
        days = 120
        for a,b,d in table:
            if a <= months < b:
                days = d; break
        if months >= 240:
            days += 30
        return days

    def calculate_daily_amount(self, monthly_salary: int, age: int) -> Dict[str, Any]:
        """일 급여액 계산 (청년 가산 포함)"""
        # None 방어
        monthly_salary = int(monthly_salary or 0)
        age = int(age or 25)
        
        if not monthly_salary:
            return {"daily_base": 0, "daily_benefit": 0, "applied": "계산불가"}
        base = monthly_salary / 30
        rate = 0.6 * (1.1 if self.YOUTH[0] <= age <= self.YOUTH[1] else 1.0)
        val = round(base * rate)
        if val > self.DAILY_MAX:
            return {"daily_base": round(base), "daily_benefit": self.DAILY_MAX, "applied": "상한액"}
        if val < self.DAILY_MIN:
            return {"daily_base": round(base), "daily_benefit": self.DAILY_MIN, "applied": "하한액"}
        return {"daily_base": round(base), "daily_benefit": val,
                "applied": "60%" + (" + 청년가산" if self.YOUTH[0] <= age <= self.YOUTH[1] else "")}

    def calculate_total_benefit(self, variables: Dict[str, Any]) -> Dict[str, Any]:
        """실업급여 계산 (None 방어 포함)"""
        # None 방어 코드
        age = int(variables.get("age") or 25)
        salary = int(variables.get("monthly_salary") or 0)
        months = int(variables.get("eligible_months") or 0)
        resignation = variables.get("resignation_category", "")
        repetition = int(variables.get("repetition_count") or 1)
        disability = bool(variables.get("disability", False))

        # 청년/장애 특례 체크
        is_youth = self.YOUTH[0] <= age <= self.YOUTH[1]
        min_months = 3 if (is_youth or disability) else 6
        
        if months < min_months:
            return {"eligible": False, "reason": f"고용보험 {months}개월 < 최소 {min_months}개월", 
                   "is_youth": is_youth, "disability": disability}
        
        if resignation == "자발적":
            return {"eligible": False, "reason": "단순 자발적 퇴사는 수급 불가"}

        benefit_days = self.calculate_benefit_days(age, months, disability)
        reduction_info = None
        if repetition and repetition >= 3:
            penalty = self.REP_PENALTY.get(repetition, 0.5)
            benefit_days = int(round(benefit_days * penalty))
            reduction_info = f"{repetition}회차 반복수급으로 {int((1-penalty)*100)}% 감액"

        daily = self.calculate_daily_amount(salary, age)
        total = daily["daily_benefit"] * benefit_days
        return {
            "eligible": True,
            "age": age,
            "monthly_salary": salary,
            "eligible_months": months,
            "resignation_type": resignation,
            "resignation_reason": variables.get("special_reason"),
            "is_youth": is_youth,
            "disability": disability,
            "daily_benefit": daily["daily_benefit"],
            "applied_limit": daily["applied"],
            "benefit_days": benefit_days,
            "total_amount": total,
            "reduction_info": reduction_info,
            "employment_type": variables.get("employment_type"),
            "employment_types": variables.get("employment_types", []),
            "confidence": variables.get("confidence", {}),
            "llm_verified": variables.get("llm_verified", False)
        }

    def format_calculation_result(self, result: Dict[str, Any]) -> str:
        """결과 포맷팅 (None 방어 포함)"""
        if not result.get("eligible"):
            return f"⛔ 수급 불가: {result.get('reason')}"
        
        # None 방어 포맷팅
        lines = [
            "✅ **실업급여 계산 완료** (2025년 기준)",
            "",
            f"📋 **기본 정보**",
            f"- 나이: {result.get('age', 0)}세" + (" (청년 특례)" if result.get('is_youth') else "") + (" (장애 특례)" if result.get('disability') else ""),
            f"- 가입 기간: {result.get('eligible_months', 0)}개월",
            f"- 월급: {(result.get('monthly_salary') or 0):,}원",
            "",
            f"💰 **수급 내역**",
            f"- 일 급여액: {(result.get('daily_benefit') or 0):,}원 ({result.get('applied_limit', '')})",
            f"- 수급 기간: {result.get('benefit_days', 0)}일",
            f"- **총 수급액: {(result.get('total_amount') or 0):,}원**",
        ]
        
        if result.get("reduction_info"):
            lines.append("")
            lines.append(f"⚠️ {result['reduction_info']}")
        
        if result.get("llm_verified"):
            lines.append("")
            lines.append("✔ LLM 검증 완료")
        
        confidence = result.get("confidence", {})
        if isinstance(confidence, dict) and confidence.get("overall"):
            lines.append("")
            lines.append(f"🎯 신뢰도: {confidence['overall']*100:.0f}%")
        
        return "\n".join(lines)

    # 내부 메서드들은 동일하게 유지
    def _is_complex_case(self, text: str) -> bool:
        signals = [
            ("첫" in text and ("두번째" in text or "세번째" in text)),
            ("체불" in text),
            (text.count("개월") >= 2 or text.count("년") >= 2),
            (text.count("만원") >= 2 or text.count("만") >= 3),
            ("프리랜서" in text and "정규직" in text),
            bool(re.search(r"[일이삼사오육칠팔구십]+년", text)),
        ]
        return sum(bool(s) for s in signals) >= 2

    def _extract_with_segments(self, text: str) -> Optional[Dict[str, Any]]:
        try:
            segments = self.segmenter.segment(text)
            if not segments:
                return None
            careers: List[Dict[str, Any]] = []
            issues: List[Dict[str, Any]] = []

            for seg in segments:
                if seg.type == "career":
                    months = self.pve._months(seg.text) if seg.text else None
                    if not months and seg.duration:
                        if seg.duration == "일년반" or seg.duration == "18개월":
                            months = 18
                        elif "년" in seg.duration:
                            m = re.search(r"(\d+)\s*년", seg.duration)
                            if m:
                                months = int(m.group(1)) * 12
                            else:
                                m = re.search(r"([일이삼사오육칠팔구십]+)\s*년", seg.duration)
                                if m:
                                    y = ko_word_to_int(m.group(1))
                                    if y: months = y * 12
                        elif "개월" in seg.duration:
                            m = re.search(r"(\d+)\s*개월", seg.duration)
                            if m: months = int(m.group(1))
                    salary = self.pve._salary(seg.text) if seg.text else None
                    if not salary and seg.salary:
                        clean = re.sub(r"(만|원)$", "", seg.salary or "")
                        if "백" in clean:
                            val = ko_hundreds_phrase_to_int(clean)
                            if val: salary = val * 10000
                        elif clean.isdigit():
                            salary = int(clean) * 10000
                    if months or salary:
                        careers.append({"ordinal": seg.ordinal, "months": months or 0, "salary": salary or 0, "text": seg.text})
                elif seg.type == "issue":
                    issues.append({"type": seg.issue, "period": seg.period, "text": seg.text})

            result = self._analyze_segment_relations(careers, issues, text)
            if result:
                result["used_segmentation"] = True
                result["segments"] = len(segments)
                return result
        except Exception as e:
            logger.error(f"Segmentation failed: {e}")
        return None

    def _analyze_segment_relations(self, careers: List[Dict], issues: List[Dict], full_text: str) -> Dict[str, Any]:
        if not careers:
            return None
        total_months = sum(c["months"] for c in careers)
        for issue in issues:
            if re.search(r"(체불|못\s*받|미지급)", issue.get("type") or ""):
                if issue["period"] and isinstance(issue["period"], str):
                    m = re.search(r"(\d+)\s*개월", issue["period"])
                    if m:
                        chebul = int(m.group(1))
                        total_months = max(0, total_months - chebul)

        valid_salaries = [c["salary"] for c in careers if c["salary"] > 0]
        monthly_salary = valid_salaries[-1] if valid_salaries else 0

        p = self.pve.extract_all(full_text)
        if not monthly_salary and p.get("monthly_salary"):
            monthly_salary = p["monthly_salary"]

        resignation_category, special_reason = None, None
        for issue in issues:
            it = issue["type"] or ""
            if re.search(r"(체불|못\s*받|미지급)", it):
                resignation_category, special_reason = "정당한자발적", "임금체불"; break
            if re.search(r"(폐업|부도|파산|망했|망함)", it):
                resignation_category, special_reason = "비자발적", "회사폐업"; break
            if re.search(r"(갑질|괴롭힘)", it):
                resignation_category, special_reason = "정당한자발적", "직장내괴롭힘"; break

        result = {
            "age": p.get("age", 25),
            "eligible_months": total_months,
            "monthly_salary": monthly_salary,
            "resignation_category": resignation_category or p.get("resignation_category"),
            "special_reason": special_reason or p.get("special_reason"),
            "repetition_count": p.get("repetition_count"),
            "disability": p.get("disability", False),
            "employment_type": p.get("employment_type"),
            "career_history": careers,
            "issues": issues,
        }
        result["confidence"] = self._compute_confidence(result, used_segmentation=True)
        return result

    def _extract_traditional(self, text: str, query_info: Optional[Dict]) -> Dict[str, Any]:
        p = self.pve.extract_all(text)
        mres: Dict[str, Any] = {}
        if query_info and "morphs" in query_info and isinstance(query_info["morphs"], list):
            mres = self.morph.extract(query_info["morphs"])
        career_info = self._parse_career_history(text)

        if career_info:
            age = p.get("age") if p.get("age") is not None else mres.get("age")
            months = career_info.get("eligible_months", p.get("eligible_months"))
            salary = career_info.get("monthly_salary", p.get("monthly_salary"))
        else:
            age = p.get("age") if p.get("age") is not None else mres.get("age")
            months = p.get("eligible_months") if p.get("eligible_months") is not None else mres.get("eligible_months")
            salary = p.get("monthly_salary") if p.get("monthly_salary") is not None else mres.get("monthly_salary")

        repetition = p.get("repetition_count") if p.get("repetition_count") is not None else mres.get("repetition_count")
        disability = p.get("disability") if p.get("disability") is not None else mres.get("disability")
        employment_type = p.get("employment_type") or mres.get("employment_type")

        emp_types: List[str] = []
        if p.get("employment_type"): emp_types.append(p["employment_type"])
        emp_types += p.get("employment_types", [])
        emp_types += mres.get("employment_types", [])
        employment_types = list(dict.fromkeys([t for t in emp_types if t]))

        ambiguous = {}
        for k, (v, _, conf) in self.AMBIGUOUS.items():
            if k in text:
                if k in ["적당히","괜찮게","많이","조금"] and (salary is None or salary == 0):
                    salary = v
                    ambiguous["monthly_salary"] = {"rule": k, "value": v, "confidence": conf}
                if k in ["얼마전","최근","꽤오래","오래전","작년쯤"] and (months is None or months == 0):
                    months = v
                    ambiguous["eligible_months"] = {"rule": k, "value": v, "confidence": conf}

        months = months or 0
        out: Dict[str, Any] = {
            "age": age if age is not None else 25,
            "monthly_salary": int(salary) if salary is not None else 0,
            "eligible_months": months,
            "resignation_category": p.get("resignation_category"),
            "special_reason": p.get("special_reason"),
            "repetition_count": repetition,
            "disability": bool(disability) if disability is not None else False,
        }
        if employment_type:
            out["employment_type"] = employment_type
        if employment_types:
            out["employment_types"] = employment_types
        if ambiguous:
            out["ambiguous_fields"] = ambiguous
        if career_info:
            out["career_history"] = career_info.get("careers", [])
        out["confidence"] = self._compute_confidence(out, used_segmentation=False)
        return self._validate(out)

    def _parse_career_history(self, text: str) -> Optional[Dict]:
        careers: List[Dict[str, Any]] = []
        text_norm = text.replace("일년반", "18개월")

        chebul_months = 0
        m = re.search(r"마지막\s*(\d+)\s*개월\s*체불", text)
        if m:
            chebul_months = int(m.group(1))

        patterns = [
            r"(?P<label>첫|두번째|세번째|마지막)?\s*(?:직장|회사)?\s*(?P<months>\d+)\s*개월\s*(?P<sal_ko>[이삼사오육칠팔구]백[일이삼사오육칠팔구십]*)?(?P<sal_num>\d+)?\s*만",
            r"(?P<emp>정규직|계약직|프리랜서|특고|예술인)(?:로|으로)?\s*(?P<years>\d+)\s*년",
            r"(?P<emp2>정규직|계약직|프리랜서|특고|예술인)(?:로|으로)?\s*(?P<months2>\d+)\s*개월",
        ]
        for pat in patterns:
            for m in re.finditer(pat, text_norm):
                if m.groupdict().get('years'):
                    months = int(m.group('years')) * 12
                elif m.groupdict().get('months'):
                    months = int(m.group('months'))
                elif m.groupdict().get('months2'):
                    months = int(m.group('months2'))
                else:
                    continue

                sal_ko = m.groupdict().get('sal_ko') or ''
                sal_num = m.groupdict().get('sal_num') or '0'
                if sal_ko:
                    salary = (ko_hundreds_phrase_to_int(sal_ko) or 0) * 10_000
                elif sal_num and sal_num.isdigit():
                    salary = int(sal_num) * 10_000
                else:
                    salary = 0

                if months > 0:
                    careers.append({
                        "months": months,
                        "salary": salary,
                        "offset": m.start(),
                        "employment_type": m.groupdict().get('emp') or m.groupdict().get('emp2')
                    })

        if not careers:
            return None

        last = max(careers, key=lambda x: x["offset"])
        total = sum(c["months"] for c in careers)
        if chebul_months > 0:
            total = max(0, total - chebul_months)

        return {
            "monthly_salary": last["salary"] if last["salary"] > 0 else None,
            "eligible_months": total,
            "careers": careers,
            "chebul_months": chebul_months
        }

    def _compute_confidence(self, result: Dict, used_segmentation: bool) -> Dict:
        conf = {"overall": 0.0, "age": 0.0, "salary": 0.0, "months": 0.0, "resignation": 0.0}
        if result.get("age") and result["age"] != 25:  # 25로 변경
            conf["age"] = 0.9
        if result.get("monthly_salary") and result["monthly_salary"] > 0:
            conf["salary"] = 0.9 if 1_500_000 <= result["monthly_salary"] <= 10_000_000 else 0.6
        if result.get("eligible_months") is not None:
            conf["months"] = 0.9 if 3 <= int(result["eligible_months"]) <= 360 else 0.5
        if result.get("resignation_category"): conf["resignation"] = 0.85
        weights = {"age": 0.25, "salary": 0.25, "months": 0.30, "resignation": 0.20}
        overall = sum(conf[k] * weights.get(k, 0) for k in conf if k != "overall")
        if result.get("ambiguous_fields"): overall *= 0.8
        if used_segmentation: overall *= 1.1
        conf["overall"] = min(overall, 0.95)
        return conf

    def _validate(self, r: Dict[str, Any]) -> Dict:
        if r.get("age") and not (15 <= r["age"] <= 100):
            r["age"] = 25  # 25로 변경
        salary = int(r.get("monthly_salary") or 0)
        if salary < 0: salary = 0
        r["monthly_salary"] = salary
        if r.get("eligible_months") is not None and r["eligible_months"] < 0:
            r["eligible_months"] = None
        return r

    def build_response(self, variables: Dict[str, Any]) -> Tuple[Dict[str, Any], str]:
        benefit = self.calculate_total_benefit(variables)

        if not benefit.get("eligible"):
            reason = benefit.get("reason", "사유 불명")
            txt = f"수급 불가. 사유: {reason}"
            return benefit, txt

        age = benefit["age"]
        months = benefit["eligible_months"]
        daily = benefit["daily_benefit"]
        days = benefit["benefit_days"]
        total = benefit["total_amount"]
        applied = benefit["applied_limit"]
        resign = benefit.get("resignation_type") or "불명"
        reason = benefit.get("resignation_reason")
        rep = benefit.get("reduction_info")

        head = f"{reason+'으로 인한 ' if reason else ''}{resign} 퇴사로 실업급여 수급 가능."
        core = f"{age}세 {months}개월 가입 {days}일, 일 급여 {daily:,}원({applied}), 총 {total:,}원"
        tail = f" ({rep})" if rep else ""
        txt = f"{head} {core}{tail}"
        return benefit, txt

    def respond(self, query: str, query_info: Optional[Dict]=None) -> Dict[str, Any]:
        vars_ = self.extract_variables_with_llm(query, query_info)
        benefit, text = self.build_response(vars_)
        out = {
            "extracted": vars_,
            "benefit": benefit,
            "answer": text,
        }
        return out


# 모듈 단독 실행 테스트용 경량 CLI
if __name__ == "__main__":
    import json
    import sys

    logic = UnemploymentLogic()
    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "스물셋 알바생인데 3개월 단기 계약 끝나서 나왔어요. 시급 만원으로 하루 8시간 주5일 일했어요."
    res = logic.respond(query)
    print(json.dumps(res, ensure_ascii=False, indent=2))

# 싱글톤 인스턴스
unemployment_logic = UnemploymentLogic()
