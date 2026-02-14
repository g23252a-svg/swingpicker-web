import os
import json
import re
import time
import logging
from datetime import datetime, timedelta
import google.generativeai as genai

# 로깅 설정
logger = logging.getLogger("DartAnalyzer")

try:
    import OpenDartReader
    DART_OK = True
except ImportError:
    DART_OK = False
    logger.error("⚠️ OpenDartReader 미설치")

class DartAnalyzer:
    def __init__(self, dart_api_key=None, gemini_api_key=None):
        self.dart_api_key = dart_api_key or os.environ.get("DART_API_KEY")
        self.gemini_api_key = gemini_api_key or os.environ.get("GEMINI_API_KEY")
        self.dart = None
        self.model = None

        if DART_OK and self.dart_api_key:
            try:
                self.dart = OpenDartReader(self.dart_api_key)
            except Exception as e: logger.error(f"DART 초기화 실패: {e}")

        if self.gemini_api_key:
            try:
                genai.configure(api_key=self.gemini_api_key)
                # [보강] 최신 모델 설정 및 JSON 출력 강제
                self.model = genai.GenerativeModel(
                    model_name='gemini-1.5-flash',
                    generation_config={"response_mime_type": "application/json"}
                )
            except Exception as e: logger.error(f"Gemini 초기화 실패: {e}")

    def _extract_key_facts(self, text: str) -> str:
        """[채찍 3 반영] 텍스트 정제 전 핵심 수치(금액, 비중 등)만 별도 추출"""
        facts = []
        # 매출액 대비 비중 (%)
        pcts = re.findall(r'(\d+(?:\.\d+)?\s*%)', text)
        if pcts: facts.append(f"비중: {', '.join(pcts[:3])}")
        
        # 금액 단위 (억원/백만원)
        money = re.findall(r'(\d+(?:,\d+)?\s*(?:억원|백만원))', text)
        if money: facts.append(f"금액: {', '.join(money[:3])}")
        
        # 기간/날짜 관련
        dates = re.findall(r'(\d{4}-\d{2}-\d{2})', text)
        if dates: facts.append(f"주요날짜: {', '.join(dates[:2])}")
        
        return " | ".join(facts)

    def _clean_text(self, text: str) -> str:
        """[채찍 1] XML/HTML 태그 제거 및 공백 압축"""
        if not text: return ""
        clean = re.sub(r'<[^>]*>', ' ', text)
        clean = re.sub(r'\s+', ' ', clean).strip()
        return clean[:12000]

    def get_major_disclosures(self, code, days=3):
        if not self.dart: return []
        end_d = datetime.now().strftime("%Y%m%d")
        start_d = (datetime.now() - timedelta(days=days)).strftime("%Y%m%d")
        
        try:
            df = self.dart.list(code, start=start_d, end=end_d, kind='I')
            if df is None or df.empty: return []
            
            # [채측 1 반영] str.contains() 결측치 방어(na=False)
            keywords = [
                '공급계약', '수주', '유상증자', '무상증자', '전환사채', '신주인수권', 
                '교환사채', '자사주', '취득', '처분', '최대주주', '변경'
            ]
            mask = df['report_nm'].str.contains('|'.join(keywords), na=False)
            targets = df[mask].copy()
            
            return targets[['rcept_no', 'report_nm', 'rcept_dt']].to_dict('records')
        except Exception as e:
            logger.error(f"공시 목록 조회 실패 ({code}): {e}")
            return []

    def analyze_report(self, rcept_no, report_nm):
        """[채측 2,4 반영] 팩트 기반 정밀 분석 및 재시도 로직"""
        if not self.dart or not self.model: return 0.0, ""

        # [채찍 4 반영] 네트워크/서버 재시도 로직 (최대 2회)
        for attempt in range(2):
            try:
                # 1. 공시 원문 획득
                raw_xml = self.dart.document(rcept_no)
                if not raw_xml: continue
                
                # 2. 핵심 팩트 선추출 후 텍스트 정제
                key_facts = self._extract_key_facts(raw_xml)
                content = self._clean_text(raw_xml)

                # 3. 프롬프트 강화 (수치 데이터 강조)
                prompt = f"""
                당신은 대한민국 금융감독원 공시 전문 분석관입니다. 
                제공된 [핵심 수치]를 바탕으로 [본문]의 맥락을 분석하여 주가 영향력을 평가하세요.

                [제목] {report_nm}
                [핵심 수치] {key_facts}
                [본문 내용 요약] {content}

                [평가 가이드라인 (스케일: -10 ~ +10)]
                - (+8~10): 무상증자, 매출액 30% 이상의 대규모 공급계약, 경영권 분쟁 없는 최대주주 매수.
                - (+3~7): 시설투자용 3자배정 증자, 매출액 10% 이상 계약, 자사주 소각.
                - (0): 단순 정정, 통상적인 분기보고서.
                - (-3~7): 운영자금/채무상환용 유상증자, 전환사채(CB) 대량 발행, 공급계약 해지.
                - (-8~10): 횡령/배임, 회계처리 위반, 최대주주의 대량 지분 매도.

                반드시 JSON 형식으로만 응답하십시오.
                {{"score": 0.0, "reason": "이유 요약"}}
                """

                response = self.model.generate_content(prompt)
                res_text = response.text.strip()
                
                # [채찍 2 반영] Non-greedy JSON 추출 및 유효성 검사
                json_match = re.search(r'\{.*?\}', res_text, re.DOTALL)
                if json_match:
                    data = json.loads(json_match.group())
                    if "score" in data and "reason" in data:
                        # 점수 최종 보정 및 클램프
                        final_score = max(-10.0, min(10.0, float(data["score"])))
                        return final_score, data["reason"]
                
                logger.warning(f"⚠️ 형식 오류 재시도 중... ({report_nm})")
                
            except Exception as e:
                logger.error(f"❌ 분석 시도 {attempt+1} 실패 ({report_nm}): {e}")
                time.sleep(1) # 지연 후 재시도
        
        return 0.0, "분석 불가(서버 응답 오류)"
