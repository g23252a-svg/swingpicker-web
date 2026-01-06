# dart_analyzer.py (v1.0)
import os
import json
from datetime import datetime, timedelta
import google.generativeai as genai

# 라이브러리 설치 확인
try:
    import OpenDartReader
    DART_OK = True
except ImportError:
    DART_OK = False
    print("⚠️ OpenDartReader 미설치: pip install opendartreader xmltodict")

class DartAnalyzer:
    def __init__(self, dart_api_key=None, gemini_api_key=None):
        # API 키 로드 (인자 우선 -> 환경변수 순)
        self.dart_api_key = dart_api_key or os.environ.get("DART_API_KEY")
        self.gemini_api_key = gemini_api_key or os.environ.get("GEMINI_API_KEY")
        
        self.dart = None
        self.model = None

        # DART 연결
        if DART_OK and self.dart_api_key:
            try:
                self.dart = OpenDartReader(self.dart_api_key)
                # print(f"✅ [DART] 초기화 완료 (Key: {self.dart_api_key[:4]}***)")
            except Exception as e:
                print(f"⚠️ [DART] 초기화 실패: {e}")

        # Gemini 연결
        if self.gemini_api_key:
            try:
                genai.configure(api_key=self.gemini_api_key)
                self.model = genai.GenerativeModel('gemini-pro')
            except Exception as e:
                print(f"⚠️ [DART] LLM 초기화 실패: {e}")

    def get_major_disclosures(self, code, days=3):
        """최근 N일간의 주요 공시(수시공시) 목록 조회"""
        if not self.dart: return []
        
        end_d = datetime.now().strftime("%Y%m%d")
        start_d = (datetime.now() - timedelta(days=days)).strftime("%Y%m%d")
        
        try:
            # 수시공시(I) 조회
            df = self.dart.list(code, start=start_d, end=end_d, kind='I')
            if df is None or df.empty:
                return []
            
            # 분석할만한 키워드 필터링
            keywords = ['공급계약', '유상증자', '전환사채', '무상증자', '신주인수권', '교환사채', '수주']
            mask = df['report_nm'].str.contains('|'.join(keywords))
            targets = df[mask].copy()
            
            results = []
            for _, row in targets.iterrows():
                results.append({
                    'rcept_no': row['rcept_no'], # 접수번호 (원문 조회용)
                    'report_nm': row['report_nm'],
                    'rcept_dt': row['rcept_dt']
                })
            return results
        except Exception as e:
            return []

    def analyze_report(self, rcept_no, report_nm):
        """공시 원문을 가져와 LLM 분석 수행"""
        if not self.dart or not self.model:
            return 0.0, ""

        try:
            # 원문 텍스트 추출 (XML 파싱은 OpenDartReader가 처리)
            xml_text = self.dart.document(rcept_no)
            if not xml_text: return 0.0, "본문없음"
            
            # 토큰 제한 고려하여 앞부분(핵심)만 추출
            content_snippet = xml_text[:10000]

            prompt = f"""
            당신은 주식 공시 분석 전문가입니다. 
            아래 공시가 주가에 미칠 영향을 분석하여 JSON으로 답하세요.

            [공시 제목] {report_nm}
            [원문 일부] {content_snippet}

            [기준]
            1. 공급계약: 매출액 대비 규모가 크면(10%↑) 호재.
            2. 유상증자/CB: 시설투자는 중립/호재, 채무상환은 악재.
            3. 무상증자: 강력 호재.
            
            [응답 형식]
            {{"score": 점수(-5~+5), "reason": "한줄 요약"}}
            """

            response = self.model.generate_content(prompt)
            text = response.text.strip()
            
            # JSON 파싱
            if "```" in text:
                text = text.split("```")[1].replace("json", "").strip()
            
            data = json.loads(text)
            return float(data.get("score", 0)), data.get("reason", "")

        except Exception as e:
            # print(f"⚠️ 분석 에러: {e}")
            return 0.0, ""
