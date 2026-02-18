import os
import json
import re
import time
import logging
from datetime import datetime, timedelta

# 로깅 설정
logger = logging.getLogger("DartAnalyzer")

try:
    import OpenDartReader
    DART_OK = True
except ImportError:
    DART_OK = False
    logger.error("⚠️ OpenDartReader 미설치")

# ── google-genai SDK (신규 통합 SDK, 2025~) ──
# 기존 google.generativeai 는 2025-11-30 EOL
_USE_NEW_SDK = False
_USE_LEGACY_SDK = False

try:
    from google import genai
    from google.genai import types as genai_types
    _USE_NEW_SDK = True
    GEMINI_OK = True
except ImportError:
    genai = None
    genai_types = None
    # fallback: 레거시 SDK (FutureWarning 발생하지만 동작은 함)
    try:
        import google.generativeai as genai_legacy
        _USE_LEGACY_SDK = True
        GEMINI_OK = True
    except ImportError:
        genai_legacy = None
        GEMINI_OK = False
        logger.error("⚠️ google-genai 미설치 (pip install google-genai)")


class DartAnalyzer:
    """DART 공시 분석 + Gemini LLM 점수 산출 엔진 (v3.0)"""

    # ──────────────────────────────────────────────
    # 초기화
    # ──────────────────────────────────────────────
    def __init__(self, dart_api_key=None, gemini_api_key=None):
        self.dart_api_key = dart_api_key or os.environ.get("DART_API_KEY")
        self.gemini_api_key = gemini_api_key or os.environ.get("GEMINI_API_KEY")
        self.dart = None
        self._gemini_client = None   # 신규 SDK Client
        self._gemini_model = None    # 레거시 SDK GenerativeModel
        self._model_name = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")

        # --- DART 초기화 (실제 timeout 적용) ---
        if DART_OK and self.dart_api_key:
            self._init_dart()

        # --- Gemini 초기화 ---
        if GEMINI_OK and self.gemini_api_key:
            self._init_gemini()

    # ────── DART 초기화: requests.Session 에 timeout 주입 ──────
    def _init_dart(self):
        """OpenDartReader 초기화 + 실제 HTTP timeout 적용

        OpenDartReader 는 OPENDART_TIMEOUT 환경변수를 읽지 않으므로
        내부 requests.Session 의 HTTPAdapter 를 교체하여 강제 적용.
        """
        try:
            from requests.adapters import HTTPAdapter
            import requests

            timeout_sec = int(os.environ.get("DART_TIMEOUT", "10"))

            self.dart = OpenDartReader(self.dart_api_key)

            class _TimeoutAdapter(HTTPAdapter):
                """모든 요청에 기본 timeout 을 강제하는 어댑터"""
                def __init__(self, default_timeout=10, **kw):
                    self._default_timeout = default_timeout
                    super().__init__(**kw)

                def send(self, request, **kw):
                    kw.setdefault('timeout', self._default_timeout)
                    return super().send(request, **kw)

            adapter = _TimeoutAdapter(default_timeout=timeout_sec)

            # OpenDartReader 내부 session 패치
            if hasattr(self.dart, 'session') and isinstance(self.dart.session, requests.Session):
                self.dart.session.mount('http://', adapter)
                self.dart.session.mount('https://', adapter)
                logger.info(f"DART session timeout = {timeout_sec}s (adapter 주입)")
            else:
                logger.info("DART 초기화 완료 (timeout adapter 미적용, try/except 방어)")

        except Exception as e:
            logger.error(f"DART 초기화 실패: {e}")
            self.dart = None

    # ────── Gemini 초기화: 신규 SDK 우선, 레거시 fallback ──────
    def _init_gemini(self):
        """google-genai 신규 SDK 우선 사용, 실패 시 레거시 fallback"""
        try:
            if _USE_NEW_SDK:
                # ✅ 신규 SDK: google.genai.Client
                self._gemini_client = genai.Client(api_key=self.gemini_api_key)
                logger.info(f"Gemini 신규 SDK 초기화 (model={self._model_name})")
            elif _USE_LEGACY_SDK:
                # ⚠️ 레거시 fallback
                genai_legacy.configure(api_key=self.gemini_api_key)
                self._gemini_model = genai_legacy.GenerativeModel(
                    model_name=self._model_name,
                    generation_config={"response_mime_type": "application/json"},
                )
                logger.warning("Gemini 레거시 SDK 사용 중 — google-genai 로 마이그레이션 권장")
            else:
                logger.error("Gemini SDK 없음")
        except Exception as e:
            logger.error(f"Gemini 초기화 실패: {e}")

    @property
    def _has_gemini(self) -> bool:
        return self._gemini_client is not None or self._gemini_model is not None

    # ──────────────────────────────────────────────
    # 유틸리티
    # ──────────────────────────────────────────────
    def _extract_key_facts(self, text: str) -> str:
        """텍스트에서 핵심 수치(금액, 비중, 날짜)만 별도 추출"""
        facts = []
        pcts = re.findall(r'(\d+(?:\.\d+)?\s*%)', text)
        if pcts:
            facts.append(f"비중: {', '.join(pcts[:3])}")
        money = re.findall(r'(\d+(?:,\d+)?\s*(?:억원|백만원))', text)
        if money:
            facts.append(f"금액: {', '.join(money[:3])}")
        dates = re.findall(r'(\d{4}-\d{2}-\d{2})', text)
        if dates:
            facts.append(f"주요날짜: {', '.join(dates[:2])}")
        return " | ".join(facts)

    def _clean_text(self, text: str) -> str:
        """XML/HTML 태그 제거 및 공백 압축"""
        if not text:
            return ""
        clean = re.sub(r'<[^>]*>', ' ', text)
        clean = re.sub(r'\s+', ' ', clean).strip()
        return clean[:12000]

    # ──────────────────────────────────────────────
    # 공시 목록 조회
    # ──────────────────────────────────────────────
    def get_major_disclosures(self, code: str, days: int = 3) -> list:
        """최근 N일 이내 주요 공시(투자 판단 관련) 목록 반환"""
        if not self.dart:
            return []

        end_d = datetime.now().strftime("%Y%m%d")
        start_d = (datetime.now() - timedelta(days=days)).strftime("%Y%m%d")

        try:
            df = self.dart.list(code, start=start_d, end=end_d, kind='I')
            if df is None or df.empty:
                return []

            keywords = [
                '공급계약', '수주', '유상증자', '무상증자', '전환사채', '신주인수권',
                '교환사채', '자사주', '취득', '처분', '최대주주', '변경',
            ]
            mask = df['report_nm'].str.contains('|'.join(keywords), na=False)
            targets = df[mask].copy()

            return targets[['rcept_no', 'report_nm', 'rcept_dt']].to_dict('records')
        except Exception as e:
            logger.error(f"공시 목록 조회 실패 ({code}): {e}")
            return []

    # ──────────────────────────────────────────────
    # Gemini LLM 호출 (신규/레거시 분기)
    # ──────────────────────────────────────────────
    def _call_gemini(self, prompt: str) -> str:
        """Gemini 호출 → 응답 텍스트 반환 (SDK 버전에 따라 분기)"""
        if self._gemini_client is not None:
            # ✅ 신규 SDK (google-genai)
            response = self._gemini_client.models.generate_content(
                model=self._model_name,
                contents=prompt,
                config=genai_types.GenerateContentConfig(
                    response_mime_type="application/json",
                    max_output_tokens=512,
                ),
            )
            return response.text.strip() if response.text else ""
        elif self._gemini_model is not None:
            # ⚠️ 레거시 SDK (google-generativeai)
            response = self._gemini_model.generate_content(prompt)
            return response.text.strip() if response.text else ""
        return ""

    # ──────────────────────────────────────────────
    # 개별 보고서 LLM 분석
    # ──────────────────────────────────────────────
    def analyze_report(self, rcept_no: str, report_nm: str) -> tuple:
        """공시 원문을 Gemini로 분석 → (score, reason)"""
        if not self.dart or not self._has_gemini:
            return 0.0, ""

        for attempt in range(2):
            try:
                raw_xml = self.dart.document(rcept_no)
                if not raw_xml:
                    continue

                key_facts = self._extract_key_facts(raw_xml)
                content = self._clean_text(raw_xml)

                prompt = f"""
                당신은 대한민국 금융감독원 공시 전문 분석관입니다.
                제공된 [핵심 수치]를 바탕으로 [본문]의 맥락을 분석하여 주가 영향력을 평가하세요.

                [제목] {report_nm}
                [핵심 수치] {key_facts}
                [본문 내용 요약] {content}

                [평가 가이드라인 (스케일: -10 ~ +10)]
                - (+8~10): 무상증자, 매출액 30% 이상 대규모 공급계약, 경영권 분쟁 없는 최대주주 매수.
                - (+3~7): 시설투자용 3자배정 증자, 매출액 10% 이상 계약, 자사주 소각.
                - (0): 단순 정정, 통상적인 분기보고서.
                - (-3~-7): 운영자금/채무상환용 유상증자, 전환사채(CB) 대량 발행, 공급계약 해지.
                - (-8~-10): 횡령/배임, 회계처리 위반, 최대주주의 대량 지분 매도.

                반드시 JSON 형식으로만 응답하십시오.
                {{"score": 0.0, "reason": "이유 요약"}}
                """

                res_text = self._call_gemini(prompt)

                json_match = re.search(r'\{.*?\}', res_text, re.DOTALL)
                if json_match:
                    data = json.loads(json_match.group())
                    if "score" in data and "reason" in data:
                        final_score = max(-10.0, min(10.0, float(data["score"])))
                        return final_score, str(data["reason"])

                logger.warning(f"⚠️ 형식 오류 재시도 중... ({report_nm})")

            except Exception as e:
                logger.error(f"❌ 분석 시도 {attempt + 1} 실패 ({report_nm}): {e}")
                time.sleep(1)

        return 0.0, "분석 불가(서버 응답 오류)"

    # ──────────────────────────────────────────────
    # ★ apply_dart_filter  — DataFrame 일괄 처리
    # ──────────────────────────────────────────────
    def apply_dart_filter(self, df, code_col: str = "종목코드",
                          name_col: str = "종목명",
                          days: int = 3, top_n: int = 50) -> "pd.DataFrame":
        """
        scored DataFrame 을 받아 DART 공시 점수를 추가하여 반환한다.

        처리 흐름
        ─────────
        1) 상위 top_n 종목에 대해 최근 days 일 주요 공시를 조회
        2) 각 공시를 Gemini 로 분석 → 종목별 최저(최악) 점수 채택
        3) DART_SCORE / DART_REASON 컬럼 추가
        4) 원본 DataFrame 을 그대로 반환 (컬럼만 추가)

        Parameters
        ----------
        df : pd.DataFrame
            종목코드·종목명 컬럼을 포함한 scored DataFrame
        code_col : str
            종목코드 컬럼명 (기본 '종목코드')
        name_col : str
            종목명 컬럼명 (기본 '종목명')
        days : int
            최근 며칠 이내 공시를 조회할지 (기본 3)
        top_n : int
            상위 몇 종목까지 분석할지 (API 부하 방지, 기본 50)

        Returns
        -------
        pd.DataFrame
            DART_SCORE, DART_REASON 컬럼이 추가된 DataFrame
        """
        import pandas as pd

        # 기본 컬럼 세팅 (모든 종목 0점 / 특이사항 없음)
        df = df.copy()
        df["DART_SCORE"] = 0.0
        df["DART_REASON"] = "특이사항 없음"

        # DART 또는 Gemini 사용 불가 → 즉시 반환
        if not self.dart:
            logger.info("DART 미연결 — 공시 필터 스킵")
            return df
        if not self._has_gemini:
            logger.info("Gemini 미연결 — 공시 분석 스킵 (목록만 조회)")

        if code_col not in df.columns:
            logger.warning(f"'{code_col}' 컬럼 없음 — DART 필터 스킵")
            return df

        # 분석 대상 (상위 top_n 행)
        targets = df.head(top_n)
        analyzed = 0

        for idx in targets.index:
            code = str(df.at[idx, code_col]).strip().zfill(6)
            name = str(df.at[idx, name_col]) if name_col in df.columns else code

            try:
                disclosures = self.get_major_disclosures(code, days=days)
                if not disclosures:
                    continue

                worst_score = 0.0
                worst_reason = ""

                for disc in disclosures:
                    rcept_no = disc.get("rcept_no", "")
                    report_nm = disc.get("report_nm", "")

                    if self._has_gemini:
                        score, reason = self.analyze_report(rcept_no, report_nm)
                    else:
                        # Gemini 없으면 공시 제목만 기록 (점수 0)
                        score, reason = 0.0, f"[공시감지] {report_nm}"

                    # 가장 부정적인(최악) 점수를 채택
                    if score < worst_score:
                        worst_score = score
                        worst_reason = reason
                    # 긍정 공시도 기록 (기존 0점이면 갱신)
                    elif worst_score == 0.0 and score > 0.0:
                        worst_score = score
                        worst_reason = reason

                df.at[idx, "DART_SCORE"] = worst_score
                df.at[idx, "DART_REASON"] = worst_reason or "특이사항 없음"
                analyzed += 1

                # API 과부하 방지 (종목 간 0.3초 쿨다운)
                time.sleep(0.3)

            except Exception as e:
                logger.error(f"DART 필터 처리 오류 ({name}/{code}): {e}")
                continue

        logger.info(f"DART 필터 완료: {analyzed}/{len(targets)}건 분석")
        return df
