import os
import json
import re
import time
import random
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Optional, Tuple, List

logger = logging.getLogger("DartAnalyzer")

try:
    import OpenDartReader
    DART_OK = True
except ImportError:
    DART_OK = False
    logger.error("⚠️ OpenDartReader 미설치")

# ── google-genai SDK (신규 통합 SDK, 2025~) ──
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
    try:
        import google.generativeai as genai_legacy
        _USE_LEGACY_SDK = True
        GEMINI_OK = True
    except ImportError:
        genai_legacy = None
        GEMINI_OK = False
        logger.error("⚠️ google-genai 미설치 (pip install google-genai)")


# ═══════════════════════════════════════════════════
#  Structured Output 스키마
# ═══════════════════════════════════════════════════

@dataclass
class DartScore:
    """Gemini 응답 스키마 — JSON 파싱 에러 방지"""
    score: float
    reason: str


# Pydantic 가능하면 사용, 아니면 dataclass fallback
_PYDANTIC_SCHEMA = None
try:
    from pydantic import BaseModel as _PydanticBase

    class DartScoreSchema(_PydanticBase):
        score: float
        reason: str

    _PYDANTIC_SCHEMA = DartScoreSchema
except ImportError:
    pass


class DartAnalyzer:
    """DART 공시 분석 + Gemini LLM 점수 산출 엔진 (v4.0)

    [v4.0] 4건 리팩터링:
      #1 원문 12000자 절단 → 섹션 기반 지능형 발췌
      #2 JSON 정규식 파싱 → Structured Output (response_schema)
      #3 최악 점수(worst) → 최대 임팩트(max abs) 채택
      #4 정규식 수치 추출 폐기 → LLM에 원문 맥락 직접 전달
    """

    # 공시 원문에서 우선 추출할 섹션 키워드
    _PRIORITY_SECTIONS = [
        "자금조달의 목적", "자금의 사용목적", "발행조건", "배정대상자",
        "계약내용", "계약금액", "계약기간", "주요내용", "결정내용",
        "취득목적", "처분사유", "변경내용", "사유", "결의내용",
    ]

    # 최대 LLM 입력 길이 (Gemini 2.0 Flash 토큰 여유 충분)
    _MAX_CONTENT_LEN = 8000

    def __init__(self, dart_api_key=None, gemini_api_key=None):
        self.dart_api_key = dart_api_key or os.environ.get("DART_API_KEY")
        self.gemini_api_key = gemini_api_key or os.environ.get("GEMINI_API_KEY")
        self.dart = None
        self._gemini_client = None
        self._gemini_model = None
        self._model_name = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")

        if DART_OK and self.dart_api_key:
            self._init_dart()
        if GEMINI_OK and self.gemini_api_key:
            self._init_gemini()

    # ──────────────────────────────────────────────
    # DART 초기화
    # ──────────────────────────────────────────────
    def _init_dart(self):
        try:
            from requests.adapters import HTTPAdapter
            import requests

            timeout_sec = int(os.environ.get("DART_TIMEOUT", "10"))
            self.dart = OpenDartReader(self.dart_api_key)

            class _TimeoutAdapter(HTTPAdapter):
                def __init__(self, default_timeout=10, **kw):
                    self._default_timeout = default_timeout
                    super().__init__(**kw)

                def send(self, request, **kw):
                    kw.setdefault('timeout', self._default_timeout)
                    return super().send(request, **kw)

            adapter = _TimeoutAdapter(default_timeout=timeout_sec)

            if hasattr(self.dart, 'session') and isinstance(self.dart.session, requests.Session):
                self.dart.session.mount('http://', adapter)
                self.dart.session.mount('https://', adapter)
                logger.info(f"DART session timeout = {timeout_sec}s (adapter 주입)")
        except Exception as e:
            logger.error(f"DART 초기화 실패: {e}")
            self.dart = None

    # ──────────────────────────────────────────────
    # Gemini 초기화
    # ──────────────────────────────────────────────
    def _init_gemini(self):
        try:
            if _USE_NEW_SDK:
                self._gemini_client = genai.Client(api_key=self.gemini_api_key)
                logger.info(f"Gemini 신규 SDK 초기화 (model={self._model_name})")
            elif _USE_LEGACY_SDK:
                genai_legacy.configure(api_key=self.gemini_api_key)
                self._gemini_model = genai_legacy.GenerativeModel(
                    model_name=self._model_name,
                    generation_config={"response_mime_type": "application/json"},
                )
                logger.warning("Gemini 레거시 SDK 사용 중")
        except Exception as e:
            logger.error(f"Gemini 초기화 실패: {e}")

    @property
    def _has_gemini(self) -> bool:
        return self._gemini_client is not None or self._gemini_model is not None

    # ──────────────────────────────────────────────
    # [v4.0 #1] 섹션 기반 지능형 발췌 (12000자 절단 제거)
    # ──────────────────────────────────────────────
    def _extract_sections(self, raw_xml: str) -> str:
        """XML 원문에서 우선 섹션을 지능적으로 발췌.

        1) <TITLE> 태그 기준으로 섹션 분리
        2) _PRIORITY_SECTIONS 키워드와 매칭되는 섹션 우선 추출
        3) 매칭 없으면 전체를 태그 제거 후 앞부분 반환
        """
        if not raw_xml:
            return ""

        # 방법 1: <TITLE> 태그 기반 섹션 추출 시도
        sections = re.split(r'<TITLE[^>]*>', raw_xml, flags=re.IGNORECASE)
        if len(sections) > 1:
            priority_texts = []
            for section in sections[1:]:  # 첫 번째는 헤더
                # 섹션 제목 추출
                title_end = section.find('</TITLE>')
                if title_end == -1:
                    title_end = section.find('<')
                title = section[:title_end].strip() if title_end > 0 else ""

                # 우선 섹션 키워드 매칭
                if any(kw in title for kw in self._PRIORITY_SECTIONS):
                    # 태그 제거 후 본문 추출
                    body = re.sub(r'<[^>]*>', ' ', section)
                    body = re.sub(r'\s+', ' ', body).strip()
                    if body:
                        priority_texts.append(f"[{title}] {body[:2000]}")

            if priority_texts:
                result = "\n\n".join(priority_texts)
                return result[:self._MAX_CONTENT_LEN]

        # 방법 2: 폴백 — 태그 제거 후 전체 텍스트 앞부분
        clean = re.sub(r'<[^>]*>', ' ', raw_xml)
        clean = re.sub(r'\s+', ' ', clean).strip()
        return clean[:self._MAX_CONTENT_LEN]

    # ──────────────────────────────────────────────
    # [v4.0 #2] Structured Output으로 Gemini 호출
    # ──────────────────────────────────────────────
    def _call_gemini(self, prompt: str) -> str:
        """Gemini 호출 → 응답 텍스트 반환"""
        from llm_retry_utils import _llm_call_with_retry

        def _raw_call():
            if self._gemini_client is not None:
                # Structured Output: response_schema로 JSON 강제
                config_kwargs = {
                    "response_mime_type": "application/json",
                    "max_output_tokens": 2048,   # [v20.3] 2.5-flash thinking 토큰 여유
                }
                # Pydantic 스키마가 있으면 주입 (파싱 에러율 0%)
                if _PYDANTIC_SCHEMA is not None:
                    config_kwargs["response_schema"] = _PYDANTIC_SCHEMA

                response = self._gemini_client.models.generate_content(
                    model=self._model_name,
                    contents=prompt,
                    config=genai_types.GenerateContentConfig(**config_kwargs),
                )
                return response.text.strip() if response.text else ""
            elif self._gemini_model is not None:
                response = self._gemini_model.generate_content(prompt)
                return response.text.strip() if response.text else ""
            return ""

        try:
            return _llm_call_with_retry(_raw_call, max_retries=3, base_delay=2.0, cap=30.0)
        except Exception as e:
            logger.warning(f"Gemini 호출 최종 실패: {e}")
            return ""

    def _parse_gemini_response(self, res_text: str) -> Tuple[float, str]:
        """Gemini 응답 텍스트 → (score, reason) 파싱.

        [v20.3] 잘린 JSON 복구 + regex 폴백 3단계:
          1) 정상 JSON 파싱
          2) 잘린 JSON 복구 시도
          3) regex로 score 숫자만 추출
        """
        if not res_text:
            return 0.0, ""

        # 마크다운 코드블록 제거
        cleaned = re.sub(r'```json\s*', '', res_text)
        cleaned = re.sub(r'```\s*', '', cleaned)
        cleaned = cleaned.strip()

        # ── 1단계: 정상 JSON 파싱 ──
        try:
            data = json.loads(cleaned)
            if "score" in data and "reason" in data:
                score = max(-10.0, min(10.0, float(data["score"])))
                return score, str(data["reason"])
        except (json.JSONDecodeError, ValueError):
            pass

        # ── 2단계: 잘린 JSON 복구 ──
        # {"score": 4.5, "reason": → 누락된 끝 보완
        try:
            repaired = cleaned
            # 열린 따옴표 닫기
            if repaired.count('"') % 2 != 0:
                repaired += '"'
            # 닫는 중괄호 없으면 추가
            if '{' in repaired and '}' not in repaired:
                repaired += '}'
            data = json.loads(repaired)
            if "score" in data:
                score = max(-10.0, min(10.0, float(data["score"])))
                return score, str(data.get("reason", "파싱 복구"))
        except (json.JSONDecodeError, ValueError):
            pass

        # ── 3단계: regex로 score 숫자만 추출 ──
        # {"score": 4.5 또는 "score":4.5 패턴
        score_match = re.search(r'"score"\s*:\s*(-?\d+\.?\d*)', cleaned)
        if score_match:
            try:
                score = max(-10.0, min(10.0, float(score_match.group(1))))
                # reason도 시도
                reason_match = re.search(r'"reason"\s*:\s*"([^"]*)', cleaned)
                reason = reason_match.group(1) if reason_match else "regex 폴백"
                return score, reason
            except ValueError:
                pass

        logger.warning(f"⚠️ Gemini 응답 파싱 실패: {res_text[:200]}")
        return 0.0, ""

    # ──────────────────────────────────────────────
    # 공시 목록 조회
    # ──────────────────────────────────────────────
    def get_major_disclosures(self, code: str, days: int = 3) -> list:
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
    # [v4.0 #1+#4] 개별 보고서 LLM 분석 (섹션 발췌 + 맥락 직접 전달)
    # ──────────────────────────────────────────────
    def analyze_report(self, rcept_no: str, report_nm: str) -> Tuple[float, str]:
        """공시 원문을 Gemini로 분석 → (score, reason)"""
        if not self.dart or not self._has_gemini:
            return 0.0, ""

        for attempt in range(2):
            try:
                raw_xml = self.dart.document(rcept_no)
                if not raw_xml:
                    continue

                # [v4.0 #1] 섹션 기반 발췌 (12000자 절단 대신)
                content = self._extract_sections(raw_xml)

                # [v4.0 #4] 정규식 수치 추출 폐기 → LLM에 원문 맥락 직접 전달
                prompt = f"""당신은 대한민국 금융감독원 공시 전문 분석관입니다.
아래 공시의 핵심 내용을 분석하여 주가 영향력을 평가하세요.
수치(금액, 비율, 날짜)는 본문에서 직접 파악하십시오.

[제목] {report_nm}
[본문]
{content}

[평가 가이드라인 (스케일: -10 ~ +10)]
- (+8~10): 무상증자, 매출액 30% 이상 대규모 공급계약, 경영권 분쟁 없는 최대주주 매수.
- (+3~7): 시설투자용 3자배정 증자, 매출액 10% 이상 계약, 자사주 소각.
- (0): 단순 정정, 통상적인 분기보고서.
- (-3~-7): 운영자금/채무상환용 유상증자, 전환사채(CB) 대량 발행, 공급계약 해지.
- (-8~-10): 횡령/배임, 회계처리 위반, 최대주주의 대량 지분 매도.

반드시 JSON 형식으로만 응답하십시오.
{{"score": 0.0, "reason": "이유 요약"}}"""

                res_text = self._call_gemini(prompt)
                score, reason = self._parse_gemini_response(res_text)

                if score != 0.0 or reason:
                    return score, reason

                logger.warning(f"⚠️ 형식 오류 재시도 중... ({report_nm})")

            except Exception as e:
                from llm_retry_utils import _is_retryable
                if _is_retryable(e):
                    wait = min(30, 2 ** (attempt + 1))
                    logger.warning(f"⚠️ 429/재시도 {attempt+1}/2 ({report_nm}): wait={wait}s")
                    time.sleep(wait)
                else:
                    logger.error(f"❌ 비재시도 에러 ({report_nm}): {e}")
                    break

        return 0.0, "분석 불가(서버 응답 오류)"

    # ──────────────────────────────────────────────
    # [v4.0 #3] 최대 임팩트 기반 점수 병합
    # ──────────────────────────────────────────────
    def apply_dart_filter(self, df, code_col: str = "종목코드",
                          name_col: str = "종목명",
                          days: int = 3, top_n: int = 50) -> "pd.DataFrame":
        """scored DataFrame에 DART 공시 점수를 추가하여 반환.

        [v4.0 #1] ThreadPoolExecutor 병렬 처리 (5~8분 → ~20초)
        [v4.0 #2] 0점 공시도 scores에 보존 (사유 유실 방지)
        [v4.0 #3] 최대 임팩트(max abs) 채택
        """
        import pandas as pd
        from concurrent.futures import ThreadPoolExecutor, as_completed

        df = df.copy()
        df["DART_SCORE"] = 0.0
        df["DART_REASON"] = "특이사항 없음"

        if not self.dart:
            logger.info("DART 미연결 — 공시 필터 스킵")
            return df
        if not self._has_gemini:
            logger.info("Gemini 미연결 — 공시 분석 스킵")

        if code_col not in df.columns:
            logger.warning(f"'{code_col}' 컬럼 없음 — DART 필터 스킵")
            return df

        targets = df.head(top_n)
        if targets.empty:
            logger.info("DART 분석 대상 종목 없음 — 스킵")
            return df

        def _process_ticker(idx, code, name):
            """단일 종목 공시 분석 (스레드에서 실행)"""
            # Thundering Herd 방지: 스레드 출발 시점 분산
            time.sleep(random.uniform(0.1, 1.5))
            try:
                disclosures = self.get_major_disclosures(code, days=days)
                if not disclosures:
                    return idx, 0.0, "공시 없음"

                scores: List[Tuple[float, str]] = []
                for disc in disclosures:
                    rcept_no = disc.get("rcept_no", "")
                    report_nm = disc.get("report_nm", "")

                    if self._has_gemini:
                        score, reason = self.analyze_report(rcept_no, report_nm)
                    else:
                        score, reason = 0.0, f"[공시감지] {report_nm}"

                    # 0점이어도 무조건 담음 (LLM 분석 사유 보존)
                    scores.append((score, reason))
                    time.sleep(0.3)  # API Rate Limit 방어

                if scores:
                    best_score, best_reason = max(scores, key=lambda x: abs(x[0]))
                    return idx, best_score, best_reason
                return idx, 0.0, "특이사항 없음"

            except Exception as e:
                logger.error(f"DART 필터 처리 오류 ({name}/{code}): {e}")
                return idx, 0.0, f"분석 오류: {e}"

        # ThreadPoolExecutor 병렬 처리
        analyzed = 0
        max_workers = min(5, len(targets))

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    _process_ticker,
                    idx,
                    str(df.at[idx, code_col]).strip().zfill(6),
                    str(df.at[idx, name_col]) if name_col in df.columns else "",
                ): idx
                for idx in targets.index
            }

            for future in as_completed(futures):
                try:
                    idx, best_score, best_reason = future.result()
                    df.at[idx, "DART_SCORE"] = best_score
                    df.at[idx, "DART_REASON"] = best_reason or "특이사항 없음"
                    analyzed += 1
                except Exception as e:
                    logger.error(f"DART 병렬 처리 오류: {e}")

        logger.info(f"DART 필터 완료: {analyzed}/{len(targets)}건 분석 (병렬 처리)")
        return df
