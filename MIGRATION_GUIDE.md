# 🚀 Streamlit → NiceGUI 마이그레이션 가이드

## 개요

LDY Pro Trader를 Streamlit Cloud에서 NiceGUI + Railway로 전환하는 가이드입니다.

## 파일 목록 (추가/수정)

```
새로 추가하는 파일:
├── main.py                  ← NiceGUI 진입점 (Phase 0 PoC)
├── Dockerfile               ← Railway 배포용
├── railway.toml             ← Railway 설정
├── requirements_nicegui.txt ← NiceGUI용 의존성
├── .dockerignore            ← Docker 빌드 최적화

기존 유지 (변경 없음):
├── collector.py             ← GitHub Actions 그대로
├── scoring_engine.py
├── ml_engine.py
├── chart_components.py      ← Plotly 차트 (ui.plotly()로 표시)
├── shared_utils.py
├── db_utils.py
├── indicators.py
├── schema.py
├── data/                    ← CSV, 모델 파일 그대로

Phase 3에서 삭제:
├── dashboard.py             ← Streamlit 버전 — [v85] 삭제 완료 (아래 "레거시 Streamlit 대시보드 제거" 참고)
├── .streamlit/              ← Streamlit 설정
├── requirements.txt         ← requirements_nicegui.txt로 대체
```

## Railway 배포 절차

### 1단계: Railway 계정 생성
1. https://railway.app 접속 → GitHub 계정으로 가입
2. Hobby Plan ($5/월) 선택 (무료 크레딧 포함)

### 2단계: GitHub Repo 연결
1. Railway 대시보드 → New Project → Deploy from GitHub
2. `swingpicker-web-main` 레포 선택
3. Railway가 Dockerfile을 자동 감지 → 빌드 시작

### 3단계: 환경변수 설정
Railway 대시보드 → Variables 탭에서 추가:

```
GEMINI_API_KEY=xxx           # AI 분석용
DART_API_KEY=xxx             # 공시 분석용
LDY_GIST_TOKEN=xxx           # 포트폴리오 Gist 연동
LDY_GIST_ID=xxx
```

(기존 .streamlit/secrets.toml에 있던 값들)

### 4단계: 도메인 설정
- Railway가 자동으로 `xxx.up.railway.app` 도메인 생성
- Settings → Networking → Generate Domain
- (선택) 커스텀 도메인 연결 가능

### 5단계: 자동 배포 확인
- 이후 `git push`할 때마다 Railway가 자동 빌드+배포 (1~2분)
- Streamlit Cloud와 동시 운영 가능 (충돌 없음)

## 전환 기간 안전망

```
Phase 0~2 기간:
  Streamlit Cloud → dashboard.py 실행 (기존 사용자용)
  Railway         → main.py 실행 (테스트용)

Phase 3 완료 후:
  Railway만 운영 → Streamlit Cloud 비활성화
```

### [v85] 레거시 Streamlit 대시보드 제거

- v85에서 구 Streamlit 진입점 `dashboard.py`와 백업 폴더 `backup_v205c_20260310/`
  (dashboard.py·ml_engine.py·scoring_engine.py·version_info.py 사본)를 저장소에서 **삭제**했습니다.
  라이브 앱은 Railway의 `main.py`(NiceGUI) 하나뿐이며, Dockerfile·CI·테스트 어디에서도 이 파일들을 쓰지 않았습니다.
- ⚠️ **운영자 확인 필요:** 위 "전환 기간 안전망"대로 Streamlit Community Cloud 앱이 아직
  `main` 브랜치의 `dashboard.py`를 가리키고 있다면, 다음 재부팅/재배포 때 파일을 찾지 못해 깨집니다.
  저장소만으로는 외부 Cloud 앱의 활성 여부를 확인할 수 없으니 Streamlit Cloud 콘솔에서 앱을 비활성화(삭제)하세요.
- 복구가 필요하면 git 이력에서 꺼내면 됩니다 (삭제 직전 버전 = 삭제 커밋의 부모):

  ```bash
  # 삭제 커밋 찾기
  git log --oneline --diff-filter=D -- dashboard.py
  # 그 커밋의 부모(^)에서 파일 복원
  git checkout <삭제커밋>^ -- dashboard.py
  git checkout <삭제커밋>^ -- backup_v205c_20260310/   # 백업 폴더가 필요할 때만
  ```

  복원한 dashboard.py를 실행하려면 `requirements.txt`(streamlit 포함), `auth_user.py`, `.streamlit/config.toml`이
  필요합니다 — 이 셋은 v85에서도 그대로 남아 있습니다.

## Streamlit → NiceGUI 변환 패턴 요약

| Streamlit                      | NiceGUI                                    |
|-------------------------------|--------------------------------------------|
| `st.title("x")`              | `ui.label("x").classes('text-3xl')`        |
| `st.metric("이름", 값)`       | `metric_card("이름", 값, delta)`            |
| `st.plotly_chart(fig)`        | `ui.plotly(fig).classes('w-full')`         |
| `st.dataframe(df)`           | `ui.table(columns, rows)`                  |
| `st.tabs([...])`             | `ui.tabs()` + `ui.tab_panels()`            |
| `st.selectbox(...)`          | `ui.select(opts, on_change=fn)`            |
| `st.checkbox(...)`           | `ui.checkbox(on_change=fn)`                |
| `st.columns(N)`              | `ui.row()` + `ui.column()`                 |
| `st.expander("...")`         | `ui.expansion("...")`                      |
| `st.sidebar`                 | `ui.left_drawer()`                          |
| `st.session_state`           | `app.storage.user`                          |
| `st.cache_data`              | 전역변수 / `@lru_cache`                     |
| `st.secrets["KEY"]`          | `os.environ["KEY"]`                         |
| `st.set_page_config(...)`    | `ui.run(title=..., favicon=...)`            |
| `st.spinner("...")`          | `ui.spinner()` 또는 `ui.notify()`           |
| `st.success/warning/error()` | `ui.notify("msg", type="positive/warning")` |
| `st.divider()`               | `ui.separator()`                            |
| `st.caption("...")`          | `ui.label("...").classes('text-xs')`        |
| `st.markdown("# ...")`       | `ui.markdown("# ...")`                     |

## Phase별 작업 상세

### Phase 0 (오늘) ✅
- [x] main.py PoC 작성 (Tab1 시장 + Tab2 종목분석)
- [x] Dockerfile + railway.toml
- [x] requirements_nicegui.txt
- [ ] GitHub push → Railway 배포 확인

### Phase 1 (2~3일)
- [ ] Tab2 완성: 캔들차트 (plot_interactive_chart 연동)
- [ ] Tab2 완성: 칸반 보기 모드
- [ ] 매물대 분석 섹션

### Phase 2 (2~3일)
- [ ] Tab3 포트폴리오: 종목 입력 → AI 진단
- [ ] 인증 시스템: auth_user.py → app.storage 변환
- [ ] 회원 등급별 접근 제어

### Phase 3 (1~2일)
- [ ] Tab4~8 나머지 탭 변환
- [ ] Streamlit Cloud 비활성화
- [x] dashboard.py → 삭제 (v85, git 이력으로 복구 가능)
- [ ] requirements.txt → requirements_nicegui.txt로 교체
