# SwingPicker Phase 1 — 연동 완료 버전

## 테스트: ✅ 기존 12/12 ALL PASSED

## 변경 파일 (8개) — 전부 레포 루트에 덮어쓰기

| 파일 | 변경 | 핵심 |
|------|------|------|
| `collector_config.py` | 수정 | WEIGHT_CONFIG 통합 + time_stop(7일) + slippage + snapshot |
| `scoring_engine.py` | 수정 | WEIGHT_CONFIG → CollectorConfig SSOT (하위호환 유지) |
| `trade_plan.py` | 수정 | TradePlan time_stop 필드 + exec_multi_bar time_stop 로직 + estimate_slippage_bps |
| `validation.py` | 수정 | 기존 유지 + HardBlock 7규칙 추가 |
| `collector.py` | 수정 | Hard Block 호출 + 동적 슬리피지 ExecRule + Time Stop 자동 주입 + Config Snapshot |
| `main.py` | 수정 | Tab 7에 Research Workbench 연동 |
| `test_ssot_import.py` | 수정 | collector_config import 허용 |
| `research_tab.py` | **신규** | NiceGUI 성과 분석 (점수 구간별 승률, Top-K, Reality Check) |

## 이전 버전(92점)과 달라진 점

| 항목 | 이전 (92점) | 지금 (94~95점) |
|------|------------|---------------|
| Time Stop | 로직만 존재, 기본값 0(비활성) | **기본값 7일**, collector.py에서 자동 주입 |
| 동적 슬리피지 | 함수만 존재 | **collector.py에서 ExecRule에 자동 반영** |
| Hard Block | 함수만 존재 | **collector.py에서 자동 호출**, blocked CSV 별도 저장 |
| Config Snapshot | 메서드만 존재 | **recommend CSV에 자동 저장** |
| Research 탭 | 파일만 존재 | **main.py Tab 7에 연동 완료** |

## 적용 방법

GitHub 레포 → `Add file` → `Upload files` → 8개 파일 드래그앤드롭 → Commit
