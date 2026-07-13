# 단계 1 — 국소 방향전환 제거 (switch repair)

**결과 이미지:** `reference_path_repaired.png` — Switch-repaired path: switches 28→26, len 120.6→120.0m, violations=0

원본 Hybrid A* 경로(단계 2의 `reference_path.json`)의 28개 방향전환 경계를 각각
"제거 가능한가" 독립 검사하여, 원본 좌표는 고정한 채 전환 주변만 국소 CMA-ES로
다시 이어 붙였습니다. 결과적으로 방향전환 2개를 안전하게 제거(28→26).

## 파일
- `switch_repair.py` — 국소 방향전환 제거 핵심 로직(`try_connect`, `repair_switches`, `search_cluster`)
- `run_switch_repair.py` — 각 방향전환의 제거 가능 여부 탐색 실행
- `apply_switch_repair.py` — 확인된 제거 가능 전환을 원본에 스플라이스 후 정밀 폴리곤으로 재검증
- `repair_cluster.py`, `repair_loop.py` — 더 넓은 자유기어 클러스터/루프 창 탐색(밀집 헤어핀 구간에는 실패)
- `reference_path_repaired.json` — 최종 산출물(28→26, 위반 0, 전진 도착)
- `cma_es_investigation_2026-07-08.md` — CMA-ES 회차 경로 탐색 전체 경과 보고(단계 1·3 공통)

> 이 단계의 결과(26회)도 아직 최적이 아니라는 판단에서 단계 4(바퀴 모델
> 재설계)로 이어졌습니다.
