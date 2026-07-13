# 단계 4 — Hybrid A* 바퀴(축거) 충돌모델 재실행

**결과 이미지:** `hybrid_astar_wheelmodel.png` — WHEEL-model Hybrid A*: gear_changes=5, len=62.6m, wheel_viol=0, bumper-over-curb nodes=93

핵심 통찰: 노면 경계는 **연석(緣石)**이라 바퀴는 못 넘지만 차체 오버행은 그 위로
넘어가도 됩니다. 그래서 충돌 판정을 차체/범퍼 전체가 아니라 **바퀴/축거 기준**
(front_oh=L=6.29m, rear_oh=0, 반폭 1.2475m)으로 바꿔 Hybrid A*를 재실행했습니다.
결과: 방향전환 **26~28 → 5회**, 길이 **120.6 → 62.6m(-48%)**, 바퀴 위반 0.
209개 노드 중 93개에서 범퍼가 연석 위로 오버행 — 예전 차체 모델이 잘못 금지하던
바로 그 영역이며, 이것이 물리적 "해금"이었습니다.

## 파일
- `run_hybrid_astar_wheel.py` — 바퀴 footprint로 Hybrid A* 재실행(핵심 플래너는 단계 2의 `hybrid_astar.py` 재사용)
- `reference_path_wheelmodel.json` — 회차 골격 산출물(방향전환 5회)
- `hybrid_astar_wheelmodel.png` — 결과 이미지(탐색 노드 포함)
- `zones_wheelmodel.png` — 바퀴 모델 구역 시각화
- `reference_path_wheelmodel.dxf` — DXF 추출본
- `repair_path_safe.py`, `restore_original.py` — 바퀴 모델 경로 점검/복원 유틸
- `path_optimization_deep_plan_2026-07-08.md` — 이 재설계를 도출한 심층 계획 보고

> 이 골격에 우측통행 진출을 반영한 것이 단계 5의 최종본입니다.
