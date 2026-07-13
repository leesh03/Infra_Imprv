# 단계 2 — 원본 Hybrid A* (차체 충돌모델)

**결과 이미지:** `full_path_check.png` — Full path check: 403 waypoints, gear_changes=28, end-to-goal dpos=0.000m, 전진 도착

전/후진을 모두 허용하는 양방향 Hybrid A*로 start→goal 경로를 처음으로 찾은
단계입니다. 미세 해상도(arc_len 0.3 / xy_res 0.2 / theta_res 2°)에서
버스 앞면 중앙이 목표에 전진으로 도달하는 경로를 탐색했습니다. 충돌 판정은
**차체/범퍼 전체(front_oh=d_ref)** 기준이었고, 이 과도하게 조인 모델이 뒤에서
밝혀진 다회 방향전환의 근본 원인입니다(→ 단계 4).

## 파일
- `hybrid_astar.py` — Hybrid A* 플래너 핵심(단계 4에서도 재사용)
- `run_hybrid_astar.py` — 본 탐색 실행 스크립트(해상도·gear_switch_penalty 설정)
- `plot_full_path_check.py` — 저장된 `reference_path.json`을 그림으로 도착·기어 검증
- `plot_search_frontier.py` / `search_frontier.png` — 탐색 프런티어 시각화
- `plot_best_partial_path.py` / `best_partial_path.png` — 도달 전 최선 부분경로 진단
- `_check_constraints.py` — `reference_path.json` 제약(경계/곡률) 점검 유틸
- `reference_path.json` — 403 웨이포인트, 방향전환 28회 (차체 모델 산출물)
- `reference_path.json.bak_wiggly` — 이전 38회 전환 버전 백업
- `full_path_check.png.bak_wiggly`, `reference_path_check.png` — 관련 검증 이미지
- `hybrid_astar_arrival_constraint_lessons.md` — 목표 도달 판정 시행착오 정리(2026-07-04)

> ⚠️ `reference_path.json`은 폐기된 RL 환경 로더(`simulation/envs/reference_path.py`)가
> 실행 시 읽던 파일입니다. 이 폴더로 옮겼으므로 남아 있는 RL 코드는 이 파일을
> 더 이상 찾지 못합니다(RL은 이미 폐기).
