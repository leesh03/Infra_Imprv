# misc_utils — 초기 셋업·진단 유틸리티

5단계 어느 하나에 속하지 않는, 초기 지오메트리 셋업과 일회성 진단·시각화
스크립트 및 그 출력물입니다. 재현 기록으로 남길 가치는 있으나 필수는 아닙니다.

## 스크립트
- `recompute_start_goal.py`, `refine_ramp_pose.py`, `plan_arcs.py` — 시작/목표 자세·경사로 자세 재정의, 원호 계획
- `draw_zones.py` — 구역(zone) 시각화
- `verify_swept_envelope.py` — 스윕 포락선(swept envelope) 검증
- `delta_max_sensitivity.py` — 최대 조향각 민감도 분석
- `manual_drive_demo.py`, `plot_steering_diagram.py` — 수동 주행 데모·조향 기하 다이어그램

## 출력물
- `final_check.png`, `final_start_goal.png`, `start_goal_redefinition.png`, `goal_pocket_zoom.png`,
  `road_boundary.png`, `steering_geometry_diagram.png`, `manual_drive_demo.png`
- `comparison.csv` — 시나리오별 조정 횟수·통과 가능 여부·최소 요구 폭 비교표
