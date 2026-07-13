# 단계 5 — 우측통행 진출 (최종 채택)

**결과 이미지:** `keepright_exit.png` — 우측통행 진출(오른쪽 차선): 방향전환 4회, 길이 62.9m, 바퀴위반 47*, 최대곡률 0.359/0.136

**최종 채택된 경로입니다.** 윗 경사로(진출)는 2차선 양방향 도로이므로, 회차
골격(단계 4)의 마지막 전진 진출 구간(run4)을 진행방향 오른쪽 차선으로 붙이고
오른쪽 차선에 도착하도록 다듬었습니다. 곡선(포켓 탈출) 구간은 원본 곡률로
보호하고 직진 구간만 우측으로 옆이동 + 스무딩했습니다.

## 파일
- `exit_keepright.py` — 우측통행 진출 최종 스크립트(clean v2)
- `exit_rightlane.py`, `replan_exit_keepright.py`, `shift_exit_keepright.py` — 이전 시도 변형들
- `generate_rightbias_path.py` — 목표 이동으로 우측 편향 경로 생성
- `find_max_safe_offset.py` — 제약을 만족하는 최대 우측 옵셋 이진탐색
- `export_path_dxf.py` — 회차 경로를 Vehicle Tracking용 DXF로 추출
- `reference_path_keepright.json` — **최종 산출물(우측통행 진출 반영)**
- `reference_path_keepright.dwg` / `.dxf` / `_backup.dxf` — CAD 추출본
- `keepright_exit.png` — **최종 결과 이미지**
- `keepright_dryrun.png`, `zones_keepright.png` — 진행 중 검증 이미지
- `tmp_keepright_*.{json,png}`, `tmp_arrive_right_*.{json,png}`, `tmp_right_arrival.*` — 옵셋/도착각 스윕 중간 산출물
- `path_design_report.md` — 회차 경로 최적화 설계 보고(산출물=`reference_path_keepright.json`)

> `*` 결과 이미지 제목의 "바퀴위반 47"은 옆이동 스윕 중간 진단값 표기이며,
> 최종 설계 검증은 `path_design_report.md` 및 리포지토리 루트에 남겨둔
> `reports/final_algorithm_spec.md`를 참고하세요.
