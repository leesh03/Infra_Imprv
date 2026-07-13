# 명지야호 — AI 기반 셔틀버스 통행 안전 시스템

∧자 분기점에서 셔틀버스가 **아래 경사로로 진입 → 회차 → 윗 경사로로 진출**하는
회차 경로를 계획하는 프로젝트입니다. 최종 채택 경로는 Hybrid A*(바퀴 충돌모델)로
회차 골격을 만들고 진출 구간을 우측통행에 맞춰 다듬은
[`solution_steps/step5_keepright_exit/`](solution_steps/step5_keepright_exit/)의
`reference_path_keepright.json` 입니다.

## 폴더 구조

저장소는 **해결 과정 아카이브**와 **성격별 정리 버킷**으로 구성되어 있습니다.
각 폴더에는 무엇이 왜 들어있는지 설명하는 자체 `README.md`가 있습니다.

| 폴더 | 성격 | 내용 |
| --- | --- | --- |
| [`solution_steps/`](solution_steps/) | 해결 과정 (핵심) | 경로를 찾아온 과정을 5단계로 정리. 각 단계 폴더에 대표 결과 이미지 + 스크립트·데이터·리포트 |
| [`project_core/`](project_core/) | 🔴 공용 토대 | 5단계 전부가 공유하는 기반 코드/데이터 (`envs`·`kinematics`·`geometry`·`config`). **삭제·분산 금지** |
| [`docs/`](docs/) | 문서 | 최종 알고리즘 명세, 종합 현황 보고서, 발표자료(PDF) |
| [`rl_abandoned/`](rl_abandoned/) | 폐기 갈래 | 중간에 접은 강화학습(PPO) 코드·학습 산출물 (약 8MB, 삭제 무방) |
| [`misc_utils/`](misc_utils/) | 보조 유틸 | 초기 셋업·진단·시각화 일회성 스크립트와 그 출력물 |
| [`scratch/`](scratch/) | 임시 | 임시 메모·로그·크래시덤프 (삭제 무방) |
| `data/raw/cad/` | 원본 자료 | 원본 CAD (`road_model.dxf`, `bus_model.dxf`). `project_core/config.yaml`이 `../data`로 참조하므로 루트에 유지 |
| `src/` | (예약) | `trajectory_to_display/` 자리표시자 |

## 해결 과정 5단계 (`solution_steps/`)

| 단계 | 폴더 | 대표 결과 | 핵심 |
| --- | --- | --- | --- |
| 1 | `step1_switch_repair` | 방향전환 28→26 | 원본 경로의 방향전환을 국소적으로 제거 |
| 2 | `step2_hybrid_astar_bodymodel` | 403 웨이포인트, 28회 | 차체 충돌모델 Hybrid A*로 첫 경로 탐색 |
| 3 | `step3_cma_es_fromscratch` | **실패** | CMA-ES로 처음부터 탐색(좁은 통로 통과 실패) |
| 4 | `step4_hybrid_astar_wheelmodel` | 5회, 62.6m | 바퀴(축거) 충돌모델로 재실행 → 대폭 개선 |
| 5 | `step5_keepright_exit` | **최종 채택** | 우측통행 진출 반영본 |

> 실제 작업 순서는 2 → 3 → 1 → 4 → 5 였고, 단계 번호는 설명용 순서입니다.
> 자세한 내용은 [`solution_steps/README.md`](solution_steps/README.md) 참고.

## 재현 관련 주의

- 최종 경로 생성·검증·DXF 추출은 모두 `project_core/`의 코드·데이터에 의존합니다.
  이 폴더는 지우거나 내부 구조를 흩뜨리면 안 됩니다.
- `solution_steps/`·`rl_abandoned/`의 스크립트는 아카이브(보존)용이라,
  옮겨진 자리에서는 import 경로가 달라 **그대로 재실행되지 않습니다**.
  재현하려면 원위치로 되돌리거나 경로를 맞춰야 합니다.

## 협업 규칙

- 개인 브랜치에서 작업하고 `main`은 직접 건드리지 말 것 (마지막에 `main`에서 merge).
- 위 폴더 재정리는 개인 브랜치(`Leesh`)의 아카이브 정리입니다. `main` merge 시점에
  팀과 폴더 구조를 조율하세요. (원래 팀 규칙은 "경로 최적화 작업은 `simulation/`
  폴더 내에서만" 이었고, 이 정리로 `simulation/`은 위 버킷들로 분리되었습니다.)
