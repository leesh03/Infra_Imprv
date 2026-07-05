# 셔틀버스 ∧자 분기점 회차 강화학습 — 종합 설계 보고서

> 기준일: 2026-07-03 · 리포지토리: Intra_Imprv / simulation · Python 3.11.5 (venv)
> 단계: TASK 1–7 완료 · TASK 8 재학습 대기 · 경로탐색 블로커 진단 중

DXF 원본 도면에서 노면 마킹 설계 도면까지, 전체 파이프라인·기술 스택·지금까지의 시행착오·현재 구현 상태와 최종 산출물 배치 계획을 정리한다.

---

## 1. 전체 설계 파이프라인 (실제 산출물 포함)

DXF 원본 → 기하 추출 → 물리·환경 → 학습 → 경로 추출 → 마킹 도면까지 TASK 1–11 순서. 각 단계의 실제 코드 파일과 현재 저장소에 존재하는 산출물을 함께 표기한다.

| TASK | 내용 | 파일 / 산출물 | 상태 |
|---|---|---|---|
| T1 | DXF 파싱 & 기하 추출 | `geometry/dxf_parser.py`, `build_geometry.py` → `raw_plan.json`(1.38MB), `grades.json` | ✅ 완료 |
| T2 | 평면 경계 → 닫힌 폴리곤 | `geometry/build_boundary.py`, `polygon_merge.py` → `road_boundary.json`, `outputs/road_boundary.png` | ✅ 완료 (v3 도면 재작업) |
| T3 | 경사 맵(grade field) | `geometry/grade_field.py` → `grade_at`/`grade_ahead` | 🟥 회귀 버그 (§3, §4) |
| T4 | Kinematic Bicycle 모듈 | `kinematics/bicycle.py` → `BicycleModel` (PHASE1 δ_max 보정 반영) | ✅ 완료 · 재검증됨 |
| T5 | Gymnasium 환경 | `envs/bus_hairpin_env.py` → `BusHairpinEnv` (PHASE2 스윕충돌 반영) | ✅ 완료 · 재검증됨 |
| T6 | 환경 정합성 검증 | `tests/manual_drive_demo.py` → `outputs/manual_drive_demo.png` | ✅ 완료 |
| T7 | 보상 함수 | `config.yaml`의 `reward:`, `tests/test_reward.py` (PHASE3 반영) | ✅ 완료 |
| T8 | PPO 학습 | `train_ppo.py` → `outputs/bus_hairpin_policy.zip`, `reward_curve.png`, `vecnormalize.pkl`, `tb_logs/` | 🟥 구도면 기준 1차만 완료, 재학습 대기 |
| T9 | 도메인 랜덤화 | `bus_hairpin_env.py` curriculum(선반영), 장애물 랜덤화는 의도적 생략 | 🟨 일부 선반영 |
| T10 | 정책 평가 & 경로 추출 | `evaluate.py` → `outputs/policy_rollout.png`(구정책 기준) | ⬜ 경로 확정 후 재수행 |
| T11 | 마킹 도면 출력 | `road_marking_design.dxf`, `analysis_dashboard.pdf` | ⬜ 미착수 (§4 배치안 설계 완료) |

### 보조 모듈 — Hybrid A* 경로 플래너

`planning/hybrid_astar.py` + `tests/run_hybrid_astar.py`. TASK 명세에는 없던 보조 도구로, 손튜닝 pure-pursuit+K턴 컨트롤러가 좁은 훅 노치를 못 넘어서 도입했다. 뒷축 (x,y,θ) 그리드 위에서 R_min을 정확히 지키는 전/후진 원호 프리미티브로 탐색한다. 현재 `geometry/reference_path.json`은 아직 이 플래너가 찾은 결과가 아니라 이전 세션의 잠정 경로이며, §3·§4에서 다루는 블로커가 풀려야 정식으로 갱신된다.

---

## 2. 기술 스택

Python 3.11.5, 프로젝트 로컬 venv(`.venv/`). 핵심 의존성 9개 + 그 전이 의존성까지 포함해 전체 43개 패키지가 설치되어 있다(`pip freeze` 기준). 역할별로 묶어 정리한다.

**RL 코어**
- `gymnasium` 1.3.0
- `stable_baselines3` 2.9.0
- `torch` 2.12.1 — SB3의 정책망(MlpPolicy, net_arch=[256,256]) 백엔드. CPU 전용(8코어, GPU 없음)으로 학습.

**기하 / CAD**
- `shapely` 2.1.2 — 폴리곤 충돌·clearance 전 구간
- `ezdxf` 1.4.4 — DXF 파싱(TASK1) 및 향후 마킹 도면 출력(TASK11)

**수치 / 과학계산**
- `numpy` 2.4.6
- `sympy` 1.14.0, `networkx` 3.6.1 — torch·ezdxf의 전이 의존성(직접 호출 없음)

**시각화 / 로깅**
- `matplotlib` 3.11.0 (전이: contourpy, cycler, fonttools, kiwisolver, pillow, pyparsing, python-dateutil)
- `tensorboard` 2.20.0 (전이: tensorboard-data-server, grpcio, protobuf, absl-py, Markdown, Werkzeug)

**설정 / 직렬화**
- `PyYAML` 6.0.3 — `config.yaml` 로딩 전 모듈 공통
- `cloudpickle` 3.1.2 — SB3 VecNormalize/모델 저장 의존성

**테스트**
- `pytest` 9.1.1 (전이: pluggy, iniconfig, packaging)

**그 외 전이 의존성 (직접 호출 없음, 참고용)**
colorama, Farama-Notifications(gymnasium), filelock, fsspec, Jinja2, MarkupSafe, mpmath(torch/sympy), rich·markdown-it-py·mdurl·Pygments(rich 렌더링 체인), six, tqdm, typing_extensions.

> **설치 관련 이슈**: **Reeds-Shepp 거리 계산 패키지**는 아직 미설치 — Windows에서 `pip install` 시 setup.py를 cp949 코덱으로 읽으려다 실패하는 인코딩 버그를 겪었다(`RL_NEXT_STEPS.md` PHASE A에서 `set PYTHONUTF8=1` 후 재시도 예정). 같은 이유로 이 프로젝트의 진단·검증 스크립트(`delta_max_sensitivity.py` 등) 실행 시에도 한글 출력이 깨지는 걸 막기 위해 `PYTHONUTF8=1`을 매번 앞에 붙여 실행하고 있다.

---

## 3. 시행착오 및 디버깅 기록

발생 순서대로. 각 항목은 "무엇이 실패했고 → 왜였고 → 어떻게 고쳤는지"로 구성.

**1차 PPO 학습 (구 도면) — 절대거리 페널티 → 퇴화 정책 (ep_len_mean=13, 성공률 0%)**
목표까지 절대거리에 매 스텝 페널티를 매기던 초기 보상 설계는, 충돌 페널티가 유한하면 몇 스텝만 방황해도 누적 페널티가 그걸 넘어서 "빨리 충돌해 끝내는" 정책이 더 유리해지는 구조였다. → potential-based shaping(이전 스텝 대비 가까워진 정도만 보상)으로 재설계.

**3M steps, ramp_fraction=0.6 — 커리큘럼 학습 중 성공률 100%→0% 붕괴, 회복 못함**
쉬운 구간에서 성공률 100%까지 올랐지만, 난이도가 실제 START까지 차오르는 시점(1.8M step)에서 정책이 정지 행동(action≈0)에 갇혀 여유 공간을 소진하고 충돌하는 패턴으로 무너짐 — 분포 밖 입력에 대한 붕괴로 추정. → `ramp_fraction 0.6→0.85`, `total_timesteps 3M→5M`로 완만하게 조정, 매 에피소드 난이도를 구간 내 무작위로 섞는 방식 도입.

**2026-07-02 — road_model.dxf v3(통합 도면) 교체 → 지오메트리 전면 재작업**
3개로 나뉘어 있던 도면이 하나로 통합되며 레이어명·경계 형상이 바뀜. 새 시작/목표 포즈를 "코너에서 1m+오버행 인셋"처럼 naive하게 잡았더니 실제로는 버스 전체 OBB가 충돌 — 실제 경계 폴리곤 위에서 전체 OBB가 완전히 들어가는 지점을 직접 탐색해 재확정(여유 ≈0.70m/≈1.13m).

**경로 탐색 1차 시도 — 수동 컨트롤러·기하학적 원호 구성 모두 훅 노치에서 55–80° 회전에 막힘**
pure-pursuit+K턴 손튜닝 컨트롤러(`plan_reference_path.py`)와 원호 직접 구성(`plan_arcs.py`) 둘 다 동일한 지점에서 막혀, 컨트롤러 튜닝 문제가 아니라 기하 자체의 한계라고 판단 → Hybrid A* 그리드 탐색으로 전환.

**Hybrid A* 1차 구현 — 디버그 트래커가 위치만 추적, "근접 실패"와 "막다른 길"을 구분 못함**
탐색이 실패로 종료될 때 "목표에 위치상 가장 가까웠던 노드"만 기록해, 위치는 가깝지만 방향이 전혀 다른 노드를 최적으로 오인할 소지가 있었다. → 위치 오차와 방향 오차를 결합한 점수(`dpos/tol_pos + dhead/tol_heading`)로 별도 추적하도록 수정.

**바퀴 기하 재검토 피드백 반영 — δ_max 과소평가 발견 및 보정, 스윕 충돌 근사 도입·검증**
공표 R_min(바깥 앞바퀴 자국 기준)을 뒷축 반경으로 직접 써온 것이 오류였음을 확인 — 보정 후 δ_max 30.92°→40.48°(+9.56°). 충돌 판정도 단일 시점 OBB에서 스텝 사이 4-서브스텝 보간 + 5점 clearance로 확장, 이론 스윕 반경과 대조해 오차 <5mm로 검증(자세한 수치는 `RL_WHEEL_GEOMETRY_REPORT.md`).

**δ_max 보정 후 재탐색 — 여전히 경로 없음, 그러나 이번엔 원인이 명확함**
위치는 0.345m까지 근접했지만 heading이 178.5° 어긋남(사실상 정반대). 40만 회 예산 중 4,911회·닫힌 셀 1,966개에서 탐색공간 자체가 고갈 — 예산 부족이 아니라 현재 프리미티브로 갈 수 있는 곳을 다 가본 상태. "조금만 더 찾으면 된다"가 아니라 **탐색 전략을 바꿔야 하는 문제**로 판정, 무한 파라미터 스윕 대신 여기서 멈추고 보고하는 쪽을 택함.

**검증 중 발견 (이번 작업과 무관) — grade_field 회귀 버그, v3 이관 세션의 미해결 잔여물**
`GradeField.grade_at()`이 위치와 무관하게 거의 같은 값(≈0.051)을 반환하고, `test_reward.py`의 `LOWER_PT` 테스트 좌표가 이미 새 도로 경계 밖에 있음을 확인. 단일 상태 충돌검사(보간 없이)만으로도 재현돼 이번 스윕충돌 변경과는 무관함을 직접 검증. 경사비례 보상이 이 함수에 의존하므로 재학습 전 수정 필수 — `RL_NEXT_STEPS.md` PHASE B로 이관.

---

## 4. 구현 현황 및 이후 일정 계획

| 영역 | 상태 | 비고 |
|---|---|---|
| 제원 라벨링 / vehicle_spec.json | ✅ 완료 | R_min·T_f는 `estimated` 라벨 유지 (자기인증 제원표 미확보) |
| δ_max 재유도 | ✅ 완료·검증 | 30.92°→40.48°, 민감도 스크립트로 근거 문서화 |
| 스윕 포락선 충돌/clearance | ✅ 완료·검증 | 이론 대비 오차 <5mm |
| 보상 재보정 | ✅ 완료 | `boundary_margin_m`은 자리만, 실측 전(Δ=0) |
| Hybrid A* 경로탐색 | 🟥 블로커 | 탐색공간 고갈, 전략 전환 필요 (PHASE A) |
| grade_field 회귀 | 🟥 미수정 | 무관 기존 버그, 재학습 전 필수 (PHASE B) |
| PPO 재학습 | ⬜ 대기 | 위 두 블로커 해소 후 (PHASE C) |
| Vehicle Tracking 외부 검증 | ⬜ 미착수 | 외부 CAD 툴 필요 (PHASE D) |

### 최종 산출물 배치안 — road_marking_design.dxf + 3D 검증 결과

TASK 11의 목표는 "학습된 정책 → 결정론적 최적 경로 → 실제 시공 가능한 마킹 도면"이다. 아래는 그 최종 패키지가 실제로 어떤 레이어·요소로 구성되고, Vehicle Tracking에서 나온 3D 스윕 검증 결과를 어떻게 배치해 보여줄지에 대한 구체안이다.

**배치 원칙은 세 겹이다.**

1. **바닥층 — 실측 도면.** 기존 `road_boundary` 경계선(회색 굵은 선, 실제 도로 폭)을 그대로 깔아, 마킹이 실제 시공 도면 위에 얹히는 모습을 그대로 보여준다.
2. **중간층 — 경로·마킹.** `optimal_path.csv`의 front-center 좌표를 중심선으로, FWD 구간은 흰 점선(실제 도색색), REV 구간은 대비색(주황/노랑 점선)으로 구분해 그린다. 조향전환점(S1, S2…)은 원형 마커+라벨, clearance 경고(≤1.0m)/위험(≤0.5m) 구간은 반투명 해치 폴리곤으로 얹는다.
3. **주석층 — 3D 검증 인셋.** Vehicle Tracking에서 나온 3D 버스 모델을 대표 지점(시작·각 조향전환점·정점 통과·목표 도달)에 배치해 캡처한 스냅샷을, 해당 지점 옆에 확대 인셋으로 붙인다 — 같은 도면 위에 여러 시점의 버스 위치를 겹쳐 보여주는 대신, 실제 시공 판단에 필요한 지점만 골라 확대해 clearance를 눈으로 확인시키는 방식. 스윕 포락선(회전 중 실제로 쓸고 간 면적)은 반투명 채움으로 깔아 도로 경계와 얼마나 겹치는지/여유가 있는지를 같은 화면에서 대조한다.

이 세 겹을 `simulation.dxf`(Vehicle Tracking 입력/검증용, CAD에서 열림)와 `road_marking_design.dxf`(실제 시공 마킹용, 위 세 겹 중 바닥층+중간층만) 두 파일로 분리해 낸다. 여기에 `analysis_dashboard.pdf` 한 장을 더해, 상단에는 위 평면도+3D 인셋 조합, 하단에는 `reward_curve.png`(학습곡선)와 구간별 clearance 최소값 라인차트(경고/위험 구간 음영)를 나란히 배치한다 — "AI가 학습했다는 증거"와 "실제로 안전하게 통과 가능하다는 증거"를 한 페이지에서 같이 보여주는 구성이다. 우측 여백에는 R_min/T_f/Δ 등 추정값·근사 라벨 요약 박스를 둬 정직성 기준(`RL_WHEEL_GEOMETRY_REPORT.md`)을 문서 전체에 유지한다.

### 향후 일정 (PHASE A–E, `RL_NEXT_STEPS.md` 승계)

| Phase | 내용 | 선행조건 | 상태 |
|---|---|---|---|
| A | Hybrid A* 재시도 — 양방향 탐색 / Reeds-Shepp 휴리스틱 / 수동 K턴 시드 | 없음, 최우선 | 🟥 다음 착수 |
| B | grade_field 회귀 버그 수정 | A와 독립적으로 진행 가능 | 🟥 다음 착수 |
| C | PPO 재학습 (5M steps, ramp_fraction=0.85) | A·B 완료 | ⬜ 대기 |
| D | Vehicle Tracking 외부 검증, Δ 실측, 위 배치안 실제 제작 | C 완료 | ⬜ 대기 |
| E | 문서화·정직성 라벨링 갱신 | 병렬 진행 가능 | 🟨 상시 |

> **분기 조건**: PHASE A가 "기하학적으로 불가능"으로 판정되면 B~D는 현재 도로 형상 기준으로 진행할 수 없다 — 이 경우 개념설계 단계로 돌아가 회차 구간 형상(훅 노치 폭 등) 자체를 재검토해야 하므로, 그 판정이 나오는 즉시 추가 코드 작업 전에 먼저 보고한다.

---
*Intra_Imprv / simulation — `RL_BUILD_INSTRUCTIONS.md`·`RL_WHEEL_GEOMETRY_REPLAN.md`·`RL_NEXT_STEPS.md` 종합. 상세 수치 근거는 각 문서 참조.*
