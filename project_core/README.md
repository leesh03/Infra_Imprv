# project_core — 공용 토대 (🔴 삭제·분산 금지)

5단계 경로계획 전부가 공유하는 기반 코드/데이터입니다. 최종 채택 경로
(keepright)도 이 코드·데이터로 생성·검증·DXF 추출됐으므로, 이 폴더는
지우거나 내부 구조를 흩뜨리면 안 됩니다.

## 구성 (원래 `simulation/` 패키지 구조를 그대로 보존)
- `envs/` — `bus_hairpin_env.py`(환경), `reference_path.py`, `sensors.py`(env가 import), `grade_field`는 geometry에
- `kinematics/` — `bicycle.py`(자전거 모델)
- `geometry/` — DXF 파서·경계 빌더(`dxf_parser`, `build_boundary`, `build_geometry`, `polygon_merge`, `grade_field`)와 산출 데이터(`road_boundary.json`, `grades.json`, `obstacles.json`, `raw_plan.json`)
- `config/vehicle_spec.json`, `config.yaml` — 차량 제원·경로·DXF 레이어 설정
- `tests/` — 토대 단위테스트(`test_bicycle.py`, `test_grade_field.py`)
- `scenarios/`, `planning/` — 패키지 골격(플래너 구현체는 각 단계 폴더로 이동됨)

## 주의
- `config.yaml`이 `../data/raw/cad/road_model.dxf`를 상대경로로 참조합니다.
  그래서 원본 CAD(`data/raw/cad/`)는 저장소 **루트에** 두어야 하고, 이 폴더는
  루트 바로 아래(`project_core/`)에 있어야 그 경로가 맞습니다.
