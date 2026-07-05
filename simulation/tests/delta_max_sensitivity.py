"""RL_WHEEL_GEOMETRY_REPLAN.md PHASE 1 민감도 비교.

기존(오류) 공식 atan(L/R_min) vs 측정기준별 보정 공식의 delta_max·뒷축
회전반경을 표로 비교하고, 조향 여유가 얼마나 늘었는지(=훅 노치 통과
가능성에 대한 영향)를 출력한다. 학습/실행에 개입하지 않는 진단 스크립트."""
import math
from pathlib import Path

import yaml

from kinematics.bicycle import (
    corrected_rear_axle_radius,
    delta_max_from_Rmin,
    delta_max_from_Rmin_spec,
)

SIM_DIR = Path(__file__).resolve().parent.parent


def main():
    with open(SIM_DIR / "config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    bus = cfg["bus"]
    L = bus["wheelbase_L"]
    R_spec = bus["R_min"]
    T_f = bus.get("T_f")
    W = bus["width"]
    f_oh = bus["overhang_front"]

    rows = []

    old_delta = delta_max_from_Rmin(L, R_spec)
    rows.append(("기존(오류): atan(L/R_min), R=R_spec", math.degrees(old_delta), R_spec))

    if T_f is not None:
        d1, R1 = delta_max_from_Rmin_spec(L, R_spec, T_f=T_f, basis="outer_front_tire_track")
        rows.append(("보정: outer_front_tire_track (T_f 사용)", math.degrees(d1), R1))

    d2, R2 = delta_max_from_Rmin_spec(L, R_spec, W=W, f_oh=f_oh, basis="outer_body")
    rows.append(("보정: outer_body (W, f_oh 사용)", math.degrees(d2), R2))

    print(f"L(축거)={L}m, R_spec(공표 R_min)={R_spec}m, T_f={T_f}m, W={W}m, f_oh={f_oh}m")
    print(f"{'공식':45s} {'delta_max[deg]':>15s} {'뒷축 R[m]':>10s}")
    for name, deg, R in rows:
        print(f"{name:45s} {deg:15.2f} {R:10.3f}")

    base_deg = math.degrees(old_delta)
    print("\n기존 대비 조향 여유 증가폭:")
    for name, deg, R in rows[1:]:
        print(f"  {name}: +{deg - base_deg:.2f} deg (뒷축 반경 {R_spec - R:+.3f}m 만큼 더 작게 돎)")

    print(
        "\n해석: 두 보정식 모두 delta_max가 커지고 실제 뒷축 회전반경 R은 "
        "R_spec보다 작다 - 기존 모델은 실제보다 더 크게 도는 것으로 과소평가했다. "
        "훅 노치(폭 ~4.3m) 통과에는 조향 여유가 늘어나는 쪽이 유리하게 작용할 "
        "가능성이 있으나, 실제 통과 여부는 PHASE 4 Hybrid A* 재탐색으로 확인해야 한다."
    )


if __name__ == "__main__":
    main()
