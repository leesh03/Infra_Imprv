"""RL_WHEEL_GEOMETRY_REPLAN.md PHASE 2 검증: 최대 조향으로 도는 대표 회전
자세에서, 촘촘히 샘플링한 버스 폴리곤들의 합집합(스윕 폭)이 이론식
R_out - R_in과 일치하는지 대조한다. 학습/실행에 개입하지 않는 진단 스크립트."""
import math
from pathlib import Path

import yaml
from shapely.ops import unary_union

from envs.bus_hairpin_env import BusHairpinEnv
from kinematics.bicycle import BicycleState

SIM_DIR = Path(__file__).resolve().parent.parent


def main():
    env = BusHairpinEnv()
    model = env.model
    R, L, W = model.R, model.L, env.half_width * 2
    # env.front_oh(=d_ref)는 이미 "뒷축 -> 앞면" 전체 거리(L + 전방 오버행)다.
    # RL_WHEEL_GEOMETRY_REPLAN.md 식의 (L + f_oh)와 동일한 양이므로 그대로 쓴다
    # (L을 다시 더하면 축거를 이중으로 반영하는 오류가 됨).
    d_ref = env.front_oh

    R_out_theory = math.sqrt((R + W / 2) ** 2 + d_ref ** 2)
    R_in_theory = R - W / 2

    # 원점에서 최대 조향으로 1/4바퀴(90도) 회전하는 궤적을 촘촘히 샘플링
    state = BicycleState(x=0.0, y=0.0, theta=0.0, v=model.v_max, delta=model.delta_max)
    polys = []
    n_steps = 400
    quarter_turn_time = (math.pi / 2) * R / model.v_max
    dt_sim = quarter_turn_time / n_steps
    orig_dt = model.dt
    model.dt = dt_sim
    for _ in range(n_steps):
        polys.append(env._bus_polygon(state))
        state = model.step(state, a_accel=0.0, a_steer=0.0)
    model.dt = orig_dt

    swept = unary_union(polys)
    cx, cy = 0.0, R  # 회전 중심(뒷축 기준 좌회전 시 (0, R))
    coords = list(swept.exterior.coords)
    dists = [math.hypot(px - cx, py - cy) for px, py in coords]
    R_out_measured = max(dists)
    R_in_measured = min(dists)

    print(f"R(뒷축 보정 반경)={R:.3f}m  L={L}m  W={W}m  d_ref(뒷축->앞면)={d_ref:.3f}m")
    print(f"{'항목':20s} {'이론값[m]':>12s} {'폴리곤 실측[m]':>14s} {'오차[m]':>10s}")
    print(f"{'R_out (바깥 스윕)':20s} {R_out_theory:12.3f} {R_out_measured:14.3f} {abs(R_out_theory - R_out_measured):10.3f}")
    print(f"{'R_in (안쪽 스윕)':20s} {R_in_theory:12.3f} {R_in_measured:14.3f} {abs(R_in_theory - R_in_measured):10.3f}")
    print(f"스윕 폭(R_out - R_in): 이론 {R_out_theory - R_in_theory:.3f}m vs 실측 {R_out_measured - R_in_measured:.3f}m"
          f" (전폭 {W}m 대비 {(R_out_theory - R_in_theory) - W:+.3f}m 더 넓게 쓸고 감)")


if __name__ == "__main__":
    main()
