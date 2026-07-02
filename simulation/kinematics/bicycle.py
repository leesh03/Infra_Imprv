"""Kinematic Bicycle Model — 환경의 물리 엔진 (TASK 4).

상태 (x, y, theta, v, delta)는 **뒷축 중심** 기준(§2.6). 적분 후 목표 판정에
필요하면 front_center()로 앞면 하단 중심 좌표를 별도 계산한다(절대 theta
적분에 앞면중심을 쓰지 않을 것).
"""
import math
from dataclasses import dataclass
from pathlib import Path

import yaml

SIM_DIR = Path(__file__).resolve().parent.parent


def delta_max_from_Rmin(L, R_min):
    return math.atan(L / R_min)


def front_center(x, y, theta, d_ref):
    return x + d_ref * math.cos(theta), y + d_ref * math.sin(theta)


def _clip(v, lo, hi):
    return max(lo, min(hi, v))


def bicycle_step(
    x, y, theta, v, delta, a_accel, a_steer, dt, L, v_min, v_max, delta_max,
    grade=0.0, use_dynamics_term=False, g=9.81,
):
    """단일 스텝 전진. a_accel[m/s^2]: 가감속, a_steer[rad/s]: 조향각속도.
    grade: 현재 위치의 종단 경사(분율, 예 0.0824). use_dynamics_term=True면
    -g*sin(atan(grade)) 항으로 경사 저항/가속을 물리적으로 반영(§TASK7 옵션).
    """
    a_eff = a_accel - g * math.sin(math.atan(grade)) if use_dynamics_term else a_accel

    x2 = x + v * math.cos(theta) * dt
    y2 = y + v * math.sin(theta) * dt
    theta2 = theta + (v / L) * math.tan(delta) * dt
    v2 = _clip(v + a_eff * dt, v_min, v_max)
    delta2 = _clip(delta + a_steer * dt, -delta_max, delta_max)
    return x2, y2, theta2, v2, delta2


@dataclass
class BicycleState:
    x: float
    y: float
    theta: float
    v: float = 0.0
    delta: float = 0.0


class BicycleModel:
    """config.yaml 파라미터를 읽어 step()을 제공하는 래퍼."""

    def __init__(self, config=None):
        if config is None:
            with open(SIM_DIR / "config.yaml", "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
        self.L = config["bus"]["wheelbase_L"]
        self.R_min = config["bus"]["R_min"]
        self.delta_max = delta_max_from_Rmin(self.L, self.R_min)
        self.d_ref = config["bus"]["d_ref"]
        self.dt = config["kinematics"]["dt"]
        self.v_min = config["kinematics"]["v_min"]
        self.v_max = config["kinematics"]["v_max"]
        self.a_max = config["kinematics"]["a_max"]
        self.use_dynamics_term = config["slope"]["use_dynamics_term"]
        self.g = config["slope"]["g"]

    def step(self, state: BicycleState, a_accel, a_steer, grade=0.0) -> BicycleState:
        x2, y2, theta2, v2, delta2 = bicycle_step(
            state.x, state.y, state.theta, state.v, state.delta,
            a_accel, a_steer, self.dt, self.L, self.v_min, self.v_max, self.delta_max,
            grade=grade, use_dynamics_term=self.use_dynamics_term, g=self.g,
        )
        return BicycleState(x2, y2, theta2, v2, delta2)

    def front_center(self, state: BicycleState):
        return front_center(state.x, state.y, state.theta, self.d_ref)
