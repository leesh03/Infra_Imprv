"""TASK 4 완료 기준: 최대 조향(±delta_max) 고정·일정속도로 한 바퀴 돌렸을 때
뒷축 궤적 반경이 R_min과 ±2% 내 일치하는지 검증."""
import math

import numpy as np

from kinematics.bicycle import BicycleModel, BicycleState, delta_max_from_Rmin


def _fit_circle_radius(xs, ys):
    """Kasa 알고리즘 (대수적 최소제곱 원 피팅) -> 반경."""
    xs, ys = np.asarray(xs), np.asarray(ys)
    A = np.column_stack([xs, ys, np.ones_like(xs)])
    b = xs ** 2 + ys ** 2
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    cx, cy = sol[0] / 2, sol[1] / 2
    r = math.sqrt(sol[2] + cx ** 2 + cy ** 2)
    return r


def test_delta_max_matches_Rmin():
    model = BicycleModel()
    assert math.isclose(model.delta_max, delta_max_from_Rmin(model.L, model.R_min))


def test_turning_radius_matches_Rmin():
    model = BicycleModel()
    state = BicycleState(x=0.0, y=0.0, theta=0.0, v=model.v_max, delta=model.delta_max)

    circumference = 2 * math.pi * model.R_min
    duration = circumference / model.v_max
    n_steps = int(duration / model.dt) + 1

    xs, ys = [state.x], [state.y]
    for _ in range(n_steps):
        state = model.step(state, a_accel=0.0, a_steer=0.0)
        xs.append(state.x)
        ys.append(state.y)

    fitted_radius = _fit_circle_radius(xs, ys)
    rel_err = abs(fitted_radius - model.R_min) / model.R_min
    assert rel_err <= 0.02, f"fitted radius {fitted_radius:.3f}m vs R_min {model.R_min}m (rel_err={rel_err:.3%})"
