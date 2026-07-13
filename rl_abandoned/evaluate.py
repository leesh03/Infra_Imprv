"""TASK 10 1단계 — 학습된 PPO 정책을 결정론적으로 굴려 실제 주행 경로를 PNG로
확인. (steer_points.csv/clearance_profile.csv 등 전체 산출물은 이후 단계.)

사용:
    python evaluate.py                         (기본: 실제 START, 노이즈 없음)
    python evaluate.py --max-steps 2200
"""
import argparse
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from envs.bus_hairpin_env import BusHairpinEnv

SIM_DIR = Path(__file__).resolve().parent


def make_env():
    return BusHairpinEnv()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-steps", type=int, default=1800)
    parser.add_argument("--out", default="outputs/policy_rollout.png")
    args = parser.parse_args()

    outputs_dir = SIM_DIR / "outputs"
    # VecNormalize는 학습 때 쓴 관측 정규화 통계(obs_rms)를 그대로 재현하려고만
    # 쓴다. 실제 step/reset/render는 DummyVecEnv의 자동 reset(done이면 다음
    # step() 호출 전에 내부적으로 reset() 해버려 충돌 직후 상태를 못 봄)을
    # 피하려고 래핑되지 않은 raw env를 직접 제어한다.
    venv = DummyVecEnv([make_env])
    venv = VecNormalize.load(str(outputs_dir / "vecnormalize.pkl"), venv)
    venv.training = False
    venv.norm_reward = False
    env = venv.envs[0]

    # 평가는 항상 실제 START에서, 노이즈 없이 (학습 중 커리큘럼이 마지막으로
    # 남긴 값과 무관하게 고정).
    env.set_curriculum_level(1.0)
    env.set_start_noise(0.0, 0.0)

    model = PPO.load(str(outputs_dir / "bus_hairpin_policy.zip"))

    obs, info = env.reset()
    terminated = truncated = False
    t = 0
    for t in range(args.max_steps):
        norm_obs = venv.normalize_obs(obs[None, :])
        action, _ = model.predict(norm_obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action[0])
        if terminated or truncated:
            break

    status = "terminated" if terminated else ("truncated" if truncated else "max_steps 도달")
    print(f"step={t+1} {status} info={info}")

    fig = env.render()
    out_path = SIM_DIR / args.out
    fig.savefig(out_path, dpi=150)
    print(f"저장: {out_path}")


if __name__ == "__main__":
    main()
