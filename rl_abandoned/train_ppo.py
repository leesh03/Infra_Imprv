"""TASK 8 — PPO 베이스라인 학습 스크립트.

도로 전체(경사로 포함)로 확장되며 한 에피소드 길이가 600->1800 step으로
늘어났고, 과제 자체도 apex 인근 좁은 통로를 통과해야 해서 더 어려워졌다.
이에 맞춰(도로 확장 작업 중 재조정):
  - n_steps/batch_size를 키워 업데이트당 더 많은 에피소드를 봄
  - gamma를 0.995로 올려 더 긴 horizon의 신용 할당을 안정화
  - policy net을 키움(256,256) — 관측 차원이 늘어났고(센서 12->16+경로진행률 등) 과제가 복잡해짐
  - total_timesteps 기본값을 키움 — 로컬(8코어 CPU, GPU 없음) 기준 벤치마크
    (약 650 env-step/s, n_envs=8) 그대로면 5,000,000 steps에 약 2시간 소요.
    더 길게/짧게 학습하려면 --total-timesteps로 조정.

장애물(불법주정차 등) 도메인 랜덤화는 사용하지 않음 — 순수 도로 형상/경사만
고려 (TASK9 중 장애물 랜덤화 항목은 의도적으로 생략, 사용자 확인됨). 대신
시작 자세 커리큘럼(GOAL 근처 쉬운 자세 -> 실제 START)과 시작 자세 노이즈는
TASK9 선반영으로 CurriculumCallback이 처리한다.

> 1차 학습(3M steps, ramp_fraction=0.6) 결과: 커리큘럼이 쉬운 구간일 때는
> 성공률 100%까지 올라갔지만, 0.6*3M=1.8M step 부근에서 난이도가 실제
> START까지 차오르자 성공률이 0%로 무너진 채 회복 못함 — 정책이 거의 정지
> 행동(action≈0)에 갇혀 START 근처 여유 공간을 다 까먹고 충돌. 난이도 전환이
> 너무 가팔랐던 것으로 보여 ramp_fraction을 0.6->0.85로 늦추고
> total_timesteps를 5M으로 늘림 (정책이 각 난이도 단계에 적응할 시간을 더 줌).
"""
import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

from envs.bus_hairpin_env import BusHairpinEnv

SIM_DIR = Path(__file__).resolve().parent


def make_env_fn():
    def _f():
        return Monitor(BusHairpinEnv())
    return _f


class EpisodeStatsCallback(BaseCallback):
    """Monitor가 info["episode"]에 채워주는 에피소드 보상/성공 여부를 모아
    reward_curve.png로 그릴 수 있게 기록."""

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_timesteps = []
        self.success_flags = []

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            ep = info.get("episode")
            if ep is not None:
                self.episode_rewards.append(ep["r"])
                self.episode_timesteps.append(self.num_timesteps)
                self.success_flags.append(bool(info.get("success", False)))
        return True


class CurriculumCallback(BaseCallback):
    """학습 진행도에 맞춰 시작 자세를 GOAL 근처(쉬움)에서 실제 START(어려움)로
    점진적으로 늘리고, 동시에 시작 자세에 약간의 가우시안 노이즈를 더해
    강건성을 키운다 (TASK 9 선반영). curriculum_level은 ramp_fraction 시점까지
    선형으로 0.1->1.0 증가한 뒤 1.0으로 고정.
    env_method는 SubprocVecEnv IPC 비용이 있어 매 스텝이 아니라 update_freq
    스텝마다만 갱신한다."""

    def __init__(self, total_timesteps, ramp_fraction=0.85, update_freq=2048, verbose=0):
        super().__init__(verbose)
        self.total_timesteps = total_timesteps
        self.ramp_fraction = ramp_fraction
        self.update_freq = update_freq
        self._last_update = -update_freq

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_update < self.update_freq:
            return True
        self._last_update = self.num_timesteps

        progress = min(1.0, self.num_timesteps / max(1, self.total_timesteps * self.ramp_fraction))
        level = 0.1 + 0.9 * progress
        self.training_env.env_method("set_curriculum_level", level)
        self.training_env.env_method("set_start_noise", level * 0.5, level * 0.08)
        if self.verbose:
            print(f"[curriculum] t={self.num_timesteps} level={level:.3f}")
        return True


def main():
    # 2026-07-06: 8논리코어 머신에서 SubprocVecEnv n_envs=8을 그대로 쓰면
    # 메인 프로세스의 PyTorch가 기본적으로 코어 수만큼 스레드를 잡아먹어
    # 8개 env 워커 프로세스와 CPU를 두고 경합한다(관측된 실제 fps가 8배
    # 병렬화 기대치에 한참 못 미쳤음 — 단일 프로세스 fps와 거의 같았음).
    # 메인 프로세스의 torch 스레드를 1로 제한해 env 워커에 코어를 양보한다.
    import torch
    torch.set_num_threads(1)

    parser = argparse.ArgumentParser()
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--total-timesteps", type=int, default=5_000_000)
    parser.add_argument("--n-steps", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--gamma", type=float, default=0.995)
    parser.add_argument("--ent-coef", type=float, default=0.01,
                         help="탐색 유지용 엔트로피 계수. 커리큘럼으로 매 에피소드 난이도가 섞여 나오므로"
                              " (멀리/가까이 모두 등장) 정책이 한쪽으로 과확신해 행동이 무너지는 것을 막는 용도.")
    parser.add_argument("--no-curriculum", action="store_true", help="커리큘럼/시작자세 노이즈 끄고 항상 실제 START에서 시작")
    parser.add_argument("--fixed-curriculum-level", type=float, default=None,
                         help="2026-07-07 진단용: CurriculumCallback의 램프업 없이 난이도 하한을 이 값(0=GOAL"
                              " 근처, 1=실제 START)으로 고정하고 시작자세 노이즈도 끈다. --fixed-curriculum-ceiling"
                              " 없이 이것만 주면 상한도 같은 값으로 고정돼(무작위 폭 0) 항상 정확히 이 난이도로만"
                              " 나온다. 5M-step 학습이 성공률 0%로 나온 원인이 커리큘럼 램프 속도/난이도 자체인지,"
                              " 아니면 가장 쉬운 조건에서도 애초에 성공을 못 배우는지 구분하기 위함.")
    parser.add_argument("--fixed-curriculum-ceiling", type=float, default=None,
                         help="--fixed-curriculum-level과 같이 써서 난이도 샘플링 상한을 지정(기본은 하한과"
                              " 동일 = 무작위 폭 0). 예: --fixed-curriculum-level 0.0 --fixed-curriculum-ceiling 0.15"
                              " -> 항상 GOAL 근처~약간 어려운 정도까지만 나오게 강제.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--outputs-dir", default="outputs",
                         help="모델/곡선 저장 위치 (SIM_DIR 기준 상대경로 또는 절대경로)."
                              " 스모크 테스트 등으로 실제 학습 결과물을 덮어쓰지 않으려면 별도 경로 지정.")
    args = parser.parse_args()

    venv = SubprocVecEnv([make_env_fn() for _ in range(args.n_envs)])
    venv = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10.0)
    if args.fixed_curriculum_level is not None:
        ceiling = args.fixed_curriculum_ceiling if args.fixed_curriculum_ceiling is not None else args.fixed_curriculum_level
        venv.env_method("set_curriculum_level", args.fixed_curriculum_level)
        venv.env_method("set_curriculum_ceiling", ceiling)
        venv.env_method("set_start_noise", 0.0, 0.0)

    outputs_dir = Path(args.outputs_dir)
    if not outputs_dir.is_absolute():
        outputs_dir = SIM_DIR / outputs_dir

    model = PPO(
        "MlpPolicy",
        venv,
        verbose=1,
        seed=args.seed,
        gamma=args.gamma,
        learning_rate=3e-4,
        clip_range=0.2,
        ent_coef=args.ent_coef,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=10,
        policy_kwargs=dict(net_arch=[256, 256]),
        tensorboard_log=str(outputs_dir / "tb_logs"),
    )

    callback = EpisodeStatsCallback()
    callbacks = [callback]
    if not args.no_curriculum and args.fixed_curriculum_level is None:
        callbacks.append(CurriculumCallback(args.total_timesteps, verbose=1))
    model.learn(total_timesteps=args.total_timesteps, callback=callbacks, progress_bar=True)

    outputs_dir.mkdir(parents=True, exist_ok=True)
    model.save(str(outputs_dir / "bus_hairpin_policy.zip"))
    venv.save(str(outputs_dir / "vecnormalize.pkl"))

    _plot_reward_curve(callback, outputs_dir / "reward_curve.png")
    venv.close()


def _plot_reward_curve(callback: EpisodeStatsCallback, save_path):
    if not callback.episode_rewards:
        print("기록된 에피소드가 없어 reward_curve.png를 생성하지 않음")
        return

    rewards = np.array(callback.episode_rewards)
    steps = np.array(callback.episode_timesteps)
    success = np.array(callback.success_flags, dtype=float)

    window = max(1, len(rewards) // 50)
    kernel = np.ones(window) / window
    smoothed = np.convolve(rewards, kernel, mode="valid")
    success_rate = np.convolve(success, kernel, mode="valid")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
    ax1.plot(steps, rewards, alpha=0.25, color="tab:blue")
    ax1.plot(steps[: len(smoothed)], smoothed, color="tab:blue", label=f"reward (movavg {window})")
    ax1.set_ylabel("episode reward")
    ax1.legend()

    ax2.plot(steps[: len(success_rate)], success_rate, color="tab:green", label="success rate (movavg)")
    ax2.set_xlabel("timesteps")
    ax2.set_ylabel("success rate")
    ax2.set_ylim(-0.05, 1.05)
    ax2.legend()

    fig.suptitle("PPO training curve (TASK 8)")
    fig.tight_layout()
    fig.savefig(save_path, dpi=140)
    print(f"저장: {save_path}")


if __name__ == "__main__":
    main()
