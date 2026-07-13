"""CMA-ES 기반 K-구간 기동 플래너 실행 — PPO(train_ppo.py) 대신 미분-프리
진화적 최적화로 "전/후진 반복을 최소화하는" 최종 기동을 직접 찾는다.

제약조건(경계 폴리곤, 버스 풋프린트, R_min 기반 곡률한계, 목표는 반드시
전진으로 도착)은 하이브리드 A*/RL이 쓴 것과 동일하며, 방법만 그리드 탐색·
정책학습에서 CMA-ES로 바꿨다.

사용:
    python tests/run_evolutionary_planner.py
    python tests/run_evolutionary_planner.py --k-max 8 --maxiter 150 --n-jobs 4
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

from envs.bus_hairpin_env import BusHairpinEnv
from kinematics.bicycle import front_center
from planning.evolutionary import MIN_SEG_LEN, refine_from_seed, search, segments_from_path_json

SIM_DIR = Path(__file__).resolve().parent.parent


def build_ctx_kwargs(env):
    return dict(
        start_pose=(env.start_x, env.start_y, env.start_heading),
        goal_x=env.goal_x, goal_y=env.goal_y, goal_heading=env.goal_heading,
        tol_pos=env.tol_pos, tol_heading=env.tol_heading,
        footprint=(env.front_oh, env.rear_oh, env.half_width),
        d_ref=env.d_ref, L=env.model.L,
        kappa_max=1.0 / env.model.R, len_max=50.0,
        boundary_polygon=env.boundary_polygon, obstacle_polygons=env.obstacle_polygons,
        grade_field=env.grade_field, reference_path=env.reference_path,
        w_reversal=env.rcfg["w_reversal"], reversal_slope_k=env.rcfg["reversal_slope_k"],
    )


def describe_plan(segments):
    lines = []
    step = 0
    for kappa, length in segments:
        if abs(length) < MIN_SEG_LEN:
            continue
        step += 1
        gear = "FWD" if length > 0 else "REV"
        if abs(kappa) < 1e-6:
            shape = "straight"
        else:
            shape = f"R={1.0 / abs(kappa):.2f}m {'left' if kappa > 0 else 'right'}"
        lines.append(f"  {step:2d}. {gear:3s}  len={abs(length):6.2f}m  {shape}")
    return lines


def save_path_json(meta, env, out_path):
    waypoints, gears, rear_waypoints = [], [], []
    for (x, y, theta, gear) in meta["dense_path"]:
        fc = front_center(x, y, theta, env.d_ref)
        waypoints.append([round(fc[0], 3), round(fc[1], 3)])
        gears.append(gear if gear != "START" else "FWD")
        rear_waypoints.append([round(x, 3), round(y, 3), round(theta, 6)])
    out = {"waypoints": waypoints, "gears": gears, "rear_waypoints": rear_waypoints}
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)


def plot_path(env, meta, success, out_path):
    wp = meta["dense_path"]
    fig, ax = plt.subplots(figsize=(11, 9))
    px, py = env.boundary_polygon.exterior.xy
    ax.plot(px, py, "k-", linewidth=1)

    for i in range(1, len(wp)):
        color = "tab:red" if wp[i][3] == "REV" else "tab:blue"
        ax.plot([wp[i - 1][0], wp[i][0]], [wp[i - 1][1], wp[i][1]], color=color, linewidth=2.0)
    ax.plot([], [], color="tab:blue", linewidth=2.5, label="FWD (forward)")
    ax.plot([], [], color="tab:red", linewidth=2.5, label="REV (reverse)")

    ax.plot(wp[0][0], wp[0][1], "g^", markersize=12, label="start (rear axle)")
    ax.plot(wp[-1][0], wp[-1][1], "m*", markersize=16, label="path end (rear axle)")
    ax.plot(env.goal_x, env.goal_y, "kx", markersize=14, mew=2, label="goal target (front-center)")

    status = "SUCCESS" if success else "FAILED (best effort)"
    ax.set_title(
        f"CMA-ES maneuver: {status}  switches={meta['n_switches']}  "
        f"len={meta['total_length']:.1f}m  violation={meta['violation']:.4f}m^2\n"
        f"goal_pos_err={meta['goal_pos_err']:.3f}m  "
        f"goal_heading_err={math.degrees(meta['goal_heading_err']):.2f}deg"
    )
    ax.set_aspect("equal")
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    print(f"저장: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["seed", "scratch"], default="seed",
                         help="seed: 기존 geometry/reference_path.json(hybrid A*)을 CMA-ES 웜스타트로 "
                              "정제(권장, 좁은 헤어핀 포켓을 처음부터 다시 뚫지 않음). "
                              "scratch: 완전히 무작위로 K=1부터 늘려가며 새로 탐색.")
    parser.add_argument("--seed-path", default="geometry/reference_path.json")
    parser.add_argument("--chunk-len", type=float, default=1.0,
                         help="seed 경로를 상수-곡률 조각으로 재근사할 때의 조각 길이[m] "
                              "(작을수록 재구성 정확도↑, 대신 K(=차원)↑)")
    parser.add_argument("--k-start", type=int, default=1)
    parser.add_argument("--k-max", type=int, default=12)
    parser.add_argument("--n-restarts", type=int, default=3)
    parser.add_argument("--maxiter", type=int, default=200)
    parser.add_argument("--sigma0", type=float, default=None)
    parser.add_argument("--box-eps", type=float, default=None,
                         help="(seed 모드) 각 유전자를 seed값 ± box-eps로 하드 클립 — "
                              "사슬형 민감도로 인한 CMA-ES 폭주 방지")
    parser.add_argument("--n-jobs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-json", default="geometry/cma_path.json")
    parser.add_argument("--out-png", default="outputs/cma_path.png")
    args = parser.parse_args()

    env = BusHairpinEnv()
    ctx_kwargs = build_ctx_kwargs(env)

    print(f"start={ctx_kwargs['start_pose']} goal=({env.goal_x},{env.goal_y},{env.goal_heading})")
    print(f"kappa_max={ctx_kwargs['kappa_max']:.4f} (R={env.model.R:.2f}m)  len_max={ctx_kwargs['len_max']}m")

    t0 = time.time()
    if args.mode == "seed":
        with open(SIM_DIR / args.seed_path, encoding="utf-8") as f:
            seed_json = json.load(f)
        seed_segments = segments_from_path_json(seed_json, chunk_len=args.chunk_len)
        n_seed_switches = sum(
            1 for i in range(1, len(seed_segments))
            if (seed_segments[i][1] > 0) != (seed_segments[i - 1][1] > 0)
        )
        print(f"웜스타트: {args.seed_path} -> K={len(seed_segments)} runs, "
              f"seed switches={n_seed_switches}, seed total_length={sum(abs(l) for _, l in seed_segments):.1f}m")
        refine_kwargs = dict(n_restarts=args.n_restarts, maxiter=args.maxiter,
                              seed0=args.seed, n_jobs=args.n_jobs)
        if args.sigma0 is not None:
            refine_kwargs["sigma0"] = args.sigma0
        if args.box_eps is not None:
            refine_kwargs["box_eps"] = args.box_eps
        K, xbest, meta, success = refine_from_seed(ctx_kwargs, seed_segments, **refine_kwargs)
    else:
        K, xbest, meta, success = search(
            ctx_kwargs, K_start=args.k_start, K_max=args.k_max, n_restarts=args.n_restarts,
            maxiter=args.maxiter, seed0=args.seed, n_jobs=args.n_jobs,
        )
    elapsed = time.time() - t0
    print(f"\nelapsed {elapsed:.1f}s  K={K}  success={success}")

    print("최종 기동 계획:")
    for line in describe_plan(meta["segments"]):
        print(line)

    print(
        f"violation={meta['violation']:.4f}m^2  goal_pos_err={meta['goal_pos_err']:.3f}m  "
        f"goal_heading_err={math.degrees(meta['goal_heading_err']):.2f}deg  "
        f"switches={meta['n_switches']}  total_length={meta['total_length']:.1f}m  "
        f"arrival_ok={meta['arrival_ok']}"
    )

    out_json = SIM_DIR / args.out_json
    out_png = SIM_DIR / args.out_png
    save_path_json(meta, env, out_json)
    print(f"저장: {out_json}")
    plot_path(env, meta, success, out_png)

    if not success:
        print(f"\n경고: K_max={args.k_max}까지 성공 조건을 만족하는 해를 못 찾음 — 위는 최선 시도.")


if __name__ == "__main__":
    main()
