"""국소 방향전환 제거 시도 실행 — 원본 reference_path.json(hybrid A*, 위반
0으로 이미 검증됨)의 28개 방향전환 경계 각각이 "제거 가능한가"를 독립적으로
검사한다. 원본 좌표는 건드리지 않고, 각 전환을 감싸는 두 런의 시작/끝 pose를
고정한 채 그 사이를 국소 CMA-ES로 다시 잇는다."""
import argparse
import json
import sys
import time
from pathlib import Path

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

from envs.bus_hairpin_env import BusHairpinEnv
from planning.switch_repair import repair_switches

SIM_DIR = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-path", default="geometry/reference_path.json")
    parser.add_argument("--k-local", type=int, default=3)
    parser.add_argument("--len-cap", type=float, default=15.0)
    parser.add_argument("--maxiter", type=int, default=250)
    parser.add_argument("--n-restarts", type=int, default=4)
    parser.add_argument("--n-jobs", type=int, default=None)
    args = parser.parse_args()

    env = BusHairpinEnv()
    with open(SIM_DIR / args.seed_path, encoding="utf-8") as f:
        path_json = json.load(f)

    footprint = (env.front_oh, env.rear_oh, env.half_width)
    kappa_max = 1.0 / env.model.R

    t0 = time.time()
    runs, results = repair_switches(
        path_json, env.boundary_polygon, footprint, kappa_max,
        k_local=args.k_local, len_cap=args.len_cap, maxiter=args.maxiter,
        n_restarts=args.n_restarts, n_jobs=args.n_jobs,
    )
    elapsed = time.time() - t0

    n_removable = sum(1 for r in results if r is not None)
    print(f"\nelapsed {elapsed:.1f}s")
    print(f"총 {len(results)}개 방향전환 중 {n_removable}개 제거 가능")
    if n_removable:
        for i, r in enumerate(results):
            if r is not None:
                run_a, run_b = runs[i], runs[i + 1]
                print(f"  switch {i+1}: run({run_a[2]}) <-> run({run_b[2]}) -> gear={r[0]}")


if __name__ == "__main__":
    main()
