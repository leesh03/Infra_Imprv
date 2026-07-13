"""Shift the final exit run to the vehicle's right side.

The wheel-model Hybrid A* path is already valid, but the final forward exit
run after the last reverse tends to hug the left side of the corridor. This
script keeps the existing gear-change structure and yaw profile, then applies
a smooth lateral offset to the final run only so the path stays closer to the
right wall while still respecting curvature and wheel-envelope containment.
"""

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
from shapely.geometry import Point, Polygon

SIM_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SIM_DIR))

from envs.bus_hairpin_env import BusHairpinEnv

matplotlib.rcParams["font.family"] = "Malgun Gothic"
matplotlib.rcParams["axes.unicode_minus"] = False


def _wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def split_runs(rw, gears):
    n = len(rw)
    edge = gears[1:n]
    runs = []
    k = 0
    while k < len(edge):
        g = edge[k]
        j = k
        while j < len(edge) and edge[j] == g:
            j += 1
        runs.append((k, j, g))
        k = j
    return runs


def smoothstep(t):
    t = max(0.0, min(1.0, t))
    return t * t * (3 - 2 * t)


def taper(u, ramp, tail):
    if u < ramp:
        return smoothstep(u / ramp)
    if u > 1 - tail:
        return smoothstep((1 - u) / tail)
    return 1.0


def _fit_within_boundary(desired_xy, anchor_xy, boundary_polygon):
    x, y = desired_xy
    if boundary_polygon.covers(Point(x, y)):
        return [x, y]
    ax, ay = anchor_xy
    for frac in [1.0, 0.95, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]:
        cand = [ax + frac * (x - ax), ay + frac * (y - ay)]
        if boundary_polygon.covers(Point(*cand)):
            return cand
    return [ax, ay]


def wheel_poly(x, y, th, wheelbase, half_width):
    c, sn = math.cos(th), math.sin(th)
    loc = [(wheelbase, half_width), (wheelbase, -half_width), (0.0, -half_width), (0.0, half_width)]
    return Polygon([(x + lx * c - ly * sn, y + lx * sn + ly * c) for lx, ly in loc])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--offset-right", type=float, default=2.2, help="lateral shift to the right [m]")
    ap.add_argument("--ramp", type=float, default=0.1, help="start taper ratio")
    ap.add_argument("--tail", type=float, default=0.9, help="legacy end taper ratio")
    ap.add_argument("--k-thr", type=float, default=0.03, help="do not shift strongly curved sections")
    ap.add_argument("--smooth-m", type=float, default=10.0, help="legacy moving-average smoothing distance [m]")
    ap.add_argument("--goal-heading-deg", type=float, default=173.9,
                    help="final approach heading while keeping the front-center goal fixed")
    ap.add_argument("--offset-end-u", type=float, default=1.0,
                    help="arc-length fraction where the right offset reaches full strength")
    ap.add_argument("--morph-start", type=float, default=0.45,
                    help="arc-length fraction where the final goal-pose morph starts")
    ap.add_argument("--goal-offset-right", type=float, default=3.5,
                    help="extra rightward shift applied to the final arrival pose [m]")
    ap.add_argument("--heading-mode", choices=["original", "tangent"], default="original",
                    help="original keeps validated yaw; tangent recomputes yaw from shifted xy")
    ap.add_argument("--out-json", default="geometry/reference_path_keepright.json")
    ap.add_argument("--out-png", default="outputs/keepright_exit.png")
    args = ap.parse_args()

    env = BusHairpinEnv()
    wheelbase = env.model.L
    half_width = env.half_width
    kappa_max = 1.0 / env.model.R

    with open(SIM_DIR / "geometry" / "reference_path_wheelmodel.json", encoding="utf-8") as f:
        data = json.load(f)
    rw = data["rear_waypoints"]
    gears = data["gears"]
    runs = split_runs(rw, gears)
    exit_start = runs[-1][0]
    exit_poses = [tuple(p) for p in rw[exit_start:]]

    s = [0.0]
    for i in range(1, len(exit_poses)):
        s.append(s[-1] + math.hypot(exit_poses[i][0] - exit_poses[i - 1][0],
                                    exit_poses[i][1] - exit_poses[i - 1][1]))
    exit_len = s[-1]

    original_kappa = [0.0] * len(exit_poses)
    for i in range(1, len(exit_poses) - 1):
        ds = s[i + 1] - s[i - 1]
        if ds > 1e-6:
            original_kappa[i] = abs(_wrap(exit_poses[i + 1][2] - exit_poses[i - 1][2])) / ds
    straight_weight = [max(0.0, min(1.0, (args.k_thr - k) / args.k_thr)) for k in original_kappa]
    straight_weight[0] = straight_weight[-1] = 0.0

    shifted = []
    for i, (x, y, th) in enumerate(exit_poses):
        u = s[i] / exit_len if exit_len > 0 else 0.0
        if u < args.offset_end_u:
            offset = args.offset_right * smoothstep((u - args.ramp) / (args.offset_end_u - args.ramp))
        else:
            offset = args.offset_right
        right_x, right_y = math.sin(th), -math.cos(th)
        desired = [x + offset * right_x, y + offset * right_y]
        shifted.append(_fit_within_boundary(desired, [x, y], env.boundary_polygon))
    shifted[0] = [exit_poses[0][0], exit_poses[0][1]]

    target_heading = math.radians(args.goal_heading_deg)
    base_target = [
        env.goal_x - env.d_ref * math.cos(target_heading),
        env.goal_y - env.d_ref * math.sin(target_heading),
    ]
    desired_target = [
        base_target[0] + args.goal_offset_right * math.sin(target_heading),
        base_target[1] - args.goal_offset_right * math.cos(target_heading),
    ]
    target_pose = [* _fit_within_boundary(desired_target, base_target, env.boundary_polygon), target_heading]
    dx = target_pose[0] - shifted[-1][0]
    dy = target_pose[1] - shifted[-1][1]
    dth = _wrap(target_pose[2] - exit_poses[-1][2])
    for i in range(len(shifted)):
        u = s[i] / exit_len if exit_len > 0 else 0.0
        w = smoothstep((u - args.morph_start) / (1.0 - args.morph_start))
        shifted[i][0] += w * dx
        shifted[i][1] += w * dy

    new_exit = []
    for i in range(len(shifted)):
        if args.heading_mode == "original":
            u = s[i] / exit_len if exit_len > 0 else 0.0
            w = smoothstep((u - args.morph_start) / (1.0 - args.morph_start))
            th = exit_poses[i][2] + w * dth
        elif i == 0:
            th = exit_poses[0][2]
        elif i == len(shifted) - 1:
            th = exit_poses[-1][2]
        else:
            th = math.atan2(shifted[i + 1][1] - shifted[i - 1][1],
                            shifted[i + 1][0] - shifted[i - 1][0])
        xy = _fit_within_boundary([shifted[i][0], shifted[i][1]], [exit_poses[i][0], exit_poses[i][1]], env.boundary_polygon)
        new_exit.append([round(xy[0], 3), round(xy[1], 3), round(th, 6)])
    new_exit[-1] = [round(target_pose[0], 3), round(target_pose[1], 3), round(target_pose[2], 6)]

    max_kappa = 0.0
    for i in range(1, len(new_exit)):
        ds = math.hypot(new_exit[i][0] - new_exit[i - 1][0], new_exit[i][1] - new_exit[i - 1][1])
        if ds > 1e-6:
            max_kappa = max(max_kappa, abs(_wrap(new_exit[i][2] - new_exit[i - 1][2])) / ds)

    new_rw = [list(rw[i]) for i in range(exit_start)] + new_exit
    new_gears = gears[:exit_start] + ["FWD"] * len(new_exit)

    violations = sum(
        1 for (x, y, th) in new_rw
        if not env.boundary_polygon.contains(wheel_poly(x, y, th, wheelbase, half_width))
    )
    n_switch = sum(1 for i in range(1, len(new_gears)) if new_gears[i] != new_gears[i - 1])
    total_len = sum(
        math.hypot(new_rw[i][0] - new_rw[i - 1][0], new_rw[i][1] - new_rw[i - 1][1])
        for i in range(1, len(new_rw))
    )

    print(f"exit nodes {len(new_exit)}, exit length {exit_len:.1f}m, offset {args.offset_right}m")
    print(f"max curvature {max_kappa:.4f} (limit {kappa_max:.4f}) -> "
          f"{'OK' if max_kappa <= kappa_max * 1.05 else 'exceeds'}")
    print(f"merged: nodes {len(new_rw)}, switches {n_switch}, length {total_len:.1f}m, wheel violations {violations}")
    final_fc = (
        new_rw[-1][0] + env.d_ref * math.cos(new_rw[-1][2]),
        new_rw[-1][1] + env.d_ref * math.sin(new_rw[-1][2]),
    )
    final_dpos = math.hypot(final_fc[0] - env.goal_x, final_fc[1] - env.goal_y)
    final_dhead = abs(_wrap(new_rw[-1][2] - env.goal_heading))
    print(f"goal error: front-center {final_dpos:.4f}m, heading {math.degrees(final_dhead):.2f}deg")
    if violations > 0:
        print("warning: wheel-envelope violations remain; reduce offset or adjust taper.")

    waypoints = [
        [round(x + env.d_ref * math.cos(th), 3), round(y + env.d_ref * math.sin(th), 3)]
        for (x, y, th) in new_rw
    ]
    out = {
        "waypoints": waypoints,
        "gears": new_gears,
        "rear_waypoints": [[round(x, 3), round(y, 3), round(th, 6)] for x, y, th in new_rw],
        "collision_model": "wheel",
    }
    out_json = SIM_DIR / args.out_json
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("saved", out_json)

    fig, ax = plt.subplots(figsize=(13, 9))
    px, py = env.boundary_polygon.exterior.xy
    ax.plot(px, py, "k-", lw=1.2)
    ax.plot([p[0] for p in exit_poses], [p[1] for p in exit_poses], "b--", lw=1.3, alpha=0.6,
            label="previous exit")
    for i in range(1, len(new_rw)):
        color = "tab:red" if new_gears[i] == "REV" else "tab:blue"
        ax.plot([new_rw[i - 1][0], new_rw[i][0]], [new_rw[i - 1][1], new_rw[i][1]],
                color=color, lw=1.9, zorder=5)
    ax.plot([], [], color="tab:blue", lw=2, label="forward")
    ax.plot([], [], color="tab:red", lw=2, label="reverse")
    ax.plot(new_rw[0][0], new_rw[0][1], "g^", ms=12, zorder=6)
    ax.plot(env.goal_x, env.goal_y, "kx", ms=13, mew=2, zorder=6)
    ax.legend(loc="upper right", fontsize=10)
    ax.set_aspect("equal")
    ax.set_title(
        f"Keep-right exit shift {args.offset_right}m: switches {n_switch}, length {total_len:.1f}m, "
        f"violations {violations}, max curvature {max_kappa:.3f}/{kappa_max:.3f}"
    )
    fig.tight_layout()
    out_png = SIM_DIR / args.out_png
    fig.savefig(out_png, dpi=150)
    print("saved", out_png)


if __name__ == "__main__":
    main()
