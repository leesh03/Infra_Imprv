"""2026-07-05: 미세 해상도(arc_len 0.3/xy_res 0.2/theta_res 2deg)로 양방향
Hybrid A*가 실제로 경로를 찾았다(388 nodes, violations=0). 레포트가 강조한
"숫자만 믿지 말고 그림으로 검증" 원칙에 따라, 저장된 reference_path.json이
실제로 start->goal을 전진으로 도착하며 잇는지 육안 확인."""
import json
import math

import matplotlib.pyplot as plt

from envs.bus_hairpin_env import BusHairpinEnv

SIM_DIR = __import__("pathlib").Path(__file__).resolve().parent.parent


def main():
    env = BusHairpinEnv()
    with open(SIM_DIR / "geometry" / "reference_path.json", encoding="utf-8") as f:
        data = json.load(f)
    wp = data["waypoints"]
    gears = data["gears"]

    fig, ax = plt.subplots(figsize=(11, 9))
    px, py = env.boundary_polygon.exterior.xy
    ax.plot(px, py, "k-", linewidth=1)

    for i in range(1, len(wp)):
        color = "tab:red" if gears[i] == "REV" else "tab:blue"
        ax.plot([wp[i - 1][0], wp[i][0]], [wp[i - 1][1], wp[i][1]], color=color, linewidth=2.0)
    ax.plot([], [], color="tab:blue", linewidth=2.5, label="FWD (forward)")
    ax.plot([], [], color="tab:red", linewidth=2.5, label="REV (reverse)")

    ax.plot(wp[0][0], wp[0][1], "g^", markersize=12, label="path start (front-center)")
    ax.plot(wp[-1][0], wp[-1][1], "m*", markersize=16, label="path end (front-center)")
    ax.plot(env.goal_x, env.goal_y, "kx", markersize=14, mew=2, label="goal target (front-center)")

    dpos = math.hypot(wp[-1][0] - env.goal_x, wp[-1][1] - env.goal_y)
    n_rev_tail = 0
    for g in reversed(gears):
        if g == "REV":
            n_rev_tail += 1
        else:
            break
    ax.set_title(
        f"Full path check: {len(wp)} waypoints, gear_changes={sum(1 for i in range(1, len(gears)) if gears[i] != gears[i-1])}\n"
        f"end-to-goal dpos={dpos:.3f}m, trailing consecutive REV before arrival={n_rev_tail}"
    )
    ax.set_aspect("equal")
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    out_path = SIM_DIR / "outputs" / "full_path_check.png"
    fig.savefig(out_path, dpi=140)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
