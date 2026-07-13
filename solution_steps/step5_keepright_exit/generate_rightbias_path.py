"""Generate constraint-satisfied path with rightward bias by shifting goal."""
import json
import math
import time
from pathlib import Path

import matplotlib.pyplot as plt
from shapely.geometry import Polygon

from envs.bus_hairpin_env import BusHairpinEnv
from kinematics.bicycle import BicycleState
from planning.hybrid_astar import HybridAStarPlanner

SIM_DIR = Path(__file__).resolve().parent.parent

def _rect(x, y, theta, front, rear, half_w):
    c, s = math.cos(theta), math.sin(theta)
    local = [(front, half_w), (front, -half_w), (-rear, -half_w), (-rear, half_w)]
    return Polygon([(x + lx * c - ly * s, y + lx * s + ly * c) for lx, ly in local])

def main():
    import sys
    goal_offset_y = float(sys.argv[1]) if len(sys.argv) > 1 else 1.5
    
    env = BusHairpinEnv()
    
    # Wheel footprint for collision checking
    wheel_fp = (env.model.L, 0.0, env.half_width)
    print(f"wheel footprint: L={wheel_fp[0]:.2f}m, half_w={wheel_fp[2]:.3f}m")
    
    # Fine steering for smooth paths
    fine_steer = tuple(round(-1.0 + 0.1 * i, 2) for i in range(21))
    planner = HybridAStarPlanner(
        boundary_polygon=env.boundary_polygon,
        footprint=wheel_fp,
        delta_max=env.model.delta_max,
        arc_len=0.3,
        steer_fracs=fine_steer,
        xy_res=0.2,
        theta_res_deg=2.0,
        reverse_penalty=1.3,
        gear_switch_penalty=3.0,
        tol_pos=0.6,
        tol_heading_deg=12.0,
        obstacle_polygons=env.obstacle_polygons,
        debug=False,
    )
    
    start = (env.start_x, env.start_y, env.start_heading)
    
    # Shift goal rightward (increase y)
    goal_rear_x = env.goal_x - env.d_ref * math.cos(env.goal_heading)
    goal_rear_y = env.goal_y - env.d_ref * math.sin(env.goal_heading) + goal_offset_y
    goal = (goal_rear_x, goal_rear_y, env.goal_heading)
    
    print(f"start: {start}")
    print(f"goal (shifted by y+{goal_offset_y:.2f}m): {goal}")
    
    t0 = time.time()
    path = planner.plan(start, goal, L=env.model.L, max_iter=400000, bidirectional=True)
    elapsed = time.time() - t0
    
    if path is None:
        print(f"NO PATH FOUND after {elapsed:.1f}s")
        return False
    
    print(f"✓ Path found in {elapsed:.1f}s, nodes: {len(path)}")
    
    # Validate with correct wheel model
    wheel_viol = sum(
        1 for (x, y, theta, gear) in path
        if not env.boundary_polygon.contains(_rect(x, y, theta, *wheel_fp))
    )
    print(f"wheel violations: {wheel_viol}")
    
    if wheel_viol > 0:
        print(f"⚠ Path still has {wheel_viol} violations")
        return False
    
    # Generate output
    gear_changes = sum(1 for i in range(1, len(path)) if path[i][3] != path[i-1][3] and path[i][3] != "START")
    total_len = sum(math.hypot(path[i][0]-path[i-1][0], path[i][1]-path[i-1][1]) for i in range(1, len(path)))
    print(f"gear changes: {gear_changes}, total length: {total_len:.1f}m")
    
    waypoints, gears, rear_waypoints = [], [], []
    for (x, y, theta, gear) in path:
        fc = env.front_center(BicycleState(x=x, y=y, theta=theta, v=0.0, delta=0.0))
        waypoints.append([round(fc[0], 3), round(fc[1], 3)])
        gears.append(gear if gear != "START" else "FWD")
        rear_waypoints.append([round(x, 3), round(y, 3), round(theta, 6)])
    
    out = {
        "waypoints": waypoints,
        "gears": gears,
        "rear_waypoints": rear_waypoints,
        "collision_model": "wheel",
        "footprint": list(wheel_fp),
        "goal_offset_y": goal_offset_y,
        "metadata": {
            "planner": "HybridAStar-rightward-biased",
            "constraint_violations": wheel_viol,
            "path_length_m": round(total_len, 1),
        }
    }
    
    # Ensure geometry directory exists
    geom_dir = SIM_DIR / "geometry"
    geom_dir.mkdir(exist_ok=True)
    
    out_path = geom_dir / "reference_path_rightbias.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"saved {out_path}")
    
    # Activate as reference path
    ref_path = geom_dir / "reference_path.json"
    with open(ref_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"✓ Activated as reference_path.json")
    
    # Visualize
    fig, ax = plt.subplots(figsize=(13, 9))
    px, py = env.boundary_polygon.exterior.xy
    ax.plot(px, py, "k-", lw=1)
    
    colors = {"FWD": "tab:blue", "REV": "tab:red", "START": "gray"}
    for i in range(1, len(path)):
        color = colors.get(path[i][3], "black")
        ax.plot([path[i-1][0], path[i][0]], [path[i-1][1], path[i][1]], color=color, lw=1.5)
    
    ax.plot(start[0], start[1], "g^", ms=12, label="start", zorder=5)
    ax.plot(env.goal_x, env.goal_y, "kx", ms=12, mew=2, label="goal (original)", zorder=5)
    ax.plot(goal[0], goal[1], "b*", ms=15, label=f"goal (shifted y+{goal_offset_y:.1f}m)", zorder=5)
    
    ax.plot([], [], color="tab:blue", lw=2, label="forward")
    ax.plot([], [], color="tab:red", lw=2, label="reverse")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_aspect("equal")
    ax.set_title(f"Rightward-biased path (0 violations, goal_offset={goal_offset_y}m)")
    fig.tight_layout()
    
    out_png = SIM_DIR / "outputs" / "path_rightbias_final.png"
    fig.savefig(out_png, dpi=120)
    print(f"saved {out_png}")
    
    return True

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
