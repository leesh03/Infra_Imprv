"""Binary search to find maximum rightward offset that satisfies all constraints."""
import json
import math
import subprocess
import sys
from pathlib import Path

SIM_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SIM_DIR))

from envs.bus_hairpin_env import BusHairpinEnv
from shapely.geometry import Polygon

env = BusHairpinEnv()
front_oh = env.front_oh
rear_oh = env.rear_oh
half_width = env.half_width

def wheel_poly(x, y, th):
    """Create wheel envelope with full front/rear overhangs."""
    c, s = math.cos(th), math.sin(th)
    loc = [(front_oh, half_width), (front_oh, -half_width), (-rear_oh, -half_width), (-rear_oh, half_width)]
    return Polygon([(x + lx * c - ly * s, y + lx * s + ly * c) for lx, ly in loc])

def count_violations(path_file):
    """Count wheel envelope violations in a path."""
    with open(path_file, encoding='utf-8') as f:
        data = json.load(f)
    rw = data['rear_waypoints']
    viol = sum(1 for (x, y, th) in rw 
               if not env.boundary_polygon.contains(wheel_poly(x, y, th)))
    return viol

def generate_path(offset_right, goal_offset_right):
    """Generate path with given offsets."""
    result = subprocess.run(
        [sys.executable, 'tests/shift_exit_keepright.py',
         '--offset-right', str(offset_right),
         '--goal-offset-right', str(goal_offset_right)],
        capture_output=True, text=True, cwd=SIM_DIR
    )
    if result.returncode != 0:
        print(f"Error generating path: {result.stderr}", file=sys.stderr)
        return None
    return SIM_DIR / 'geometry' / 'reference_path_keepright.json'

# Binary search for maximum safe offset
low, high = 0.0, 1.0
best_offset = 0.0
best_violations = count_violations(SIM_DIR / 'geometry' / 'reference_path_wheelmodel.json')
print(f"Original wheel-model violations: {best_violations}")

for iteration in range(20):
    mid = (low + high) / 2
    goal_offset = mid * 0.5  # Scale goal offset proportionally
    
    print(f"\nIteration {iteration+1}: Testing offset={mid:.4f}m, goal_offset={goal_offset:.4f}m")
    
    path_file = generate_path(mid, goal_offset)
    if path_file is None:
        high = mid
        continue
    
    violations = count_violations(path_file)
    print(f"  Violations: {violations}")
    
    if violations == 0:
        # Can safely increase offset
        best_offset = mid
        best_violations = violations
        low = mid
        print(f"  ✓ Safe! Setting as new best: {best_offset:.4f}m")
    else:
        # Too many violations, reduce offset
        high = mid
        print(f"  ✗ Too many violations, reducing offset")
    
    if high - low < 0.001:
        break

print(f"\n{'='*60}")
print(f"BEST SAFE OFFSET: {best_offset:.4f}m with {best_violations} violations")
print(f"{'='*60}")

# Generate final path with best offset
if best_offset > 0:
    goal_offset = best_offset * 0.5
    print(f"\nGenerating final path with offset={best_offset:.4f}m...")
    generate_path(best_offset, goal_offset)
    
    with open(SIM_DIR / 'geometry' / 'reference_path.json', 'w', encoding='utf-8') as fout:
        with open(SIM_DIR / 'geometry' / 'reference_path_keepright.json', 'r', encoding='utf-8') as fin:
            fout.write(fin.read())
    print(f"Updated reference_path.json")
else:
    print(f"\nNo safe offset found. Keeping original path.")
