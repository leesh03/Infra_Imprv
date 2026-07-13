import json
import math
from shapely.geometry import Polygon
from envs.bus_hairpin_env import BusHairpinEnv

env = BusHairpinEnv()
with open('geometry/reference_path_wheelmodel.json', encoding='utf-8') as f:
    data = json.load(f)
rw = [list(p) for p in data['rear_waypoints']]

front_oh = env.front_oh
rear_oh = env.rear_oh
half_width = env.half_width

def wheel_poly(x, y, th):
    c, s = math.cos(th), math.sin(th)
    local = [(front_oh, half_width), (front_oh, -half_width), (-rear_oh, -half_width), (-rear_oh, half_width)]
    return Polygon([(x + lx * c - ly * s, y + lx * s + ly * c) for lx, ly in local])

for i, (x, y, th) in enumerate(rw):
    if env.boundary_polygon.contains(wheel_poly(x, y, th)):
        continue
    for frac in [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]:
        cx = x * (1.0 - frac)
        cy = y * (1.0 - frac)
        if env.boundary_polygon.contains(wheel_poly(cx, cy, th)):
            rw[i] = [round(cx, 3), round(cy, 3), round(th, 6)]
            break

with open('geometry/reference_path.json', 'w', encoding='utf-8') as f:
    json.dump(
        {
            'waypoints': [[x + env.d_ref * math.cos(th), y + env.d_ref * math.sin(th)] for x, y, th in rw],
            'rear_waypoints': rw,
            'gears': data['gears'],
            'collision_model': 'wheel',
        },
        f,
        ensure_ascii=False,
        indent=2,
    )
print('wrote geometry/reference_path.json')
