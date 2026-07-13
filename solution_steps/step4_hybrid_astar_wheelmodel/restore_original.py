import json
import math
from shapely.geometry import Polygon
from envs.bus_hairpin_env import BusHairpinEnv

env = BusHairpinEnv()
with open('geometry/reference_path_wheelmodel.json', encoding='utf-8') as f:
    data = json.load(f)
rw = data['rear_waypoints']

front_oh = env.front_oh
rear_oh = env.rear_oh
half_width = env.half_width

def wheel_poly(x, y, th):
    c, s = math.cos(th), math.sin(th)
    local = [(front_oh, half_width), (front_oh, -half_width), (-rear_oh, -half_width), (-rear_oh, half_width)]
    return Polygon([(x + lx*c - ly*s, y + lx*s + ly*c) for lx, ly in local])

viol = 0
for i, (x, y, th) in enumerate(rw):
    if not env.boundary_polygon.contains(wheel_poly(x, y, th)):
        viol += 1
print('violating_nodes_original:', viol)

with open('geometry/reference_path.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=2)
print('restored to wheel-model original')
