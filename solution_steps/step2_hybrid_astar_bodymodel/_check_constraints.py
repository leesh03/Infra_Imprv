import json
import math
from shapely.geometry import Point, Polygon
from envs.bus_hairpin_env import BusHairpinEnv

env = BusHairpinEnv()
with open('geometry/reference_path.json', encoding='utf-8') as f:
    rw = json.load(f)['rear_waypoints']

front_oh = env.front_oh
rear_oh = env.rear_oh
half_width = env.half_width

def wheel_poly(x, y, th):
    c, s = math.cos(th), math.sin(th)
    local = [(front_oh, half_width), (front_oh, -half_width), (-rear_oh, -half_width), (-rear_oh, half_width)]
    return Polygon([(x + lx * c - ly * s, y + lx * s + ly * c) for lx, ly in local])

viol = []
for i, (x, y, th) in enumerate(rw):
    p = wheel_poly(x, y, th)
    if not env.boundary_polygon.contains(p):
        viol.append((i, (x, y, th)))

print('total_nodes', len(rw))
print('violating_nodes', len(viol))
for i, pose in viol[:20]:
    print('violation', i, pose)
print('last_pose', rw[-1])
print('contains_last', env.boundary_polygon.contains(wheel_poly(*rw[-1])))
print('distance_to_boundary', env.boundary_polygon.exterior.distance(Point(*rw[-1][:2])))
