"""Hybrid A* — 뒷축 기준 (x,y,theta) 상태공간에서 R_min을 정확히 지키는
전/후진 원호 프리미티브로 탐색하는 경로 플래너.

기존 손튜닝 휴리스틱(nav_action + 제자리 K턴)이 좁은 루프에서 반복적으로
경계를 침범해 실패했기 때문에(약 80도 회전이 한계), 그리드 기반 탐색으로
전역적으로 전/후진을 섞어가며 경계를 피하는 경로를 직접 찾는다.

상태: 뒷축 (x, y, theta). 각 프리미티브는 고정 아크 길이(ARC_LEN)를 최대조향
중 일부(steer in STEER_SET)로 전진 또는 후진하는 원호. 매 프리미티브 끝에서
버스 전체 OBB(뒷축 기준 footprint)가 경계 폴리곤 안에 있는지 검사한다.
"""
import heapq
import math
from dataclasses import dataclass, field

from shapely.geometry import Polygon
from shapely.prepared import prep


@dataclass(order=True)
class _PQItem:
    f: float
    counter: int
    node: "Node" = field(compare=False)


@dataclass
class Node:
    x: float
    y: float
    theta: float
    gear: str  # "FWD" / "REV" / "START"
    g: float
    parent: "Node" = None
    depth: int = 0  # root(START)=0, 이후 에지마다 +1 — root 근처 기어 제약에 사용


def _wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


class HybridAStarPlanner:
    def __init__(self, boundary_polygon: Polygon, footprint, delta_max,
                 arc_len=2.0, steer_fracs=(1.0, 0.6, 0.0, -0.6, -1.0),
                 xy_res=0.75, theta_res_deg=10.0,
                 reverse_penalty=1.6, gear_switch_penalty=3.0,
                 tol_pos=0.6, tol_heading_deg=12.0, obstacle_polygons=None, debug=False):
        """delta_max는 호출측(BicycleModel.delta_max, RL_WHEEL_GEOMETRY_REPLAN.md
        PHASE 1 보정식 적용값)에서 이미 확정된 값을 그대로 받는다 — 여기서
        R_min으로부터 재계산하지 않는다(이전에는 atan(L/R_min)을 내부에서
        중복 계산해 PHASE 1 보정과 어긋날 위험이 있었음)."""
        self.debug = debug
        self.boundary = boundary_polygon
        self.boundary_prep = prep(boundary_polygon)
        self.obstacles = obstacle_polygons or []
        self.footprint = footprint  # (front_oh, rear_oh, half_width)
        self.delta_max = delta_max
        self.arc_len = arc_len
        self.steer_fracs = steer_fracs
        self.xy_res = xy_res
        self.theta_res = math.radians(theta_res_deg)
        self.reverse_penalty = reverse_penalty
        self.gear_switch_penalty = gear_switch_penalty
        self.tol_pos = tol_pos
        self.tol_heading = math.radians(tol_heading_deg)

    def _bus_polygon(self, x, y, theta):
        front_oh, rear_oh, half_w = self.footprint[:3]
        c, s = math.cos(theta), math.sin(theta)
        local = [(front_oh, half_w), (front_oh, -half_w), (-rear_oh, -half_w), (-rear_oh, half_w)]
        world = [(x + lx * c - ly * s, y + lx * s + ly * c) for lx, ly in local]
        return Polygon(world)

    def _collision_free(self, x, y, theta):
        poly = self._bus_polygon(x, y, theta)
        if not self.boundary_prep.contains(poly):
            return False
        for obs in self.obstacles:
            if poly.intersects(obs):
                return False
        return True

    def _cell(self, x, y, theta):
        return (round(x / self.xy_res), round(y / self.xy_res), round(_wrap(theta) / self.theta_res))

    def _arc_step(self, x, y, theta, gear, steer_frac, L):
        delta = steer_frac * self.delta_max
        v_sign = 1.0 if gear == "FWD" else -1.0
        ds = self.arc_len / 6
        pts = []
        cx, cy, cth = x, y, theta
        for _ in range(6):
            cth2 = cth + v_sign * (ds / L) * math.tan(delta)
            cx2 = cx + v_sign * ds * math.cos(cth)
            cy2 = cy + v_sign * ds * math.sin(cth)
            cx, cy, cth = cx2, cy2, cth2
            pts.append((cx, cy, cth))
        return pts

    def _fwd_run_length(self, node):
        """node로부터 parent를 거슬러 올라가며 연속으로 gear=='FWD'인
        구간의 길이(에지 개수)를 센다. 마지막 한 스텝만 FWD고 그 직전까지
        REV로 목표를 훨씬 지나쳤다가 살짝 되돌아오는 식의 "눈속임 전진
        도착"을 걸러내기 위함(사용자 지적, 2026-07-04) — 단일 에지만
        보면 이런 경우도 통과해버린다."""
        run = 0
        n = node
        while n is not None and n.gear == "FWD":
            run += 1
            n = n.parent
        return run

    def _expand(self, root, target, L, max_iter, label="",
                arrival_gear_required=None, arrival_min_fwd_run=1,
                root_run_gears=None, root_run_length=1):
        """root에서 target을 향해 표준 Hybrid A* 확장을 1회 수행한다(단방향
        탐색 본체). target에 정확히 도달하면 그 노드를 반환하고, 예산 소진 시
        None을 반환하되 닫힌 노드 전체(cell -> Node)를 함께 돌려준다 — 양방향
        탐색(RL_NEXT_STEPS.md PHASE A 항목1)에서 두 트리의 겹침을 찾는 데 쓴다.

        arrival_gear_required: target에 정확히 도달했다고 인정하는 마지막
        기어를 제한한다(예: "FWD") — 사용자 지정: 목표는 반드시 "앞면이
        닿는" 전진 도착이어야 하고, 후진으로 뒷걸음질쳐 자리만 맞추는 건
        무효. None이면 제한 없음(주로 target=start처럼 도착 방향 제약이
        없는 경우).
        arrival_min_fwd_run: arrival_gear_required="FWD"일 때, 마지막
        에지 하나만이 아니라 최소 이만큼 연속으로 FWD여야 유효한 도착으로
        인정한다(예: 3 -> 최근 3스텝 모두 전진). 기본 1은 이전 동작(단일
        에지 체크)과 동일.
        root_run_gears/root_run_length: root로부터 depth < root_run_length인
        동안만 허용할 기어 집합(예: root_run_gears=("REV",), length=3)
        — 역방향 트리(goal을 root로 삼는 탐색)에서, 실시간 기준 "goal
        도착은 최근 N스텝 연속 FWD여야 한다"는 조건은 되감았을 때 "goal에서
        되감기 전 시간으로 떠나는 첫 N스텝은 REV여야 한다"로 뒤집혀
        나타난다(_reverse_chain의 기어 반전 규칙과 대응, arrival_min_fwd_run과
        길이를 맞춰 써야 함). None이면 제한 없음."""
        if not self._collision_free(*root):
            raise ValueError(f"{label} root pose collides")
        if not self._collision_free(*target):
            raise ValueError(f"{label} target pose collides")

        root_node = Node(root[0], root[1], root[2], "START", 0.0, None)
        open_heap = []
        counter = 0
        heapq.heappush(open_heap, _PQItem(self._heuristic(root, target), counter, root_node))
        best_g = {self._cell(*root): 0.0}
        closed_nodes = {}
        best_dpos = math.inf
        best_dpos_node = None
        best_combined = math.inf
        best_combined_node = None
        # arrival_gear_required가 있으면(예: "FWD") 그 기어로 도착한 노드
        # 중에서만 별도로 최선을 추적한다 — REV로 근접한 노드는 사용자
        # 정의상 무효한 도착이므로 "거의 다 왔다"는 착시를 주면 안 된다.
        best_dpos_valid = math.inf
        best_dpos_valid_node = None
        best_combined_valid = math.inf
        best_combined_valid_node = None
        # 레포트(hybrid_astar_arrival_constraint_lessons.md) §5-1: 트리 전체에서
        # "도착조건 만족 노드 중 최솟값"을 뽑으면 엉뚱한 지점(목표와 무관)이
        # 뽑힐 수 있음이 실증됨. 따라서 "근접 후보"로 인정하는 범위를
        # tol의 NEAR_MULT배 이내로 제한하고, 그 범위 밖의 valid 노드는
        # near-candidate로 치지 않는다(그래도 최솟값 자체는 참고용으로 계속
        # 추적하되, 보고 시 near 범위 밖이면 "근접 후보 없음"으로 취급).
        NEAR_MULT = 4.0
        near_valid_node = None
        near_valid_score = math.inf

        while open_heap and counter < max_iter:
            item = heapq.heappop(open_heap)
            node = item.node
            cell = self._cell(node.x, node.y, node.theta)
            if cell in closed_nodes:
                continue
            closed_nodes[cell] = node

            dpos = math.hypot(node.x - target[0], node.y - target[1])
            dhead = abs(_wrap(node.theta - target[2]))
            if dpos < best_dpos:
                best_dpos, best_dpos_node = dpos, node
            combined = dpos / self.tol_pos + dhead / self.tol_heading
            if combined < best_combined:
                best_combined, best_combined_node = combined, node
            is_valid_arrival = (
                arrival_gear_required is None
                or (node.gear == arrival_gear_required
                    and self._fwd_run_length(node) >= arrival_min_fwd_run)
            )
            if is_valid_arrival:
                if dpos < best_dpos_valid:
                    best_dpos_valid, best_dpos_valid_node = dpos, node
                if combined < best_combined_valid:
                    best_combined_valid, best_combined_valid_node = combined, node
                if dpos <= self.tol_pos * NEAR_MULT and dhead <= self.tol_heading * NEAR_MULT:
                    if combined < near_valid_score:
                        near_valid_score, near_valid_node = combined, node
            if dpos <= self.tol_pos and dhead <= self.tol_heading:
                if is_valid_arrival:
                    return node, closed_nodes, counter, None
                # 위치+방향은 맞지만 도착 기어 조건(예: 최근 N스텝 연속
                # FWD)을 못 지켰다 — 유효한 도착이 아니므로 계속 확장한다
                # (이 노드에서 더 뻗어나가 다른 경로로 재도전할 기회는 남김).

            if root_run_gears and node.depth < root_run_length:
                allowed_gears = root_run_gears
            else:
                allowed_gears = ("FWD", "REV")
            for gear in allowed_gears:
                for sf in self.steer_fracs:
                    pts = self._arc_step(node.x, node.y, node.theta, gear, sf, L)
                    ok = True
                    for (px, py, pth) in pts:
                        if not self._collision_free(px, py, pth):
                            ok = False
                            break
                    if not ok:
                        continue
                    ex, ey, eth = pts[-1]
                    step_cost = self.arc_len
                    if gear == "REV":
                        step_cost *= self.reverse_penalty
                    if node.gear != "START" and node.gear != gear:
                        step_cost += self.gear_switch_penalty
                    step_cost += abs(sf) * 0.3  # 급조향 약한 페널티(부드러운 경로 선호)
                    new_g = node.g + step_cost
                    ncell = self._cell(ex, ey, eth)
                    if ncell in closed_nodes:
                        continue
                    if ncell in best_g and best_g[ncell] <= new_g:
                        continue
                    best_g[ncell] = new_g
                    child = Node(ex, ey, eth, gear, new_g, node, depth=node.depth + 1)
                    counter += 1
                    f = new_g + self._heuristic((ex, ey, eth), target)
                    heapq.heappush(open_heap, _PQItem(f, counter, child))
            counter += 1

        diag = {"best_dpos_node": best_dpos_node, "best_dpos": best_dpos,
                "best_combined_node": best_combined_node, "best_combined": best_combined,
                "best_dpos_valid_node": best_dpos_valid_node, "best_dpos_valid": best_dpos_valid,
                "best_combined_valid_node": best_combined_valid_node, "best_combined_valid": best_combined_valid,
                "near_valid_node": near_valid_node, "near_valid_score": near_valid_score}
        if self.debug:
            def _describe(n):
                if n is None:
                    return None
                dpos = math.hypot(n.x - target[0], n.y - target[1])
                dhead_deg = math.degrees(abs(_wrap(n.theta - target[2])))
                return {"pose": (round(n.x, 3), round(n.y, 3), round(math.degrees(n.theta), 1)),
                        "gear": n.gear, "fwd_run": self._fwd_run_length(n),
                        "dpos_m": round(dpos, 3), "dhead_deg": round(dhead_deg, 1)}

            gear_note = (f" (arrival_gear_required={arrival_gear_required}, min_fwd_run={arrival_min_fwd_run})"
                         if arrival_gear_required else "")
            near_desc = (_describe(near_valid_node) if near_valid_node is not None
                         else f"없음 (tol의 {NEAR_MULT}배 이내이면서 도착조건도 만족하는 노드 전무)")
            print(
                f"[hybrid_astar{(' ' + label) if label else ''}] exhausted: "
                f"closed={len(closed_nodes)} counter={counter}\n"
                f"  best_dpos(위치전용, 기어무관, 참고용-도착조건 무시)={_describe(best_dpos_node)}\n"
                f"  best_combined(위치+방향, 기어무관, 참고용)={_describe(best_combined_node)} combined_score={best_combined}\n"
                f"  [주의] 아래 두 값은 트리 전체 최솟값이라 목표와 무관한 지점일 수 있음(레포트 1-3) - "
                f"신뢰하려면 반드시 near_valid(바로 아래) 존재 여부를 먼저 볼 것:\n"
                f"  best_dpos_VALID(참고용, 근접 보장 없음){gear_note}={_describe(best_dpos_valid_node)}\n"
                f"  best_combined_VALID(참고용, 근접 보장 없음){gear_note}={_describe(best_combined_valid_node)} combined_score={best_combined_valid}\n"
                f"  near_valid_candidate(tol*{NEAR_MULT} 이내 AND 도착조건 만족, 이것만 '근접 후보'로 신뢰){gear_note}={near_desc}"
            )
        return None, closed_nodes, counter, diag

    def plan(self, start, goal, L, max_iter=60000, bidirectional=False, arrival_min_fwd_run=3):
        """start=(x,y,theta), goal=(x,y,theta). L=wheelbase(축거).
        반환: [(x,y,theta,gear), ...] 또는 None.

        bidirectional=True면 단방향 탐색(start->goal)이 실패할 경우, goal을
        루트로 삼아 **실공간 그대로**(좌표 반전 없이) 두 번째 트리를 키운 뒤
        두 트리가 겹치는 지점에서 경로를 잇는다(RL_NEXT_STEPS.md PHASE A
        항목1).

        주의: 처음에는 "heading을 180도 뒤집은 goal에서 출발해 같은 전진
        탐색을 돌리면 역방향 트리가 된다"는 반사(reflection) 트릭을 썼으나,
        이 차량의 풋프린트가 뒷축 기준으로 전/후 오버행이 비대칭(9.06m vs
        2.72m)이라 heading을 180도 뒤집으면 완전히 다른(뒤집힌) 차체 모양이
        되어 충돌검사 자체가 무의미해지는 버그가 있었다. 그래서 반사 없이,
        goal을 그대로 루트로 한 실공간 탐색을 쓰고, 발견된 체인을 되감을 때
        (`_reverse_chain`) **위치·헤딩은 그대로 두고 gear만 FWD<->REV로
        뒤집는다** — 원호는 같은 곡률로 반대 기어를 쓰면 정확히 되짚어지므로
        (원의 성질), 좌표 반사가 전혀 필요 없다.

        사용자 지정 제약(2026-07-04): goal 도착은 반드시 "앞면이 닿는"
        전진(FWD) 도착이어야 한다 — 후진으로 뒷걸음질쳐 위치만 맞추는 건
        무효. 처음엔 마지막 에지 하나만 FWD인지 봤는데, 그러면 "REV로 목표를
        지나쳤다가 마지막에 살짝 전진으로 되돌아오는" 눈속임도 통과해버리는
        결함이 있었다(사용자 지적) — arrival_min_fwd_run으로 최근 N스텝
        전부 FWD여야 하도록 강화. forward tree는 target=goal 도착 시
        arrival_gear_required="FWD"+arrival_min_fwd_run으로 강제하고,
        backward tree(goal이 root)는 되감았을 때 최근 N스텝이 FWD가 되도록
        root에서부터 depth<N인 구간을 REV로만 제한한다."""
        node, closed_a, counter_a, diag_a = self._expand(
            start, goal, L, max_iter, label="fwd",
            arrival_gear_required="FWD", arrival_min_fwd_run=arrival_min_fwd_run,
        )
        if node is not None:
            return self._reconstruct(node)
        if not bidirectional:
            return None

        # 역방향 트리: goal을 루트로, start를 목표로 실공간 그대로 탐색.
        # root_run_gears=("REV",), root_run_length=arrival_min_fwd_run:
        # goal에서 되감기-전 시간으로 떠나는 첫 N스텝은 REV여야, 되감았을 때
        # (_reverse_chain) goal 도착 직전 N스텝이 FWD가 된다.
        node_b, closed_b, counter_b, diag_b = self._expand(
            goal, start, L, max_iter, label="bwd",
            root_run_gears=("REV",), root_run_length=arrival_min_fwd_run,
        )
        if node_b is not None:
            # goal->start 전체를 곧바로 찾은 경우: 되감으면 start->goal 완성.
            return self._reverse_chain(self._reconstruct(node_b))

        meet_a, meet_b = self._find_meeting(closed_a, closed_b)
        if meet_a is None:
            if self.debug:
                print(
                    f"[hybrid_astar bidirectional] no meeting cell — "
                    f"|closed_a|={len(closed_a)} |closed_b|={len(closed_b)}"
                )
            return None
        return self._splice_bidirectional(meet_a, meet_b)

    def _find_meeting(self, closed_a, closed_b):
        """트리 A(start에서 확장)와 트리 B(goal에서 확장)가 같은 그리드
        셀에서 만나면, A는 start에서 그 자세까지, B는 goal에서 그 자세까지
        갈 수 있다는 뜻이므로 B쪽을 되감아 이으면 start->goal 전체 경로가
        된다. 여러 겹침이 있으면 g비용 합이 가장 작은 쌍을 고른다."""
        best = None
        best_cost = math.inf
        for cell, node_a in closed_a.items():
            node_b = closed_b.get(cell)
            if node_b is None:
                continue
            cost = node_a.g + node_b.g
            if cost < best_cost:
                best_cost = cost
                best = (node_a, node_b)
        if best is None:
            return None, None
        return best

    def _reverse_chain(self, chain):
        """[(x,y,theta,gear)...] (root->node 순서)를 node->root 순서로 뒤집고
        gear를 FWD<->REV로 바꾼다. 위치·헤딩은 실공간 그대로 두므로(반사
        없음) 좌표 변환 버그 여지가 없다 — 원호를 반대 기어로 되짚는 것뿐."""
        out = []
        n = len(chain)
        for i in range(n - 1, -1, -1):
            x, y, th, _ = chain[i]
            if i == n - 1:
                gear = "START"
            else:
                edge_gear = chain[i + 1][3]
                gear = "REV" if edge_gear == "FWD" else "FWD"
            out.append((x, y, th, gear))
        return out

    def _splice_bidirectional(self, meet_a, meet_b):
        """트리 A(start->meet_a)와 트리 B(goal->meet_b, meet_b ≈ meet_a 같은
        셀)를 잇는다. B쪽 체인은 `_reverse_chain`으로 실공간 그대로(반사 없이)
        되감되, meet_b 자신은 meet_a와 중복이므로 제외한다."""
        path_a = self._reconstruct(meet_a)  # start ... meet_a
        chain_b = self._reconstruct(meet_b)  # goal ... meet_b
        reversed_b = self._reverse_chain(chain_b)  # meet_b ... goal (gear 반영)
        return path_a + reversed_b[1:]

    def _heuristic(self, pose, goal):
        dx, dy = goal[0] - pose[0], goal[1] - pose[1]
        d = math.hypot(dx, dy)
        dhead = abs(_wrap(goal[2] - pose[2]))
        return d + 1.5 * dhead

    def _reconstruct(self, node):
        path = []
        n = node
        while n is not None:
            path.append((n.x, n.y, n.theta, n.gear))
            n = n.parent
        path.reverse()
        return path
