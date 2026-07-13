"""셔틀버스 ∧자 분기점 회차 학습용 Gymnasium 환경 (TASK 5, 보상은 TASK 7 사양 반영)."""
import json
import math
from pathlib import Path

import gymnasium as gym
import numpy as np
import yaml
from gymnasium import spaces
from shapely.geometry import LineString, Point, Polygon

from envs.reference_path import ReferencePath
from envs.sensors import build_sensor_lines, cast_rays
from geometry.grade_field import GradeField
from kinematics.bicycle import BicycleModel, BicycleState

SIM_DIR = Path(__file__).resolve().parent.parent


def _wrap_angle(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi


class BusHairpinEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, config=None, render_mode=None):
        super().__init__()
        if config is None:
            with open(SIM_DIR / "config.yaml", "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
        self.cfg = config
        self.render_mode = render_mode

        geometry_dir = SIM_DIR / config["paths"]["geometry_dir"]
        with open(geometry_dir / "road_boundary.json", "r", encoding="utf-8") as f:
            self.boundary_polygon = Polygon(json.load(f)["polygon"])
        self.boundary_line = LineString(self.boundary_polygon.exterior.coords)

        with open(geometry_dir / "obstacles.json", "r", encoding="utf-8") as f:
            obstacles = json.load(f)["obstacles"]
        self.obstacle_polygons = [Polygon(o) for o in obstacles]
        self.obstacle_lines = [LineString(p.exterior.coords) for p in self.obstacle_polygons]
        # boundary_line이 16,391개 정점짜리라 매 ray를 통짜로 intersection하면
        # 너무 느려(PPO fps 51 -> 5M step에 27시간) ray-casting 전용 단순화
        # 사본을 한 번만 만들어 쓴다(충돌판정용 원본 polygon은 그대로 둠).
        self.sensor_boundary_line, self.sensor_obstacle_lines = build_sensor_lines(
            self.boundary_line, self.obstacle_lines
        )

        self.grade_field = GradeField.from_files(geometry_dir=geometry_dir)
        self.reference_path = ReferencePath.from_file(geometry_dir / "reference_path.json")
        self.model = BicycleModel(config)

        bus_cfg = config["bus"]
        self.half_width = bus_cfg["width"] / 2.0
        self.front_oh = bus_cfg["d_ref"]          # 뒷축 -> 앞면 (heading 방향)
        self.rear_oh = bus_cfg["length_total"] - bus_cfg["d_ref"]  # 뒷축 -> 뒷면
        self.d_ref = bus_cfg["d_ref"]

        goal_cfg = config["goal"]
        self.goal_x = goal_cfg["x"]
        self.goal_y = goal_cfg["y"]
        self.goal_heading = goal_cfg["heading"]
        self.tol_pos = goal_cfg["tol_pos"]
        self.tol_heading = goal_cfg["tol_heading"]
        self.v_stop = goal_cfg["v_stop"]

        start_cfg = config["start"]
        self.start_x, self.start_y, self.start_heading = start_cfg["x"], start_cfg["y"], start_cfg["heading"]

        sensor_cfg = config["sensors"]
        self.n_rays = sensor_cfg["n_rays"]
        self.sensor_range = sensor_cfg["max_range"]
        self.lookahead_d = sensor_cfg["lookahead_d"]

        self.rcfg = config["reward"]
        self.max_episode_steps = config["env"]["max_episode_steps"]
        # RL_WHEEL_GEOMETRY_REPLAN.md PHASE 2: 스텝 사이 보간 서브스텝 수
        self.collision_substeps = config["env"].get("collision_substeps", 1)

        self.a_max = config["kinematics"]["a_max"]
        self.steer_rate_max = config["kinematics"]["steer_rate_max"]

        # 시작 자세 커리큘럼/노이즈 (TASK 9 선반영). curriculum_level=1.0이면
        # config의 start와 동일한 실제 시작점(노이즈 0일 때) — 기존 동작과 호환.
        curriculum_cfg = config.get("curriculum", {})
        self.curriculum_level = curriculum_cfg.get("level_default", 1.0)
        self.curriculum_ceiling = 1.0
        self.start_noise_pos = curriculum_cfg.get("noise_pos_std", 0.0)
        self.start_noise_heading = curriculum_cfg.get("noise_heading_std", 0.0)

        # obs: (dx,dy,dtheta,v,delta) + rays + (grade, grade_ahead) + (path_progress, lateral_offset)
        obs_dim = 5 + self.n_rays + 2 + 2
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        self.state = None
        self._step_count = 0
        self._trajectory = []
        self._rng = np.random.default_rng()

    def set_curriculum_level(self, level):
        """난이도 하한(floor). 0.0=GOAL 근처도 허용(가장 쉬움) ~
        1.0=항상 실제 START(가장 어려움). 실제 에피소드 난이도는 매 reset()마다
        [level, curriculum_ceiling] 구간에서 무작위로 뽑힌다(_sample_start_pose)."""
        self.curriculum_level = float(np.clip(level, 0.0, 1.0))

    def set_curriculum_ceiling(self, ceiling):
        """2026-07-07 진단용: episode_level 샘플링 상한을 기본값 1.0보다 낮춰
        '항상 쉬운 쪽만' 나오도록 강제한다 — floor(curriculum_level)만 낮추면
        여전히 [floor, 1.0] 사이에서 뽑혀 가끔 진짜 START(가장 어려움)가 섞여
        나오므로, 커리큘럼이 원인인지(난이도 뒤섞임) 정책이 쉬운 조건에서도
        아예 성공을 못 배우는지 구분하려면 상한도 같이 낮춰야 한다."""
        self.curriculum_ceiling = float(np.clip(ceiling, 0.0, 1.0))

    def set_start_noise(self, pos_std, heading_std):
        self.start_noise_pos = max(0.0, float(pos_std))
        self.start_noise_heading = max(0.0, float(heading_std))

    # ------------------------------------------------------------------ #
    def front_center(self, state: BicycleState):
        return self.model.front_center(state)

    def _bus_polygon(self, state: BicycleState):
        c, s = math.cos(state.theta), math.sin(state.theta)
        local_corners = [
            (self.front_oh, self.half_width),
            (self.front_oh, -self.half_width),
            (-self.rear_oh, -self.half_width),
            (-self.rear_oh, self.half_width),
        ]
        world_corners = [
            (state.x + lx * c - ly * s, state.y + lx * s + ly * c) for lx, ly in local_corners
        ]
        return Polygon(world_corners)

    def _collision_at(self, state: BicycleState):
        bus_poly = self._bus_polygon(state)
        if not self.boundary_polygon.contains(bus_poly):
            return True
        for obs in self.obstacle_polygons:
            if bus_poly.intersects(obs):
                return True
        return False

    def _check_collision(self, state: BicycleState, prev_state: BicycleState = None):
        """RL_WHEEL_GEOMETRY_REPLAN.md PHASE 2: prev_state가 주어지면 그 사이를
        collision_substeps회 선형보간해 중간 자세도 검사한다(스텝당 이동량이
        커도 벽을 통과해 지나치는 걸 놓치지 않기 위함). prev_state가 없으면
        (예: 시작 자세 샘플링) 단일 자세만 검사."""
        if prev_state is None or self.collision_substeps <= 1:
            return self._collision_at(state)
        for k in range(1, self.collision_substeps + 1):
            t = k / self.collision_substeps
            ix = prev_state.x + (state.x - prev_state.x) * t
            iy = prev_state.y + (state.y - prev_state.y) * t
            ith = prev_state.theta + _wrap_angle(state.theta - prev_state.theta) * t
            if self._collision_at(BicycleState(x=ix, y=iy, theta=ith)):
                return True
        return False

    def _pose_error(self, state: BicycleState):
        fc_x, fc_y = self.front_center(state)
        dx, dy = self.goal_x - fc_x, self.goal_y - fc_y
        dtheta = _wrap_angle(self.goal_heading - state.theta)
        return dx, dy, dtheta, math.hypot(dx, dy)

    def _progress_metric(self, state: BicycleState):
        """경로(중심선) 진행률 기반 shaping 지표. 진입/진출 경사로가 apex에서
        거의 반대 방향으로 꺾여 있어, GOAL까지의 직선거리는 더 이상 실제
        주행거리를 반영하지 못한다(직선거리가 줄어드는 방향이 벽을 가로지를
        수 있음) — 그래서 GOAL까지의 유클리드 거리 대신 reference_path에 투영한
        "남은 경로 길이"를 쓴다."""
        fc_x, fc_y = self.front_center(state)
        remaining = self.reference_path.remaining(fc_x, fc_y)
        dtheta = _wrap_angle(self.goal_heading - state.theta)
        return remaining + self.model.L * abs(dtheta)

    def _is_success(self, state: BicycleState):
        dx, dy, dtheta, pos_err = self._pose_error(state)
        return pos_err <= self.tol_pos and abs(dtheta) <= self.tol_heading and abs(state.v) <= self.v_stop

    def _observation(self, state: BicycleState):
        dx, dy, dtheta, _ = self._pose_error(state)
        c, s = math.cos(state.theta), math.sin(state.theta)
        dx_body = dx * c + dy * s
        dy_body = -dx * s + dy * c

        rays = cast_rays(
            state.x, state.y, state.theta, self.n_rays, self.sensor_range,
            self.sensor_boundary_line, self.sensor_obstacle_lines,
        )
        grade = self.grade_field.grade_at(state.x, state.y)
        grade_ahead = self.grade_field.grade_ahead(state.x, state.y, state.theta, self.lookahead_d)

        fc_x, fc_y = self.front_center(state)
        path_progress = self.reference_path.progress_fraction(fc_x, fc_y)
        lateral_offset = self.reference_path.lateral_offset(fc_x, fc_y)

        obs = np.array(
            [dx_body, dy_body, dtheta, state.v, state.delta, *rays, grade, grade_ahead,
             path_progress, lateral_offset],
            dtype=np.float32,
        )
        return obs

    def _clearance(self, state: BicycleState):
        """RL_WHEEL_GEOMETRY_REPLAN.md PHASE 2/3: 뒷축 한 점이 아니라 차체
        4모서리 + 앞면중심에서 각각 ray를 쏴 최솟값을 취한다 — 완전한
        Minkowski 스윕 밴드 거리는 아니지만(그런 계산은 스텝마다 하기엔
        비싸다), 단일점 대비 오버행 자락의 여유 부족을 반영하는 실질적
        근사다. clear_term이 참조하는 clearance의 정의가 여기서 달라진다."""
        fc_x, fc_y = self.front_center(state)
        bus_poly = self._bus_polygon(state)
        origins = [(state.x, state.y), (fc_x, fc_y), *list(bus_poly.exterior.coords)[:-1]]
        min_clear = math.inf
        for ox, oy in origins:
            rays = cast_rays(
                ox, oy, state.theta, self.n_rays, self.sensor_range,
                self.sensor_boundary_line, self.sensor_obstacle_lines,
            )
            min_clear = min(min_clear, min(rays))
        return min_clear

    # ------------------------------------------------------------------ #
    def _sample_start_pose(self):
        """curriculum_level은 "이번 에피소드가 최소 이 정도는 어려워야 한다"는
        하한(floor)이다 — 0=GOAL 근처도 허용(가장 쉬움) ~ 1=항상 실제 START만
        허용(가장 어려움). 매 에피소드 [curriculum_level, 1.0] 구간에서 난이도를
        무작위로 뽑아 reference_path 위 지점을 고른다.

        모든 환경이 동시에 같은 고정 난이도로만 학습하면, 학습 후반 난이도가
        실제 START로 고정되는 순간 정책이 그동안 거의 본 적 없는 먼 거리
        상태를 마주하며 행동이 무너지는 문제가 있었다(분포 밖 입력에 가까운
        action으로 응답 — 1차 학습에서 성공률이 100%->0%로 붕괴한 뒤 회복하지
        못한 원인으로 추정). 매 에피소드 난이도를 섞어 뽑으면 학습 후반에도
        쉬운/어려운 자세가 계속 같이 나와 이 분포 이동 문제를 줄인다."""
        episode_level = (self._rng.uniform(self.curriculum_level, self.curriculum_ceiling)
                         if self.curriculum_level < self.curriculum_ceiling else self.curriculum_ceiling)
        # 2026-07-06: 이전에는 front-center 경로의 접선 방향을 뒷축 heading으로
        # 역산해 썼는데, 전방오버행(d_ref=9.06m)이 R_min(7.37m)보다 커서
        # 급회전 중에는 front-center 경로가 나아가는 방향이 실제 뒷축
        # heading과 크게 어긋난다(직진이 아니면 근사가 거의 항상 깨짐 —
        # 임의 200곳 샘플 중 64.5%가 충돌, 5M-step 학습 성공률 0%의 원인으로
        # 확인됨). reference_path.json에 저장된 뒷축 원본(rear_waypoints)을
        # 역산 없이 직접 보간해서 쓴다.
        s_total = self.reference_path.rear_axle_total_length
        s_target = s_total * (1.0 - episode_level)
        px, py, heading = self.reference_path.rear_axle_pose_at(s_target)

        if self.start_noise_pos <= 0 and self.start_noise_heading <= 0:
            return px, py, heading

        for _ in range(5):
            nx = px + self._rng.normal(0, self.start_noise_pos)
            ny = py + self._rng.normal(0, self.start_noise_pos)
            nh = heading + self._rng.normal(0, self.start_noise_heading)
            candidate = BicycleState(x=nx, y=ny, theta=nh, v=0.0, delta=0.0)
            if not self._check_collision(candidate):
                return nx, ny, nh
        return px, py, heading  # 노이즈 추가 시 충돌 회피 실패 -> 노이즈 없는 자세로 폴백

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        x, y, heading = self._sample_start_pose()
        self.state = BicycleState(x=x, y=y, theta=heading, v=0.0, delta=0.0)
        self._step_count = 0
        self._trajectory = [(self.state.x, self.state.y)]
        # 0=아직 유의미하게 움직인 적 없음, 이후 +1=마지막으로 FWD로 움직였음,
        # -1=마지막으로 REV로 움직였음 (reversal_term 판정용, _reward 참고)
        self._last_gear_sign = 0
        obs = self._observation(self.state)
        info = {}
        return obs, info

    def step(self, action):
        action = np.clip(action, -1.0, 1.0)
        a_accel = float(action[0]) * self.a_max
        a_steer = float(action[1]) * self.steer_rate_max

        prev_state = self.state
        grade_here = self.grade_field.grade_at(prev_state.x, prev_state.y)
        next_state = self.model.step(prev_state, a_accel, a_steer, grade=grade_here)

        reward, terminated, info = self._reward(prev_state, next_state)

        self.state = next_state
        self._step_count += 1
        self._trajectory.append((next_state.x, next_state.y))

        truncated = self._step_count >= self.max_episode_steps
        obs = self._observation(next_state)
        return obs, reward, terminated, truncated, info

    def _reward(self, prev_state: BicycleState, state: BicycleState):
        r = self.rcfg
        info = {}
        terminated = False

        collided = self._check_collision(state, prev_state)
        if collided:
            return -r["collision_penalty"], True, {"collision": True}

        # potential-based shaping: 절대거리가 아니라 "이전 스텝보다 가까워진
        # 정도"에 보상한다. 절대거리 페널티(-w*pose_err 매 스텝)는 충돌 페널티가
        # 유한하면 몇 스텝만 방황해도 누적 페널티가 충돌 페널티를 넘어서서
        # "빨리 충돌해 끝내는" 퇴화 정책이 더 유리해지는 문제가 있었다(TASK8
        # 첫 학습에서 ep_len_mean=13, 성공률 0%로 수렴해 발견). potential
        # shaping은 에피소드 전체 누적합이 (시작거리-끝거리)로 유계되어 이
        # 문제가 없다. 거리 지표는 GOAL까지의 직선거리가 아니라 reference_path
        # 진행률(_progress_metric) 기준 — 진입/진출 경사로가 apex에서 거의
        # 반대 방향으로 꺾여 있어 직선거리 shaping은 벽을 가로지르는 방향을
        # 보상해 학습을 막을 수 있다(도로 전체 확장 작업 중 발견).
        prev_metric = self._progress_metric(prev_state)
        curr_metric = self._progress_metric(state)
        pose_term = r["w_pose"] * (prev_metric - curr_metric)

        success = self._is_success(state)
        goal_term = r["goal_bonus"] if success else 0.0
        if success:
            terminated = True
            info["success"] = True

        clearance = self._clearance(state)
        clear_term = 0.0
        effective_threshold = r["clearance_threshold"] + r.get("boundary_margin_m", 0.0)
        if clearance < effective_threshold:
            clear_term = -r["w_clear"] * (effective_threshold - clearance)

        steer_term = -r["w_steer_smooth"] * abs(state.delta - prev_state.delta)

        grade_here = self.grade_field.grade_at(state.x, state.y)
        # 2026-07-06: 이전에는 prev_state.v/state.v의 부호를 한 스텝만 비교했는데,
        # dt=0.1s·a_max=1.0 하에서는 스텝당 속도 변화가 최대 0.1m/s뿐이라
        # 방향전환은 항상 중간에 v=0.0을 거쳐간다 — "감속->정지->반대가속"이라는
        # 지극히 정상적인 전환 패턴에서 단 한 번도 안 걸렸다(시뮬레이션으로 확인,
        # 예: 0.5->0.4->...->0.0->-0.1->...->-0.5 전체 구간에서 reversed_dir 전부
        # False). 그래서 "마지막으로 유의미하게 움직인 방향"을 기억해뒀다가, 그와
        # 반대 방향으로 다시 움직이기 시작하는 순간(중간에 몇 스텝을 멈춰
        # 있었든 상관없이) 정확히 한 번만 페널티를 매기도록 바꾼다.
        reversal_term = 0.0
        if abs(state.v) > r["stop_speed_thresh"]:
            cur_gear_sign = 1 if state.v > 0 else -1
            if self._last_gear_sign != 0 and cur_gear_sign != self._last_gear_sign:
                reversal_term = -r["w_reversal"] * (1.0 + r["reversal_slope_k"] * grade_here)
            self._last_gear_sign = cur_gear_sign

        stop_term = 0.0
        if abs(state.v) < r["stop_speed_thresh"] and not success:
            stop_term = -r["w_stop_slope"] * grade_here

        time_term = -r["time_penalty"]

        total = pose_term + goal_term + clear_term + steer_term + reversal_term + stop_term + time_term
        return total, terminated, info

    # ------------------------------------------------------------------ #
    def render(self):
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 8))
        px, py = self.boundary_polygon.exterior.xy
        ax.plot(px, py, "k-", linewidth=1)
        for obs in self.obstacle_polygons:
            ox, oy = obs.exterior.xy
            ax.fill(ox, oy, color="gray", alpha=0.5)

        bus_poly = self._bus_polygon(self.state)
        bx, by = bus_poly.exterior.xy
        ax.fill(bx, by, color="orange", alpha=0.6)

        tx, ty = zip(*self._trajectory)
        ax.plot(tx, ty, "b--", linewidth=1)

        ax.plot(self.state.x, self.state.y, "ko", markersize=4, label="rear axle")
        fc_x, fc_y = self.front_center(self.state)
        ax.plot(fc_x, fc_y, "r^", markersize=6, label="front center")
        ax.plot(self.goal_x, self.goal_y, "g*", markersize=14, label="goal")

        ax.set_aspect("equal")
        ax.legend(loc="upper right")
        ax.set_title(f"step={self._step_count} v={self.state.v:.2f} delta={math.degrees(self.state.delta):.1f}deg")

        if self.render_mode == "rgb_array":
            fig.canvas.draw()
            buf = np.asarray(fig.canvas.buffer_rgba())
            plt.close(fig)
            return buf
        return fig
