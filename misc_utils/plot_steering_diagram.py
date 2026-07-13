"""사용자 요청: 버스 몸체 전체가 아니라 "바퀴가 실제로 꺾이는 각도"
기준으로 조향 기하를 도식화. 전륜(앞바퀴)만 조향하는 표준 바이시클
모델에서, 뒷축 중심 회전반경 R과 앞바퀴 조향각 delta_max, 축거 L의
관계(R = L / tan(delta_max))를 그림으로 확인."""
import math

import matplotlib.pyplot as plt
import numpy as np
import yaml

from kinematics.bicycle import BicycleModel

with open("config.yaml", "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

model = BicycleModel(cfg)
L = model.L
R = model.R  # 보정된 뒷축 중심 회전반경
delta_max = model.delta_max
T_f = model.T_f
R_spec = model.R_min

fig, axes = plt.subplots(1, 2, figsize=(13, 6.5))

# --- 좌: 평면도 (뒷축, 앞축, 조향각, 회전중심) ---
ax = axes[0]
# 뒷축 (원점), 앞축 (0, L) 방향으로 놓고 위에서 본 그림 (x=옆, y=진행방향)
rear_axle = (0, 0)
front_axle = (0, L)
turn_center = (-R, 0)  # 좌회전 기준

ax.plot([rear_axle[0]], [rear_axle[1]], "ko", markersize=8)
ax.annotate("rear axle\n(state (x,y,theta) reference)", rear_axle, textcoords="offset points", xytext=(10, -25))
ax.plot([front_axle[0]], [front_axle[1]], "bo", markersize=8)
ax.annotate("front axle", front_axle, textcoords="offset points", xytext=(10, 5))

# 차체 축선
ax.plot([rear_axle[0], front_axle[0]], [rear_axle[1], front_axle[1]], "k-", linewidth=2, label=f"wheelbase L={L:.3f}m")

# 앞바퀴 방향 (delta_max 만큼 꺾임)
wheel_len = 1.2
wx = front_axle[0] + wheel_len * math.sin(delta_max)
wy = front_axle[1] + wheel_len * math.cos(delta_max)
ax.plot([front_axle[0], wx], [front_axle[1], wy], "r-", linewidth=3, label=f"front wheel @ delta_max={math.degrees(delta_max):.1f}deg")
ax.plot([front_axle[0], front_axle[0]], [front_axle[1], front_axle[1] + wheel_len], "b--", linewidth=1, alpha=0.5, label="wheel neutral (straight)")

# 회전 중심과 두 반경
ax.plot([turn_center[0]], [turn_center[1]], "g*", markersize=16, label=f"turn center")
theta = np.linspace(0, 2 * math.pi, 200)
ax.plot(turn_center[0] + R * np.cos(theta), turn_center[1] + R * np.sin(theta), "g--", linewidth=1, label=f"rear-axle turning circle R={R:.3f}m")
R_front = math.hypot(R, L)
ax.plot(turn_center[0] + R_front * np.cos(theta), turn_center[1] + R_front * np.sin(theta), "b:", linewidth=1, label=f"front-axle circle ~{R_front:.3f}m")

ax.plot([turn_center[0], rear_axle[0]], [turn_center[1], rear_axle[1]], "g-", linewidth=1, alpha=0.6)
ax.plot([turn_center[0], front_axle[0]], [turn_center[1], front_axle[1]], "b-", linewidth=1, alpha=0.6)

ax.set_aspect("equal")
ax.set_xlim(-R - 3, 5)
ax.set_ylim(-3, R + 3)
ax.legend(loc="lower left", fontsize=8)
ax.set_title("Front-wheel-steer bicycle model geometry (plan view)")
ax.set_xlabel("m")
ax.set_ylabel("m")

# --- 우: R_spec(공표/추정 최소회전반경) -> R(뒷축) -> delta_max 유도 텍스트 ---
ax2 = axes[1]
ax2.axis("off")
text = (
    f"INPUTS (current config.yaml)\n"
    f"  Wheelbase L                = {L:.3f} m\n"
    f"  Front track width T_f      = {T_f:.3f} m\n"
    f"  Spec/estimated R_min       = {R_spec:.3f} m  (basis: {model.R_min_basis}, ESTIMATED)\n"
    f"\n"
    f"CONVERSION (measurement basis: {model.R_min_basis})\n"
    f"  R = sqrt(R_spec^2 - L^2) - T_f/2\n"
    f"    = sqrt({R_spec:.3f}^2 - {L:.3f}^2) - {T_f:.3f}/2\n"
    f"    = {R:.3f} m   <- rear-axle turning radius\n"
    f"\n"
    f"  delta_max = atan(L / R)\n"
    f"            = atan({L:.3f} / {R:.3f})\n"
    f"            = {math.degrees(delta_max):.2f} deg   <- max FRONT WHEEL steer angle\n"
    f"\n"
    f"CAVEATS:\n"
    f"  - R_spec={R_spec:.1f}m has NO official manufacturer source\n"
    f"    (confirmed via web search of Hyundai catalogs) - still a guess,\n"
    f"    labeled 'estimated' in code, not a verified spec.\n"
    f"  - Wheelbase L={L:.3f}m comes from this project's bus_model.dxf,\n"
    f"    but Hyundai's official Universe SPACE catalog (matching width\n"
    f"    2.495m exactly) lists wheelbase = 5.850m, not {L:.3f}m.\n"
    f"    This discrepancy needs separate confirmation (see report)."
)
ax2.text(0.02, 0.98, text, transform=ax2.transAxes, fontsize=10.5, va="top", family="monospace")

fig.tight_layout()
fig.savefig("outputs/steering_geometry_diagram.png", dpi=140)
print("saved outputs/steering_geometry_diagram.png")
print(f"L={L} R_spec={R_spec} T_f={T_f} R={R} delta_max_deg={math.degrees(delta_max)}")
