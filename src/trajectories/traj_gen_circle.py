
# --- Trajectory Gen with Body Frame Axes (No Pauses) ---
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import os 
import pdb

output_dir = '/home/will/projects/EVision/robot_trajectories/data'
# --- Trajectory parameters ---

traj_time = 120       # seconds (total duration of trajectory)
N_points = traj_time * 60    # <-- control total samples here
testbed   = 6.0 #  testbed size (meters)
margin    = 0.2

n_revs    = 1 # Number of spiral revolutions
#target    = np.array([-1166.0, 561.0, 0])*1e-3 # Soho's static position

target = np.array([0,0,0])
hz = N_points / traj_time  # Effective frequency of waypoints

t  = np.linspace(0, traj_time, N_points)   # time axis spans 0→traj_time but only N points
dt = t[1] - t[0]                       # derived dt
N  = N_points

r =  1   # radius of circle 
omega = 2 * np.pi * 1 / 30     # angular velocity

s_off = 135
#px = target[0] + r * np.cos(omega * t - s_off*np.pi/180)
px = np.full((N,),-1.0)
# py = target[1] + r * np.sin(omega * t- s_off*np.pi/180)
py = np.full((N,),-1.0)
pz = np.zeros(N)

r = 1                                # constant radius
pos = np.stack([px, py, pz], axis=0)

print(f"Total trajectory duration: {t[-1]:.1f}s  ({N} samples)")

# --- Linear Velocity ---
vel = np.diff(pos, axis=1) / dt
vel = np.hstack([vel, vel[:, -1:]])

# --- Rotation Matrix Helper ---
def rotm_to_quat(R):
    trace = R[0,0] + R[1,1] + R[2,2]
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2,1] - R[1,2]) * s
        y = (R[0,2] - R[2,0]) * s
        z = (R[1,0] - R[0,1]) * s
    elif R[0,0] > R[1,1] and R[0,0] > R[2,2]:
        s = 2.0 * np.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2])
        w = (R[2,1] - R[1,2]) / s
        x = 0.25 * s
        y = (R[0,1] + R[1,0]) / s
        z = (R[0,2] + R[2,0]) / s
    elif R[1,1] > R[2,2]:
        s = 2.0 * np.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2])
        w = (R[0,2] - R[2,0]) / s
        x = (R[0,1] + R[1,0]) / s
        y = 0.25 * s
        z = (R[1,2] + R[2,1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1])
        w = (R[1,0] - R[0,1]) / s
        x = (R[0,2] + R[2,0]) / s
        y = (R[1,2] + R[2,1]) / s
        z = 0.25 * s
    return np.array([w, x, y, z])

def quat_mult(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])

def build_frame(p):
    x_body = target - p
    if np.linalg.norm(x_body) < 1e-6:
        x_body = np.array([1.0, 0.0, 0.0])
    x_body /= np.linalg.norm(x_body)

    z_body = np.array([0.0, 0.0, 1.0])

    y_body = np.cross(z_body, x_body)
    if np.linalg.norm(y_body) < 1e-6:
        y_body = np.array([0.0, 1.0, 0.0])
    y_body /= np.linalg.norm(y_body)

    z_body = np.cross(x_body, y_body)
    z_body /= np.linalg.norm(z_body)

    R = np.column_stack([x_body, y_body, z_body])
    return x_body, y_body, z_body, R

# --- Quaternions & Angular Velocity ---
quats = np.zeros((4, N))
R_all = []
for i in range(N):
    _, _, _, R = build_frame(pos[:, i])
    quats[:, i] = rotm_to_quat(R)
    R_all.append(R)

ang_vel = np.zeros((3, N))
for i in range(N - 1):
    q1 = quats[:, i]
    q2 = quats[:, i + 1]
    dq = (q2 - q1) / dt
    q1_conj = np.array([q1[0], -q1[1], -q1[2], -q1[3]])
    oq = quat_mult(q1_conj, dq)
    ang_vel[:, i] = 2 * oq[1:]
ang_vel[:, -1] = ang_vel[:, -2]

# --- Discretize ---
every_n     = max(1, int(round((N_points / traj_time / hz))))
idx         = np.arange(0, N_points, every_n)

t_d       = t[idx]
pos_d     = pos[:, idx]
vel_d     = vel[:, idx]
quats_d   = quats[:, idx]
ang_vel_d = ang_vel[:, idx]
R_d       = [R_all[i] for i in idx]


# --- Pan / Tilt ---
tilt_deg = 0.0
pan_deg  = 0.0
ramp_dur = 1.0   # seconds to reach 15°

pan_pos_deg = np.where(t < ramp_dur,
                       pan_deg * (t / ramp_dur),   # linear ramp
                       pan_deg)                     # hold at 15°
pan_pos_rad = np.deg2rad(pan_pos_deg)

# velocity: constant during ramp, zero after
pan_vel_rad = np.where(t < ramp_dur,
                       np.deg2rad(pan_deg) / ramp_dur,
                       0.0)



tilt_pos_deg = np.where(t < ramp_dur,
                       tilt_deg * (t / ramp_dur),   # linear ramp
                       tilt_deg)                     # hold at 15°
tilt_pos_rad = np.deg2rad(tilt_pos_deg)

# velocity: constant during ramp, zero after
tilt_vel_rad = np.where(t < ramp_dur,
                       np.deg2rad(tilt_deg) / ramp_dur,
                       0.0)




df = pd.DataFrame({
    'timestamp':   t,
    'px': pos[0],     'py': pos[1],     'pz': pos[2],
    'qw': quats[0],   'qx': quats[1],   'qy': quats[2],   'qz': quats[3],
    'vx': vel[0],     'vy': vel[1],     'vz': vel[2],
    'wx': ang_vel[0], 'wy': ang_vel[1], 'wz': ang_vel[2],
    'pan_pos':  pan_pos_rad,
    'tilt_pos': tilt_pos_rad,
    'pan_vel':  pan_vel_rad,
    'tilt_vel': tilt_vel_rad,
})

tilt_radians = -tilt_deg*np.pi/180

df['tilt_pos'] = tilt_radians

file_path = os.path.join(output_dir, f'trajectory_{hz:.2f}hz_circle.csv' )

df.iloc[idx].to_csv(file_path, index=False)

# --- Plots ---
from matplotlib.lines import Line2D

axis_len = 0.25
fig = plt.figure(figsize=(14, 9))

# 1. Top-down view with body axes
ax1 = fig.add_subplot(2, 3, 1)
# Testbed rect centered on target
rect = plt.Polygon([
    [-3, -3], [+3, -3],
    [+3, +3], [-3, +3]
], fill=False, edgecolor='r', linewidth=2)
ax1.add_patch(rect)
ax1.plot(px, py, 'b', linewidth=1.5)
ax1.plot(target[0], target[1], 'r*', markersize=12, label='Target')
ax1.plot(px[0], py[0], 'go', markersize=8, label='Start')
ax1.plot(px[-1], py[-1], 'rs', markersize=8, label='End')

# for i, R in enumerate(R_d):
#     ox, oy = pos_d[0, i], pos_d[1, i]
#     ax1.quiver(ox, oy, R[0,0], R[1,0],
#                color='r', scale=1/axis_len, scale_units='xy', angles='xy', width=0.004)
#     ax1.quiver(ox, oy, R[0,1], R[1,1],
#                color='g', scale=1/axis_len, scale_units='xy', angles='xy', width=0.004)

# Fix: use Line2D proxies instead of empty quiver calls
legend_elements = [
    Line2D([0], [0], color='b', linewidth=1.5, label='Trajectory'),
    Line2D([0], [0], marker='*', color='r', markersize=10, linewidth=0, label='Target'),
    Line2D([0], [0], marker='o', color='g', markersize=8, linewidth=0, label='Start'),
    Line2D([0], [0], color='r', linewidth=2, label='X (→ target)'),
    Line2D([0], [0], color='g', linewidth=2, label='Y'),
]
ax1.set_xlim(-3.5, +3.5)
ax1.set_ylim(-3.5, +3.5)
ax1.set_aspect('equal'); ax1.grid(True)
ax1.set_title('Top View — X(red)→target, Y(green)')
ax1.set_xlabel('X (m)'); ax1.set_ylabel('Y (m)')
ax1.legend(handles=legend_elements, fontsize=8)

# 2. 3D trajectory with body axes
ax2 = fig.add_subplot(2, 3, 2, projection='3d')
ax2.plot(px, py, pz, 'b', linewidth=1.5)
# Fix: plot target at correct position, not origin
ax2.scatter(target[0], target[1], target[2], color='r', s=80, marker='*', label='Target')
ax2.set_xlabel('X (m)'); ax2.set_ylabel('Y (m)'); ax2.set_zlabel('Z (m)')
ax2.set_title('3D — X(red)→target, Y(green), Z(blue)↑')
ax2.legend(fontsize=8)

# 3. Quaternion
ax3 = fig.add_subplot(2, 3, 3)
for val, label in zip(quats, ['w','x','y','z']):
    ax3.plot(t, val, label=label)
ax3.set_title('Quaternion'); ax3.set_xlabel('t (s)')
ax3.legend(); ax3.grid(True)

# 4. Linear velocity
ax4 = fig.add_subplot(2, 3, 4)
for val, label in zip(vel, ['vx','vy','vz']):
    ax4.plot(t, val, label=label)
ax4.set_title('Linear Velocity (m/s)'); ax4.set_xlabel('t (s)')
ax4.legend(); ax4.grid(True)

# 5. Angular velocity
ax5 = fig.add_subplot(2, 3, 5)
for val, label in zip(ang_vel, ['wx','wy','wz']):
    ax5.plot(t, val, label=label)
ax5.set_title('Angular Velocity (rad/s)'); ax5.set_xlabel('t (s)')
ax5.legend(); ax5.grid(True)

# 6. Orbital radius over time (measured from target, not origin)
ax6 = fig.add_subplot(2, 3, 6)
orbital_r = np.sqrt((px - target[0])**2 + (py - target[1])**2)
ax6.plot(t, orbital_r, 'b', label='Orbital radius')
ax6.axhline(r, color='r', linestyle='--', label=f'Expected r={r}m')
ax6.set_title('Orbital Radius from Target'); ax6.set_xlabel('t (s)'); ax6.set_ylabel('r (m)')
ax6.set_ylim(0, r * 1.5)
ax6.legend(); ax6.grid(True)

plt.tight_layout()
plt.show()