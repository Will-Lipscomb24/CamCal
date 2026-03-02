import os
import csv
import cv2
import cv2.aruco as aruco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R
import yaml
import glob
import re
import matplotlib.pyplot as plt

######## Inputs ################
IMAGE_PATH    = "/home/will/projects/CamCal/data/offset_images/"  # glob pattern
CONFIG_PATH    = "/home/will/projects/CamCal/configs/calibration.yaml"
SQUARES_X     = 7          # columns
SQUARES_Y     = 5          # rows
SQUARE_LEN    = 20e-3      # meters
MARKER_LEN    = 15e-3      # meters
DICTIONARY = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)

########### Load Camera Intrinsics and Distortion Coeffiecients  ############
with open(CONFIG_PATH) as f:
    data = yaml.safe_load(f)

K = np.array([
    [data['fx'], 0,            data['cx']],
    [0,            data['fy'], data['cy']],
    [0,            0,            1]
], dtype=np.float64)

dist = np.array([
    data['k1'],
    data['k2'],
    data['p1'],
    data['p2'],
    data['k3']
], dtype=np.float64).reshape(1, 5)


# --------------------------------------------------------------
# Load and Read Vicon CSV File 
# ---------------------------------------------------------------

with open("/home/will/projects/CamCal/data/vicon_data/vicon_data.csv", newline='') as f:
    rows = list(csv.DictReader(f))

image_numbers = [int(r['image_number']) for r in rows]

cam_tL  = np.array([[float(r['cam_x']),  float(r['cam_y']),  float(r['cam_z'])]  for r in rows])/1000
soho_tL = np.array([[float(r['soho_x']), float(r['soho_y']), float(r['soho_z'])] for r in rows])/1000
cam_qG  = np.array([[float(r['cam_qw']),  float(r['cam_qx']),  float(r['cam_qy']),  float(r['cam_qz'])]  for r in rows])
soho_qG = np.array([[float(r['soho_qw']), float(r['soho_qx']), float(r['soho_qy']), float(r['soho_qz'])] for r in rows])

rotations_cam  = R.from_quat(cam_qG,  scalar_first=True).as_matrix()
rotations_soho = R.from_quat(soho_qG, scalar_first=True).as_matrix()

# ---------------------------------------------------------------
# Build 4x4 Transforms from Vicon Data
# ---------------------------------------------------------------

def build_transformations(rotations, translations):
    N            = len(rotations)
    T            = np.zeros((N, 4, 4))
    T[:, :3, :3] = rotations
    T[:, :3,  3] = translations
    T[:,  3,  3] = 1.0
    return T


def inv_T(T):
   T_inv = np.eye(4)
   R_transpose = T[:3, :3].T
   T_inv[:3, :3] = R_transpose
   T_inv[:3, 3] = -R_transpose @ T[:3, 3]
   return T_inv

# T_VCv is Vicon frame to camera frame
# T_VTv is Vicon frame to board frame
T_VCv = build_transformations( rotations_cam, cam_tL)   # (N, 4, 4)
T_VTv = build_transformations( rotations_soho, soho_tL)  # (N, 4, 4)

# ---------------------------------------------------------------
# ChArUco Board Setup
# ---------------------------------------------------------------

board      = cv2.aruco.CharucoBoard((SQUARES_X, SQUARES_Y),
                                     SQUARE_LEN,
                                     MARKER_LEN,
                                     DICTIONARY)
detector   = cv2.aruco.CharucoDetector(board)

def get_camera_to_board_pose(image, K, dist):
    img = cv2.imread(image)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    corners, ids, _ = aruco.detectMarkers(gray, DICTIONARY)

    if ids is None or len(ids) < 4:
        print(f"Skipping — insufficient markers: {os.path.basename(image)}")
        return None, None, None

    retval, charucoCorners, charucoIds = aruco.interpolateCornersCharuco(
        corners, ids, gray, board
    )

    if charucoCorners is None or charucoIds is None :
        print(f"Skipping — insufficient charuco corners: {os.path.basename(image)}")
        return None, None, None

    success, rvec, tvec = aruco.estimatePoseCharucoBoard(
        charucoCorners, charucoIds, board, K, dist, None, None
    )

    if not success:
        print(f"Skipping — pose estimation failed: {os.path.basename(image)}")
        return None, None, None

    T_CB = np.eye(4)
    R, _ = cv2.Rodrigues(rvec)
    T_CB[:3, :3] = R
    T_CB[:3,  3] = tvec.flatten() 
    return T_CB, rvec, tvec


# Obtain a stack of the T_CB transformation for all of the images
def get_TCB_series(image_dir, K, dist):
    image_paths = sorted(
        glob.glob(os.path.join(image_dir, "*.png")),
        key=lambda x: int(re.search(r'(\d+)', x).group(1))
    )
    T_CB_list, valid_image_numbers = [], []

    for path in image_paths:
        img_num = int(re.search(r'(\d+)', path).group(1))  # e.g. 1,2,3,5...
        T_CB, _, _ = get_camera_to_board_pose(path, K, dist)
        if T_CB is None:
            print(f"Detection failed: {os.path.basename(path)}")
            continue
        T_CB_list.append(T_CB)
        valid_image_numbers.append(img_num)

    print(f"Processed {len(T_CB_list)} / {len(image_paths)} images successfully")
    return np.stack(T_CB_list), valid_image_numbers
# ---------------------------------------------------------------
# RWHE Helpers
# ---------------------------------------------------------------

def rvec_to_T(rvec, tvec):
    """
    Converts Rodrigues vector + translation into a 4x4 transform.
    Called every iteration by the optimizer to reconstruct T_CSC and T_TST
    from the flat 12-parameter vector.
    """
    T            = np.eye(4)
    T[:3, :3], _ = cv2.Rodrigues(np.array(rvec, dtype=np.float64))
    T[:3,  3]    = np.array(tvec, dtype=np.float64).flatten()
    return T


def params_to_T(p):
    """Converts 6 params (3 rot, 3 trans) to 4x4 matrix"""
    T = np.eye(4)
    R, _ = cv2.Rodrigues(p[:3])
    T[:3, :3] = R
    T[:3, 3] = p[3:]
    return T

def T_to_params(T):
    """Convert 4x4 transform to 6-param vector [rvec(3), tvec(3)]"""
    rvec, _ = cv2.Rodrigues(T[:3, :3])
    tvec = T[:3, 3]
    return np.hstack([rvec.flatten(), tvec.flatten()])


# def residuals(params, T_CB_array, T_VCv_array, T_VTv_array, weight_rot=1.0):
#     """
#     Geodesic residual function.
#     params: 12-element vector [x, y, z, rx, ry, rz] for CvC and TvT
#     weight_rot: scalar to balance translation (e.g., meters) vs rotation (radians)
#     """
#     T_CvC = params_to_T(params[:6])      # Cv ← C
#     T_TvT = params_to_T(params[6:])      # Tv ← T
    
#     # Pre-calculate inverse if needed for the chain
#     T_CCv = inv_T(T_CvC)
    
#     res = []

#     for T_CB_obs, T_VCv, T_VTv in zip(T_CB_array, T_VCv_array, T_VTv_array):
#         # 1. Calculate Predicted Transformation
#         # Cv ← Tv
#         T_CvTv = inv_T(T_VCv) @ T_VTv
#         # Predicted: B ← C
#         T_CB_pred = T_CCv @ T_CvTv @ T_TvT

#         # 2. Extract Rotation and Translation
#         R_obs = T_CB_obs[:3, :3]
#         t_obs = T_CB_obs[:3, 3]
        
#         R_pred = T_CB_pred[:3, :3]
#         t_pred = T_CB_pred[:3, 3]

#         # 3. Translation Residual (3 components)
#         t_err = t_obs - t_pred

#         # 4. Rotation Residual (Geodesic distance / Angle)
#         # The relative rotation R_diff takes us from Pred to Obs
#         R_diff = R_obs.T @ R_pred
        
#         # Calculate the angle of rotation (clamping for numerical stability)
#         cos_theta = (np.trace(R_diff) - 1.0) / 2.0
#         angle_err = np.arccos(cos_theta)

#         # 5. Append components
#         # Note: We append the angle_err as a single value. 
#         # To keep the Jacobian 'square-ish', you could also use Axis-Angle (3 values).
#         # Here we use the 3 translation errors + 1 weighted scalar angle error.
#         res.append(np.concatenate([t_err, [weight_rot * angle_err]]))

#     return np.concatenate(res)

# def residuals(params, T_CB_array, T_VCv_array, T_VTv_array, w_rot=1.0, w_trans=1.0):
#     T_CvC = params_to_T(params[:6])      
#     T_TvT = params_to_T(params[6:])      
#     T_CCv = inv_T(T_CvC)
#     res = []
#     for T_CB_obs, T_VCv, T_VTv in zip(T_CB_array, T_VCv_array, T_VTv_array):
        
#         T_CvTv = inv_T(T_VCv) @ T_VTv
#         T_CB_pred = T_CCv @ T_CvTv @ T_TvT

#         # Split rotation (top-left 3x3) and translation (top-right 3x1)
#         rot_diff   = (T_CB_obs[:3, :3] - T_CB_pred[:3, :3]).flatten() * w_rot
#         trans_diff = (T_CB_obs[:3,  3] - T_CB_pred[:3,  3]).flatten() * w_trans

#         res.append(np.concatenate([rot_diff, trans_diff]))
#     return np.concatenate(res)

def residuals(params, T_CB_array, T_VCv_array, T_VTv_array):
    T_CvC = params_to_T(params[:6])      
    T_TvT = params_to_T(params[6:])      
    
    
    res = []

    for T_CB_obs, T_VCv, T_VTv in zip(T_CB_array, T_VCv_array, T_VTv_array):

        T_CCv = inv_T(T_CvC)
        # 1. Prediction Chain
        T_CvTv = inv_T(T_VCv) @ T_VTv
        T_CB_pred = T_CCv @ T_CvTv @ T_TvT

        # 2. Matrix Difference (Algebraic)
        # We only use the top 3x4 block because the bottom row is [0,0,0,1] 
        # for both, meaning the difference is always zero (useless for the solver).
        diff = T_CB_obs[:3, :] - T_CB_pred[:3, :]
        
        # 3. Flatten into 12 residuals per observation
        # The sum of squares of these 12 values == Frobenius Norm Squared
        res.append(diff.flatten())

    return np.concatenate(res)



def solve_rwhe(T0_flat, T_CB_array, T_VCv, T_VTv):
    """
    Jointly solves for T_CSC and T_TST via Levenberg-Marquardt.

    Returns:
        T_CSC  (4x4) — camera optical frame → Vicon camera marker frame
        T_TST  (4x4) — board/target frame   → Vicon target marker frame
        result       — full scipy OptimizeResult
    """
    print(f"Solving RWHE with {len(T_CB_array)} measurements...")

    res0 = residuals(T0_flat, T_CB_array, T_VCv, T_VTv)
    print(f"Initial cost: {0.5 * np.sum(res0**2):.6f}")

    result = least_squares(
        residuals,
        T0_flat,
        args=(T_CB_array, T_VCv, T_VTv),
        method='lm',
        verbose=1
    )

    print(f"Final cost:  {result.cost:.6f}")
    print(f"Termination: {result.message}")

    T_CvC, T_TvT = params_to_T(result.x[:6]), params_to_T(result.x[6:])
    return T_CvC, T_TvT, result

####### Manually Measured Offset Estimation for Initialization  ###########
T0_cam = np.eye(4)
T0_cam[:3, :3] = np.eye(3)
T0_cam[:3, 3]  = np.array([0, 0, 0])

target_offset = np.array([-228, -55, 30])/1000  
T0_target = np.eye(4)
T0_target[:3, :3] = np.eye(3)
T0_target[:3, 3]  = target_offset

T0_flat = np.concatenate([T_to_params(T0_cam), T_to_params(T0_target)])

####### Manually Measured Offset Estimation for Initialization  ###########


T_CB_array, valid_image_numbers = get_TCB_series(IMAGE_PATH, K, dist)



for i, (num, T) in enumerate(zip(valid_image_numbers, T_CB_array)):
    rvec, _ = cv2.Rodrigues(T[:3, :3])
    angle = np.linalg.norm(rvec) * 180 / np.pi
    tvec = T[:3, 3]
    print(f"Image {num}: angle={angle:.1f}°  t={tvec}")

# Keep only images that have a corresponding Vicon row
common_numbers = [n for n in valid_image_numbers if n in image_numbers]

T_CB_sync_idx  = [valid_image_numbers.index(n) for n in common_numbers]

vicon_sync_idx = [image_numbers.index(n) for n in common_numbers]

T_CB_sync  = T_CB_array[T_CB_sync_idx]
T_VCv_sync = T_VCv[vicon_sync_idx]
T_VTv_sync = T_VTv[vicon_sync_idx]

outlier_indices = [3, 5, 7, 22, 26, 28, 30, 32, 37, 38, 47, 49]

mask = np.ones(len(T_CB_sync), dtype=bool)
mask[outlier_indices] = False

T_CB_sync  = T_CB_sync[mask]
T_VCv_sync = T_VCv_sync[mask]
T_VTv_sync = T_VTv_sync[mask]
common_numbers = [n for i, n in enumerate(common_numbers) if mask[i]]

print(f"T_CB: {len(T_CB_sync)}, T_VCv: {len(T_VCv_sync)}, T_VTv: {len(T_VTv_sync)}")  


# 3. Solve
T_CvC, T_TvT, result = solve_rwhe(T0_flat, T_CB_sync, T_VCv_sync, T_VTv_sync)
print("\nCamera in Vicon Camera Frame:")
print(T_CvC)

print("\nTarget in Vicon Target Frame:")
print(T_TvT)
# Residual per observation (should be small and uniform)
res = result.fun.reshape(-1, 12)  # 12 residuals per image
per_obs_cost = np.sum(res**2, axis=1)
print("Per-observation costs:")
for i, c in enumerate(per_obs_cost):
    print(f"  Image {i}: {c:.6f}")

# Flag outliers
mean_cost = np.mean(per_obs_cost)
for i, c in enumerate(per_obs_cost):
    if c > 3 * mean_cost:
        print(f"  WARNING: Image {i} is an outlier ({c:.4f} vs mean {mean_cost:.4f})")

