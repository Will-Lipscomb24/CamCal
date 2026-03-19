import os, shutil
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
import json
from pathlib import Path
import pdb



######## Inputs ################
HERE                    = Path(__file__).resolve().parent
REPO_ROOT               = HERE.parent
EXAMPLES_ROOT           = REPO_ROOT.parent / "sc-pose-utils" / "src" / "sc_pose" / "examples"

DATA_FOLDER             = EXAMPLES_ROOT / "artifacts" / "offset" / "expm_003"
DATA_NAME               = DATA_FOLDER.name
IMAGE_DIR               = DATA_FOLDER / "images"
IMG_SUFFIX              = ".png"
VICON_CSV_PATH          = DATA_FOLDER / "vicon_data.csv"
CALIBRATION_YAML_PATH   = DATA_FOLDER / "calibration.yaml"

RESULT_PATH             = HERE / "results" / 'test_001'
OPENCV_POSE_EST_PATH    = RESULT_PATH / "calc_camera_poses.csv"
OUTPUT_JSON_PATH        = RESULT_PATH / "offset_results.json"
REPROJECTION_DIR        = RESULT_PATH / "reprojection"

SQUARES_X       = 9          # columns
SQUARES_Y       = 5          # rows
SQUARE_LEN      = 17e-3      # meters
MARKER_LEN      = 12e-3      # meters
DICTIONARY      = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)

########### Load Camera Intrinsics and Distortion Coeffiecients  ############
if os.path.exists(RESULT_PATH):
    print(f"Warning: {RESULT_PATH} already exists. Deleting and recreating.")
    shutil.rmtree(RESULT_PATH)
RESULT_PATH.mkdir(parents = True, exist_ok = True)
REPROJECTION_DIR.mkdir(parents = True, exist_ok = True)
IMAGE_PATH  = str(IMAGE_DIR)  # e.g. "data/images"

with open(CALIBRATION_YAML_PATH) as f:
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

with open(VICON_CSV_PATH, newline='') as f:
    rows = list(csv.DictReader(f))

image_numbers = [int(r['image_number']) for r in rows]

cam_VCv_V       = np.array([[float(r['cam_x']),  float(r['cam_y']),  float(r['cam_z'])]  for r in rows])/1000
soho_VTv_V      = np.array([[float(r['soho_x']), float(r['soho_y']), float(r['soho_z'])] for r in rows])/1000
q_V_Cv          = np.array([[float(r['cam_qw']),  float(r['cam_qx']),  float(r['cam_qy']),  float(r['cam_qz'])]  for r in rows])
q_V_Tv          = np.array([[float(r['soho_qw']), float(r['soho_qx']), float(r['soho_qy']), float(r['soho_qz'])] for r in rows])
rotations_cam   = R.from_quat(q_V_Cv,  scalar_first=True).as_matrix()
rotations_soho  = R.from_quat(q_V_Tv, scalar_first=True).as_matrix()
from sc_pose.mathtils.quaternion import q2trfm, q2rotm
Trfms_Cv_V      = np.zeros((len(q_V_Cv), 3, 3))
Trfms_Tv_V      = np.zeros((len(q_V_Tv), 3, 3))
for i in range(len(q_V_Cv)):
    Trfms_Cv_V[i] = q2rotm(q_V_Cv[i])
    Trfms_Tv_V[i] = q2rotm(q_V_Tv[i])
sc1             = np.max(Trfms_Cv_V - rotations_cam)
sc2             = np.max(Trfms_Tv_V - rotations_soho)
print(f"Max difference between q2rotm and R.from_quat for camera: {sc1:.18f}")
print(f"Max difference between q2rotm and R.from_quat for target: {sc2:.18f}")
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

T_CvV   = build_transformations( Trfms_Cv_V, cam_VCv_V   )  # (N, 4, 4)
T_TvV   = build_transformations( Trfms_Tv_V, soho_VTv_V )  # (N, 4, 4)

# ---------------------------------------------------------------
# ChArUco Board Setup
# ---------------------------------------------------------------

board       = cv2.aruco.CharucoBoard((SQUARES_X, SQUARES_Y),
                                     SQUARE_LEN,
                                     MARKER_LEN,
                                     DICTIONARY)
detector    = cv2.aruco.CharucoDetector(board)

def get_camera_to_board_pose(image, K, dist):
    img     = cv2.imread(image)
    gray    = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    corners, ids, _ = aruco.detectMarkers(gray, DICTIONARY)

    if ids is None or len(ids) < 4:
        print(f"Skipping — insufficient markers: {os.path.basename(image)}")
        return None, None, None

    retval, charucoCorners, charucoIds  = aruco.interpolateCornersCharuco(
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

    T_BC            = np.eye(4)
    Trfm_B_C, _     = cv2.Rodrigues(rvec)
    T_BC[:3, :3]    = Trfm_B_C
    T_BC[:3, 3]     = tvec.flatten() 
    return T_BC, rvec, tvec


# Obtain a stack of the T_CB transformation for all of the images
def get_TBC_series(image_dir, K, dist):
    image_paths = sorted(
        glob.glob(os.path.join(image_dir, "*.png")),
        key=lambda x: int(re.search(r'(\d+)', Path(x).stem).group(1))
    )
    T_BC_list, valid_image_numbers = [], []

    for path in image_paths:
        img_num = int(re.search(r'(\d+)', Path(path).stem).group(1))  # e.g. 1,2,3,5...
        T_BC, _, _ = get_camera_to_board_pose(path, K, dist)
        if T_BC is None:
            print(f"Detection failed: {os.path.basename(path)}")
            continue
        T_BC_list.append(T_BC)
        valid_image_numbers.append(img_num)

    print(f"Processed {len(T_BC_list)} / {len(image_paths)} images successfully")
    return np.stack(T_BC_list), valid_image_numbers
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
    T           = np.eye(4)
    R, _        = cv2.Rodrigues(p[:3])
    T[:3, :3]   = R
    T[:3, 3]    = p[3:]
    return T

def T_to_params(T):
    """Convert 4x4 transform to 6-param vector [rvec(3), tvec(3)]"""
    rvec, _     = cv2.Rodrigues(T[:3, :3])
    tvec        = T[:3, 3]
    return np.hstack([rvec.flatten(), tvec.flatten()])


def residuals(params, T_BC_array, T_CvV_array, T_TvV_array):
    T_CvC   = params_to_T(params[:6])
    T_TvT   = params_to_T(params[6:])
    res     = []

    for T_BC_obs, T_CvV, T_TvV in zip(T_BC_array, T_CvV_array, T_TvV_array):

        # T_CCv = inv_T(T_CvC)
        # # 1. Prediction Chain
        # T_TvCv = inv_T(T_VCv) @ T_VTv
        # T_BC_pred = T_CCv @ T_CvTv @ T_TvT

        # # 2. Matrix Difference (Algebraic)
        # # We only use the top 3x4 block because the bottom row is [0,0,0,1] 
        # # for both, meaning the difference is always zero (useless for the solver).
        
        T_BC_pred   = T_CvC @ (inv_T(T_CvV) @ T_TvV) @ inv_T(T_TvT)
        diff        = T_BC_obs[:3, :] - T_BC_pred[:3, :]
        
        # 3. Flatten into 12 residuals per observation
        # The sum of squares of these 12 values == Frobenius Norm Squared
        res.append(diff.flatten())

    return np.concatenate(res)



def solve_rwhe(T0_flat, T_BC_array, T_CvV_array, T_TvV_array):
    """
    Jointly solves for T_CSC and T_TST via Levenberg-Marquardt.

    Returns:
        T_CSC  (4x4) — camera optical frame → Vicon camera marker frame
        T_TST  (4x4) — board/target frame   → Vicon target marker frame
        result       — full scipy OptimizeResult
    """
    print(f"Solving RWHE with {len(T_BC_array)} measurements...")

    res0 = residuals(T0_flat, T_BC_array, T_CvV_array, T_TvV_array)
    print(f"Initial cost: {0.5 * np.sum(res0**2):.6f}")

    result = least_squares(
        residuals,
        T0_flat,
        args=(T_BC_array, T_CvV_array, T_TvV_array),
        method='lm',
        verbose=1
    )

    print(f"Final cost:  {result.cost:.6f}")
    print(f"Termination: {result.message}")

    T_CvC, T_TvT = params_to_T(result.x[:6]), params_to_T(result.x[6:])
    return T_CvC, T_TvT, result

####### Manually Measured Offset Estimation for Initialization  ###########
T0_cam              = np.eye(4)
T0_cam[:3, :3]      = np.eye(3)
T0_cam[:3, 3]       = np.array([0, 0, 0])

target_offset       = np.array([-228, -55, 30])/1000  
T0_target           = np.eye(4)
T0_target[:3, :3]   = np.eye(3)
T0_target[:3, 3]    = target_offset
T0_flat             = np.concatenate([T_to_params(T0_cam), T_to_params(T0_target)])

####### Manually Measured Offset Estimation for Initialization  ###########


T_BC_array, valid_image_numbers = get_TBC_series(IMAGE_PATH, K, dist)


# Keep only images that have a corresponding Vicon row
common_numbers  = [n for n in valid_image_numbers if n in image_numbers]

T_BC_sync_idx   = [valid_image_numbers.index(n) for n in common_numbers]

vicon_sync_idx  = [image_numbers.index(n) for n in common_numbers]

T_BC_sync       = T_BC_array[T_BC_sync_idx]
T_CvV_sync      = T_CvV[vicon_sync_idx]
T_TvV_sync      = T_TvV[vicon_sync_idx]

outlier_image_numbers   = [42, 43, 44, 45, 46, 47] # hold out

mask    = np.ones(len(T_BC_sync), dtype=bool)
for i, image_number in enumerate(common_numbers):
    if image_number in outlier_image_numbers:
        mask[i] = False

T_BC_sync       = T_BC_sync[mask]
T_CvV_sync      = T_CvV_sync[mask]
T_TvV_sync      = T_TvV_sync[mask]
common_numbers  = [n for i, n in enumerate(common_numbers) if mask[i]]

print(f"T_BC: {len(T_BC_sync)}, T_CvV: {len(T_CvV_sync)}, T_TvV: {len(T_TvV_sync)}")  


# 3. Solve
T_CvC, T_TvT, result = solve_rwhe(T0_flat, T_BC_sync, T_CvV_sync, T_TvV_sync)
print("\n Homogenous Transformation Matrix from Camera Vicon to True Camera Frame:")
print(T_CvC)

print("\n Homogenous Transformation Matrix from Target Vicon to True Target Frame:")
print(T_TvT)

with open(OUTPUT_JSON_PATH, "w") as f:
    json.dump({
        "T_CvC": T_CvC.tolist(),
        "T_TvT": T_TvT.tolist()
    }, f, indent=4)

print(result.fun.shape)
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
print(f"Results saved to {RESULT_PATH} with offets at {OUTPUT_JSON_PATH}")