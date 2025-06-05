
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.optimize import least_squares 
import pandas as pd
import cv2
from python_pose_terrier.src.pose_utils import Camera, Projection

SENSOR_SIZE = (14.13,10.35) #mm
PIXEL_SIZE = 0.00345 #mm
FOCAL_LENGTH = 25 #mm
RES= (4096, 3000)

#Import vicon data and assign headers
headers = ["dotx", "doty", "dotz", "dotqw", "dotqx", "dotqy", "dotqz",
           "camx", "camy", "camz", "camqw", "camqx", "camqy", "camqz"]
df = pd.read_csv("calibration_vicon_data/vicon_data.csv", header=None, names=headers)
df.to_csv("calibration_vicon_data/vicon_data_mod.csv", index=False)
df_G2L = pd.read_csv("calibration_vicon_data/vicon_data_mod.csv")

#Reorder columns to match the expected format for scipy
cam_tL = df[["camx", "camy", "camz"]].values
dot_tL = df[["dotx", "doty", "dotz"]].values
cam_qG = df[["camqw", "camqx", "camqy", "camqz"]]
cam_qG_df = pd.DataFrame(cam_qG)
cam_qG_reorder = cam_qG_df.loc[:, ["camqx", "camqy", "camqz", "camqw"]].values
rotations_cam = R.from_quat(cam_qG_reorder).as_matrix()

#Rotate from global frame to camera frame
P_local = np.zeros((len(rotations_cam), 3))
for i in range(len(rotations_cam)):
    rotation_G2L = rotations_cam[i]
    P_local[i,:] = rotation_G2L.T @ (dot_tL[i,:] - cam_tL[i,:])
    #Account for offset from the focal plane
    P_local[i,1] += 25.35
    P_local[i,0] -= 10.75
df2 = pd.DataFrame(P_local, columns=["cam2markerx", "cam2markery", "cam2markerz"])

#Analytical Camera Matrix
K0  = Camera.camera_matrix(RES[0], RES[1], SENSOR_SIZE[0], SENSOR_SIZE[1], FOCAL_LENGTH)
dist = np.zeros(5)

def project(
                              df: np.ndarray
                            , K: np.ndarray
                            , dist: np.ndarray = None
        ) -> np.ndarray:
        """
        Projecting 3D keypoints to 2D image coordinates
        Input:
            q: quaternion (4,) (np.ndarray)
            r: position (3,) (np.ndarray)
            K: camera intrinsic matrix (3,3) (np.ndarray)
            keypoints: 3D keypoints (N, 3) (np.ndarray)
            dist: distortion coefficients (5,) (np.ndarray), defaults to np.zeros(5) if None
        Output:
            points2D: 2D keypoints (N, 2) (np.ndarray)
        Src: https://github.com/tpark94/spnv2/blob/dbcf0de8813da56529bb7467a87c6cdacfc46d23/core/utils/postprocess.py#L78
        """
        if dist is None:
            dist        = np.zeros(5)
        # perspective division, this step is to convert from 3D to 2D
        x0        = df[:, 0] / df[:, 2]
        y0        = df[:, 1] / df[:, 2]
        D1       = dist[0]
        D2       = dist[1]
        D3       = dist[2]
        D4       = dist[3]
        D5       = dist[4]
        # radial distortion
        r2          = x0**2 + y0**2
        cdist       = 1 + D1*r2 + D2*r2**2 + D3*r2**3
        # tangential distortion
        xdist       = x0*cdist + 2*D4*x0*y0 + D5*(r2 + 2*x0**2)
        ydist       = y0*cdist + D4*(r2 + 2*y0**2) + 2*D5*x0*y0
        # apply camera matrix
        u           = K[0, 0]*xdist + K[0, 2]
        v           = K[1, 1]*ydist + K[1, 2]
        points2D    = np.vstack((u, v)).T
        return points2D
    
img_path= f"calibration_images/cal_image_{i+1}.png"
img_to_project = cv2.imread(img_path)
pixel_coords = pd.read_csv("pixels.csv")

def reprojection_residual(cam_params):
    K = [cam_params[0], 0, cam_params[1]
         , 0, cam_params[2], cam_params[3]
         , 0, 0, 1]
    K = np.array(K).reshape((3, 3))
    reproj = project(df2.to_numpy(), K, cam_params[4:])
    manual_pixels = pixel_coords.loc[:,["x", "y"]].values
    residuals = (manual_pixels - reproj).ravel() 
    return residuals

initial_guess = [K0[0, 0], K0[0, 2], K0[1, 1], K0[1, 2]] + list(dist)
lower_bounds = [0, 0, 0, 0] + [-1, -1, -1, -1, -1]
upper_bounds = [np.inf, RES[0], np.inf, RES[1]] + [1, 1, 1, 1, 1]
#result = least_squares(reprojection_residual, initial_guess,jac='3-point',loss='soft_l1', method='dogbox')
result = least_squares(reprojection_residual, initial_guess, method='lm')

K_solved = [result.x[0], 0, result.x[1]
            , 0, result.x[2], result.x[3]
            , 0, 0, 1]
K_solved = np.array(K_solved).reshape((3, 3))
dist_solved = result.x[4:]

print("Analytical Camera Matrix:\n", K0)
print(f"Optimized Camera Matrix:\n{K_solved}")
print(f"Optimized Distortion Coefficients:\n{dist_solved}")

#Project the Solved Camera Matrix and Distortion Coefficients
# onto the 3D keypoints to visualize the reprojection
analy_kps_2D_proj   = project(
                                df2.to_numpy()
                                , K_solved
                                , dist_solved
                                ).round().astype(int)

for i in range(len(pixel_coords)):
    project_coords = analy_kps_2D_proj[i,:]
    base_image = f"calibration_images/cal_image_{i+1}.png"
    img = cv2.imread(base_image)
    # Draw the projected points on the image
    cv2.circle(img, (project_coords[0], project_coords[1]), 25, (255, 0, 0), 20)
    cv2.imwrite(f"projected_images/proj_img_{i+1}.png", img)