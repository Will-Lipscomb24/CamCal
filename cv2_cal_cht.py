#!/usr/bin/env python3
"""
OpenCV camera calibration runner driven by config.

Usage:
  python calibrate_opencv.py --config cal_config.yaml
"""
import cv2
import yaml
import json
import argparse
import numpy as np
from pathlib import Path

def build_flags(cfg):
    f = 0
    C = cv2
    # pinhole flags
    if cfg["flags"].get("use_intrinsic_guess", False): f |= C.CALIB_USE_INTRINSIC_GUESS
    if cfg["flags"].get("fix_principal_point", False): f |= C.CALIB_FIX_PRINCIPAL_POINT
    if cfg["flags"].get("fix_aspect_ratio", False):    f |= C.CALIB_FIX_ASPECT_RATIO
    if cfg["flags"].get("zero_tangent_dist", False):   f |= C.CALIB_ZERO_TANGENT_DIST
    if cfg["flags"].get("rational_model", False):      f |= C.CALIB_RATIONAL_MODEL
    if cfg["flags"].get("thin_prism_model", False):    f |= C.CALIB_THIN_PRISM_MODEL
    if cfg["flags"].get("tilted_model", False):        f |= C.CALIB_TILTED_MODEL
    if cfg["flags"].get("fix_k1", False):              f |= C.CALIB_FIX_K1
    if cfg["flags"].get("fix_k2", False):              f |= C.CALIB_FIX_K2
    if cfg["flags"].get("fix_k3", False):              f |= C.CALIB_FIX_K3
    if cfg["flags"].get("fix_k4", False):              f |= C.CALIB_FIX_K4
    if cfg["flags"].get("fix_k5", False):              f |= C.CALIB_FIX_K5
    if cfg["flags"].get("fix_k6", False):              f |= C.CALIB_FIX_K6
    if cfg["flags"].get("fix_s1_s2_s3_s4", False):     f |= C.CALIB_FIX_S1_S2_S3_S4
    if cfg["flags"].get("fix_taux_tauy", False):       f |= C.CALIB_FIX_TAUX_TAUY
    if cfg["flags"].get("fix_focal_length", False):    f |= C.CALIB_FIX_FOCAL_LENGTH
    if cfg["flags"].get("fix_skew", False):            f |= C.CALIB_FIX_SKEW
    return f

def pattern_points(cfg):
    squaresX = cfg["board"]["cols"]
    squaresY = cfg["board"]["rows"]
    squareSize = float(cfg["board"]["square_size"])
    objp = np.zeros((squaresY*squaresX,3), np.float32)
    objp[:,:2] = np.mgrid[0:squaresX, 0:squaresY].T.reshape(-1,2)
    objp *= squareSize
    return objp

def detect_points(img, cfg, aruco_dict=None, aruco_params=None):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    pt = cfg["board"]["pattern"].lower()
    refine = cfg["detector"].get("refine_subpix", True)

    if pt == "chessboard":
        ret, corners = cv2.findChessboardCorners(gray,
                    (cfg["board"]["cols"], cfg["board"]["rows"]),
                    flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)
        if not ret: return False, None
        if refine:
            cv2.cornerSubPix(gray, corners, (11,11), (-1,-1),
                             (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6))
        return True, corners

    elif pt == "circles":
        ret, centers = cv2.findCirclesGrid(gray,
                    (cfg["board"]["cols"], cfg["board"]["rows"]),
                    flags=cv2.CALIB_CB_SYMMETRIC_GRID)
        return ret, centers

    elif pt == "charuco":
        corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=aruco_params)
        if len(corners) == 0: return False, None
        nrows, ncols = cfg["board"]["rows"], cfg["board"]["cols"]
        board = cv2.aruco.CharucoBoard(
            (ncols, nrows),
            cfg["board"]["square_size"],
            cfg["board"]["marker_size"],
            aruco_dict
        )
        ret, ch_corners, ch_ids = cv2.aruco.interpolateCornersCharuco(corners, ids, gray, board)
        if not ret or ch_corners is None or ch_ids is None or len(ch_corners) < 4:
            return False, None
        return True, (ch_corners, ch_ids, board)

    else:
        raise ValueError("Unknown pattern type")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    img_paths = sorted([str(p) for p in Path(cfg["data"]["images_glob_dir"]).glob(cfg["data"]["glob"])])
    if len(img_paths) == 0:
        raise RuntimeError("No images found")

    objp = pattern_points(cfg)
    objpoints = []
    imgpoints = []
    imsize = None

    # Optional ChArUco dictionary/params
    aruco_dict = aruco_params = None
    if cfg["board"]["pattern"].lower() == "charuco":
        aruco_dict = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, cfg["charuco"]["dictionary"]))
        aruco_params = cv2.aruco.DetectorParameters()

    for p in img_paths:
        img = cv2.imread(p)
        if img is None: 
            print(f"Skip unreadable {p}")
            continue
        if imsize is None:
            imsize = (img.shape[1], img.shape[0])

        ok, pts = detect_points(img, cfg, aruco_dict, aruco_params)
        if not ok:
            print(f"Pattern not found: {p}")
            continue

        if cfg["board"]["pattern"].lower() == "charuco":
            ch_corners, ch_ids, board = pts
            # Convert to chessboard-like sets per view for calibrateCameraCharuco
            objpoints.append(board.chessboardCorners)  # nominal corners (3D)
            imgpoints.append(ch_corners)
        else:
            objpoints.append(objp)
            imgpoints.append(pts)

    if cfg["fisheye"]["enable"]:
        # Fisheye model
        K = np.eye(3)
        D = np.zeros((4,1))
        flags = 0
        if cfg["fisheye"].get("recompute_extrinsic", False):
            flags |= cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC
        if cfg["fisheye"].get("fix_skew", False):
            flags |= cv2.fisheye.CALIB_FIX_SKEW
        if cfg["fisheye"].get("fix_k1k2k3k4", False):
            flags |= cv2.fisheye.CALIB_FIX_K1 + cv2.fisheye.CALIB_FIX_K2 + cv2.fisheye.CALIB_FIX_K3 + cv2.fisheye.CALIB_FIX_K4
        rms, K, D, rvecs, tvecs = cv2.fisheye.calibrate(
            objpoints, imgpoints, imsize, K, D, flags=flags,
            criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, cfg["solver"]["max_iter"], cfg["solver"]["eps"])
        )
    else:
        # Pinhole model
        flags = build_flags(cfg)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                    cfg["solver"]["max_iter"], cfg["solver"]["eps"])

        if cfg["board"]["pattern"].lower() == "charuco":
            # Calibrate via ChArUco helper
            board = cv2.aruco.CharucoBoard(
                (cfg["board"]["cols"], cfg["board"]["rows"]),
                cfg["board"]["square_size"],
                cfg["board"]["marker_size"],
                aruco_dict
            )
            # OpenCV provides a dedicated function:
            rms, K, D, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(
                charucoCorners=imgpoints,
                charucoIds=[None]*len(imgpoints),  # already baked
                board=board,
                imageSize=imsize,
                cameraMatrix=None,
                distCoeffs=None,
                flags=flags,
                criteria=criteria
            )
        else:
            rms, K, D, rvecs, tvecs = cv2.calibrateCamera(
                objpoints, imgpoints, imsize, None, None, flags=flags, criteria=criteria
            )

    print(f"RMS reprojection error: {rms:.6f}")
    out = {
        "image_size": {"w": imsize[0], "h": imsize[1]},
        "K": K.tolist(),
        "D": D.tolist(),
        "rms": float(rms),
        "flags": int(flags)
    }

    Path(cfg["output"]["dir"]).mkdir(parents=True, exist_ok=True)
    with open(Path(cfg["output"]["dir"])/cfg["output"]["json"], "w") as f:
        json.dump(out, f, indent=2)

    if cfg["output"].get("yaml"):
        fs = cv2.FileStorage(str(Path(cfg["output"]["dir"])/cfg["output"]["yaml"]), cv2.FILE_STORAGE_WRITE)
        fs.write("K", K); fs.write("D", D); fs.write("image_width", imsize[0]); fs.write("image_height", imsize[1])
        fs.write("rms", float(rms)); fs.release()

    if cfg["output"].get("save_undistorted_preview", False):
        for p in img_paths[:min(5, len(img_paths))]:
            img = cv2.imread(p)
            undist = cv2.undistort(img, K, D) if not cfg["fisheye"]["enable"] else cv2.fisheye.undistortImage(img, K, D)
            outp = Path(cfg["output"]["dir"]) / ("undist_" + Path(p).name)
            cv2.imwrite(str(outp), undist)

    print(f"Saved results to: {cfg['output']['dir']}")

if __name__ == "__main__":
    main()
