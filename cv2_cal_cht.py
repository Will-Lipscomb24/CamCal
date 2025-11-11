#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified camera calibration runner driven by YAML config.

Modes:
  - "opencv": OpenCV-based calibration (pinhole or fisheye)
  - "scipy": custom LM (SciPy) calibration with per-parameter constraints

Usage:
  python run_calibration.py --config cal_config.yaml
"""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import yaml

from tqdm import tqdm

import pdb 

try:
    import cv2
except Exception:
    cv2 = None

try:
    from scipy.optimize import least_squares
except Exception:
    least_squares = None


# ============================================================
# Helpers
# ============================================================

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def save_outputs_opencv(cfg, imsize, K, D, rms, flags_value, outdir: Path):
    ensure_dir(outdir)
    payload = {
        "mode": "opencv",
        "image_size": {"w": int(imsize[0]), "h": int(imsize[1])},
        "K": np.asarray(K).tolist(),
        "D": np.asarray(D).tolist(),
        "rms": float(rms),
        "flags": int(flags_value),
    }
    with open(outdir / cfg["output"]["json"], "w") as f:
        json.dump(payload, f, indent=2)

    if cfg["output"].get("yaml"):
        fs = cv2.FileStorage(str(outdir / cfg["output"]["yaml"]), cv2.FILE_STORAGE_WRITE)
        fs.write("K", np.asarray(K))
        fs.write("D", np.asarray(D))
        fs.write("image_width", int(imsize[0]))
        fs.write("image_height", int(imsize[1]))
        fs.write("rms", float(rms))
        fs.release()

def save_outputs_scipy(cfg, Kopt, Dopt, qopt, rms, outdir: Path):
    ensure_dir(outdir)
    payload = {
        "mode": "scipy",
        "K_like": Kopt,
        "distortion": Dopt,
        "attitude_quaternion": (qopt.tolist() if qopt is not None else None),
        "rms": float(rms),
    }
    with open(outdir / cfg["output"]["json"], "w") as f:
        json.dump(payload, f, indent=2)

    if cfg["output"].get("yaml") and cv2 is not None:
        fs = cv2.FileStorage(str(outdir / cfg["output"]["yaml"]), cv2.FILE_STORAGE_WRITE)
        K_cv = np.array([
            [Kopt["dx"], Kopt["alpha"], Kopt["up"]],
            [0.0, Kopt["dy"], Kopt["vp"]],
            [0.0, 0.0, 1.0]
        ], dtype=float)
        D_cv = np.array([Dopt["k1"], Dopt["k2"], Dopt["p1"], Dopt["p2"], Dopt["k3"]])
        fs.write("K", K_cv)
        fs.write("D", D_cv)
        fs.release()


# ============================================================
# OpenCV Mode
# ============================================================

def _opencv_build_pinhole_flags(cfg):
    flags = 0
    f = cfg.get("flags", {})
    C = cv2
    def on(k): return bool(f.get(k, False))
    if on("use_intrinsic_guess"): flags |= C.CALIB_USE_INTRINSIC_GUESS
    if on("fix_principal_point"): flags |= C.CALIB_FIX_PRINCIPAL_POINT
    if on("fix_aspect_ratio"):    flags |= C.CALIB_FIX_ASPECT_RATIO
    if on("zero_tangent_dist"):   flags |= C.CALIB_ZERO_TANGENT_DIST
    if on("rational_model"):      flags |= C.CALIB_RATIONAL_MODEL
    if on("thin_prism_model"):    flags |= C.CALIB_THIN_PRISM_MODEL
    if on("tilted_model"):        flags |= C.CALIB_TILTED_MODEL
    if on("fix_k1"): flags |= C.CALIB_FIX_K1
    if on("fix_k2"): flags |= C.CALIB_FIX_K2
    if on("fix_k3"): flags |= C.CALIB_FIX_K3
    if on("fix_k4"): flags |= C.CALIB_FIX_K4
    if on("fix_k5"): flags |= C.CALIB_FIX_K5
    if on("fix_k6"): flags |= C.CALIB_FIX_K6
    if on("fix_s1_s2_s3_s4"):    flags |= C.CALIB_FIX_S1_S2_S3_S4
    if on("fix_taux_tauy"):      flags |= C.CALIB_FIX_TAUX_TAUY
    if on("fix_focal_length"):   flags |= C.CALIB_FIX_FOCAL_LENGTH
    if on("fix_skew"):           flags |= C.CALIB_FIX_SKEW
    return flags

def _opencv_pattern_points(board_cfg):
    rows = int(board_cfg["rows"])
    cols = int(board_cfg["cols"])
    sq = float(board_cfg["square_size"])
    objp = np.zeros((rows*cols,3), np.float32)
    objp[:,:2] = np.mgrid[0:cols, 0:rows].T.reshape(-1,2)
    objp *= sq
    return objp

def _opencv_detect_points(img, cfg, aruco_dict=None, aruco_params=None):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    p = cfg["board"]["pattern"].lower()
    refine = cfg.get("detector", {}).get("refine_subpix", True)

    if p == "chessboard":
        ret, corners = cv2.findChessboardCorners(
            gray, (cfg["board"]["cols"], cfg["board"]["rows"]),
            flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
        )
        if not ret:
            return False, None
        if refine:
            cv2.cornerSubPix(
                gray, corners, (11,11), (-1,-1),
                (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,30,1e-6)
            )
        return True, corners

    elif p == "circles":
        ret, centers = cv2.findCirclesGrid(
            gray, (cfg["board"]["cols"], cfg["board"]["rows"]),
            flags=cv2.CALIB_CB_SYMMETRIC_GRID
        )
        return ret, centers

    elif p == "charuco":
        corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=aruco_params)
        if len(corners) == 0:
            return False, None
        board = cv2.aruco.CharucoBoard(
            (cfg["board"]["cols"], cfg["board"]["rows"]),
            cfg["board"]["square_size"],
            cfg["board"]["marker_size"],
            aruco_dict
        )
        ret, ch_corners, ch_ids = cv2.aruco.interpolateCornersCharuco(
            corners, ids, gray, board)
        if not ret or ch_corners is None or ch_ids is None or len(ch_corners) < 4:
            return False, None
        return True, (ch_corners, ch_ids, board)

    else:
        raise ValueError("Unknown board.pattern")

def run_opencv_from_yaml(cfg):
    if cv2 is None:
        raise RuntimeError("OpenCV not installed.")

    img_dir = Path(cfg["data"]["images_glob_dir"])
    pattern = cfg["data"]["glob"]
    img_paths = sorted(img_dir.glob(pattern))
    if not img_paths:
        raise RuntimeError(f"No images found in {img_dir} using {pattern}")

    objp = _opencv_pattern_points(cfg["board"])
    imsize = None

    # Storage
    objpoints, imgpoints = [], []
    charuco_corners_list, charuco_ids_list = [], []

    aruco_dict = None
    aruco_params = None
    if cfg["board"]["pattern"].lower() == "charuco":
        aruco_dict = cv2.aruco.getPredefinedDictionary(
            getattr(cv2.aruco, cfg["charuco"]["dictionary"]))
        aruco_params = cv2.aruco.DetectorParameters()

    for p in tqdm(img_paths, total = len(img_paths), desc = 'Processing Images'):
        img = cv2.imread(str(p))
        if img is None:
            print(f"[WARN] cannot read {p}")
            continue
        if imsize is None:
            imsize = (img.shape[1], img.shape[0])
        # pdb.set_trace()
        ok, pts = _opencv_detect_points(img, cfg, aruco_dict, aruco_params)
        if not ok:
            print(f"[INFO] pattern not found: {p}")
            continue

        if cfg["board"]["pattern"].lower() == "charuco":
            ch_c, ch_id, _b = pts
            charuco_corners_list.append(ch_c)
            charuco_ids_list.append(ch_id)
        else:
            objpoints.append(objp)
            imgpoints.append(pts)

    if cfg["board"]["pattern"].lower() == "charuco":
        if len(charuco_corners_list) < 3:
            raise RuntimeError("Not enough ChArUco detections.")
    else:
        if len(objpoints) < 3:
            raise RuntimeError("Not enough detections.")

    max_iter = cfg["solver"].get("max_iter", 200)
    eps = cfg["solver"].get("eps", 1e-8)

    if cfg["fisheye"].get("enable", False):
        K = np.eye(3)
        D = np.zeros((4,1))
        flags = 0
        if cfg["fisheye"].get("recompute_extrinsic", False):
            flags |= cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC
        if cfg["fisheye"].get("fix_skew", False):
            flags |= cv2.fisheye.CALIB_FIX_SKEW
        if cfg["fisheye"].get("fix_k1k2k3k4", False):
            flags |= (cv2.fisheye.CALIB_FIX_K1 |
                      cv2.fisheye.CALIB_FIX_K2 |
                      cv2.fisheye.CALIB_FIX_K3 |
                      cv2.fisheye.CALIB_FIX_K4)
        rms, K, D, rvecs, tvecs = cv2.fisheye.calibrate(
            objpoints, imgpoints, imsize, K, D, flags=flags,
            criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, max_iter, eps)
        )
    else:
        flags = _opencv_build_pinhole_flags(cfg)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                    max_iter, eps)
        if cfg["board"]["pattern"].lower() == "charuco":
            board = cv2.aruco.CharucoBoard(
                (cfg["board"]["cols"], cfg["board"]["rows"]),
                cfg["board"]["square_size"],
                cfg["board"]["marker_size"],
                aruco_dict
            )
            rms, K, D, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(
                charucoCorners=charuco_corners_list,
                charucoIds=charuco_ids_list,
                board=board,
                imageSize=imsize,
                cameraMatrix=None,
                distCoeffs=None,
                flags=flags,
                criteria=criteria
            )
        else:
            rms, K, D, rvecs, tvecs = cv2.calibrateCamera(
                objpoints, imgpoints, imsize, None, None,
                flags=flags, criteria=criteria
            )

    outdir = Path(cfg["output"]["dir"])
    save_outputs_opencv(cfg, imsize, K, D, rms, flags, outdir)

    if cfg["output"].get("save_undistorted_preview", False):
        for p in img_paths[:min(5, len(img_paths))]:
            img = cv2.imread(str(p))
            if cfg["fisheye"].get("enable", False):
                und = cv2.fisheye.undistortImage(img, K, D)
            else:
                und = cv2.undistort(img, K, D)
            cv2.imwrite(str(outdir / ("undist_" + p.name)), und)

    print(f"[OK] OpenCV calibration complete. RMS={rms:.6f}")
    print(f"Results stored at {outdir}")


# ============================================================
# SciPy LM Mode
# ============================================================

def _quat_to_dcm(q):
    w,x,y,z = q
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
        [2*(x*y + z*w), 1-2*(x*x+z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w), 2*(y*z + x*w), 1-2*(x*x+y*y)]
    ], dtype=float)

def _normalize(v):
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    n[n == 0] = 1.0
    return v / n

def _distort_xy(x, y, D):
    r2 = x*x + y*y
    radial = 1 + D["k1"]*r2 + D["k2"]*(r2*r2) + D["k3"]*(r2*r2*r2)
    x_tan = 2*D["p1"]*x*y + D["p2"]*(r2 + 2*x*x)
    y_tan = 2*D["p2"]*x*y + D["p1"]*(r2 + 2*y*y)
    xd = x*radial + x_tan
    yd = y*radial + y_tan
    return xd, yd

def _project_dirs_to_pixels(d_cam, K, D):
    x = d_cam[:,0] / d_cam[:,2]
    y = d_cam[:,1] / d_cam[:,2]
    xd, yd = _distort_xy(x,y,D)
    u = K["dx"]*xd + K["alpha"]*yd + K["up"]
    v = K["dy"]*yd + K["vp"]
    return np.stack([u,v], axis=1)

def _residuals(vec, names, K0, D0, q0, ref_dirs, pix, solve_att):
    # unpack
    K = dict(K0)
    D = dict(D0)
    q = None if q0 is None else q0.copy()

    i = 0
    for n in names:
        v = float(vec[i]); i += 1
        if n in K: K[n] = v
        elif n in D: D[n] = v
        elif n in ["qw","qx","qy","qz"]:
            idx = {"qw":0,"qx":1,"qy":2,"qz":3}[n]
            q[idx] = v

    if q is not None and solve_att:
        q = q / np.linalg.norm(q)

    if solve_att and q is not None:
        C = _quat_to_dcm(q)
        d_cam = (C @ ref_dirs.T).T
    else:
        d_cam = ref_dirs

    d_cam = _normalize(d_cam)
    uv_pred = _project_dirs_to_pixels(d_cam, K, D)
    return (uv_pred - pix).ravel()

def run_scipy_from_yaml(cfg):
    if least_squares is None:
        raise RuntimeError("SciPy not installed.")

    data = np.load(cfg["data"]["points_npz"])
    ref_dirs = data["ref_dirs"].astype(float)
    pix_uv = data["pix_uv"].astype(float)

    sc = cfg["scipy_calib"]
    init = sc["init"]

    K0 = {
        "dx": init["dx"], "dy": init["dy"], "alpha": init["alpha"],
        "up": init["up"], "vp": init["vp"]
    }
    D0 = {
        "k1": init["k1"], "k2": init["k2"], "k3": init["k3"],
        "p1": init["p1"], "p2": init["p2"]
    }
    q0 = None
    if init.get("q") is not None:
        q0 = np.array(init["q"], float)

    opt = sc.get("optimize", {})
    fixed_values = dict(opt.get("fixed_values", {}))
    zero_params  = list(opt.get("zero_params", []))
    free_params  = opt.get("free_params", None)
    solve_att    = bool(opt.get("solve_attitude", False))

    for z in zero_params:
        fixed_values[z] = 0.0

    all_names = list(K0.keys()) + list(D0.keys())
    if solve_att and q0 is not None:
        all_names += ["qw","qx","qy","qz"]

    Kw = dict(K0)
    Dw = dict(D0)
    qw = None if q0 is None else q0.copy()

    for n, val in fixed_values.items():
        if n in Kw: Kw[n] = val
        elif n in Dw: Dw[n] = val
        elif n in ["qw","qx","qy","qz"]:
            idx = {"qw":0,"qx":1,"qy":2,"qz":3}[n]
            if qw is None:
                qw = np.array([1,0,0,0], float)
            qw[idx] = val

    fixed_names = set(fixed_values.keys())
    if free_params is None:
        free_set = set(all_names) - fixed_names
    else:
        free_set = set(free_params) - fixed_names

    x0 = []
    names = []
    for n in all_names:
        if n not in free_set:
            continue
        if n in Kw: x0.append(Kw[n])
        elif n in Dw: x0.append(Dw[n])
        elif n in ["qw","qx","qy","qz"]:
            idx = {"qw":0,"qx":1,"qy":2,"qz":3}[n]
            x0.append(qw[idx])
        names.append(n)

    x0 = np.array(x0, float)

    solver = sc.get("solver", {})
    method = solver.get("method", "lm")
    max_iter = solver.get("max_iter", 200)
    eps = solver.get("eps", 1e-12)

    res = least_squares(
        _residuals, x0, method=method,
        args=(names, K0, D0, q0, ref_dirs, pix_uv, solve_att),
        max_nfev=max_iter,
        ftol=eps, xtol=eps, gtol=eps,
        verbose=2
    )

    Kopt = dict(Kw)
    Dopt = dict(Dw)
    qopt = None if qw is None else qw.copy()

    i = 0
    for n in names:
        v = float(res.x[i]); i += 1
        if n in Kopt: Kopt[n] = v
        elif n in Dopt: Dopt[n] = v
        elif n in ["qw","qx","qy","qz"]:
            idx = {"qw":0,"qx":1,"qy":2,"qz":3}[n]
            qopt[idx] = v

    if qopt is not None and solve_att:
        qopt = qopt / np.linalg.norm(qopt)

    if solve_att and qopt is not None:
        C = _quat_to_dcm(qopt)
        d_cam = (C @ ref_dirs.T).T
    else:
        d_cam = ref_dirs

    d_cam = _normalize(d_cam)
    uv_pred = _project_dirs_to_pixels(d_cam, Kopt, Dopt)
    rms = float(np.sqrt(np.mean(np.sum((uv_pred - pix_uv)**2, axis=1))))

    outdir = Path(cfg["output"]["dir"])
    save_outputs_scipy(cfg, Kopt, Dopt, qopt, rms, outdir)

    print(f"[OK] SciPy LM calibration complete. RMS={rms:.6f}")
    print(f"Results stored at {outdir}")


# ============================================================
# Main Entry
# ============================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    mode = cfg["mode"].lower()
    if mode == "opencv":
        run_opencv_from_yaml(cfg)
    elif mode == "scipy":
        run_scipy_from_yaml(cfg)
    else:
        raise ValueError(f"Unknown mode: {mode}")


if __name__ == "__main__":
    main()