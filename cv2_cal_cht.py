#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified camera calibration runner driven by YAML config (no ruamel.yaml).

Modes:
  - "opencv": OpenCV-based calibration (pinhole or fisheye) with robust detectors
  - "scipy": custom LM (SciPy) calibration with per-parameter constraints

Usage:
  python run_calibration.py --config cal_config.yaml
"""

import argparse
import json
import math
import pdb  # Debugger available on demand
from pathlib import Path
import sys

import numpy as np
import yaml  # for reading config only
from tqdm import tqdm

import pdb

# Optional deps
try:
    import cv2
except Exception:
    cv2 = None

try:
    from scipy.optimize import least_squares
except Exception:
    least_squares = None


# ============================================================
# Helpers & Math
# ============================================================

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def _compute_intrinsic_extras(fx, fy, skew, cx, cy, image_w=None, image_h=None):
    """
    Compute convenience metrics:
      - aspect_ratio = fx/fy
      - fx_norm = fx / image_w
      - fy_norm = fy / image_h
      - fov_x_deg, fov_y_deg, fov_diag_deg (pinhole assumptions)
      - f_theta_deg: alias for diagonal FoV
    If image size is unknown, FoVs and normalized values are None.
    """
    extras = {}
    extras["aspect_ratio"] = (fx / fy) if fy != 0 else None
    extras["fx_norm"] = (fx / float(image_w)) if image_w and image_w > 0 else None
    extras["fy_norm"] = (fy / float(image_h)) if image_h and image_h > 0 else None

    if image_w and image_h and fx > 0 and fy > 0:
        fov_x = 2.0 * math.degrees(math.atan((image_w) / (2.0 * fx)))
        fov_y = 2.0 * math.degrees(math.atan((image_h) / (2.0 * fy)))
        diag = math.hypot(image_w, image_h)
        f_eq = math.sqrt(fx * fy)  # approximate equivalent focal along diagonal
        fov_diag = 2.0 * math.degrees(math.atan(diag / (2.0 * f_eq)))
        extras["fov_x_deg"] = fov_x
        extras["fov_y_deg"] = fov_y
        extras["fov_diag_deg"] = fov_diag
        extras["f_theta_deg"] = fov_diag
    else:
        extras["fov_x_deg"] = None
        extras["fov_y_deg"] = None
        extras["fov_diag_deg"] = None
        extras["f_theta_deg"] = None

    return extras

def compute_skew_extras_from_K(K, tol=1e-9):
    """
    From a 3x3 intrinsic matrix K, return:
      - skew_from_K: raw K[0,1]
      - skew_clean: 0 if |skew| < tol, else raw
      - pixel_axes_angle_deg: angle between pixel x/y axes in degrees
        Using s = fx * cot(theta)  =>  theta = atan2(fx, s); as s->0, theta->90°
    """
    fx = float(K[0,0])
    s  = float(K[0,1])
    skew_from_K = s
    skew_clean  = 0.0 if abs(s) < tol else s
    theta_deg = math.degrees(math.atan2(fx, s)) if fx > 0 else 90.0
    return skew_from_K, skew_clean, theta_deg

def compute_skew_extras_from_alpha(dx, alpha, tol=1e-9):
    """
    SciPy-mode version:
      dx: focal in x (pixels)
      alpha: stored skew term
    Returns same triplet as above.
    """
    s = float(alpha)
    skew_from_K = s
    skew_clean  = 0.0 if abs(s) < tol else s
    theta_deg   = math.degrees(math.atan2(float(dx), s)) if dx > 0 else 90.0
    return skew_from_K, skew_clean, theta_deg

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


# ============================================================
# Output Writers (JSON + hand-written YAML with inline comments)
# ============================================================

def _fmt_yaml_scalar(val):
    """Render Python values to YAML scalars (simple, safe)."""
    if val is None:
        return "null"
    if isinstance(val, bool):
        return "true" if val else "false"
    if isinstance(val, (int, np.integer)):
        return f"{int(val)}"
    if isinstance(val, float) or isinstance(val, np.floating):
        # compact float formatting
        return f"{float(val):.12g}"
    if isinstance(val, (list, tuple)):
        inner = ", ".join(_fmt_yaml_scalar(v) for v in val)
        return f"[{inner}]"
    s = str(val)
    # quote only if necessary
    if any(c in s for c in [":", "#", "{", "}", "[", "]", ",", "'"]):
        return '"' + s.replace('"', '\\"') + '"'
    return s

def _write_yaml_kv(fh, key, value, comment=None):
    line = f"{key}: {_fmt_yaml_scalar(value)}"
    if comment:
        line += f"  # {comment}"
    fh.write(line + "\n")

def _write_yaml_blank(fh):
    fh.write("\n")

def _write_yaml_header(fh, title):
    fh.write(f"# ---- {title} ----\n")

def _writer_opencv_yaml_with_comments(yaml_path: Path, imsize, K, D, rms, flags_value):
    iw, ih = int(imsize[0]), int(imsize[1])
    fx = float(K[0,0]); fy = float(K[1,1])
    cx = float(K[0,2]); cy = float(K[1,2])

    skew_from_K, skew_clean, pixel_axes_angle_deg = compute_skew_extras_from_K(K)
    extras = _compute_intrinsic_extras(fx, fy, skew_from_K, cx, cy, iw, ih)

    # Distortion values mapped to full set
    all_dist_keys = ["k1","k2","k3","k4","k5","k6","p1","p2","s1","s2","s3","s4","tauX","tauY"]
    dist_desc = {
        "k1":"Radial distortion (primary)",
        "k2":"Radial distortion (secondary)",
        "k3":"Radial distortion (tertiary)",
        "k4":"Higher-order radial (rational model)",
        "k5":"Higher-order radial",
        "k6":"Higher-order radial",
        "p1":"Tangential distortion x (decentering/tilt)",
        "p2":"Tangential distortion y (decentering/tilt)",
        "s1":"Thin prism distortion",
        "s2":"Thin prism distortion",
        "s3":"Thin prism distortion",
        "s4":"Thin prism distortion",
        "tauX":"Tilted sensor term (x)",
        "tauY":"Tilted sensor term (y)"
    }
    D_flat = np.asarray(D).reshape(-1).tolist()
    ocv_names = ["k1","k2","p1","p2","k3","k4","k5","k6","s1","s2","s3","s4","tauX","tauY"]
    dist_vals = {k: 0.0 for k in all_dist_keys}
    for i, v in enumerate(D_flat):
        if i < len(ocv_names):
            dist_vals[ocv_names[i]] = float(v)

    with open(yaml_path, "w") as fh:
        _write_yaml_kv(fh, "mode", "opencv", "Calibration mode used (opencv or scipy)")
        _write_yaml_kv(fh, "image_w", iw, "Image width in pixels")
        _write_yaml_kv(fh, "image_h", ih, "Image height in pixels")
        _write_yaml_blank(fh)

        _write_yaml_header(fh, "Intrinsics (pinhole)")
        _write_yaml_kv(fh, "fx", fx, "Focal length in x (pixels): controls horizontal FOV")
        _write_yaml_kv(fh, "fy", fy, "Focal length in y (pixels): controls vertical FOV")
        _write_yaml_kv(fh, "cx", cx, "Principal point x-coordinate (optical center)")
        _write_yaml_kv(fh, "cy", cy, "Principal point y-coordinate (optical center)")
        _write_yaml_kv(fh, "skew_from_K", skew_from_K, "Raw skew read from K[0,1] (pixels)")
        _write_yaml_kv(fh, "skew_clean", skew_clean, "Skew after thresholding small values to 0 (numerical cleanup)")
        _write_yaml_kv(fh, "pixel_axes_angle_deg", pixel_axes_angle_deg, "Angle between pixel axes (~90° if skew≈0)")
        _write_yaml_blank(fh)

        _write_yaml_header(fh, "Distortion (always listed; zero if unused)")
        # Radials first for readability
        for k in ["k1","k2","k3","k4","k5","k6"]:
            _write_yaml_kv(fh, k, dist_vals[k], dist_desc[k])
        # Then tangentials & others
        for k in ["p1","p2","s1","s2","s3","s4","tauX","tauY"]:
            _write_yaml_kv(fh, k, dist_vals[k], dist_desc[k])
        _write_yaml_blank(fh)

        _write_yaml_header(fh, "Derived Intrinsics & Quality")
        _write_yaml_kv(fh, "aspect_ratio", extras["aspect_ratio"], "fx / fy; ~1 for square pixels")
        _write_yaml_kv(fh, "fx_norm", extras["fx_norm"], "fx normalized by image width")
        _write_yaml_kv(fh, "fy_norm", extras["fy_norm"], "fy normalized by image height")
        _write_yaml_kv(fh, "fov_x_deg", extras["fov_x_deg"], "Horizontal field of view (deg)")
        _write_yaml_kv(fh, "fov_y_deg", extras["fov_y_deg"], "Vertical field of view (deg)")
        _write_yaml_kv(fh, "fov_diag_deg", extras["fov_diag_deg"], "Diagonal field of view (deg)")
        _write_yaml_kv(fh, "f_theta_deg", extras["f_theta_deg"], "Alias of diagonal FOV")
        _write_yaml_kv(fh, "rms", rms, "Mean reprojection error (pixels) — lower is better")
        _write_yaml_kv(fh, "flags", int(flags_value), "OpenCV calibration flags bitmask used during solve")

def _writer_scipy_yaml_with_comments(yaml_path: Path, cfg, Kopt, Dopt, qopt, rms):
    # optional image size hint to compute FoVs in SciPy mode
    hint = cfg["output"].get("image_size_hint", {})
    iw = int(hint["w"]) if "w" in hint else None
    ih = int(hint["h"]) if "h" in hint else None

    fx = float(Kopt["dx"])
    fy = float(Kopt["dy"])
    cx = float(Kopt["up"])
    cy = float(Kopt["vp"])
    skew_from_K, skew_clean, pixel_axes_angle_deg = compute_skew_extras_from_alpha(fx, float(Kopt["alpha"]))
    extras = _compute_intrinsic_extras(fx, fy, skew_from_K, cx, cy, iw, ih)

    # Distortion dict to full set
    all_dist_keys = ["k1","k2","k3","k4","k5","k6","p1","p2","s1","s2","s3","s4","tauX","tauY"]
    dist_desc = {
        "k1":"Radial distortion (primary)",
        "k2":"Radial distortion (secondary)",
        "k3":"Radial distortion (tertiary)",
        "k4":"Higher-order radial (rational model)",
        "k5":"Higher-order radial",
        "k6":"Higher-order radial",
        "p1":"Tangential distortion x (decentering/tilt)",
        "p2":"Tangential distortion y (decentering/tilt)",
        "s1":"Thin prism distortion",
        "s2":"Thin prism distortion",
        "s3":"Thin prism distortion",
        "s4":"Thin prism distortion",
        "tauX":"Tilted sensor term (x)",
        "tauY":"Tilted sensor term (y)"
    }
    dist_vals = {k: 0.0 for k in all_dist_keys}
    for k, v in Dopt.items():
        if k in dist_vals:
            dist_vals[k] = float(v)

    with open(yaml_path, "w") as fh:
        _write_yaml_kv(fh, "mode", "scipy", "Calibration mode used (opencv or scipy)")
        _write_yaml_kv(fh, "image_w", iw, "Image width in pixels (hint if not known)")
        _write_yaml_kv(fh, "image_h", ih, "Image height in pixels (hint if not known)")
        _write_yaml_blank(fh)

        _write_yaml_header(fh, "Intrinsics (pinhole-like)")
        _write_yaml_kv(fh, "fx", fx, "Focal length in x (pixels): controls horizontal FOV")
        _write_yaml_kv(fh, "fy", fy, "Focal length in y (pixels): controls vertical FOV")
        _write_yaml_kv(fh, "cx", cx, "Principal point x-coordinate (optical center)")
        _write_yaml_kv(fh, "cy", cy, "Principal point y-coordinate (optical center)")
        _write_yaml_kv(fh, "skew_from_K", skew_from_K, "Raw skew from model 'alpha' (pixels)")
        _write_yaml_kv(fh, "skew_clean", skew_clean, "Skew after thresholding small values to 0 (numerical cleanup)")
        _write_yaml_kv(fh, "pixel_axes_angle_deg", pixel_axes_angle_deg, "Angle between pixel axes (~90° if skew≈0)")
        _write_yaml_blank(fh)

        _write_yaml_header(fh, "Distortion (always listed; zero if unused)")
        for k in ["k1","k2","k3","k4","k5","k6"]:
            _write_yaml_kv(fh, k, dist_vals[k], dist_desc[k])
        for k in ["p1","p2","s1","s2","s3","s4","tauX","tauY"]:
            _write_yaml_kv(fh, k, dist_vals[k], dist_desc[k])
        _write_yaml_blank(fh)

        if qopt is not None:
            _write_yaml_header(fh, "Attitude (quaternion)")
            for i, n in enumerate(["qw","qx","qy","qz"]):
                _write_yaml_kv(fh, n, float(qopt[i]), "Camera orientation quaternion component")
            _write_yaml_blank(fh)

        _write_yaml_header(fh, "Derived Intrinsics & Quality")
        _write_yaml_kv(fh, "aspect_ratio", extras["aspect_ratio"], "fx / fy; ~1 for square pixels")
        _write_yaml_kv(fh, "fx_norm", extras["fx_norm"], "fx normalized by image width")
        _write_yaml_kv(fh, "fy_norm", extras["fy_norm"], "fy normalized by image height")
        _write_yaml_kv(fh, "fov_x_deg", extras["fov_x_deg"], "Horizontal field of view (deg)")
        _write_yaml_kv(fh, "fov_y_deg", extras["fov_y_deg"], "Vertical field of view (deg)")
        _write_yaml_kv(fh, "fov_diag_deg", extras["fov_diag_deg"], "Diagonal field of view (deg)")
        _write_yaml_kv(fh, "f_theta_deg", extras["f_theta_deg"], "Alias of diagonal FOV")
        _write_yaml_kv(fh, "rms", rms, "Mean reprojection error (pixels) — lower is better")


def _outputs_opencv(cfg, imsize, K, D, rms, flags_value, outdir: Path):
    ensure_dir(outdir)
    # JSON (programmatic)
    payload = {
        "mode": "opencv",
        "image_width": int(imsize[0]),
        "image_height": int(imsize[1]),
        "K": np.asarray(K).tolist(),
        "D": np.asarray(D).tolist(),
        "rms": float(rms),
        "flags": int(flags_value),
    }
    with open(outdir / cfg["output"]["json"], "w") as f:
        json.dump(payload, f, indent=2)
    # YAML (manual with comments)
    if cfg["output"].get("yaml"):
        _writer_opencv_yaml_with_comments(outdir / cfg["output"]["yaml"], imsize, K, D, rms, flags_value)
    print(f"Results stored at {outdir}")

def _outputs_scipy(cfg, Kopt, Dopt, qopt, rms, outdir: Path):
    ensure_dir(outdir)
    payload = {
        "mode": "scipy",
        "K_like": {k: float(v) for k, v in Kopt.items()},
        "distortion": {k: float(v) for k, v in Dopt.items()},
        "attitude_quaternion": (qopt.tolist() if qopt is not None else None),
        "rms": float(rms),
    }
    with open(outdir / cfg["output"]["json"], "w") as f:
        json.dump(payload, f, indent=2)
    if cfg["output"].get("yaml"):
        _writer_scipy_yaml_with_comments(outdir / cfg["output"]["yaml"], cfg, Kopt, Dopt, qopt, rms)
    print(f"Results stored at {outdir}")


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

    # Optional preprocessing
    prep = cfg.get("detector", {}).get("preprocess", {})
    if prep.get("clahe", False):
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        gray = clahe.apply(gray)
    bk = int(prep.get("blur_ksize", 0))
    if bk and bk >= 3 and bk % 2 == 1:
        gray = cv2.GaussianBlur(gray, (bk, bk), 0)

    if p == "chessboard":
        pattern_size = (int(cfg["board"]["cols"]), int(cfg["board"]["rows"]))

        # Try robust SB detector first
        try:
            ret, corners = cv2.findChessboardCornersSB(gray, pattern_size, flags=0)
            if ret and corners is not None:
                if refine:
                    cv2.cornerSubPix(
                        gray, corners, (11,11), (-1,-1),
                        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 1e-7)
                    )
                return True, corners
        except Exception:
            pass

        # Fallback classic with stronger flags
        flags = (cv2.CALIB_CB_ADAPTIVE_THRESH |
                 cv2.CALIB_CB_NORMALIZE_IMAGE |
                 cv2.CALIB_CB_EXHAUSTIVE |
                 cv2.CALIB_CB_QUADRILATERAL)
        ret, corners = cv2.findChessboardCorners(gray, pattern_size, flags=flags)
        if not ret:
            return False, None
        if refine:
            cv2.cornerSubPix(
                gray, corners, (11,11), (-1,-1),
                (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 1e-7)
            )
        return True, corners

    elif p == "circles":
        pattern_size = (int(cfg["board"]["cols"]), int(cfg["board"]["rows"]))
        circ_cfg = cfg.get("circles", {})
        asymmetric = bool(circ_cfg.get("asymmetric", False))
        flags = cv2.CALIB_CB_ASYMMETRIC_GRID if asymmetric else cv2.CALIB_CB_SYMMETRIC_GRID
        ret, centers = cv2.findCirclesGrid(gray, pattern_size, flags=flags)
        return ret, centers

    elif p == "charuco":
        if not hasattr(cv2, "aruco"):
            raise RuntimeError("CharUco requested but cv2.aruco is missing (install opencv-contrib-python).")
        corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=aruco_params)
        if ids is None or len(corners) == 0:
            return False, None
        board = cv2.aruco.CharucoBoard(
            (int(cfg["board"]["cols"]), int(cfg["board"]["rows"])),
            float(cfg["board"]["square_size"]),
            float(cfg["board"]["marker_size"]),
            aruco_dict
        )
        ok, ch_corners, ch_ids = cv2.aruco.interpolateCornersCharuco(corners, ids, gray, board)
        if not ok or ch_corners is None or ch_ids is None or len(ch_corners) < 4:
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

    objpoints, imgpoints = [], []
    charuco_corners_list, charuco_ids_list = [], []

    aruco_dict = None
    aruco_params = None
    if cfg["board"]["pattern"].lower() == "charuco":
        aruco_dict = cv2.aruco.getPredefinedDictionary(
            getattr(cv2.aruco, cfg["charuco"]["dictionary"]))
        aruco_params = cv2.aruco.DetectorParameters()

    for p in tqdm(img_paths, total=len(img_paths), desc="Processing Images"):
        img = cv2.imread(str(p))
        if img is None:
            print(f"[WARN] cannot read {p}")
            continue
        if imsize is None:
            imsize = (img.shape[1], img.shape[0])

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
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, max_iter, eps)
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
    _outputs_opencv(cfg, imsize, K, D, rms, flags, outdir)

    # Optional preview
    if cfg["output"].get("save_undistorted_preview", False):
        for p in img_paths[:min(5, len(img_paths))]:
            img = cv2.imread(str(p))
            if cfg["fisheye"].get("enable", False):
                und = cv2.fisheye.undistortImage(img, K, D)
            else:
                und = cv2.undistort(img, K, D)
            cv2.imwrite(str(outdir / ("undist_" + p.name)), und)


# ============================================================
# SciPy LM Mode
# ============================================================

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

    K0 = {"dx": init["dx"], "dy": init["dy"], "alpha": init["alpha"], "up": init["up"], "vp": init["vp"]}
    D0 = {"k1": init["k1"], "k2": init["k2"], "k3": init["k3"], "p1": init["p1"], "p2": init["p2"]}
    q0 = None
    if init.get("q") is not None:
        q0 = np.array(init["q"], float)

    opt = sc.get("optimize", {})
    fixed_values = dict(opt.get("fixed_values", {}))
    zero_params  = list(opt.get("zero_params", []))
    free_params  = opt.get("free_params", None)
    solve_att    = bool(opt.get("solve_attitude", False))

    # zero -> fixed 0.0
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

    # Write back optimized
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

    # Report RMS
    if solve_att and qopt is not None:
        C = _quat_to_dcm(qopt)
        d_cam = (C @ ref_dirs.T).T
    else:
        d_cam = ref_dirs
    d_cam = _normalize(d_cam)
    uv_pred = _project_dirs_to_pixels(d_cam, Kopt, Dopt)
    rms = float(np.sqrt(np.mean(np.sum((uv_pred - pix_uv)**2, axis=1))))

    outdir = Path(cfg["output"]["dir"])
    _outputs_scipy(cfg, Kopt, Dopt, qopt, rms, outdir)


# ============================================================
# Main
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
