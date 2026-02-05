#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified camera calibration runner driven by YAML config (no ruamel.yaml).

Modes:
  - "opencv": OpenCV-based calibration (pinhole or fisheye) with robust detectors
  - "scipy": custom LM (SciPy) calibration with per-parameter constraints

Usage:
  python3 cv2_cal_cht.py --config configs/cal_config.yaml
"""

import argparse
import json
import math
import pdb  # Debugger available on demand
from pathlib import Path
import shutil

import numpy as np
import yaml  # for reading config only
from tqdm import tqdm

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


def pose_from_rvec_tvec(rvec, tvec):
    """
    OpenCV pose semantics (explicit):

      X^C = R_{B->C} X^B + t^C_{Co->Bo}

    where:
      - R_{B->C} rotates board-frame vectors into camera frame
      - t^C_{Co->Bo} is the vector from camera origin to board origin,
        expressed in camera frame

    Parameters
    ----------
    rvec : (3,1) Rodrigues vector
    tvec : (3,1) translation vector

    Returns
    -------
    dict with explicit frame-aware naming
    """
    if cv2 is None:
        raise ImportError("OpenCV is required for pose_from_rvec_tvec")
    R, _ = cv2.Rodrigues(rvec)

    return {
        "R_B_to_C": R.tolist(),
        "t_C_Co_to_Bo": tvec.reshape(-1).tolist(),
        "rvec_B_to_C": rvec.reshape(-1).tolist(),
    }


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
    fx = float(K[0, 0])
    s = float(K[0, 1])
    skew_from_K = s
    skew_clean = 0.0 if abs(s) < tol else s
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
    skew_clean = 0.0 if abs(s) < tol else s
    theta_deg = math.degrees(math.atan2(float(dx), s)) if dx > 0 else 90.0
    return skew_from_K, skew_clean, theta_deg


def _quat_to_dcm(q):
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w),     2*(x*z + y*w)],
        [2*(x*y + z*w),     1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w),     2*(y*z + x*w),     1 - 2*(x*x + y*y)]
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
    x = d_cam[:, 0] / d_cam[:, 2]
    y = d_cam[:, 1] / d_cam[:, 2]
    xd, yd = _distort_xy(x, y, D)
    u = K["dx"]*xd + K["alpha"]*yd + K["up"]
    v = K["dy"]*yd + K["vp"]
    return np.stack([u, v], axis=1)


# ============================================================
# Sanity checker for distortion (OpenCV pinhole form)
# ============================================================

def _sanity_check_distortion_pinhole(K, D, imsize=None, model_name="pinhole", flags_value=None):
    """
    Returns a list of human-readable warnings if distortion looks unstable/implausible.
    Heuristics:
      - Radial factor at r in {0.25, 0.5, 0.7, 1.0} should stay within reasonable envelope
      - Oscillation near the center (multiple derivative sign flips)
      - Tangential |p1|,|p2| size
      - FOV sanity (optional)
    """
    warnings = []

    # Extract coeffs by name (OpenCV order: [k1,k2,p1,p2,k3,k4,k5,k6,s1,s2,s3,s4,tauX,tauY])
    dflat = np.asarray(D).reshape(-1).tolist()
    def get(i, default=0.0): return dflat[i] if i < len(dflat) else default
    k1, k2, p1, p2, k3 = get(0), get(1), get(2), get(3), get(4)
    k4, k5, k6 = get(5), get(6), get(7)

    # Evaluate radial polynomial (r normalized)
    def radial(r):
        r2 = r*r
        r4 = r2*r2
        r6 = r4*r2
        return 1.0 + k1*r2 + k2*r4 + k3*r6 + k4*r2*r6 + k5*r4*r6 + k6*r6*r6

    # Tangential magnitude heuristic
    if abs(p1) > 0.02 or abs(p2) > 0.02:
        warnings.append(f"Tangential coefficients are large (p1={p1:.3g}, p2={p2:.3g}). "
                        "This often means poor board alignment diversity or inaccurate corners.")

    # Radial envelope checks
    radii = [0.25, 0.5, 0.7, 1.0]
    vals = [radial(r) for r in radii]
    for r, v in zip(radii, vals):
        if r <= 0.7 and not (0.6 <= v <= 1.6):
            warnings.append(f"Radial factor at r={r:.2f} is {v:.3g}, outside [0.6,1.6] — likely overfit/unstable.")
        if r == 1.0 and (v < 0.3 or v > 2.5):
            warnings.append(f"Radial factor at r=1.00 is {v:.3g} — implausible for standard pinhole lenses.")

    # Oscillation near center
    rs = np.array([0.05, 0.1, 0.15, 0.2, 0.25])
    dr = 1e-3
    deriv = []
    for r in rs:
        deriv.append((radial(r+dr) - radial(r-dr)) / (2*dr))
    flips = np.sum(np.sign(deriv[:-1]) != np.sign(deriv[1:]))
    if flips >= 2:
        warnings.append("Radial curve oscillates near the center (multiple derivative sign flips) — high-order terms likely overfitting.")

    # Rational model note
    if flags_value is not None:
        CALIB_RATIONAL_MODEL = 1 << 14  # 16384
        if (flags_value & CALIB_RATIONAL_MODEL) != 0:
            warnings.append("Rational model (k4..k6) enabled. Ensure strong edge coverage; "
                            "otherwise high-order terms may explode. Consider disabling initially.")

    # FOV sanity vs fx/fy — optional
    if imsize is not None:
        iw, ih = int(imsize[0]), int(imsize[1])
        fx = float(K[0, 0]); fy = float(K[1, 1])
        fovx = 2.0 * math.degrees(math.atan(iw/(2.0*fx))) if fx > 0 else None
        fovy = 2.0 * math.degrees(math.atan(ih/(2.0*fy))) if fy > 0 else None
        if fovx and (fovx < 5 or fovx > 140):
            warnings.append(f"FOVx={fovx:.1f}° looks unusual for this sensor size; verify fx or units.")
        if fovy and (fovy < 5 or fovy > 140):
            warnings.append(f"FOVy={fovy:.1f}° looks unusual for this sensor size; verify fy or units.")

    return warnings



def _recommendations_from_warnings(warnings_list):
    """
    Given a list of warning strings, return a list of plain-English
    recommendations on what to try next in calibration.
    """
    recommendations = []
    seen = set()

    def add(rec):
        if rec not in seen:
            seen.add(rec)
            recommendations.append(rec)

    for w in warnings_list or []:
        if "Tangential coefficients are large" in w:
            add("Capture more images with out-of-plane board tilt and strong coverage near all image edges; avoid only fronto-parallel views.")
        if "Radial factor at r=" in w and "outside [0.6,1.6]" in w:
            add("Reduce the number of free high-order radial terms (e.g., fix k4..k6 to zero) and collect more images with the board near the corners.")
        if "Radial factor at r=1.00" in w:
            add("Verify that the square_size and sensor dimensions are in consistent units; for very wide lenses, consider using the fisheye model.")
        if "oscillates near the center" in w:
            add("Constrain or zero the highest-order radial coefficients and prefer a simpler distortion model unless you have dense corner coverage across the frame.")
        if "Rational model (k4..k6) enabled" in w:
            add("Try running calibration once with rational_model disabled and compare RMS and reprojection overlays before enabling high-order terms.")
        if "FOVx=" in w or "FOVy=" in w:
            add("Double-check image_size or image_size_hint and that fx/fy are in pixel units, not normalized or sensor-millimeter units.")
        if "Not enough ChArUco detections" in w or "Not enough detections" in w:
            add("Increase the number of calibration images, vary board distance and orientation, and ensure good lighting and focus on the pattern.")

    if (warnings_list and not recommendations):
        add("Inspect reprojection overlays and consider simplifying the distortion model, increasing pose diversity, and adding more calibration images.")

    return recommendations


# ============================================================
# Output Writers (JSON + manual YAML with inline comments)
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
        return f"{float(val):.12g}"
    if isinstance(val, (list, tuple)):
        inner = ", ".join(_fmt_yaml_scalar(v) for v in val)
        return f"[{inner}]"
    s = str(val)
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


def _writer_opencv_yaml_with_comments(
    yaml_path: Path,
    imsize,
    K,
    D,
    rms,
    flags_value,
    warnings_list=None,
    recommendations_list=None,
):
    iw, ih = int(imsize[0]), int(imsize[1])
    fx = float(K[0, 0]); fy = float(K[1, 1])
    cx = float(K[0, 2]); cy = float(K[1, 2])

    skew_from_K, skew_clean, pixel_axes_angle_deg = compute_skew_extras_from_K(K)
    extras = _compute_intrinsic_extras(fx, fy, skew_from_K, cx, cy, iw, ih)

    # Distortion values mapped to full set
    all_dist_keys = ["k1", "k2", "k3", "k4", "k5", "k6",
                     "p1", "p2", "s1", "s2", "s3", "s4", "tauX", "tauY"]
    dist_desc = {
        "k1":   "Radial distortion (primary)",
        "k2":   "Radial distortion (secondary)",
        "k3":   "Radial distortion (tertiary)",
        "k4":   "Higher-order radial (rational model)",
        "k5":   "Higher-order radial",
        "k6":   "Higher-order radial",
        "p1":   "Tangential distortion x (decentering/tilt)",
        "p2":   "Tangential distortion y (decentering/tilt)",
        "s1":   "Thin prism distortion",
        "s2":   "Thin prism distortion",
        "s3":   "Thin prism distortion",
        "s4":   "Thin prism distortion",
        "tauX": "Tilted sensor term (x)",
        "tauY": "Tilted sensor term (y)"
    }
    D_flat = np.asarray(D).reshape(-1).tolist()
    ocv_names = ["k1", "k2", "p1", "p2", "k3", "k4", "k5", "k6",
                 "s1", "s2", "s3", "s4", "tauX", "tauY"]
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
        for k in ["k1", "k2", "k3", "k4", "k5", "k6"]:
            _write_yaml_kv(fh, k, dist_vals[k], dist_desc[k])
        for k in ["p1", "p2", "s1", "s2", "s3", "s4", "tauX", "tauY"]:
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
        _write_yaml_kv(fh, "rms", float(rms), "Mean reprojection error (pixels) — lower is better")
        _write_yaml_kv(fh, "flags", int(flags_value), "OpenCV calibration flags bitmask used during solve")
        _write_yaml_blank(fh)

        _write_yaml_header(fh, "Sanity Warnings")
        if warnings_list:
            fh.write("warnings:\n")
            for w in warnings_list:
                fh.write(f"  - {_fmt_yaml_scalar(w)}\n")
        else:
            fh.write("warnings: []\n")
        _write_yaml_blank(fh)

        _write_yaml_header(fh, "Sanity Recommendations")
        if recommendations_list:
            fh.write("recommendations:\n")
            for r in recommendations_list:
                fh.write(f"  - {_fmt_yaml_scalar(r)}\n")
        else:
            fh.write("recommendations: []\n")


def write_poses_json(outdir, pose_entries, calibration_path):
    """
    Writes per-image board->camera poses.

    Each pose satisfies:
      X^C = R_B_to_C X^B + t^C_{Co->Bo}
    """
    payload = {
        "calibration": {
            "path": str(calibration_path),
            "frame_convention": {
                                "equation": "X^C = R_B_to_C X^B + t^C_{Co->Bo}",
                                "frames": {
                                    "B": "Charuco board frame",
                                    "C": "Camera optical frame"
                                }
            }
        },
        "poses": pose_entries
    }

    with open(outdir / "poses.json", "w") as f:
        json.dump(payload, f, indent=2)


def _writer_scipy_yaml_with_comments(
    yaml_path: Path,
    cfg,
    Kopt,
    Dopt,
    qopt,
    rms,
    warnings_list=None,
    recommendations_list=None,
):
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
    all_dist_keys = ["k1", "k2", "k3", "k4", "k5", "k6",
                     "p1", "p2", "s1", "s2", "s3", "s4", "tauX", "tauY"]
    dist_desc = {
        "k1":   "Radial distortion (primary)",
        "k2":   "Radial distortion (secondary)",
        "k3":   "Radial distortion (tertiary)",
        "k4":   "Higher-order radial (rational model)",
        "k5":   "Higher-order radial",
        "k6":   "Higher-order radial",
        "p1":   "Tangential distortion x (decentering/tilt)",
        "p2":   "Tangential distortion y (decentering/tilt)",
        "s1":   "Thin prism distortion",
        "s2":   "Thin prism distortion",
        "s3":   "Thin prism distortion",
        "s4":   "Thin prism distortion",
        "tauX": "Tilted sensor term (x)",
        "tauY": "Tilted sensor term (y)"
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
        for k in ["k1", "k2", "k3", "k4", "k5", "k6"]:
            _write_yaml_kv(fh, k, dist_vals[k], dist_desc[k])
        for k in ["p1", "p2", "s1", "s2", "s3", "s4", "tauX", "tauY"]:
            _write_yaml_kv(fh, k, dist_vals[k], dist_desc[k])
        _write_yaml_blank(fh)

        if qopt is not None:
            _write_yaml_header(fh, "Attitude (quaternion)")
            for i, n in enumerate(["qw", "qx", "qy", "qz"]):
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
        _write_yaml_kv(fh, "rms", float(rms), "Mean reprojection error (pixels) — lower is better")
        _write_yaml_blank(fh)

        _write_yaml_header(fh, "Sanity Warnings")
        if warnings_list:
            fh.write("warnings:\n")
            for w in warnings_list:
                fh.write(f"  - {_fmt_yaml_scalar(w)}\n")
        else:
            fh.write("warnings: []\n")
        _write_yaml_blank(fh)

        _write_yaml_header(fh, "Sanity Recommendations")
        if recommendations_list:
            fh.write("recommendations:\n")
            for r in recommendations_list:
                fh.write(f"  - {_fmt_yaml_scalar(r)}\n")
        else:
            fh.write("recommendations: []\n")


def _outputs_opencv(cfg, imsize, K, D, rms, flags_value, charuco_geometry, outdir: Path, warnings=None):
    ensure_dir(outdir)
    warnings = warnings or []
    recommendations = _recommendations_from_warnings(warnings)
    # charuco_geometry is None for non-Charuco patterns
    payload = {
            "mode": "opencv",

            "intrinsics": {
                "K": np.asarray(K).tolist(),
                "D": np.asarray(D).tolist(),
                "image_width": int(imsize[0]),
                "image_height": int(imsize[1]),
            },

            "charuco_board": {
                "frame": "B",
                "square_size": cfg["board"]["square_size"],
                "marker_size": cfg["charuco"]["marker_size"],
                "corners_3d": charuco_geometry,
            } if charuco_geometry else None,

            "rms": float(rms),
            "warnings": warnings,
            "recommendations": recommendations,
    }
    with open(outdir / cfg["output"]["json"], "w") as f:
        json.dump(payload, f, indent=2)
    if cfg["output"].get("yaml"):
        _writer_opencv_yaml_with_comments(
            outdir / cfg["output"]["yaml"],
            imsize,
            K,
            D,
            rms,
            flags_value,
            warnings_list=warnings,
            recommendations_list=recommendations,
        )
    print(f"Results stored at {outdir}")


def _outputs_scipy(cfg, Kopt, Dopt, qopt, rms, outdir: Path, warnings=None):
    ensure_dir(outdir)
    warnings = warnings or []
    recommendations = _recommendations_from_warnings(warnings)

    payload = {
        "mode": "scipy",
        "K_like": {k: float(v) for k, v in Kopt.items()},
        "distortion": {k: float(v) for k, v in Dopt.items()},
        "attitude_quaternion": (qopt.tolist() if qopt is not None else None),
        "rms": float(rms),
        "warnings": warnings,
        "recommendations": recommendations,
    }
    with open(outdir / cfg["output"]["json"], "w") as f:
        json.dump(payload, f, indent=2)
    if cfg["output"].get("yaml"):
        _writer_scipy_yaml_with_comments(
            outdir / cfg["output"]["yaml"],
            cfg,
            Kopt,
            Dopt,
            qopt,
            rms,
            warnings_list=warnings,
            recommendations_list=recommendations,
        )
    print(f"Results stored at {outdir}")


# ============================================================
# OpenCV Mode
# ============================================================

def _opencv_build_pinhole_flags(cfg):
    flags = 0
    f = cfg.get("flags", {})
    # Alias: modeling.rational_model
    if "modeling" in cfg and "rational_model" in cfg["modeling"] and cfg["modeling"]["rational_model"] is not None:
        f["rational_model"] = bool(cfg["modeling"]["rational_model"])

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
    objp = np.zeros((rows*cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    objp *= sq
    return objp


def _opencv_detect_points(img, cfg, aruco_dict=None, aruco_params=None):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    p = cfg["board"]["pattern"].lower()
    refine = cfg.get("detector", {}).get("refine_subpix", True)

    # Optional preprocessing
    prep = cfg.get("detector", {}).get("preprocess", {})
    if prep.get("clahe", False):
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
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
                        gray, corners, (11, 11), (-1, -1),
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
                gray, corners, (11, 11), (-1, -1),
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

        # Detect markers: support new ArucoDetector and (if present) legacy detectMarkers
        if hasattr(cv2.aruco, "ArucoDetector"):
            detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
            corners, ids, rejected = detector.detectMarkers(gray)
        elif hasattr(cv2.aruco, "detectMarkers"):
            corners, ids, rejected = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=aruco_params)
        else:
            raise RuntimeError("cv2.aruco has neither ArucoDetector nor detectMarkers; update OpenCV contrib.")

        if ids is None or len(corners) == 0:
            return False, None

        board = cv2.aruco.CharucoBoard(
            (int(cfg["board"]["cols"]), int(cfg["board"]["rows"])),
            float(cfg["board"]["square_size"]),
            float(cfg["charuco"]["marker_size"]),
            aruco_dict
        )

        ok, ch_corners, ch_ids = cv2.aruco.interpolateCornersCharuco(corners, ids, gray, board)
        if not ok or ch_corners is None or ch_ids is None or len(ch_corners) < 4:
            return False, None
        return True, (ch_corners, ch_ids, board)

    else:
        raise ValueError("Unknown board.pattern")


# ============================================================
# Reprojection Overlays
# ============================================================

def _draw_error_line(img, pt_detect, pt_proj):
    """
    Draw a line between detected and projected points, with color based on reprojection error.

    - Green:  small error
    - Yellow: medium
    - Red:    large
    """
    u, v = pt_detect
    up, vp = pt_proj
    err = math.hypot(float(u - up), float(v - vp))

    if err < 0.5:
        color = (0, 255, 0)      # green
    elif err < 1.5:
        color = (0, 255, 255)    # yellow
    else:
        color = (0, 0, 255)      # red

    cv2.line(img, (int(u), int(v)), (int(up), int(vp)), color, 1)


def _save_reprojection_overlays_pinhole(cfg, used_paths, objpoints, imgpoints, K, D, rvecs, tvecs, outdir: Path):
    """
    For a few used calibration images, reproject 3D board points and overlay:
      - small green circle:  detected corners
      - small red circle:    reprojected points
      - colored line:        reprojection error (see _draw_error_line)
    """
    ensure_dir(outdir)
    max_n = int(cfg["output"].get("max_reprojection_images", 5))

    for idx, (path, obj, imgpts, rv, tv) in enumerate(zip(used_paths, objpoints, imgpoints, rvecs, tvecs)):
        if idx >= max_n:
            break

        img = cv2.imread(str(path))
        if img is None:
            continue
        proj, _ = cv2.projectPoints(obj, rv, tv, K, D)
        proj = proj.reshape(-1, 2)
        pts2d = imgpts.reshape(-1, 2)

        vis = img.copy()
        for (u, v), (up, vp) in zip(pts2d, proj):
            # detected (green circle)
            cv2.circle(vis, (int(u), int(v)), 5, (0, 255, 0), -1)
            # reprojected (red circle)
            cv2.circle(vis, (int(up), int(vp)), 15, (0, 0, 255), 5)
            # error line (color-coded)
            _draw_error_line(vis, (u, v), (up, vp))

        out_path = outdir / f"reproj_{path.name}"
        cv2.imwrite(str(out_path), vis)


def _get_charuco_board_corners(board):
    """
    Return Nx3 array of Charuco board chessboard corners in board coordinates.

    Handles both:
      - board.chessboardCorners (attribute, older API)
      - board.getChessboardCorners() (method, newer API)
    """
    if hasattr(board, "chessboardCorners"):
        return board.chessboardCorners
    if hasattr(board, "getChessboardCorners"):
        return board.getChessboardCorners()
    raise RuntimeError("CharucoBoard has neither 'chessboardCorners' nor 'getChessboardCorners()'.")


def extract_charuco_board_geometry(board):
    """
    Returns Charuco corner coordinates in the board frame B.

    Each corner is:
      X^B = [x, y, 0]^T  (units = square_size)
    """
    corners = _get_charuco_board_corners(board)  # (N,3)

    return {
        str(i): corners[i].tolist()
        for i in range(corners.shape[0])
    }


def _save_reprojection_overlays_charuco(cfg, used_paths, charuco_corners_list, charuco_ids_list,
                                        board, K, D, rvecs, tvecs, outdir: Path):
    """
    For a few Charuco calibration images, reproject 3D board points and overlay:
      - small green circle:  detected Charuco corners
      - small red circle:    reprojected points
      - colored line:        reprojection error
    """
    ensure_dir(outdir)
    max_n = int(cfg["output"].get("max_reprojection_images", 5))

    board_corners = _get_charuco_board_corners(board)  # (N,3)

    for idx, (path, ch_c, ch_ids, rv, tv) in enumerate(
        zip(used_paths, charuco_corners_list, charuco_ids_list, rvecs, tvecs)
    ):
        if idx >= max_n:
            break

        img = cv2.imread(str(path))
        if img is None:
            continue

        ids_flat = ch_ids.flatten().astype(int)
        obj = board_corners[ids_flat]  # (M,3)
        proj, _ = cv2.projectPoints(obj, rv, tv, K, D)
        proj = proj.reshape(-1, 2)
        pts2d = ch_c.reshape(-1, 2)

        vis = img.copy()
        for (u, v), (up, vp) in zip(pts2d, proj):
            # detected corner
            cv2.circle(vis, (int(u), int(v)), 5, (0, 255, 0), -1)
            # projected corner
            cv2.circle(vis, (int(up), int(vp)), 15, (0, 0, 255), 5)
            # error line
            _draw_error_line(vis, (u, v), (up, vp))

        out_path = outdir / f"reproj_{path.name}"
        cv2.imwrite(str(out_path), vis)


def run_opencv_from_yaml(cfg, cfg_path):
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
    used_paths = []
    charuco_corners_list, charuco_ids_list = [], []
    used_paths_charuco = []

    aruco_dict = None
    aruco_params = None
    if cfg["board"]["pattern"].lower() == "charuco":
        if not hasattr(cv2, "aruco"):
            raise RuntimeError("CharUco requested but cv2.aruco is missing (install opencv-contrib-python).")

        dict_name = cfg["charuco"]["dictionary"]
        try:
            dict_id = getattr(cv2.aruco, dict_name)
        except AttributeError:
            raise RuntimeError(f"Unknown Charuco dictionary '{dict_name}' in config.")

        aruco_dict = cv2.aruco.getPredefinedDictionary(dict_id)
        # DetectorParameters: handle both new and old API
        if hasattr(cv2.aruco, "DetectorParameters_create"):
            aruco_params = cv2.aruco.DetectorParameters_create()
        else:
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
            used_paths_charuco.append(p)
        else:
            objpoints.append(objp)
            imgpoints.append(pts)
            used_paths.append(p)

    if cfg["board"]["pattern"].lower() == "charuco":
        if len(charuco_corners_list) < 3:
            raise RuntimeError("Not enough ChArUco detections.")
    else:
        if len(objpoints) < 3:
            raise RuntimeError("Not enough detections.")

    max_iter = cfg["solver"].get("max_iter", 200)
    eps = cfg["solver"].get("eps", 1e-8)

    # Build flags (includes rational model if requested)
    flags = _opencv_build_pinhole_flags(cfg)

    # Optional seeding: if K/D provided, fix-held params are pinned to those initial values
    camM = None
    dist0 = None
    ocv_init = cfg.get("opencv_init", {})
    if "K" in ocv_init and ocv_init["K"] is not None:
        K_init = np.array(ocv_init["K"], float)
        if K_init.shape == (3, 3):
            camM = K_init.copy()
    if "D" in ocv_init and ocv_init["D"] is not None:
        D_init = np.array(ocv_init["D"], float).reshape(-1, 1)
        dist0 = D_init
    if camM is not None or dist0 is not None:
        flags |= cv2.CALIB_USE_INTRINSIC_GUESS

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, max_iter, eps)

    if cfg["fisheye"].get("enable", False):
        K = np.eye(3) if camM is None else camM
        D = np.zeros((4, 1)) if dist0 is None else dist0
        fe_flags = 0
        if cfg["fisheye"].get("recompute_extrinsic", False):
            fe_flags |= cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC
        if cfg["fisheye"].get("fix_skew", False):
            fe_flags |= cv2.fisheye.CALIB_FIX_SKEW
        if cfg["fisheye"].get("fix_k1k2k3k4", False):
            fe_flags |= (cv2.fisheye.CALIB_FIX_K1 |
                         cv2.fisheye.CALIB_FIX_K2 |
                         cv2.fisheye.CALIB_FIX_K3 |
                         cv2.fisheye.CALIB_FIX_K4)
        rms, K, D, rvecs, tvecs = cv2.fisheye.calibrate(
            objpoints, imgpoints, imsize, K, D, flags=fe_flags,
            criteria=criteria
        )
    else:
        if cfg["board"]["pattern"].lower() == "charuco":
            board = cv2.aruco.CharucoBoard(
                (int(cfg["board"]["cols"]), int(cfg["board"]["rows"])),
                float(cfg["board"]["square_size"]),
                float(cfg["charuco"]["marker_size"]),
                aruco_dict
            )
            rms, K, D, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(
                charucoCorners=charuco_corners_list,
                charucoIds=charuco_ids_list,
                board=board,
                imageSize=imsize,
                cameraMatrix=camM,
                distCoeffs=dist0,
                flags=flags,
                criteria=criteria
            )
        else:
            rms, K, D, rvecs, tvecs = cv2.calibrateCamera(
                objpoints, imgpoints, imsize, camM, dist0,
                flags=flags, criteria=criteria
            )

    # ------------------------------------------------------------
    # Collect per-image poses (explicit image paths)
    # ------------------------------------------------------------
    pose_entries = []

    if cfg["board"]["pattern"].lower() == "charuco":
        used = used_paths_charuco
    else:
        used = used_paths

    for path, rvec, tvec in zip(used, rvecs, tvecs):
        pose = pose_from_rvec_tvec(rvec, tvec)
        pose_entries.append({
            "image_path": str(path),
            **pose
        })
    
    # assert len(pose_entries) == len(rvecs)
    assert len(pose_entries) == len(rvecs)
    
    # ------------------------------------------------------------
    # Collect Charuco board geometry if applicable
    # ------------------------------------------------------------
    charuco_geometry = None
    if cfg["board"]["pattern"].lower() == "charuco":
        charuco_geometry = extract_charuco_board_geometry(board)



    # Sanity checks and warnings
    sanity = _sanity_check_distortion_pinhole(K, D, imsize=imsize, flags_value=flags)
    if sanity:
        print("\n[CALIB WARNING] Distortion sanity checks flagged issues:")
        for w in sanity:
            print("  -", w)

    outdir = Path(cfg["output"]["dir"])
    _outputs_opencv(cfg, imsize, K, D, rms, flags, charuco_geometry, outdir, warnings=sanity)
    
    try:
        shutil.copy(cfg_path, outdir / f"config_used_{cfg_path.stem}.yaml")
    except Exception as e:
        print(f"[WARN] Could not copy config file: {e}")

    # Optional undistorted preview (subset)
    if cfg["output"].get("save_undistorted_preview", False):
        for p in img_paths[:min(5, len(img_paths))]:
            img = cv2.imread(str(p))
            if img is None:
                continue
            if cfg["fisheye"].get("enable", False):
                und = cv2.fisheye.undistortImage(img, K, D)
            else:
                und = cv2.undistort(img, K, D)
            cv2.imwrite(str(outdir / ("undist_" + p.name)), und)

    # Optional reprojection overlays (few images, error-colored edges)
    if cfg["output"].get("save_reprojection_preview", False) and not cfg["fisheye"].get("enable", False):
        if cfg["board"]["pattern"].lower() == "charuco":
            _save_reprojection_overlays_charuco(
                cfg,
                used_paths_charuco,
                charuco_corners_list,
                charuco_ids_list,
                board,
                K,
                D,
                rvecs,
                tvecs,
                outdir
            )
        else:
            _save_reprojection_overlays_pinhole(
                cfg,
                used_paths,
                objpoints,
                imgpoints,
                K,
                D,
                rvecs,
                tvecs,
                outdir
            )
    
    calibration_json_path = Path(cfg["output"]["json"]).as_posix()
    write_poses_json(
        outdir,
        pose_entries,
        calibration_path=calibration_json_path
    )


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
        if n in K:
            K[n] = v
        elif n in D:
            D[n] = v
        elif n in ["qw", "qx", "qy", "qz"]:
            idx = {"qw": 0, "qx": 1, "qy": 2, "qz": 3}[n]
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


def run_scipy_from_yaml(cfg, cfg_path):
    if least_squares is None:
        raise RuntimeError("SciPy not installed.")

    data = np.load(cfg["data"]["points_npz"])
    ref_dirs = data["ref_dirs"].astype(float)
    pix_uv = data["pix_uv"].astype(float)

    sc = cfg["scipy_calib"]
    init = sc["init"]

    # K0 initial (dx,dy,alpha,up,vp)
    K0 = {"dx": init["dx"], "dy": init["dy"], "alpha": init["alpha"], "up": init["up"], "vp": init["vp"]}
    # D0 initial (radial: k1..k3, tangential: p1,p2) — extendable by fixed_values later
    D0 = {"k1": init["k1"], "k2": init["k2"], "k3": init["k3"], "p1": init["p1"], "p2": init["p2"]}
    q0 = None
    if init.get("q") is not None:
        q0 = np.array(init["q"], float)

    opt = sc.get("optimize", {})
    fixed_values = dict(opt.get("fixed_values", {}))  # exact values to pin
    zero_params = list(opt.get("zero_params", []))    # parameters pinned to 0.0
    free_params = opt.get("free_params", None)        # if None -> all (minus fixed)
    solve_att = bool(opt.get("solve_attitude", False))

    # zero_params -> fixed 0.0
    for z in zero_params:
        fixed_values[z] = 0.0

    # Collect all names reachable (extend by any fixed values keys not in D0/K0)
    all_names = list(K0.keys()) + list(D0.keys())
    for n in fixed_values.keys():
        if n not in all_names and n in [
            "k1", "k2", "k3", "k4", "k5", "k6",
            "p1", "p2", "s1", "s2", "s3", "s4", "tauX", "tauY",
            "dx", "dy", "alpha", "up", "vp",
            "qw", "qx", "qy", "qz"
        ]:
            all_names.append(n)
    if solve_att and q0 is not None:
        all_names += ["qw", "qx", "qy", "qz"]

    # Working copies incorporating fixed values
    Kw = dict(K0)
    Dw = dict(D0)
    qw = None if q0 is None else q0.copy()

    for n, val in fixed_values.items():
        if n in Kw:
            Kw[n] = float(val)
        elif n in Dw:
            Dw[n] = float(val)
        elif n in ["qw", "qx", "qy", "qz"]:
            idx = {"qw": 0, "qx": 1, "qy": 2, "qz": 3}[n]
            if qw is None:
                qw = np.array([1, 0, 0, 0], float)
            qw[idx] = float(val)
        else:
            # if a distortion key not in Dw, add it (e.g., k4..k6 in custom model)
            if n.startswith("k") or n in ["p1", "p2", "s1", "s2", "s3", "s4", "tauX", "tauY"]:
                Dw[n] = float(val)

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
        if n in Kw:
            x0.append(Kw[n])
        elif n in Dw:
            x0.append(Dw[n])
        elif n in ["qw", "qx", "qy", "qz"]:
            idx = {"qw": 0, "qx": 1, "qy": 2, "qz": 3}[n]
            if qw is None:
                qw = np.array([1, 0, 0, 0], float)
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
        if n in Kopt:
            Kopt[n] = v
        elif n in Dopt:
            Dopt[n] = v
        elif n in ["qw", "qx", "qy", "qz"]:
            idx = {"qw": 0, "qx": 1, "qy": 2, "qz": 3}[n]
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
    
    Dvec = np.array([
        Dopt.get("k1", 0), Dopt.get("k2", 0), Dopt.get("p1", 0), Dopt.get("p2", 0),
        Dopt.get("k3", 0), Dopt.get("k4", 0), Dopt.get("k5", 0), Dopt.get("k6", 0)
    ], float).reshape(-1, 1)

    # Optional image size hint
    hint = cfg["output"].get("image_size_hint", {})
    iw = int(hint["w"]) if "w" in hint else None
    ih = int(hint["h"]) if "h" in hint else None

    K_like = np.array([[Kopt["dx"], Kopt.get("alpha", 0.0), Kopt["up"]],
                       [0.0,          Kopt["dy"],            Kopt["vp"]],
                       [0.0,          0.0,                   1.0]], float)

    sanity = _sanity_check_distortion_pinhole(K_like, Dvec, imsize=(iw, ih) if iw and ih else None)
    if sanity:
        print("\n[CALIB WARNING] Distortion sanity checks flagged issues (SciPy):")
        for w in sanity:
            print("  -", w)

    outdir = Path(cfg["output"]["dir"])
    _outputs_scipy(cfg, Kopt, Dopt, qopt, rms, outdir, warnings=sanity)
    try:
        shutil.copy(cfg_path, outdir / f"config_used_{cfg_path.stem}.yaml")
    except Exception as e:
        print(f"[WARN] Could not copy config file: {e}")


# ============================================================
# Main
# ============================================================

def main():
    ap          = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg_path = Path(args.config).resolve()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    mode = cfg["mode"].lower()
    if mode == "opencv":
        run_opencv_from_yaml(cfg, cfg_path)
    elif mode == "scipy":
        run_scipy_from_yaml(cfg, cfg_path)
    else:
        raise ValueError(f"Unknown mode: {mode}")


if __name__ == "__main__":
    main()
