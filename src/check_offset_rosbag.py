"""Process expm_004 using rosbag Vicon poses and notation-branch offsets.

This script reuses `check_offset.py` v01 Vicon offset processing and applies it
directly to expm_004. The expm_004 imagery does not contain the ChArUco board,
so this is intentionally a Vicon-only pipeline:

- nearest-time Vicon cam/target poses from the rosbag
- notation-branch offset corrected `T_T_C = ^C T_T`
- reprojection overlays onto the images
- a JSON file containing the per-frame `T_T_C` matrices
"""

from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from rosbags.highlevel import AnyReader

from sc_pose.examples.check_offset import (
    _load_offset_estimates,
    _process_vicon_offset_v01,
)
from sc_pose.sensors.camera import PinholeCamera
from sc_pose.sensors.camera_projections import PoseProjector, draw_uv_points_on_image


@dataclass
class PoseStampedRow:
    stamp_ns: int
    x: float
    y: float
    z: float
    qw: float
    qx: float
    qy: float
    qz: float


def _parse_image_timestamp_ns(image_path: Path) -> int:
    match = re.fullmatch(r"frame_(\d{8})_(\d{6})_(\d{3})ms", image_path.stem)
    if match is None:
        raise ValueError(f"Unsupported image name format: {image_path.name}")

    date_part, time_part, ms_part = match.groups()
    dt = datetime.strptime(f"{date_part}{time_part}", "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1e9) + int(ms_part) * 1_000_000


def _load_pose_rows_from_rosbag(
    bag_dir      : Path,
    topic_name   : str,
) -> list[PoseStampedRow]:
    rows: list[PoseStampedRow] = []

    with AnyReader([bag_dir]) as reader:
        connections = [conn for conn in reader.connections if conn.topic == topic_name]
        if not connections:
            raise RuntimeError(f"Topic not found in rosbag: {topic_name}")

        for connection, _, rawdata in reader.messages(connections=connections):
            msg = reader.deserialize(rawdata, connection.msgtype)
            stamp_ns = int(msg.header.stamp.sec) * 1_000_000_000 + int(msg.header.stamp.nanosec)
            rows.append(
                PoseStampedRow(
                    stamp_ns = stamp_ns,
                    x = float(msg.pose.position.x),
                    y = float(msg.pose.position.y),
                    z = float(msg.pose.position.z),
                    qw = float(msg.pose.orientation.w),
                    qx = float(msg.pose.orientation.x),
                    qy = float(msg.pose.orientation.y),
                    qz = float(msg.pose.orientation.z),
                )
            )

    rows.sort(key=lambda row: row.stamp_ns)
    if not rows:
        raise RuntimeError(f"No messages found for topic: {topic_name}")
    return rows


def _find_nearest_pose_row(
    image_stamp_ns  : int,
    pose_rows       : list[PoseStampedRow],
) -> tuple[PoseStampedRow, int]:
    stamp_array = np.array([row.stamp_ns for row in pose_rows], dtype=np.int64)
    idx = int(np.searchsorted(stamp_array, image_stamp_ns))

    candidate_indices = []
    if idx < len(pose_rows):
        candidate_indices.append(idx)
    if idx > 0:
        candidate_indices.append(idx - 1)

    best_idx = min(candidate_indices, key=lambda i: abs(int(stamp_array[i]) - int(image_stamp_ns)))
    best_row = pose_rows[best_idx]
    delta_ns = int(best_row.stamp_ns) - int(image_stamp_ns)
    return best_row, delta_ns


def _build_vicon_dataframe_from_rosbag(
    image_paths      : list[Path],
    bag_dir          : Path,
) -> pd.DataFrame:
    cam_rows = _load_pose_rows_from_rosbag(bag_dir, "/vicon/basler_cam/basler_cam")
    soho_rows = _load_pose_rows_from_rosbag(bag_dir, "/vicon/soho/soho")

    vicon_rows: list[dict[str, object]] = []
    for idx, image_path in enumerate(image_paths, start=1):
        image_stamp_ns = _parse_image_timestamp_ns(image_path)
        cam_row, cam_delta_ns = _find_nearest_pose_row(image_stamp_ns, cam_rows)
        soho_row, soho_delta_ns = _find_nearest_pose_row(image_stamp_ns, soho_rows)

        vicon_rows.append(
            {
                "frame": image_path.name,
                "image_number": idx,
                "image_timestamp_ns": image_stamp_ns,
                "cam_timestamp_ns": cam_row.stamp_ns,
                "soho_timestamp_ns": soho_row.stamp_ns,
                "cam_delta_ms": cam_delta_ns / 1e6,
                "soho_delta_ms": soho_delta_ns / 1e6,
                # convert meters -> millimeters to match existing check_offset v01 inputs
                "cam_x": cam_row.x * 1e3,
                "cam_y": cam_row.y * 1e3,
                "cam_z": cam_row.z * 1e3,
                "cam_qw": cam_row.qw,
                "cam_qx": cam_row.qx,
                "cam_qy": cam_row.qy,
                "cam_qz": cam_row.qz,
                "soho_x": soho_row.x * 1e3,
                "soho_y": soho_row.y * 1e3,
                "soho_z": soho_row.z * 1e3,
                "soho_qw": soho_row.qw,
                "soho_qx": soho_row.qx,
                "soho_qy": soho_row.qy,
                "soho_qz": soho_row.qz,
            }
        )

    return pd.DataFrame(vicon_rows)


def _draw_origin(
    image,
    uv_points: np.ndarray,
):
    if len(uv_points) == 0:
        return image
    return draw_uv_points_on_image(
        img_or_path     = image,
        points_uv       = uv_points[:1],
        point_color     = (0, 255, 0),
        point_radius    = 20,
        point_thickness = 3,
    )


def main():
    here = Path(__file__).resolve().parent

    ##################################### Inputs #####################################
    data_folder      = here / "artifacts" / "offset" / "expm_004"
    data_name        = data_folder.name
    image_folder     = data_folder / "images"
    rosbag_dir       = data_folder / "rosbag2_2026_03_18-22_29_25"
    kps_file         = here / "artifacts" / "soho_reframed_mesh_pose_pack" / "mesh_points_50000.json"
    calib_data       = here / "artifacts" / "offset" / "expm_003" / "calibration.yaml"
    offset_data      = here / "artifacts" / "offset" / "expm_003" / "offset_results_notation_branch.json"
    res_path         = here / "results" / f"{data_name}_notation_branch_v01"

    img_width        = 4096
    img_height       = 3000
    focal_length     = 25.0
    sensor_width     = 14.13
    sensor_height    = 10.35

    offset_keys      = {
        "Trf_4x4_CamViconDef_Cam": "T_CvC",
        "Trf_4x4_TargetViconDef_Target": "T_TvT",
    }
    vicon_keys       = {
        "frame": "frame",
        "x_target": "soho_x",
        "y_target": "soho_y",
        "z_target": "soho_z",
        "qw_target": "soho_qw",
        "qx_target": "soho_qx",
        "qy_target": "soho_qy",
        "qz_target": "soho_qz",
        "x_cam": "cam_x",
        "y_cam": "cam_y",
        "z_cam": "cam_z",
        "qw_cam": "cam_qw",
        "qx_cam": "cam_qx",
        "qy_cam": "cam_qy",
        "qz_cam": "cam_qz",
    }
    ##################################### Inputs #####################################

    if res_path.exists():
        shutil.rmtree(res_path)
    res_path.mkdir(parents=True, exist_ok=True)

    cam = PinholeCamera(
        sensor_width_mm  = sensor_width,
        sensor_height_mm = sensor_height,
        image_width_px   = img_width,
        image_height_px  = img_height,
        focal_length_mm  = focal_length,
    )
    cam.set_calibration_yaml(calib_data)
    Kmat_cal = cam.calc_Kmat()
    dist_coeffs = cam._dist_coeffs_as_array()
    proj = PoseProjector(camera=cam)

    with open(kps_file, "r", encoding="utf-8") as handle:
        kps_mm = np.array(json.load(handle), dtype=float)
    target_BFF_pts_with_origin = np.vstack((np.zeros((1, 3)), kps_mm / 1e3))

    image_paths = sorted(image_folder.glob("*.png"))
    if not image_paths:
        raise FileNotFoundError(f"No images found in {image_folder}")

    vicon_df = _build_vicon_dataframe_from_rosbag(image_paths, rosbag_dir)
    vicon_csv_path = res_path / "vicon_from_rosbag.csv"
    vicon_df.to_csv(vicon_csv_path, index=False)

    T_CvC, T_TvT = _load_offset_estimates(offset_data, offset_keys)

    ttc_rows: list[dict[str, object]] = []

    for image_path in image_paths:
        vicon_match = vicon_df[vicon_df["frame"] == image_path.name]
        if vicon_match.empty:
            print(f"Frame {image_path.name} not found in rosbag-derived Vicon data, skipping.")
            continue
        vicon_row = vicon_match.iloc[0]

        q_C_2_T, r_Co2To_C, T_T_C = _process_vicon_offset_v01(
            row = vicon_row,
            T_CvC = T_CvC,
            T_TvT = T_TvT,
            vicon_keys = vicon_keys,
        )

        uv_vicon = proj.classless_pinhole_project_T4x4_2_uv(
            T_TARGET_CAM = T_T_C,
            Kmat = Kmat_cal,
            BC_dist_coeffs = dist_coeffs,
            points_xyz_TARGET = target_BFF_pts_with_origin,
        )

        img_base = image_path.stem
        img_vicon = draw_uv_points_on_image(
            img_or_path = str(image_path),
            points_uv = uv_vicon,
            point_color = (255, 0, 0),
            point_radius = 15,
            point_thickness = 2,
        )
        img_vicon = _draw_origin(img_vicon, uv_vicon)
        cv2.imwrite(str(res_path / f"vicon_reproj_{img_base}.png"), img_vicon)
        ttc_rows.append(
            {
                "frame": image_path.name,
                "image_timestamp_ns": int(vicon_row["image_timestamp_ns"]),
                "cam_delta_ms": float(vicon_row["cam_delta_ms"]),
                "soho_delta_ms": float(vicon_row["soho_delta_ms"]),
                "q_C_2_T": np.asarray(q_C_2_T, dtype=float).tolist(),
                "r_Co2To_C": np.asarray(r_Co2To_C, dtype=float).tolist(),
                "T_T_C": np.asarray(T_T_C, dtype=float).tolist(),
            }
        )

    with (res_path / "T_T_C_results.json").open("w", encoding="utf-8") as handle:
        json.dump(ttc_rows, handle, indent=4)

    print(f"Results located at: {res_path}")
    print(f"Rosbag-derived Vicon CSV written to: {vicon_csv_path}")


if __name__ == "__main__":
    main()
