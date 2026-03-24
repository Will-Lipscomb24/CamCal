from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd

############################# How This Script Works #############################
# 1) load camera calibration, solved offsets, and target keypoints
# 2) collect the rosbag-extracted images, then sync each image to the nearest
#    Vicon camera / target PoseStamped rows from the bag (or use a precomputed csv)
# 3) apply the solved camera / target Vicon offsets to build one truth T_T_C per image
# 4) apply the target body-fixed origin shift to build the shifted-truth variant
# 5) write both trajectory packs:
#       - truth
#       - shifted truth
# 6) optionally write sanity overlays for the generated packs so the projected
#    poses can be checked quickly
# 7) optionally call run_megapose_on_trajectory_pack_VC.py on the shifted-truth
#    lifted json so MegaPose refines from the shifted truth pose initialization
# 8) write one summary json that points to the rosbag sync outputs, generated packs,
#    and the MegaPose refinement summary
############################# How This Script Works #############################

from offset_utils.camera_io import (
                                        collect_indxed_image_paths,
                                        ensure_clean_dir,
                                        load_camera_calibration,
                                        parse_img_saver_ros_timestamp_v01,
                                        select_image_paths,
                                    )
from offset_utils.metrics_and_pack import (
                                            build_vicon_dataframe_from_rosbag,
                                            build_frame_record_from_vicon_row,
                                            write_topic_yamls,
                                            write_json_payload,
                                            write_trajectory_pack,
                                        )
from offset_utils.pose_ops import (
                                    apply_identity_target_origin_shift_to_T_T_C,
                                    load_offset_estimates,
                                    vicon_row_to_T_T_C_v01,
                                )
from offset_utils.reprojection import load_target_points, write_sanity_overlays

########################## Inputs ##########################
HERE                        = Path(__file__).resolve()
PARENT_ROOT                 = HERE.parent
CAMCAL_ROOT                 = PARENT_ROOT.parent
AGNC_ROOT                   = CAMCAL_ROOT.parent
NAVROS_ROOT                 = AGNC_ROOT / "nav_ros"
test_num                    = "001" # change this as needed
DEFAULT_RUN_NAME            = "run_007" # change this as needed
# DEFAULT_DATA_DIR            = CAMCAL_ROOT / "data" / "rosbag_data" / DEFAULT_RUN_NAME
DEFAULT_DATA_DIR            = NAVROS_ROOT / "testing" / "live_tests" / DEFAULT_RUN_NAME
DEFAULT_CALIBRATION_YAML    = CAMCAL_ROOT / "data" / "offset" / "collection_001" / "calibration.yaml" # change this as needed
DEFAULT_OFFSET_JSON         = CAMCAL_ROOT / "results" / "collection_001" / "calc_and_verify_offset_001" / "calc_offset_results.json" # change this as needed
DEFAULT_RESULT_ROOT         = CAMCAL_ROOT / "results" / DEFAULT_RUN_NAME / f"process_truth_rosbag_and_refine_VC_{DEFAULT_RUN_NAME}_{test_num}" # change this as needed

DEFAULT_CAM_TOPIC           = "/vicon/basler_cam/basler_cam" # change this as needed, should be the compresed? 
DEFAULT_TARGET_TOPIC        = "/vicon/soho/soho"

# keypoints
DEFAULT_TARGET_KPS_FILE             = PARENT_ROOT / "mesh_keypoints" / "soho" / "rendered_keypoints.json"
DEFAULT_TARGET_KPS_UNITS            = "m"
DEFAULT_SHIFTED_TARGET_KPS_FILE     = PARENT_ROOT / "mesh_keypoints" / "soho_centered" / "rendered_keypoints.json"
DEFAULT_SHIFTED_TARGET_KPS_UNITS    = "m"
DEFAULT_TARGET_BODY_FIXED_OFFSET_M  = [0.2374940214582, 0.0523612819210, -0.0269223600937]

# meshes
DEFAULT_MEGAPOSE_MESHES_DIR             = AGNC_ROOT / "nav_ros" / "testing" / "pose_model_artifacts" / "meshes"
DEFAULT_MEGAPOSE_MESH_LABEL             = "soho_centered"
DEFAULT_MEGAPOSE_MODEL                  = "megapose-1.0-RGB-multi-hypothesis"
DEFAULT_MEGAPOSE_OUTPUT_PACK_DIR_NAME   = "trajectory_pack_truth_shifted_megapose_h5_r8"
DEFAULT_MEGAPOSE_OUTPUT_LIFTED_FILENAME = "trajectory_lifted_truth_shifted_megapose_h5_r8.json"

DEFAULT_SENSOR_WIDTH_MM         = 14.13
DEFAULT_SENSOR_HEIGHT_MM        = 10.35
DEFAULT_CALIBRATION_WIDTH_PX    = 4096
DEFAULT_CALIBRATION_HEIGHT_PX   = 3000
DEFAULT_FOCAL_LENGTH_MM         = 25.0
########################## Inputs ##########################

def _infer_image_size_wh(image_paths: list[Path]) -> tuple[int, int]:
    """ Infer image width and height in pixels by reading the first image in the list """
    if len(image_paths) == 0:
        raise ValueError("Cannot infer image size from an empty image list.")
    image   = cv2.imread(str(image_paths[0]), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Failed to read image for size inference: {image_paths[0]}")
    height_px, width_px = image.shape[:2]
    return int(width_px), int(height_px)


def _resolve_rosbag_dir(image_dir: Path, rosbag_dir: Path | None) -> Path:
    """ Resolve the rosbag directory, defaulting to the single child bag under image_dir """
    if rosbag_dir is not None:
        resolved = rosbag_dir.expanduser().resolve()
        if not resolved.is_dir():
            raise FileNotFoundError(f"rosbag directory not found: {resolved}")
        if not (resolved / "metadata.yaml").is_file():
            raise FileNotFoundError(f"rosbag metadata.yaml not found under: {resolved}")
        return resolved

    candidates = []
    for child in sorted(image_dir.iterdir()):
        if child.is_dir() and (child / "metadata.yaml").is_file():
            candidates.append(child.resolve())

    if len(candidates) == 0:
        raise FileNotFoundError(
            f"No rosbag directory with metadata.yaml found directly under image_dir: {image_dir}"
        )
    if len(candidates) > 1:
        raise RuntimeError(
            "Expected exactly one rosbag directory under image_dir, found "
            f"{len(candidates)}: {[str(path) for path in candidates]}"
        )
    return candidates[0]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Process rosbag truth into trajectory packs, apply the target shift, and run MegaPose refinement."
    )
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument(
        "--rosbag-dir",
        type=Path,
        default=None,
        help="Optional rosbag directory. When omitted, detect the single child rosbag under --image-dir.",
    )
    parser.add_argument("--offset-json", type=Path, default=DEFAULT_OFFSET_JSON)
    parser.add_argument("--calibration-yaml", type=Path, default=DEFAULT_CALIBRATION_YAML)
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)
    parser.add_argument(
        "--precomputed-vicon-csv",
        type=Path,
        default=None,
        help="Optional precomputed rosbag-synced Vicon-style CSV. When provided, skip live rosbag topic reading.",
    )
    parser.add_argument("--cam-topic", type=str, default=DEFAULT_CAM_TOPIC)
    parser.add_argument("--target-topic", type=str, default=DEFAULT_TARGET_TOPIC)
    parser.add_argument("--target-kps-file", type=Path, default=DEFAULT_TARGET_KPS_FILE)
    parser.add_argument("--target-kps-units", type=str, default=DEFAULT_TARGET_KPS_UNITS)
    parser.add_argument("--shifted-target-kps-file", type=Path, default=DEFAULT_SHIFTED_TARGET_KPS_FILE)
    parser.add_argument("--shifted-target-kps-units", type=str, default=DEFAULT_SHIFTED_TARGET_KPS_UNITS)
    parser.add_argument(
        "--target-body-fixed-offset-m",
        type=float,
        nargs=3,
        default=DEFAULT_TARGET_BODY_FIXED_OFFSET_M,
        metavar=("TX_M", "TY_M", "TZ_M"),
    )
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--truth-only",
        action="store_true",
        help="Write only the unshifted truth pack and skip shifted truth / MegaPose.",
    )
    mode_group.add_argument(
        "--shifted-truth-only",
        action="store_true",
        help="Write only the shifted-truth pack. MegaPose still runs unless --skip-megapose is also set.",
    )
    parser.add_argument(
        "--skip-sanity-overlays",
        action="store_true",
        help="Skip writing sanity_check overlays for truth / shifted truth and pass the same choice to MegaPose.",
    )
    parser.add_argument(
        "--skip-megapose",
        action="store_true",
        help="Skip MegaPose refinement after writing the shifted-truth pack.",
    )
    parser.add_argument("--record-indices", type=int, nargs="*", default=None)
    parser.add_argument("--sensor-width-mm", type=float, default=DEFAULT_SENSOR_WIDTH_MM)
    parser.add_argument("--sensor-height-mm", type=float, default=DEFAULT_SENSOR_HEIGHT_MM)
    parser.add_argument("--focal-length-mm", type=float, default=DEFAULT_FOCAL_LENGTH_MM)
    parser.add_argument("--square-pixels", action="store_true")
    parser.add_argument("--disable-scaled-k", action="store_true")
    parser.add_argument("--calibration-width-px", type=int, default=DEFAULT_CALIBRATION_WIDTH_PX)
    parser.add_argument("--calibration-height-px", type=int, default=DEFAULT_CALIBRATION_HEIGHT_PX)

    parser.add_argument("--megapose-meshes-dir", type=Path, default=DEFAULT_MEGAPOSE_MESHES_DIR)
    parser.add_argument("--megapose-mesh-label", type=str, default=DEFAULT_MEGAPOSE_MESH_LABEL)
    parser.add_argument("--megapose-mesh-units", type=str, default="mm")
    parser.add_argument("--megapose-model", type=str, default=DEFAULT_MEGAPOSE_MODEL)
    parser.add_argument("--n-pose-hypotheses", type=int, default=5)
    parser.add_argument("--n-refiner-iterations", type=int, default=8)
    parser.add_argument("--megapose-output-pack-dir-name", type=str, default=DEFAULT_MEGAPOSE_OUTPUT_PACK_DIR_NAME)
    parser.add_argument(
        "--megapose-output-lifted-filename",
        type=str,
        default=DEFAULT_MEGAPOSE_OUTPUT_LIFTED_FILENAME,
    )
    return parser.parse_args()


def _run_megapose_refinement(
    trajectory_export_dir: Path,
    shifted_lifted_path: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    command = [
        sys.executable,
        str(PARENT_ROOT / "run_megapose_on_trajectory_pack_VC.py"),
        "--lifted-json",
        str(shifted_lifted_path),
        "--meshes-dir",
        str(args.megapose_meshes_dir),
        "--mesh-label",
        str(args.megapose_mesh_label),
        "--mesh-units",
        str(args.megapose_mesh_units),
        "--model",
        str(args.megapose_model),
        "--n-pose-hypotheses",
        str(args.n_pose_hypotheses),
        "--n-refiner-iterations",
        str(args.n_refiner_iterations),
        "--output-pack-dir-name",
        str(args.megapose_output_pack_dir_name),
        "--output-lifted-filename",
        str(args.megapose_output_lifted_filename),
        "--sanity-calibration-yaml",
        str(args.calibration_yaml),
        "--sanity-kps-file",
        str(args.shifted_target_kps_file),
        "--sanity-kps-units",
        str(args.shifted_target_kps_units),
    ]
    if bool(args.skip_sanity_overlays):
        command.append("--skip-sanity-overlays")
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH", "")
    pythonpath_parts = [str(PARENT_ROOT)]
    if existing_pythonpath:
        pythonpath_parts.append(existing_pythonpath)
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_parts)

    subprocess.run(command, cwd=str(trajectory_export_dir), env=env, check=True)

    summary_stem = Path(args.megapose_output_lifted_filename).stem
    summary_path = trajectory_export_dir / f"{summary_stem}_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"MegaPose summary JSON not found after refinement: {summary_path}")
    with summary_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _has_direct_rosbag_frames(image_dir: Path) -> bool:
    """Return True when the directory itself contains rosbag-saver style frames."""
    return any(image_dir.glob("frame_*.png"))


def _segment_sort_key(path: Path) -> tuple[int, str]:
    """Sort `images_<n>` directories numerically when possible."""
    suffix = path.name.split("_")[-1]
    try:
        return int(suffix), path.name
    except ValueError:
        return sys.maxsize, path.name


def _parse_segment_index_from_dir(segment_dir: Path) -> int:
    """Parse the numeric suffix from an `images_<n>` directory name."""
    suffix = segment_dir.name.split("_")[-1]
    try:
        return int(suffix)
    except ValueError as exc:
        raise ValueError(
            f"Expected segmented image directory to end with an integer suffix, got: {segment_dir.name}"
        ) from exc


def _discover_segment_image_dirs(image_root: Path) -> tuple[list[Path], list[str]]:
    """Discover non-empty `images_*` segment directories directly under the root."""
    segment_dirs: list[Path] = []
    skipped_empty: list[str] = []
    for child in sorted(image_root.iterdir(), key = _segment_sort_key):
        if not child.is_dir() or not child.name.startswith("images_"):
            continue
        if any(child.glob("frame_*.png")):
            segment_dirs.append(child.resolve())
        else:
            skipped_empty.append(child.name)
    return segment_dirs, skipped_empty


def _resolve_segment_mcap_path(
    rosbag_dir: Path | None,
    segment_index: int | None,
    input_mode: str,
) -> Path | None:
    """Resolve the split-mcap file corresponding to a segmented image directory."""
    if rosbag_dir is None or segment_index is None:
        return None
    if input_mode != "multi_segment_image_dirs":
        return None
    candidate = rosbag_dir / f"{rosbag_dir.name}_{int(segment_index)}.mcap"
    if not candidate.is_file():
        raise FileNotFoundError(f"Expected segment mcap not found for segment {segment_index}: {candidate}")
    if candidate.stat().st_size <= 0:
        raise RuntimeError(f"Resolved segment mcap is empty for segment {segment_index}: {candidate}")
    return candidate.resolve()


def _process_single_image_dir(
    *,
    args: argparse.Namespace,
    image_dir: Path,
    result_root: Path,
    resolved_rosbag_dir: Path | None,
    input_mode: str,
    run_root_dir: Path,
    segment_name: str = "",
    segment_index: int | None = None,
) -> dict[str, Any]:
    """Process one image directory into truth / shifted truth / megapose packs."""
    result_root = ensure_clean_dir(result_root)
    trajectory_export_dir = result_root / "trajectory_export"
    topic_yamls_dir = result_root / "topic_yamls"
    trajectory_export_dir.mkdir(parents=True, exist_ok=True)
    topic_yamls_dir.mkdir(parents=True, exist_ok=True)

    all_image_paths = collect_indxed_image_paths(
        image_dir=image_dir,
        img_suffix=".png",
        rosbag_style=True,
        img_name_parser=parse_img_saver_ros_timestamp_v01,
    )
    selected_image_paths, selected_record_indices = select_image_paths(all_image_paths, args.record_indices)
    image_width_px, image_height_px = _infer_image_size_wh(selected_image_paths)

    K, dist_coeffs, camera_settings = load_camera_calibration(
        calibration_yaml_path=args.calibration_yaml,
        sensor_width_mm=args.sensor_width_mm,
        sensor_height_mm=args.sensor_height_mm,
        image_width_px=image_width_px,
        image_height_px=image_height_px,
        focal_length_mm=args.focal_length_mm,
        square_pixels=bool(args.square_pixels),
        scale_loaded_K=not bool(args.disable_scaled_k),
        calibration_width_px=args.calibration_width_px,
        calibration_height_px=args.calibration_height_px,
    )
    target_pts_with_origin = load_target_points(
        args.target_kps_file,
        with_origin=True,
        units=args.target_kps_units,
    )
    shifted_target_pts_with_origin = load_target_points(
        args.shifted_target_kps_file,
        with_origin=True,
        units=args.shifted_target_kps_units,
    )
    selected_frame_names = {path.name for path in selected_image_paths}
    rosbag_input_path = resolved_rosbag_dir
    if args.precomputed_vicon_csv is None:
        rosbag_input_path = _resolve_segment_mcap_path(
            rosbag_dir = resolved_rosbag_dir,
            segment_index = segment_index,
            input_mode = input_mode,
        )
        if rosbag_input_path is None:
            rosbag_input_path = resolved_rosbag_dir
    if args.precomputed_vicon_csv is None:
        topic_yaml_info = write_topic_yamls(
            bag_dir=rosbag_input_path,
            topic_names=[args.cam_topic, args.target_topic],
            output_dir=topic_yamls_dir,
        )
        vicon_df = build_vicon_dataframe_from_rosbag(
            image_paths=selected_image_paths,
            bag_dir=rosbag_input_path,
            cam_topic=args.cam_topic,
            target_topic=args.target_topic,
        )
    else:
        resolved_rosbag_dir = args.rosbag_dir
        topic_yaml_info = {
            "output_dir": str(topic_yamls_dir),
            "manifest_path": "",
            "topics": [],
        }
        vicon_df = pd.read_csv(args.precomputed_vicon_csv)
        if "frame" not in vicon_df.columns:
            raise KeyError(f"precomputed Vicon CSV missing required 'frame' column: {args.precomputed_vicon_csv}")
        vicon_df = vicon_df[vicon_df["frame"].isin(selected_frame_names)].copy()
        vicon_df = vicon_df.sort_values("frame_index").reset_index(drop=True)
    vicon_csv_path = result_root / "vicon_from_rosbag.csv"
    vicon_df.to_csv(vicon_csv_path, index=False)

    T_CvC, T_TvT = load_offset_estimates(args.offset_json)
    target_body_fixed_offset_m = np.asarray(args.target_body_fixed_offset_m, dtype=float).reshape(3,)

    vicon_row_by_frame = {str(row["frame"]): row for _, row in vicon_df.iterrows()}
    frame_records: list[dict[str, Any]] = []
    shifted_frame_records: list[dict[str, Any]] = []
    frame_metric_rows: list[dict[str, Any]] = []
    T_T_C_rows: list[dict[str, Any]] = []

    for image_path in selected_image_paths:
        vicon_row = vicon_row_by_frame.get(image_path.name)
        if vicon_row is None:
            raise RuntimeError(f"Missing rosbag-derived truth row for image: {image_path.name}")

        q_CAM_2_TARGET, r_Co2To_C, q_TARGET_2_CAM, T_T_C = vicon_row_to_T_T_C_v01(
            row=vicon_row,
            T_CvC=T_CvC,
            T_TvT=T_TvT,
        )
        frame_record = build_frame_record_from_vicon_row(
                                                        image_path = image_path,
                                                        vicon_row = vicon_row,
                                                        q_CAM_2_TARGET = q_CAM_2_TARGET,
                                                        q_TARGET_2_CAM = q_TARGET_2_CAM,
                                                        r_Co2To_C = r_Co2To_C,
                                                        T_T_C = T_T_C,
                                                     )
        frame_records.append(frame_record)

        (
            q_CAM_2_TARGET_shifted,
            r_Co2To_C_shifted,
            q_TARGET_2_CAM_shifted,
            T_T_C_shifted,
        ) = apply_identity_target_origin_shift_to_T_T_C(
            T_T_C=T_T_C,
            r_To_2_T1_T=target_body_fixed_offset_m,
        )
        shifted_frame_records.append(
            build_frame_record_from_vicon_row(
                                                image_path = image_path,
                                                vicon_row = vicon_row,
                                                q_CAM_2_TARGET = q_CAM_2_TARGET_shifted,
                                                q_TARGET_2_CAM = q_TARGET_2_CAM_shifted,
                                                r_Co2To_C = r_Co2To_C_shifted,
                                                T_T_C = T_T_C_shifted,
                                             )
        )

        frame_index, image_stamp_ns, _ = parse_img_saver_ros_timestamp_v01(image_path)
        frame_metric_rows.append(
            {
                "frame": str(image_path.name),
                "frame_index": int(frame_index),
                "image_timestamp_ns": int(image_stamp_ns),
                "cam_timestamp_ns": int(vicon_row["cam_timestamp_ns"]),
                "soho_timestamp_ns": int(vicon_row["soho_timestamp_ns"]),
                "cam_delta_ms": float(vicon_row["cam_delta_ms"]),
                "soho_delta_ms": float(vicon_row["soho_delta_ms"]),
                "truth_tx_m": float(r_Co2To_C[0]),
                "truth_ty_m": float(r_Co2To_C[1]),
                "truth_tz_m": float(r_Co2To_C[2]),
                "truth_shifted_tx_m": float(r_Co2To_C_shifted[0]),
                "truth_shifted_ty_m": float(r_Co2To_C_shifted[1]),
                "truth_shifted_tz_m": float(r_Co2To_C_shifted[2]),
            }
        )
        T_T_C_rows.append(
            {
                "frame": str(image_path.name),
                "frame_index": int(frame_index),
                "q_CAM_2_TARGET": np.asarray(q_CAM_2_TARGET, dtype=float).tolist(),
                "q_TARGET_2_CAM": np.asarray(q_TARGET_2_CAM, dtype=float).tolist(),
                "r_Co2To_C": np.asarray(r_Co2To_C, dtype=float).tolist(),
                "T_T_C": np.asarray(T_T_C, dtype=float).tolist(),
                "q_CAM_2_TARGET_shifted": np.asarray(q_CAM_2_TARGET_shifted, dtype=float).tolist(),
                "q_TARGET_2_CAM_shifted": np.asarray(q_TARGET_2_CAM_shifted, dtype=float).tolist(),
                "r_Co2To_C_shifted": np.asarray(r_Co2To_C_shifted, dtype=float).tolist(),
                "T_T_C_shifted": np.asarray(T_T_C_shifted, dtype=float).tolist(),
            }
        )

    frame_metrics_df = pd.DataFrame(frame_metric_rows)
    frame_metrics_csv = result_root / "frame_metrics.csv"
    frame_metrics_json = result_root / "frame_metrics.json"
    frame_metrics_df.to_csv(frame_metrics_csv, index=False)
    write_json_payload(frame_metrics_json, frame_metric_rows)
    T_T_C_results_path = result_root / "T_T_C_results.json"
    write_json_payload(T_T_C_results_path, T_T_C_rows)

    truth_traj_dir            = None
    truth_lifted_path         = None
    truth_sanity_dir          = None
    shifted_truth_traj_dir    = None
    shifted_truth_lifted_path = None
    shifted_truth_sanity_dir  = None
    run_name = str(run_root_dir.name)
    source_rosbag_mcap_path = _resolve_segment_mcap_path(
        rosbag_dir = resolved_rosbag_dir,
        segment_index = segment_index,
        input_mode = input_mode,
    )
    record_metadata = {
        "source_run_name": run_name,
    }
    if len(segment_name) > 0:
        record_metadata["source_segment_name"] = str(segment_name)
    if segment_index is not None:
        record_metadata["source_segment_index"] = int(segment_index)

    if not bool(args.shifted_truth_only):
        truth_traj_dir, truth_lifted_path = write_trajectory_pack(
            output_dir=trajectory_export_dir,
            frame_records=frame_records,
            camera_settings=camera_settings,
            K=K,
            dist_coeffs=dist_coeffs,
            image_width_px=image_width_px,
            image_height_px=image_height_px,
            pack_dir_name="trajectory_pack_truth",
            lifted_filename="trajectory_lifted_truth.json",
            metadata_extras={
                "origin_shift_applied": False,
                "source_pose_variant": "truth",
                **record_metadata,
            },
            source_rosbag_dir=resolved_rosbag_dir,
            source_rosbag_mcap_path=source_rosbag_mcap_path,
        )
        if not bool(args.skip_sanity_overlays):
            truth_sanity_dir = write_sanity_overlays(
                traj_dir=truth_traj_dir,
                frame_records=frame_records,
                K=K,
                dist_coeffs=dist_coeffs,
                target_pts_with_origin=target_pts_with_origin,
            )

    if not bool(args.truth_only):
        shifted_truth_traj_dir, shifted_truth_lifted_path = write_trajectory_pack(
            output_dir=trajectory_export_dir,
            frame_records=shifted_frame_records,
            camera_settings=camera_settings,
            K=K,
            dist_coeffs=dist_coeffs,
            image_width_px=image_width_px,
            image_height_px=image_height_px,
            pack_dir_name="trajectory_pack_truth_shifted",
            lifted_filename="trajectory_lifted_truth_shifted.json",
            metadata_extras={
                "target_origin_shift_m": target_body_fixed_offset_m.tolist(),
                "target_origin_shift_frame": "target",
                "origin_shift_applied": True,
                "source_pose_variant": "truth_translation_shifted",
                **record_metadata,
            },
            source_rosbag_dir=resolved_rosbag_dir,
            source_rosbag_mcap_path=source_rosbag_mcap_path,
        )
        if not bool(args.skip_sanity_overlays):
            shifted_truth_sanity_dir = write_sanity_overlays(
                traj_dir=shifted_truth_traj_dir,
                frame_records=shifted_frame_records,
                K=K,
                dist_coeffs=dist_coeffs,
                target_pts_with_origin=shifted_target_pts_with_origin,
            )

    if bool(args.truth_only):
        megapose_summary = {
            "skipped": True,
            "reason": "--truth-only requested; shifted truth pack was not written.",
        }
    elif bool(args.skip_megapose):
        megapose_summary = {
            "skipped": True,
            "reason": "--skip-megapose requested.",
        }
    else:
        assert shifted_truth_lifted_path is not None
        megapose_summary = _run_megapose_refinement(
            trajectory_export_dir=trajectory_export_dir,
            shifted_lifted_path=shifted_truth_lifted_path,
            args=args,
        )

    summary = {
        "result_root": str(result_root),
        "bag_dir": str(resolved_rosbag_dir) if resolved_rosbag_dir is not None else "",
        "image_dir": str(image_dir),
        "run_root_dir": str(run_root_dir),
        "input_mode": str(input_mode),
        "source_run_name": str(run_name),
        "source_segment_name": str(segment_name),
        "source_segment_index": int(segment_index) if segment_index is not None else -1,
        "source_rosbag_mcap_path": "" if source_rosbag_mcap_path is None else str(source_rosbag_mcap_path),
        "source_rosbag_mcap_name": "" if source_rosbag_mcap_path is None else str(source_rosbag_mcap_path.name),
        "offset_json": str(args.offset_json),
        "precomputed_vicon_csv": str(args.precomputed_vicon_csv) if args.precomputed_vicon_csv is not None else "",
        "used_precomputed_vicon_csv": bool(args.precomputed_vicon_csv is not None),
        "topic_yamls_dir": str(topic_yamls_dir),
        "topic_yamls_manifest": str(topic_yaml_info["manifest_path"]),
        "vicon_csv": str(vicon_csv_path),
        "frame_metrics_csv": str(frame_metrics_csv),
        "frame_metrics_json": str(frame_metrics_json),
        "T_T_C_results_json": str(T_T_C_results_path),
        "num_images_total_discovered": int(len(all_image_paths)),
        "num_images_processed": int(len(selected_image_paths)),
        "selected_record_indices": [int(idx) for idx in selected_record_indices],
        "image_size_wh": [int(image_width_px), int(image_height_px)],
        "scaled_K_enabled": bool(not args.disable_scaled_k),
        "truth_only": bool(args.truth_only),
        "shifted_truth_only": bool(args.shifted_truth_only),
        "skip_sanity_overlays": bool(args.skip_sanity_overlays),
        "skip_megapose": bool(args.skip_megapose),
        "trajectory_pack_truth_dir": str(truth_traj_dir) if truth_traj_dir is not None else "",
        "trajectory_lifted_truth_json": str(truth_lifted_path) if truth_lifted_path is not None else "",
        "trajectory_truth_sanity_dir": str(truth_sanity_dir) if truth_sanity_dir is not None else "",
        "trajectory_pack_truth_shifted_dir": str(shifted_truth_traj_dir) if shifted_truth_traj_dir is not None else "",
        "trajectory_lifted_truth_shifted_json": str(shifted_truth_lifted_path) if shifted_truth_lifted_path is not None else "",
        "trajectory_truth_shifted_sanity_dir": str(shifted_truth_sanity_dir) if shifted_truth_sanity_dir is not None else "",
        "target_body_fixed_offset_m": target_body_fixed_offset_m.tolist(),
        "megapose_summary": megapose_summary,
    }
    write_json_payload(result_root / "summary.json", summary)
    print(json.dumps(summary, indent=2))
    print(f"All done! Results written to: {result_root}")
    return summary


def main() -> None:
    args = _parse_args()
    args.image_dir = args.image_dir.expanduser().resolve()
    args.offset_json = args.offset_json.expanduser().resolve()
    args.calibration_yaml = args.calibration_yaml.expanduser().resolve()
    args.result_root = args.result_root.expanduser().resolve()
    if args.rosbag_dir is not None:
        args.rosbag_dir = args.rosbag_dir.expanduser().resolve()
    if args.precomputed_vicon_csv is not None:
        args.precomputed_vicon_csv = args.precomputed_vicon_csv.expanduser().resolve()
    args.target_kps_file = args.target_kps_file.expanduser().resolve()
    args.shifted_target_kps_file = args.shifted_target_kps_file.expanduser().resolve()
    args.megapose_meshes_dir = args.megapose_meshes_dir.expanduser().resolve()

    if _has_direct_rosbag_frames(args.image_dir):
        resolved_rosbag_dir = None
        if args.precomputed_vicon_csv is None:
            resolved_rosbag_dir = _resolve_rosbag_dir(
                image_dir = args.image_dir,
                rosbag_dir = args.rosbag_dir,
            )
        _process_single_image_dir(
            args = args,
            image_dir = args.image_dir,
            result_root = args.result_root,
            resolved_rosbag_dir = resolved_rosbag_dir,
            input_mode = "single_image_dir",
            run_root_dir = args.image_dir,
        )
        return

    segment_dirs, skipped_empty = _discover_segment_image_dirs(args.image_dir)
    if len(segment_dirs) == 0:
        raise RuntimeError(
            "No direct frame_*.png files and no non-empty images_* segment directories found under "
            f"{args.image_dir}"
        )

    base_result_root = ensure_clean_dir(args.result_root)
    resolved_rosbag_dir = None
    if args.precomputed_vicon_csv is None:
        resolved_rosbag_dir = _resolve_rosbag_dir(
            image_dir = args.image_dir,
            rosbag_dir = args.rosbag_dir,
        )

    segment_summaries: list[dict[str, Any]] = []
    for segment_ordinal, segment_dir in enumerate(segment_dirs):
        segment_index = _parse_segment_index_from_dir(segment_dir)
        segment_result_root = base_result_root / segment_dir.name
        print(
            f"Processing segment {segment_ordinal + 1}/{len(segment_dirs)}: "
            f"{segment_dir.name} (segment index {segment_index}) -> {segment_result_root}",
            flush = True,
        )
        segment_summary = _process_single_image_dir(
            args = args,
            image_dir = segment_dir,
            result_root = segment_result_root,
            resolved_rosbag_dir = resolved_rosbag_dir,
            input_mode = "multi_segment_image_dirs",
            run_root_dir = args.image_dir,
            segment_name = str(segment_dir.name),
            segment_index = int(segment_index),
        )
        segment_summaries.append(
            {
                "segment_name": str(segment_dir.name),
                "segment_index": int(segment_index),
                "segment_ordinal": int(segment_ordinal),
                "image_dir": str(segment_dir),
                "result_root": str(segment_result_root),
                "num_images_processed": int(segment_summary["num_images_processed"]),
                "source_rosbag_mcap_name": str(segment_summary["source_rosbag_mcap_name"]),
                "source_rosbag_mcap_path": str(segment_summary["source_rosbag_mcap_path"]),
                "trajectory_pack_truth_dir": str(segment_summary["trajectory_pack_truth_dir"]),
                "trajectory_pack_truth_shifted_dir": str(segment_summary["trajectory_pack_truth_shifted_dir"]),
            }
        )

    aggregate_summary = {
        "result_root": str(base_result_root),
        "run_root_dir": str(args.image_dir),
        "input_mode": "multi_segment_image_dirs",
        "bag_dir": str(resolved_rosbag_dir) if resolved_rosbag_dir is not None else "",
        "offset_json": str(args.offset_json),
        "precomputed_vicon_csv": str(args.precomputed_vicon_csv) if args.precomputed_vicon_csv is not None else "",
        "used_precomputed_vicon_csv": bool(args.precomputed_vicon_csv is not None),
        "num_segments_processed": int(len(segment_summaries)),
        "segment_names_processed": [entry["segment_name"] for entry in segment_summaries],
        "skipped_empty_segments": [str(name) for name in skipped_empty],
        "segments": segment_summaries,
    }
    write_json_payload(base_result_root / "summary.json", aggregate_summary)
    print(json.dumps(aggregate_summary, indent=2))
    print(f"All done! Results written to: {base_result_root}")


if __name__ == "__main__":
    main()

# test runs:

# truth + shifted truth + megapose:
# need to switch envs
# source /home/saa4743/.venvs/navy/bin/activate
# python /home/saa4743/agnc_repos/CamCal/src/process_truth_rosbag_and_refine_VC.py \
#   --megapose-mesh-label middle_soho_real

# truth + shifted truth, no megapose:
# python3 /home/saa4743/agnc_repos/CamCal/src/process_truth_rosbag_and_refine_VC.py --skip-megapose

# truth only:
# python3 /home/saa4743/agnc_repos/CamCal/src/process_truth_rosbag_and_refine_VC.py --truth-only

# shifted truth only:
# python3 /home/saa4743/agnc_repos/CamCal/src/process_truth_rosbag_and_refine_VC.py --shifted-truth-only --skip-sanity-overlays --skip-megapose


# cd /home/saa4743/agnc_repos/CamCal

# /home/saa4743/.venvs/navy/bin/python src/process_truth_rosbag_and_refine_VC.py \
#   --image-dir /home/saa4743/agnc_repos/nav_ros/testing/live_tests/run_010 \
#   --result-root /home/saa4743/agnc_repos/CamCal/results/run_010/process_truth_rosbag_and_refine_VC_run_010_001

# # overwriting mesh:
# cd /home/saa4743/agnc_repos/CamCal
# /home/saa4743/.venvs/navy/bin/python src/process_truth_rosbag_and_refine_VC.py \
#   --image-dir /home/saa4743/agnc_repos/nav_ros/testing/live_tests/run_011 \
#   --result-root /home/saa4743/agnc_repos/CamCal/results/run_011/process_truth_rosbag_and_refine_VC_run_011_001 \
#   --megapose-mesh-label middle_soho_real \
#   --megapose-mesh-units mm \
#   --megapose-meshes-dir /home/saa4743/agnc_repos/nav_ros/testing/pose_model_artifacts/meshes

# # overwriting mesh + using different keypoints 
# cd /home/saa4743/agnc_repos/CamCal
# /home/saa4743/.venvs/navy/bin/python src/process_truth_rosbag_and_refine_VC.py \
#   --image-dir /home/saa4743/agnc_repos/nav_ros/testing/live_tests/run_013 \
#   --result-root /home/saa4743/agnc_repos/CamCal/results/run_013/process_truth_rosbag_and_refine_VC_run_013_001 \
#   --megapose-mesh-label middle_soho_real \
#   --megapose-mesh-units mm \
#   --megapose-meshes-dir /home/saa4743/agnc_repos/nav_ros/testing/pose_model_artifacts/meshes \
#   --shifted-target-kps-file /home/saa4743/agnc_repos/nav_ros/testing/pose_model_artifacts/mesh_keypoints/middle_soho_real/mesh_points_5000.json \
#   --shifted-target-kps-units mm


# # soho with colors overwriting mesh + using different keypoints 
# cd /home/saa4743/agnc_repos/CamCal
# /home/saa4743/.venvs/navy/bin/python src/process_truth_rosbag_and_refine_VC.py \
#   --image-dir /home/saa4743/agnc_repos/nav_ros/testing/live_tests/run_014 \
#   --result-root /home/saa4743/agnc_repos/CamCal/results/run_014/process_truth_rosbag_and_refine_VC_run_014_001 \
#   --megapose-mesh-label soho_centered_color \
#   --megapose-mesh-units mm \
#   --megapose-meshes-dir /home/saa4743/agnc_repos/nav_ros/testing/pose_model_artifacts/meshes \
#   --shifted-target-kps-file /home/saa4743/agnc_repos/nav_ros/testing/pose_model_artifacts/mesh_keypoints/soho_centered/rendered_keypoints.json \
#   --shifted-target-kps-units mm


# # soho with colors overwriting mesh + using different keypoints, less jitter
# cd /home/saa4743/agnc_repos/CamCal
# /home/saa4743/.venvs/navy/bin/python src/process_truth_rosbag_and_refine_VC.py \
#   --image-dir /home/saa4743/agnc_repos/nav_ros/testing/live_tests/run_016 \
#   --result-root /home/saa4743/agnc_repos/CamCal/results/run_016/process_truth_rosbag_and_refine_VC_run_016_001 \
#   --megapose-mesh-label soho_centered_color \
#   --megapose-mesh-units mm \
#   --megapose-meshes-dir /home/saa4743/agnc_repos/nav_ros/testing/pose_model_artifacts/meshes \
#   --shifted-target-kps-file /home/saa4743/agnc_repos/nav_ros/testing/pose_model_artifacts/mesh_keypoints/soho_centered/rendered_keypoints.json \
#   --shifted-target-kps-units mm
