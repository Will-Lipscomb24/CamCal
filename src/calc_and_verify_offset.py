from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import pdb

# local imports
from offset_utils.camera_io import (
                                        ensure_clean_dir,
                                        load_camera_calibration,
                                        build_charuco_board,
                                        build_camera_pose_rows_from_reprojection_rows,
                                        get_charuco_T_T_C_series,
                                        parse_image_number,
                                    )
from offset_utils.metrics_and_pack import (
                                                build_pose_metric_rows,
                                                write_json_payload,
                                                write_metric_bundle,
                                           )
from offset_utils.pose_ops import (
                                    apply_camera_target_vicon_offset,
                                    build_vicon_transform_series,
                                    load_vicon_dataframe,
                                    solve_rwhe,
                                    sync_charuco_vicon_measurements,
                                    T_to_params,
                                 )
from offset_utils.reprojection import (
                                            draw_combined_pose_overlay,
                                            draw_truth_charuco_overlay,
                                            draw_pose_overlay,
                                            load_target_points,
                                            project_points_T_T_C,
                                       )

##################################### Inputs #####################################
HERE                    = Path(__file__).resolve()
PARENT_ROOT             = HERE.parent
CAMCAL_ROOT             = PARENT_ROOT.parent

DATA_DIR                = CAMCAL_ROOT / "data" / "offset" / "collection_001"
IMAGE_DIR               = DATA_DIR / "images"
VAL_IMAGE_DIR           = DATA_DIR / "val_images"
VICON_CSV_PATH          = DATA_DIR / "vicon_data.csv"
CALIBRATION_YAML_PATH   = DATA_DIR / "calibration.yaml"
RESULT_PATH             = CAMCAL_ROOT / "results" / "collection_001" / "calc_and_verify_offset_001"
TARGET_KPS_FILE_1       = PARENT_ROOT / "mesh_keypoints" / "soho" / "rendered_keypoints.json"
TARGET_KPS_UNITS_1      = "m"
TARGET_KPS_FILE_2       = PARENT_ROOT / "mesh_keypoints" / "soho_centered" / "rendered_keypoints.json"
TARGET_KPS_UNITS_2      = "m"

# holdout precedence:
# 1) explicit HOLDOUT_IMAGE_NUMBERS if not None
# 2) HOLDOUT_FRACTION if not None
HOLDOUT_IMAGE_NUMBERS   = [42, 43, 44, 45, 46, 47]
HOLDOUT_FRACTION        = None
HOLDOUT_SEED            = 0

# camera raw intrinsics 
SENSOR_WIDTH_MM         = 14.13
SENSOR_HEIGHT_MM        = 10.35
IMAGE_WIDTH_PX          = 4096
IMAGE_HEIGHT_PX         = 3000
FOCAL_LENGTH_MM         = 25.0
SQUARE_PIXELS           = False

# chrauco board geometry
SQUARES_X               = 9
SQUARES_Y               = 5
SQUARE_LEN_M            = 17e-3
MARKER_LEN_M            = 12e-3
ARUCO_DICT_ID           = cv2.aruco.DICT_5X5_100
AXIS_LENGTH_M           = 3.0 * SQUARE_LEN_M
TvT0_GUESS_MM           = [-228, -55, 30]
r_To2T1_T               = [237.4940214582, 52.3612819210, -26.9223600937]
##################################### Inputs #####################################
print(f"Running {HERE.name}")
print(f"Results will be stored in {RESULT_PATH}")

def _select_holdout_numbers(
                                matched_image_numbers      : list[int],
                                requested_holdout_numbers  : list[int] | None,
                                holdout_fraction           : float | None,
                                holdout_seed               : int,
                                val_image_numbers          : list[int],
                            ) -> tuple[list[int], dict[str, Any]]:
    matched_sorted  = sorted(int(n) for n in matched_image_numbers)

    if requested_holdout_numbers is not None and len(requested_holdout_numbers) > 0:
        requested       = sorted({int(n) for n in requested_holdout_numbers})
        source          = "explicit"
    elif holdout_fraction is not None:
        if not (0.0 <= float(holdout_fraction) < 1.0):
            raise ValueError("--holdout-fraction must be in [0.0, 1.0).")
        source          = "fraction"
        requested_count = int(np.floor(len(matched_sorted) * float(holdout_fraction)))
        if float(holdout_fraction) > 0.0 and requested_count == 0 and len(matched_sorted) > 1:
            requested_count = 1
        rng             = np.random.default_rng(int(holdout_seed))
        requested       = sorted(int(n) for n in rng.choice(matched_sorted, size = requested_count, replace = False).tolist())
    else:
        requested       = sorted({int(n) for n in val_image_numbers})
        source          = "val_images"

    matched_set = set(matched_sorted)
    effective   = [image_number for image_number in requested if image_number in matched_set]
    dropped     = [image_number for image_number in requested if image_number not in matched_set]

    return effective, {
                        "holdout_source"                           : source,
                        "requested_holdout_image_numbers"          : requested,
                        "effective_holdout_image_numbers"          : effective,
                        "dropped_holdout_image_numbers_not_matched": dropped,
                        "holdout_fraction"                         : None if holdout_fraction is None else float(holdout_fraction),
                        "holdout_seed"                             : int(holdout_seed),
                      }


def main() -> None:
    # setup input, ouput dirs 
    image_dir               = IMAGE_DIR.expanduser().resolve()
    val_image_dir           = VAL_IMAGE_DIR.expanduser().resolve()
    vicon_csv_path          = VICON_CSV_PATH.expanduser().resolve()
    calibration_yaml_path   = CALIBRATION_YAML_PATH.expanduser().resolve()
    result_root             = ensure_clean_dir(RESULT_PATH.expanduser().resolve())
    target_kps_file         = TARGET_KPS_FILE_1.expanduser().resolve()
    images_discovered       = sorted(image_dir.glob("*.png"))

    truth_charuco_dir       = result_root / "truth_charuco_reprojection"
    truth_target_dir        = result_root / "truth_target_reprojection"
    holdout_combined_dir    = result_root / "holdout_combined_reprojection"
    truth_charuco_dir.mkdir(parents = True, exist_ok = True)
    truth_target_dir.mkdir(parents = True, exist_ok = True)
    holdout_combined_dir.mkdir(parents = True, exist_ok = True)

    # load camera intrinsics, board geometry, and the target keypoints used for the visual verification overlays
    K, dist_coeffs, _   = load_camera_calibration(
                                                    calibration_yaml_path = calibration_yaml_path,
                                                    sensor_width_mm = SENSOR_WIDTH_MM,
                                                    sensor_height_mm = SENSOR_HEIGHT_MM,
                                                    image_width_px = IMAGE_WIDTH_PX,
                                                    image_height_px = IMAGE_HEIGHT_PX,
                                                    focal_length_mm = FOCAL_LENGTH_MM,
                                                    square_pixels = bool(SQUARE_PIXELS),
                                                )
    board, aruco_dict   = build_charuco_board(
                                                squares_x = SQUARES_X,
                                                squares_y = SQUARES_Y,
                                                square_len_m = SQUARE_LEN_M,
                                                marker_len_m = MARKER_LEN_M,
                                                aruco_dict_id = ARUCO_DICT_ID,
    )
    target_pts_with_origin  = load_target_points(
                                                    target_kps_file,
                                                    with_origin = True,
                                                    units = TARGET_KPS_UNITS_1,
                                            )

    # run OpenCV ChArUco pose estimation across the full calibration image set first,
    # then later split the matched measurements into train / holdout for the actual solve
    (
        T_T_C_array,
        valid_image_numbers,
        valid_image_paths,
        reprojection_rows,
        invalid_image_paths,
                            )   = get_charuco_T_T_C_series(
                                    image_dir = image_dir,
                                    img_suffix = ".png",
                                    K = K,
                                    dist = dist_coeffs,
                                    board = board,
                                    aruco_dict = aruco_dict,
                                )

    camera_pose_rows        = build_camera_pose_rows_from_reprojection_rows(reprojection_rows)
    camera_pose_df          = pd.DataFrame(camera_pose_rows)
    camera_pose_csv_path    = result_root / "camera_poses.csv"
    camera_pose_df.to_csv(camera_pose_csv_path, index = False)

    # sync the valid ChArUco measurements with the Vicon rows by image number before we choose the holdout set
    vicon_df                = load_vicon_dataframe(vicon_csv_path)
    T_CvV_by_image, T_TvV_by_image, vicon_image_numbers                         = build_vicon_transform_series(vicon_df)
    T_T_C_sync_all, T_CvV_sync_all, T_TvV_sync_all, matched_image_numbers, _    = sync_charuco_vicon_measurements(
                                                                                                                    T_T_C_array = T_T_C_array,
                                                                                                                    valid_image_numbers = valid_image_numbers,
                                                                                                                    T_CvV_by_image = T_CvV_by_image,
                                                                                                                    T_TvV_by_image = T_TvV_by_image,
                                                                                                                    excluded_image_numbers = None,
                                                                                                                )

    matched_set                 = set(matched_image_numbers)
    valid_index_by_image        = {int(image_number): idx for idx, image_number in enumerate(valid_image_numbers)}
    matched_image_paths         = [Path(valid_image_paths[valid_index_by_image[image_number]]) for image_number in matched_image_numbers]
    matched_reprojection_rows   = [reprojection_rows[valid_index_by_image[image_number]] for image_number in matched_image_numbers]
    # TODO: make holdout easier, just store validation images in separate dir
    val_image_numbers           = (
                                    sorted({parse_image_number(path) for path in val_image_dir.glob("*.png")})
                                    if val_image_dir.exists()
                                    else []
                                )
    holdout_image_numbers, holdout_manifest = _select_holdout_numbers(
                                                                        matched_image_numbers = matched_image_numbers,
                                                                        requested_holdout_numbers = HOLDOUT_IMAGE_NUMBERS,
                                                                        holdout_fraction = HOLDOUT_FRACTION,
                                                                        holdout_seed = HOLDOUT_SEED,
                                                                        val_image_numbers = val_image_numbers,
                                                                    )
    holdout_set     = set(holdout_image_numbers)
    train_indices   = [idx for idx, image_number in enumerate(matched_image_numbers) if image_number not in holdout_set]
    holdout_indices = [idx for idx, image_number in enumerate(matched_image_numbers) if image_number in holdout_set]
    if len(train_indices) == 0:
        raise RuntimeError("No training frames remain after applying the requested holdout split.")

    # initial guess of zero offset for camera and rough guess for offset between target vicon frame and true target frame based on visual inspection
    T0_CvC          = np.eye(4, dtype = np.float64)
    T0_TvT          = np.eye(4, dtype = np.float64)
    T0_TvT[:3, 3]   = np.asarray(TvT0_GUESS_MM, dtype = np.float64).reshape(3,) / 1000.0
    print(f"Starting solve with initial guess T0_CvC:\n{T0_CvC}")
    print(f"Starting solve with initial guess T0_TvT:\n{T0_TvT}")
    T_CvC, T_TvT, result, initial_cost  = solve_rwhe(
                                                        T0_CvC = T0_CvC,
                                                        T0_TvT = T0_TvT,
                                                        T_T_C_array = T_T_C_sync_all[train_indices],
                                                        T_CvV_array = T_CvV_sync_all[train_indices],
                                                        T_TvV_array = T_TvV_sync_all[train_indices],
                                                    )

    truth_T_T_C_all = np.stack(
                                [
                                    apply_camera_target_vicon_offset(
                                                                        T_CvC = T_CvC,
                                                                        T_TvT = T_TvT,
                                                                        T_CvV = T_CvV_sync_all[idx],
                                                                        T_TvV = T_TvV_sync_all[idx],
                                                                    )
                                    for idx in range(len(matched_image_numbers))
                                ]
                            )

    train_metric_rows, train_summary        = build_pose_metric_rows(
                                                                        split_name = "train",
                                                                        image_numbers = [matched_image_numbers[idx] for idx in train_indices],
                                                                        image_paths = [matched_image_paths[idx] for idx in train_indices],
                                                                        T_T_C_est = T_T_C_sync_all[train_indices],
                                                                        T_T_C_truth = truth_T_T_C_all[train_indices],
                                                                    )
    holdout_metric_rows, holdout_summary    = build_pose_metric_rows(
                                                                        split_name = "holdout",
                                                                        image_numbers = [matched_image_numbers[idx] for idx in holdout_indices],
                                                                        image_paths = [matched_image_paths[idx] for idx in holdout_indices],
                                                                        T_T_C_est = (
                                                                                        T_T_C_sync_all[holdout_indices]
                                                                                        if len(holdout_indices) > 0
                                                                                        else np.zeros((0, 4, 4), dtype = np.float64)
                                                                                    ),
                                                                        T_T_C_truth = (
                                                                                            truth_T_T_C_all[holdout_indices]
                                                                                            if len(holdout_indices) > 0
                                                                                            else np.zeros((0, 4, 4), dtype = np.float64)
                                                                                    ),
                                                                    )

    write_metric_bundle(result_root / "train_metrics", train_metric_rows, train_summary)

    combined_overlay_paths: dict[int, str]  = {}
    for idx, image_number in enumerate(matched_image_numbers):
        image_path  = matched_image_paths[idx]
        T_T_C_truth = truth_T_T_C_all[idx]
        detection   = matched_reprojection_rows[idx]["detection"]

        truth_charuco_overlay   = draw_truth_charuco_overlay(
                                                                image_path = image_path,
                                                                detection = detection,
                                                                board = board,
                                                                T_T_C_truth = T_T_C_truth,
                                                                K = K,
                                                                dist_coeffs = dist_coeffs,
                                                                axis_length_m = AXIS_LENGTH_M,
                                                            )
        truth_charuco_path      = truth_charuco_dir / f"truth_charuco_{image_path.name}"
        cv2.imwrite(str(truth_charuco_path), truth_charuco_overlay)

        uv_truth                = project_points_T_T_C(T_T_C_truth, K, dist_coeffs, target_pts_with_origin)
        truth_target_overlay    = draw_pose_overlay(
                                                        image = str(image_path),
                                                        uv_points = uv_truth,
                                                        point_color = (255, 0, 0),
                                                        point_radius = 10,
                                                        point_thickness = 2,
                                                        origin_color = (0, 255, 0),
                                                        origin_radius = 20,
                                                        origin_thickness = 3,
                                                        text_label = image_path.name,
                                                    )
        truth_target_path       = truth_target_dir / f"truth_target_{image_path.name}"
        cv2.imwrite(str(truth_target_path), truth_target_overlay)

        if image_number in holdout_set:
            uv_opencv           = project_points_T_T_C(T_T_C_sync_all[idx], K, dist_coeffs, target_pts_with_origin)
            combined_overlay    = draw_combined_pose_overlay(
                                                                image_path = image_path,
                                                                uv_opencv = uv_opencv,
                                                                uv_truth = uv_truth,
                                                                text_label = image_path.name,
                                                            )
            combined_overlay_path   = holdout_combined_dir / f"holdout_combined_{image_path.name}"
            cv2.imwrite(str(combined_overlay_path), combined_overlay)
            combined_overlay_paths[image_number] = str(combined_overlay_path)

    for row in holdout_metric_rows:
        row["combined_overlay_path"] = combined_overlay_paths.get(int(row["image_number"]), "")
    write_metric_bundle(result_root / "holdout_metrics", holdout_metric_rows, holdout_summary)

    calc_offset_results = {
                            "T_CvC": np.asarray(T_CvC, dtype=np.float64).tolist(),
                            "T_TvT": np.asarray(T_TvT, dtype=np.float64).tolist(),
                            "T_CvC_params": T_to_params(T_CvC).tolist(),
                            "T_TvT_params": T_to_params(T_TvT).tolist(),
                            "initial_cost": float(initial_cost),
                            "final_cost": float(result.cost),
                            "termination_message": str(result.message),
                            "num_measurements_used_in_solve": int(len(train_indices)),
                    }
    calc_offset_results_path    = result_root / "calc_offset_results.json"
    write_json_payload(calc_offset_results_path, calc_offset_results)

    invalid_image_numbers           = sorted(parse_image_number(path) for path in invalid_image_paths)
    unmatched_valid_image_numbers   = sorted(set(int(n) for n in valid_image_numbers) - matched_set)
    split_manifest                  = {
                                        "images_dir": str(image_dir),
                                        "val_images_dir": str(val_image_dir),
                                        "vicon_csv": str(vicon_csv_path),
                                        "images_discovered": int(len(images_discovered)),
                                        "valid_charuco_detections": int(len(valid_image_numbers)),
                                        "invalid_charuco_image_numbers": invalid_image_numbers,
                                        "vicon_image_numbers": [int(n) for n in vicon_image_numbers],
                                        "matched_image_numbers": [int(n) for n in matched_image_numbers],
                                        "unmatched_valid_charuco_image_numbers": unmatched_valid_image_numbers,
                                        "train_image_numbers": [int(matched_image_numbers[idx]) for idx in train_indices],
                                        "holdout_image_numbers": [int(n) for n in holdout_image_numbers],
                                        **holdout_manifest,
                                    }
    split_manifest_path             = result_root / "split_manifest.json"
    write_json_payload(split_manifest_path, split_manifest)

    summary = {
                "result_root": str(result_root),
                "calc_offset_results_json": str(calc_offset_results_path),
                "camera_poses_csv": str(camera_pose_csv_path),
                "split_manifest_json": str(split_manifest_path),
                "train_metrics_csv": str(result_root / "train_metrics.csv"),
                "train_metrics_json": str(result_root / "train_metrics.json"),
                "holdout_metrics_csv": str(result_root / "holdout_metrics.csv"),
                "holdout_metrics_json": str(result_root / "holdout_metrics.json"),
                "truth_charuco_reprojection_dir": str(truth_charuco_dir),
                "truth_target_reprojection_dir": str(truth_target_dir),
                "holdout_combined_reprojection_dir": str(holdout_combined_dir),
                "images_discovered": int(len(images_discovered)),
                "valid_charuco_detections": int(len(valid_image_numbers)),
                "matched_measurements": int(len(matched_image_numbers)),
                "train_measurements": int(len(train_indices)),
                "holdout_measurements": int(len(holdout_indices)),
                "train_summary": train_summary,
                "holdout_summary": holdout_summary,
                "T_CvC_params": T_to_params(T_CvC).tolist(),
                "T_TvT_params": T_to_params(T_TvT).tolist(),
                "initial_cost": float(initial_cost),
                "final_cost": float(result.cost),
                "termination_message": str(result.message),
            }
    write_json_payload(result_root / "summary.json", summary)
    print(json.dumps(summary, indent = 2))
    print(f"Finished. Results stored in {result_root}")


if __name__ == "__main__":
    main()
