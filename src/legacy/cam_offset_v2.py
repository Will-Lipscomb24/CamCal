import json
from pathlib import Path

import cv2
import pandas as pd
import numpy as np

from offset_utils.camera_io import (
                                        ensure_clean_dir,
                                        load_camera_calibration,
                                        build_charuco_board,
                                        get_charuco_T_T_C_series,
                                    )
from offset_utils.pose_ops import (
                                    load_vicon_dataframe,
                                    build_vicon_transform_series,
                                    sync_measurements,
                                    solve_rwhe,
                                    per_observation_costs,
                                    T_to_params,
                                 )
from offset_utils.reprojection import (
                                            write_charuco_reprojection_overlays,
                                            write_overlay_stats,
                                       )


##################################### Inputs #####################################
HERE                    = Path(__file__).resolve().parent
PARENT                  = HERE.parent
DATA_ROOT               = PARENT / "data" / "offset"
EXP_NAME                = "collection_001"
DATA_FOLDER             = DATA_ROOT / EXP_NAME
IMAGE_DIR               = DATA_FOLDER / "images"
IMG_SUFFIX              = ".png"
VICON_CSV_PATH          = DATA_FOLDER / "vicon_data.csv"
CALIBRATION_YAML_PATH   = DATA_FOLDER / "calibration.yaml"

RESULT_PATH             = PARENT / "results" / EXP_NAME / "cam_offset_v2"
OPENCV_POSE_EST_PATH    = RESULT_PATH / "calc_camera_poses.csv"
OUTPUT_JSON_PATH        = RESULT_PATH / "calc_offset_results.json"
REPROJECTION_DIR        = RESULT_PATH / "reprojection"

SQUARES_X               = 9
SQUARES_Y               = 5
SQUARE_LEN_M            = 17e-3
MARKER_LEN_M            = 12e-3
ARUCO_DICT_ID           =  cv2.aruco.DICT_5X5_100
AXIS_LENGTH_M           = 3.0 * SQUARE_LEN_M

SENSOR_WIDTH_MM         = 14.13
SENSOR_HEIGHT_MM        = 10.35
IMAGE_WIDTH_PX          = 4096
IMAGE_HEIGHT_PX         = 3000
FOCAL_LENGTH_MM         = 25.0
SQUARE_PIXELS           = False

EXCLUDED_IMAGE_NUMBERS  = [42, 43, 44, 45, 46, 47]
##################################### Inputs #####################################


def main():
    ############################## Secondary Input Setup #############################
    ensure_clean_dir(RESULT_PATH)
    REPROJECTION_DIR.mkdir(parents = True, exist_ok = True)

    K, dist_coeffs, _  = load_camera_calibration(
                                                    calibration_yaml_path = CALIBRATION_YAML_PATH,
                                                    sensor_width_mm = SENSOR_WIDTH_MM,
                                                    sensor_height_mm = SENSOR_HEIGHT_MM,
                                                    image_width_px = IMAGE_WIDTH_PX,
                                                    image_height_px = IMAGE_HEIGHT_PX,
                                                    focal_length_mm = FOCAL_LENGTH_MM,
                                                    square_pixels = SQUARE_PIXELS,
                                                )
    board, aruco_dict  = build_charuco_board(
                                                squares_x = SQUARES_X,
                                                squares_y = SQUARES_Y,
                                                square_len_m = SQUARE_LEN_M,
                                                marker_len_m = MARKER_LEN_M,
                                                aruco_dict_id = ARUCO_DICT_ID,
                                            )

    T_T_C_array, valid_image_numbers, _, reprojection_rows = get_charuco_T_T_C_series(
                                                                                        image_dir = IMAGE_DIR,
                                                                                        img_suffix = IMG_SUFFIX,
                                                                                        K = K,
                                                                                        dist = dist_coeffs,
                                                                                        board = board,
                                                                                        aruco_dict = aruco_dict,
                                                                                    )

    opencv_pose_rows = []
    for row in reprojection_rows:
        rvec        = np.asarray(row["rvec"], dtype = float).reshape(3,)
        tvec        = np.asarray(row["tvec"], dtype = float).reshape(3,)
        image_path  = Path(row["image_path"])
        opencv_pose_rows.append(
                                {
                                    "frame"  : image_path.name,
                                    "rvec_x" : float(rvec[0]),
                                    "rvec_y" : float(rvec[1]),
                                    "rvec_z" : float(rvec[2]),
                                    "tvec_x" : float(tvec[0]),
                                    "tvec_y" : float(tvec[1]),
                                    "tvec_z" : float(tvec[2]),
                                }
                              )
    opencv_pose_df = pd.DataFrame(opencv_pose_rows)
    opencv_pose_df.to_csv(OPENCV_POSE_EST_PATH, index = False)

    vicon_df                        = load_vicon_dataframe(VICON_CSV_PATH)
    T_CvV_by_image, T_TvV_by_image, _ = build_vicon_transform_series(vicon_df)
    T_T_C_sync, T_CvV_sync, T_TvV_sync, matched_image_numbers, dropped_numbers = sync_measurements(
                                                                                                            T_T_C_array = T_T_C_array,
                                                                                                            valid_image_numbers = valid_image_numbers,
                                                                                                            T_CvV_by_image = T_CvV_by_image,
                                                                                                            T_TvV_by_image = T_TvV_by_image,
                                                                                                            excluded_image_numbers = EXCLUDED_IMAGE_NUMBERS,
                                                                                                        )

    T0_CvC                  = np.eye(4, dtype = np.float64)
    T0_TvT                  = np.eye(4, dtype = np.float64)
    T0_TvT[:3, 3]           = np.array([-228.0, -55.0, 30.0], dtype = np.float64) / 1000.0

    print(f"Vicon rows loaded: {len(vicon_df)}")
    print(f"Matched Vicon frames: {len(matched_image_numbers) + len(dropped_numbers)}")
    print(f"Excluded image numbers: {dropped_numbers if dropped_numbers else 'none'}")
    print(f"Measurements used in solve: {len(matched_image_numbers)}")

    T_CvC, T_TvT, result, initial_cost = solve_rwhe(
                                                        T0_CvC = T0_CvC,
                                                        T0_TvT = T0_TvT,
                                                        T_T_C_array = T_T_C_sync,
                                                        T_CvV_array = T_CvV_sync,
                                                        T_TvV_array = T_TvV_sync,
                                                    )

    print("\nHomogenous Transformation Matrix from Camera Vicon to True Camera Frame:")
    print(T_CvC)
    print("\nHomogenous Transformation Matrix from Target Vicon to True Target Frame:")
    print(T_TvT)

    with OUTPUT_JSON_PATH.open("w", encoding = "utf-8") as handle:
        json.dump(
                    {
                        "T_CvC" : T_CvC.tolist(),
                        "T_TvT" : T_TvT.tolist(),
                    },
                    handle,
                    indent = 4,
                 )

    overlay_stats                = write_charuco_reprojection_overlays(
                                                                        output_dir = REPROJECTION_DIR,
                                                                        reprojection_rows = reprojection_rows,
                                                                        board = board,
                                                                        K = K,
                                                                        dist_coeffs = dist_coeffs,
                                                                        axis_length_m = AXIS_LENGTH_M,
                                                                    )
    overlay_csv_path, overlay_json_path = write_overlay_stats(
                                                              output_dir = RESULT_PATH,
                                                              overlay_stats = overlay_stats,
                                                              basename = "charuco_reprojection_metrics",
                                                          )

    obs_costs                    = per_observation_costs(result)
    solve_cost_rows              = []
    for image_number, cost in zip(matched_image_numbers, obs_costs):
        solve_cost_rows.append(
                                {
                                    "image_number" : int(image_number),
                                    "solve_cost"   : float(cost),
                                }
                              )
    solve_cost_df                = pd.DataFrame(solve_cost_rows)
    solve_cost_df.to_csv(RESULT_PATH / "solve_frame_costs.csv", index = False)

    summary = {
                "images_discovered"        : int(len(list(IMAGE_DIR.glob(f"*{IMG_SUFFIX}")))),
                "valid_charuco_detections" : int(len(valid_image_numbers)),
                "matched_image_numbers"    : [int(n) for n in matched_image_numbers],
                "excluded_image_numbers"   : [int(n) for n in dropped_numbers],
                "initial_cost"             : float(initial_cost),
                "final_cost"               : float(result.cost),
                "opencv_pose_csv"          : str(OPENCV_POSE_EST_PATH),
                "offset_json"              : str(OUTPUT_JSON_PATH),
                "reprojection_dir"         : str(REPROJECTION_DIR),
                "reprojection_metrics_csv" : str(overlay_csv_path),
                "reprojection_metrics_json": str(overlay_json_path),
                "solve_frame_costs_csv"    : str(RESULT_PATH / "solve_frame_costs.csv"),
                "T_CvC_params"             : T_to_params(T_CvC).tolist(),
                "T_TvT_params"             : T_to_params(T_TvT).tolist(),
              }
    with (RESULT_PATH / "summary.json").open("w", encoding = "utf-8") as handle:
        json.dump(summary, handle, indent = 4)

    print(f"Results saved to {RESULT_PATH}")


if __name__ == "__main__":
    main()
