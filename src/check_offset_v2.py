import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from offset_utils.camera_io import (
                                        ensure_clean_dir,
                                        load_camera_calibration,
                                        parse_image_number,
                                    )
from offset_utils.pose_ops import (
                                    DEFAULT_VICON_KEYS,
                                    load_vicon_dataframe,
                                    load_offset_estimates,
                                    opencv_rvec_tvec_to_T_T_C,
                                    vicon_row_to_T_T_C_v01,
                                 )
from offset_utils.reprojection import (
                                            load_target_points,
                                            project_points_T_T_C,
                                            draw_pose_overlay,
                                            draw_combined_pose_overlay,
                                        )
from offset_utils.metrics_and_pack import (
                                                compute_pose_error_metrics,
                                                summarize_frame_metrics,
                                                write_error_histograms,
                                           )


##################################### Inputs #####################################
HERE                        = Path(__file__).resolve().parent
PARENT                      = HERE.parent
DATA_ROOT                   = PARENT / "data" / "offset"
EXP_NAME                    = "collection_001"
DATA_FOLDER                 = DATA_ROOT / EXP_NAME
IMAGE_DIR                   = DATA_FOLDER / "images"
VICON_CSV_PATH              = DATA_FOLDER / "vicon_data.csv"
CALIBRATION_YAML_PATH       = DATA_FOLDER / "calibration.yaml"

SOLVE_RESULT_DIR            = PARENT / "results" / EXP_NAME / "cam_offset_v2"
INPUT_JSON_PATH             = SOLVE_RESULT_DIR / "calc_offset_results.json"
INPUT_OPENCV_PATH           = SOLVE_RESULT_DIR / "calc_camera_poses.csv"
RESULT_PATH                 = PARENT / "results" / EXP_NAME / "check_offset_v2"
OPENCV_REPROJ_DIR           = RESULT_PATH / "opencv_reprojection"
VICON_REPROJ_DIR            = RESULT_PATH / "vicon_reprojection"
COMBINED_REPROJ_DIR         = RESULT_PATH / "combined_reprojection"
HISTOGRAM_DIR               = RESULT_PATH / "error_histograms"

SC_POSE_EXAMPLES_ROOT       = PARENT.parent / "sc-pose-utils" / "src" / "sc_pose" / "examples"
KPS_FILE                    = SC_POSE_EXAMPLES_ROOT / "artifacts" / "soho_reframed_mesh_pose_pack" / "mesh_points_50000.json"

SENSOR_WIDTH_MM             = 14.13
SENSOR_HEIGHT_MM            = 10.35
IMAGE_WIDTH_PX              = 4096
IMAGE_HEIGHT_PX             = 3000
FOCAL_LENGTH_MM             = 25.0
SQUARE_PIXELS               = False

SCALE_LOADED_K              = False
CALIBRATION_WIDTH_PX        = 4096
CALIBRATION_HEIGHT_PX       = 3000
##################################### Inputs #####################################


def main():
    ############################## Secondary Input Setup #############################
    ensure_clean_dir(RESULT_PATH)
    OPENCV_REPROJ_DIR.mkdir(parents = True, exist_ok = True)
    VICON_REPROJ_DIR.mkdir(parents = True, exist_ok = True)
    COMBINED_REPROJ_DIR.mkdir(parents = True, exist_ok = True)
    HISTOGRAM_DIR.mkdir(parents = True, exist_ok = True)

    K, dist_coeffs, _  = load_camera_calibration(
                                                    calibration_yaml_path = CALIBRATION_YAML_PATH,
                                                    sensor_width_mm = SENSOR_WIDTH_MM,
                                                    sensor_height_mm = SENSOR_HEIGHT_MM,
                                                    image_width_px = IMAGE_WIDTH_PX,
                                                    image_height_px = IMAGE_HEIGHT_PX,
                                                    focal_length_mm = FOCAL_LENGTH_MM,
                                                    square_pixels = SQUARE_PIXELS,
                                                    scale_loaded_K = SCALE_LOADED_K,
                                                    calibration_width_px = CALIBRATION_WIDTH_PX,
                                                    calibration_height_px = CALIBRATION_HEIGHT_PX,
                                                )
    target_pts_with_origin = load_target_points(KPS_FILE, with_origin = True)
    vicon_df               = load_vicon_dataframe(VICON_CSV_PATH)
    T_CvC, T_TvT           = load_offset_estimates(INPUT_JSON_PATH)

    opencv_df              = pd.read_csv(INPUT_OPENCV_PATH)
    opencv_df["frame_idx"] = opencv_df["frame"].apply(parse_image_number)
    opencv_df              = opencv_df.sort_values("frame_idx").reset_index(drop = True)

    frame_metrics_rows     = []

    for _, row in opencv_df.iterrows():
        img_name                     = str(row["frame"])
        img_num                      = int(row["frame_idx"])
        img_path                     = IMAGE_DIR / img_name
        vicon_match                  = vicon_df[vicon_df[DEFAULT_VICON_KEYS["frame"]] == img_num]
        if vicon_match.empty:
            print(f"Image number {img_num} for {img_name} not found in Vicon data, skipping")
            continue
        vicon_row                    = vicon_match.iloc[0]

        rvec                         = np.array([row["rvec_x"], row["rvec_y"], row["rvec_z"]], dtype = float)
        tvec                         = np.array([row["tvec_x"], row["tvec_y"], row["tvec_z"]], dtype = float)
        q_cam_target_cv, r_cv, q_target_cam_cv, T_T_C_cv = opencv_rvec_tvec_to_T_T_C(rvec, tvec)
        q_cam_target_vc, r_vc, q_target_cam_vc, T_T_C_vc = vicon_row_to_T_T_C_v01(vicon_row, T_CvC, T_TvT)

        uv_cv                        = project_points_T_T_C(T_T_C_cv, K, dist_coeffs, target_pts_with_origin)
        uv_vc                        = project_points_T_T_C(T_T_C_vc, K, dist_coeffs, target_pts_with_origin)

        img_base                     = img_path.stem
        opencv_overlay               = draw_pose_overlay(
                                                            image = str(img_path),
                                                            uv_points = uv_cv,
                                                            point_color = (0, 0, 255),
                                                            point_radius = 12,
                                                            point_thickness = 2,
                                                            origin_color = (0, 255, 255),
                                                            origin_radius = 20,
                                                            origin_thickness = 3,
                                                            text_label = img_name,
                                                        )
        vicon_overlay                = draw_pose_overlay(
                                                            image = str(img_path),
                                                            uv_points = uv_vc,
                                                            point_color = (255, 0, 0),
                                                            point_radius = 12,
                                                            point_thickness = 2,
                                                            origin_color = (0, 255, 0),
                                                            origin_radius = 20,
                                                            origin_thickness = 3,
                                                            text_label = img_name,
                                                        )
        combined_overlay             = draw_combined_pose_overlay(
                                                                    image_path = img_path,
                                                                    uv_opencv = uv_cv,
                                                                    uv_truth = uv_vc,
                                                                    text_label = img_name,
                                                                )

        cv2.imwrite(str(OPENCV_REPROJ_DIR / f"opencv_reproj_{img_base}.png"), opencv_overlay)
        cv2.imwrite(str(VICON_REPROJ_DIR / f"vicon_reproj_{img_base}.png"), vicon_overlay)
        cv2.imwrite(str(COMBINED_REPROJ_DIR / f"combined_reproj_{img_base}.png"), combined_overlay)

        metric_row                   = compute_pose_error_metrics(T_T_C_cv, T_T_C_vc)
        metric_row.update(
                            {
                                "frame"                      : img_name,
                                "image_number"               : img_num,
                                "opencv_tx_m"                : float(r_cv[0]),
                                "opencv_ty_m"                : float(r_cv[1]),
                                "opencv_tz_m"                : float(r_cv[2]),
                                "truth_tx_m"                 : float(r_vc[0]),
                                "truth_ty_m"                 : float(r_vc[1]),
                                "truth_tz_m"                 : float(r_vc[2]),
                                "opencv_q_cam_2_target_w"    : float(q_cam_target_cv[0]),
                                "opencv_q_cam_2_target_x"    : float(q_cam_target_cv[1]),
                                "opencv_q_cam_2_target_y"    : float(q_cam_target_cv[2]),
                                "opencv_q_cam_2_target_z"    : float(q_cam_target_cv[3]),
                                "truth_q_cam_2_target_w"     : float(q_cam_target_vc[0]),
                                "truth_q_cam_2_target_x"     : float(q_cam_target_vc[1]),
                                "truth_q_cam_2_target_y"     : float(q_cam_target_vc[2]),
                                "truth_q_cam_2_target_z"     : float(q_cam_target_vc[3]),
                                "opencv_q_target_2_cam_w"    : float(q_target_cam_cv[0]),
                                "opencv_q_target_2_cam_x"    : float(q_target_cam_cv[1]),
                                "opencv_q_target_2_cam_y"    : float(q_target_cam_cv[2]),
                                "opencv_q_target_2_cam_z"    : float(q_target_cam_cv[3]),
                                "truth_q_target_2_cam_w"     : float(q_target_cam_vc[0]),
                                "truth_q_target_2_cam_x"     : float(q_target_cam_vc[1]),
                                "truth_q_target_2_cam_y"     : float(q_target_cam_vc[2]),
                                "truth_q_target_2_cam_z"     : float(q_target_cam_vc[3]),
                                "opencv_overlay_path"        : str(OPENCV_REPROJ_DIR / f"opencv_reproj_{img_base}.png"),
                                "vicon_overlay_path"         : str(VICON_REPROJ_DIR / f"vicon_reproj_{img_base}.png"),
                                "combined_overlay_path"      : str(COMBINED_REPROJ_DIR / f"combined_reproj_{img_base}.png"),
                            }
                          )
        frame_metrics_rows.append(metric_row)

    frame_metrics_df = pd.DataFrame(frame_metrics_rows)
    frame_metrics_df.to_csv(RESULT_PATH / "frame_metrics.csv", index = False)
    with (RESULT_PATH / "frame_metrics.json").open("w", encoding = "utf-8") as handle:
        json.dump(frame_metrics_rows, handle, indent = 4)

    summary = summarize_frame_metrics(frame_metrics_df)
    histogram_manifest = write_error_histograms(
                                                frame_metrics_df = frame_metrics_df,
                                                output_dir = HISTOGRAM_DIR,
                                              )
    summary.update(
                    {
                        "input_offset_json"  : str(INPUT_JSON_PATH),
                        "input_opencv_csv"   : str(INPUT_OPENCV_PATH),
                        "input_vicon_csv"    : str(VICON_CSV_PATH),
                        "opencv_reproj_dir"  : str(OPENCV_REPROJ_DIR),
                        "vicon_reproj_dir"   : str(VICON_REPROJ_DIR),
                        "combined_reproj_dir": str(COMBINED_REPROJ_DIR),
                        "histogram_dir"      : str(HISTOGRAM_DIR),
                        "num_histograms"     : int(len(histogram_manifest)),
                    }
                  )
    with (RESULT_PATH / "summary.json").open("w", encoding = "utf-8") as handle:
        json.dump(summary, handle, indent = 4)

    print(f"Results located at: {RESULT_PATH}")


if __name__ == "__main__":
    main()
