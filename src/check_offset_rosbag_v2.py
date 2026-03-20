import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from offset_utils.camera_io import (
                                        ensure_clean_dir,
                                        load_camera_calibration,
                                        collect_image_paths,
                                        parse_rosbag_frame_name,
                                    )
from offset_utils.pose_ops import load_offset_estimates, vicon_row_to_T_T_C_v01
from offset_utils.reprojection import load_target_points, project_points_T_T_C, draw_pose_overlay
from offset_utils.metrics_and_pack import (
                                                build_vicon_dataframe_from_rosbag,
                                                write_topic_yamls,
                                                write_trajectory_pack,
                                           )


##################################### Inputs #####################################
HERE                        = Path(__file__).resolve().parent
PARENT                      = HERE.parent
RUN_NAME                    = "run_004"
DATA_FOLDER                 = PARENT / "data" / "rosbag_data" / RUN_NAME
IMAGE_DIR                   = DATA_FOLDER
ROSBAG_DIR                  = DATA_FOLDER / "rosbag2_2026_03_19-23_06_16"
CALIBRATION_YAML_PATH       = PARENT / "data" / "offset" / "collection_001" / "calibration.yaml"
OFFSET_JSON_PATH            = PARENT / "results" / "collection_001" / "cam_offset_v2" / "calc_offset_results.json"

RESULT_PATH                 = PARENT / "results" / RUN_NAME / "check_offset_rosbag_v2"
REPROJECTION_DIR            = RESULT_PATH / "reprojection"
TRAJECTORY_EXPORT_DIR       = RESULT_PATH / "trajectory_export"
TOPIC_YAMLS_DIR             = RESULT_PATH / "topic_yamls"
VICON_CSV_OUT_PATH          = RESULT_PATH / "vicon_from_rosbag.csv"
T_T_C_RESULTS_PATH          = RESULT_PATH / "T_T_C_results.json"

SC_POSE_EXAMPLES_ROOT       = PARENT.parent / "sc-pose-utils" / "src" / "sc_pose" / "examples"
KPS_FILE                    = SC_POSE_EXAMPLES_ROOT / "artifacts" / "soho_reframed_mesh_pose_pack" / "mesh_points_50000.json"

CAM_TOPIC                   = "/vicon/basler_cam/basler_cam"
TARGET_TOPIC                = "/vicon/soho/soho"

SENSOR_WIDTH_MM             = 14.13
SENSOR_HEIGHT_MM            = 10.35
CALIBRATION_WIDTH_PX        = 4096
CALIBRATION_HEIGHT_PX       = 3000
IMAGE_WIDTH_PX              = 1024
IMAGE_HEIGHT_PX             = 750
FOCAL_LENGTH_MM             = 25.0
SQUARE_PIXELS               = False
SCALE_LOADED_K              = True
##################################### Inputs #####################################


def main():
    ############################## Secondary Input Setup #############################
    ensure_clean_dir(RESULT_PATH)
    REPROJECTION_DIR.mkdir(parents = True, exist_ok = True)
    TRAJECTORY_EXPORT_DIR.mkdir(parents = True, exist_ok = True)
    TOPIC_YAMLS_DIR.mkdir(parents = True, exist_ok = True)

    K, dist_coeffs, camera_settings = load_camera_calibration(
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
    target_pts_with_origin  = load_target_points(KPS_FILE, with_origin = True)
    image_paths             = collect_image_paths(IMAGE_DIR, img_suffix = ".png", rosbag_style = True)
    topic_yaml_info         = write_topic_yamls(
                                                bag_dir = ROSBAG_DIR,
                                                topic_names = [CAM_TOPIC, TARGET_TOPIC],
                                                output_dir = TOPIC_YAMLS_DIR,
                                            )
    vicon_df                = build_vicon_dataframe_from_rosbag(
                                                            image_paths = image_paths,
                                                            bag_dir = ROSBAG_DIR,
                                                            cam_topic = CAM_TOPIC,
                                                            target_topic = TARGET_TOPIC,
                                                        )
    vicon_df.to_csv(VICON_CSV_OUT_PATH, index = False)

    T_CvC, T_TvT            = load_offset_estimates(OFFSET_JSON_PATH)

    frame_records           = []
    frame_metric_rows       = []
    T_T_C_rows              = []

    for image_path in image_paths:
        frame_index, image_stamp_ns    = parse_rosbag_frame_name(image_path)
        vicon_match                    = vicon_df[vicon_df["frame"] == image_path.name]
        if vicon_match.empty:
            print(f"Frame {image_path.name} not found in rosbag-derived Vicon dataframe, skipping")
            continue

        vicon_row                      = vicon_match.iloc[0]
        q_CAM_2_TARGET, r_Co2To_C, q_TARGET_2_CAM, T_T_C = vicon_row_to_T_T_C_v01(
                                                                                        row = vicon_row,
                                                                                        T_CvC = T_CvC,
                                                                                        T_TvT = T_TvT,
                                                                                    )
        uv_truth                       = project_points_T_T_C(T_T_C, K, dist_coeffs, target_pts_with_origin)
        overlay                        = draw_pose_overlay(
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
        reproj_path                    = REPROJECTION_DIR / f"vicon_reproj_{image_path.stem}.png"
        cv2.imwrite(str(reproj_path), overlay)

        frame_metric_rows.append(
                                    {
                                        "frame"                     : image_path.name,
                                        "frame_index"               : int(frame_index),
                                        "image_timestamp_ns"        : int(image_stamp_ns),
                                        "cam_timestamp_ns"          : int(vicon_row["cam_timestamp_ns"]),
                                        "soho_timestamp_ns"         : int(vicon_row["soho_timestamp_ns"]),
                                        "cam_delta_ms"              : float(vicon_row["cam_delta_ms"]),
                                        "soho_delta_ms"             : float(vicon_row["soho_delta_ms"]),
                                        "tx_m"                      : float(r_Co2To_C[0]),
                                        "ty_m"                      : float(r_Co2To_C[1]),
                                        "tz_m"                      : float(r_Co2To_C[2]),
                                        "q_cam_2_target_w"          : float(q_CAM_2_TARGET[0]),
                                        "q_cam_2_target_x"          : float(q_CAM_2_TARGET[1]),
                                        "q_cam_2_target_y"          : float(q_CAM_2_TARGET[2]),
                                        "q_cam_2_target_z"          : float(q_CAM_2_TARGET[3]),
                                        "q_target_2_cam_w"          : float(q_TARGET_2_CAM[0]),
                                        "q_target_2_cam_x"          : float(q_TARGET_2_CAM[1]),
                                        "q_target_2_cam_y"          : float(q_TARGET_2_CAM[2]),
                                        "q_target_2_cam_z"          : float(q_TARGET_2_CAM[3]),
                                        "reprojection_path"         : str(reproj_path),
                                    }
                                  )
        T_T_C_rows.append(
                            {
                                "frame"                 : image_path.name,
                                "frame_index"           : int(frame_index),
                                "image_timestamp_ns"    : int(image_stamp_ns),
                                "cam_delta_ms"          : float(vicon_row["cam_delta_ms"]),
                                "soho_delta_ms"         : float(vicon_row["soho_delta_ms"]),
                                "q_CAM_2_TARGET"        : np.asarray(q_CAM_2_TARGET, dtype = float).tolist(),
                                "q_TARGET_2_CAM"        : np.asarray(q_TARGET_2_CAM, dtype = float).tolist(),
                                "r_Co2To_C"             : np.asarray(r_Co2To_C, dtype = float).tolist(),
                                "T_T_C"                 : np.asarray(T_T_C, dtype = float).tolist(),
                            }
                         )
        frame_records.append(
                                {
                                    "image_path"          : image_path,
                                    "frame"               : image_path.name,
                                    "frame_index"         : int(frame_index),
                                    "image_timestamp_ns"  : int(image_stamp_ns),
                                    "cam_timestamp_ns"    : int(vicon_row["cam_timestamp_ns"]),
                                    "soho_timestamp_ns"   : int(vicon_row["soho_timestamp_ns"]),
                                    "cam_delta_ms"        : float(vicon_row["cam_delta_ms"]),
                                    "soho_delta_ms"       : float(vicon_row["soho_delta_ms"]),
                                    "q_CAM_2_TARGET"      : np.asarray(q_CAM_2_TARGET, dtype = float).tolist(),
                                    "q_TARGET_2_CAM"      : np.asarray(q_TARGET_2_CAM, dtype = float).tolist(),
                                    "r_Co2To_C"           : np.asarray(r_Co2To_C, dtype = float).tolist(),
                                    "T_T_C"               : np.asarray(T_T_C, dtype = float).tolist(),
                                }
                             )

    frame_metrics_df = pd.DataFrame(frame_metric_rows)
    frame_metrics_df.to_csv(RESULT_PATH / "frame_metrics.csv", index = False)
    with (RESULT_PATH / "frame_metrics.json").open("w", encoding = "utf-8") as handle:
        json.dump(frame_metric_rows, handle, indent = 4)
    with T_T_C_RESULTS_PATH.open("w", encoding = "utf-8") as handle:
        json.dump(T_T_C_rows, handle, indent = 4)

    traj_dir, lifted_path = write_trajectory_pack(
                                                output_dir = TRAJECTORY_EXPORT_DIR,
                                                frame_records = frame_records,
                                                camera_settings = camera_settings,
                                                K = K,
                                                image_width_px = IMAGE_WIDTH_PX,
                                                image_height_px = IMAGE_HEIGHT_PX,
                                            )

    sanity_check_dir = traj_dir / "sanity_check"
    ensure_clean_dir(sanity_check_dir)
    for idx, record in enumerate(frame_records):
        token                   = f"{idx:05d}"
        sanity_image_path       = traj_dir / f"image_{token}.png"
        T_T_C                   = np.asarray(record["T_T_C"], dtype = float)
        uv_truth                = project_points_T_T_C(T_T_C, K, dist_coeffs, target_pts_with_origin)
        sanity_overlay          = draw_pose_overlay(
                                                    image = str(sanity_image_path),
                                                    uv_points = uv_truth,
                                                    point_color = (255, 0, 0),
                                                    point_radius = 10,
                                                    point_thickness = 2,
                                                    origin_color = (0, 255, 0),
                                                    origin_radius = 20,
                                                    origin_thickness = 3,
                                                    text_label = sanity_image_path.name,
                                                 )
        cv2.imwrite(str(sanity_check_dir / f"sanity_{token}.png"), sanity_overlay)

    summary = {
                "num_images"             : int(len(image_paths)),
                "num_records"            : int(len(frame_records)),
                "bag_dir"                : str(ROSBAG_DIR),
                "image_dir"              : str(IMAGE_DIR),
                "offset_json"            : str(OFFSET_JSON_PATH),
                "scaled_K_enabled"       : bool(SCALE_LOADED_K),
                "calibration_size_wh"    : [int(CALIBRATION_WIDTH_PX), int(CALIBRATION_HEIGHT_PX)],
                "image_size_wh"          : [int(IMAGE_WIDTH_PX), int(IMAGE_HEIGHT_PX)],
                "mean_cam_delta_ms"      : float(frame_metrics_df["cam_delta_ms"].mean()) if len(frame_metrics_df) > 0 else float("nan"),
                "max_abs_cam_delta_ms"   : float(frame_metrics_df["cam_delta_ms"].abs().max()) if len(frame_metrics_df) > 0 else float("nan"),
                "mean_soho_delta_ms"     : float(frame_metrics_df["soho_delta_ms"].mean()) if len(frame_metrics_df) > 0 else float("nan"),
                "max_abs_soho_delta_ms"  : float(frame_metrics_df["soho_delta_ms"].abs().max()) if len(frame_metrics_df) > 0 else float("nan"),
                "vicon_csv"              : str(VICON_CSV_OUT_PATH),
                "frame_metrics_csv"      : str(RESULT_PATH / "frame_metrics.csv"),
                "frame_metrics_json"     : str(RESULT_PATH / "frame_metrics.json"),
                "T_T_C_results_json"     : str(T_T_C_RESULTS_PATH),
                "reprojection_dir"       : str(REPROJECTION_DIR),
                "topic_yamls_dir"        : str(TOPIC_YAMLS_DIR),
                "topic_yamls_manifest"   : str(topic_yaml_info["manifest_path"]),
                "trajectory_pack_dir"    : str(traj_dir),
                "trajectory_sanity_dir"  : str(sanity_check_dir),
                "trajectory_lifted_json" : str(lifted_path),
              }
    with (RESULT_PATH / "summary.json").open("w", encoding = "utf-8") as handle:
        json.dump(summary, handle, indent = 4)

    print(f"Results located at: {RESULT_PATH}")


if __name__ == "__main__":
    main()
