from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.optimize import least_squares

from sc_pose.mathtils.quaternion import q2rotm, rotm2q


DEFAULT_VICON_KEYS = {
                        "frame"         : "image_number",
                        "x_target"      : "soho_x",
                        "y_target"      : "soho_y",
                        "z_target"      : "soho_z",
                        "qw_target"     : "soho_qw",
                        "qx_target"     : "soho_qx",
                        "qy_target"     : "soho_qy",
                        "qz_target"     : "soho_qz",
                        "x_cam"         : "cam_x",
                        "y_cam"         : "cam_y",
                        "z_cam"         : "cam_z",
                        "qw_cam"        : "cam_qw",
                        "qx_cam"        : "cam_qx",
                        "qy_cam"        : "cam_qy",
                        "qz_cam"        : "cam_qz",
                    }

DEFAULT_OFFSET_KEYS = {
                        "Trf_4x4_CamViconDef_Cam"      : "T_CvC",
                        "Trf_4x4_TargetViconDef_Target": "T_TvT",
                     }


def build_transform(rotation_block: NDArray[np.floating], translation: NDArray[np.floating]) -> NDArray[np.float64]:
    transform            = np.eye(4, dtype = np.float64)
    transform[:3, :3]    = np.asarray(rotation_block, dtype = np.float64)
    transform[:3, 3]     = np.asarray(translation, dtype = np.float64).reshape(3,)
    return transform


def inv_T(transform: NDArray[np.floating]) -> NDArray[np.float64]:
    transform            = np.asarray(transform, dtype = np.float64)
    rotation_block       = transform[:3, :3]
    translation          = transform[:3, 3]
    rotation_inv         = rotation_block.T
    translation_inv      = -rotation_inv @ translation
    return build_transform(rotation_inv, translation_inv)


def params_to_T(params: NDArray[np.floating]) -> NDArray[np.float64]:
    params              = np.asarray(params, dtype = np.float64).reshape(6,)
    rotation_block, _   = cv2.Rodrigues(params[:3])
    return build_transform(rotation_block, params[3:])


def T_to_params(transform: NDArray[np.floating]) -> NDArray[np.float64]:
    transform           = np.asarray(transform, dtype = np.float64)
    rvec, _             = cv2.Rodrigues(transform[:3, :3])
    return np.hstack([rvec.reshape(3,), transform[:3, 3]])


def T_T_C_to_pose(T_T_C: NDArray[np.floating]) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """ Convert ^C T_T into canonical translation and both quaternion directions """
    T_T_C               = np.asarray(T_T_C, dtype = np.float64)
    Rotm_T_2_C          = T_T_C[:3, :3]
    Rotm_C_2_T          = Rotm_T_2_C.T
    r_Co2To_C           = T_T_C[:3, 3].reshape(3,)
    q_CAM_2_TARGET      = rotm2q(Rotm_C_2_T)
    q_TARGET_2_CAM      = rotm2q(Rotm_T_2_C)
    return q_CAM_2_TARGET, r_Co2To_C, q_TARGET_2_CAM


def opencv_rvec_tvec_to_T_T_C(
                                rvec: NDArray[np.floating],
                                tvec: NDArray[np.floating],
                             ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """ Convert OpenCV board pose outputs into canonical T_T_C and quaternion outputs """
    rvec                = np.asarray(rvec, dtype = np.float64).reshape(3,)
    tvec                = np.asarray(tvec, dtype = np.float64).reshape(3,)
    Rotm_T_2_C, _       = cv2.Rodrigues(rvec)
    T_T_C               = build_transform(Rotm_T_2_C, tvec)
    q_CAM_2_TARGET, r_Co2To_C, q_TARGET_2_CAM = T_T_C_to_pose(T_T_C)
    return q_CAM_2_TARGET, r_Co2To_C, q_TARGET_2_CAM, T_T_C


def load_offset_estimates(
                            offset_json_path : Path,
                            offset_keys      : dict[str, str] | None = None,
                          ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    offset_json_path = Path(offset_json_path)
    if not offset_json_path.exists():
        raise FileNotFoundError(f"Offset JSON not found: {offset_json_path}")

    keys = DEFAULT_OFFSET_KEYS if offset_keys is None else offset_keys
    with offset_json_path.open("r", encoding = "utf-8") as handle:
        offset_json = json.load(handle)

    T_CvC = np.asarray(offset_json[keys["Trf_4x4_CamViconDef_Cam"]], dtype = np.float64)
    T_TvT = np.asarray(offset_json[keys["Trf_4x4_TargetViconDef_Target"]], dtype = np.float64)
    return T_CvC, T_TvT


def load_vicon_dataframe(
                            vicon_csv_path   : Path,
                            vicon_keys       : dict[str, str] | None = None,
                        ) -> pd.DataFrame:
    vicon_csv_path = Path(vicon_csv_path)
    if not vicon_csv_path.exists():
        raise FileNotFoundError(f"Vicon CSV not found: {vicon_csv_path}")

    keys            = DEFAULT_VICON_KEYS if vicon_keys is None else vicon_keys
    required_cols   = [
                        keys["frame"],
                        keys["x_target"],
                        keys["y_target"],
                        keys["z_target"],
                        keys["qw_target"],
                        keys["qx_target"],
                        keys["qy_target"],
                        keys["qz_target"],
                        keys["x_cam"],
                        keys["y_cam"],
                        keys["z_cam"],
                        keys["qw_cam"],
                        keys["qx_cam"],
                        keys["qy_cam"],
                        keys["qz_cam"],
                      ]

    vicon_df        = pd.read_csv(vicon_csv_path)
    missing_cols    = [column for column in required_cols if column not in vicon_df.columns]
    if len(missing_cols) > 0:
        raise KeyError(f"Vicon CSV missing required columns: {missing_cols}")

    for column in required_cols:
        vicon_df[column] = pd.to_numeric(vicon_df[column], errors = "raise")

    vicon_df[keys["frame"]] = vicon_df[keys["frame"]].astype(int)
    return vicon_df.sort_values(keys["frame"]).reset_index(drop = True)


def build_vicon_transform_series(
                                    vicon_df     : pd.DataFrame,
                                    vicon_keys   : dict[str, str] | None = None,
                                ) -> tuple[dict[int, NDArray[np.float64]], dict[int, NDArray[np.float64]], list[int]]:
    """ Build the same raw Vicon transforms used by the current working cam_offset.py """
    keys            = DEFAULT_VICON_KEYS if vicon_keys is None else vicon_keys
    T_CvV_by_image  = {}
    T_TvV_by_image  = {}
    image_numbers   = []

    for row in vicon_df.itertuples(index = False):
        image_number    = int(getattr(row, keys["frame"]))
        image_numbers.append(image_number)

        soho_VTv        = np.array(
                                    [
                                        float(getattr(row, keys["x_target"])),
                                        float(getattr(row, keys["y_target"])),
                                        float(getattr(row, keys["z_target"])),
                                    ],
                                    dtype = np.float64,
                                 ) / 1000.0
        soho_quatVTv    = np.array(
                                    [
                                        float(getattr(row, keys["qw_target"])),
                                        float(getattr(row, keys["qx_target"])),
                                        float(getattr(row, keys["qy_target"])),
                                        float(getattr(row, keys["qz_target"])),
                                    ],
                                    dtype = np.float64,
                                 )
        Trfm_TvV        = q2rotm(soho_quatVTv)

        cam_VCv         = np.array(
                                    [
                                        float(getattr(row, keys["x_cam"])),
                                        float(getattr(row, keys["y_cam"])),
                                        float(getattr(row, keys["z_cam"])),
                                    ],
                                    dtype = np.float64,
                                 ) / 1000.0
        cam_quatVCv     = np.array(
                                    [
                                        float(getattr(row, keys["qw_cam"])),
                                        float(getattr(row, keys["qx_cam"])),
                                        float(getattr(row, keys["qy_cam"])),
                                        float(getattr(row, keys["qz_cam"])),
                                    ],
                                    dtype = np.float64,
                                 )
        Trfm_CvV        = q2rotm(cam_quatVCv)

        T_TvV_by_image[image_number] = build_transform(Trfm_TvV, soho_VTv)
        T_CvV_by_image[image_number] = build_transform(Trfm_CvV, cam_VCv)

    return T_CvV_by_image, T_TvV_by_image, image_numbers


def vicon_row_to_T_T_C_v01(
                                row         : pd.Series,
                                T_CvC       : NDArray[np.floating],
                                T_TvT       : NDArray[np.floating],
                                vicon_keys  : dict[str, str] | None = None,
                           ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """ Apply the current v01 offset-processing chain from check_offset.py """
    keys            = DEFAULT_VICON_KEYS if vicon_keys is None else vicon_keys

    soho_VTv        = np.array(
                                [
                                    float(row[keys["x_target"]]),
                                    float(row[keys["y_target"]]),
                                    float(row[keys["z_target"]]),
                                ],
                                dtype = np.float64,
                             ) / 1000.0
    soho_quatVTv    = np.array(
                                [
                                    float(row[keys["qw_target"]]),
                                    float(row[keys["qx_target"]]),
                                    float(row[keys["qy_target"]]),
                                    float(row[keys["qz_target"]]),
                                ],
                                dtype = np.float64,
                             )
    Trfm_TvV        = q2rotm(soho_quatVTv)

    cam_VCv         = np.array(
                                [
                                    float(row[keys["x_cam"]]),
                                    float(row[keys["y_cam"]]),
                                    float(row[keys["z_cam"]]),
                                ],
                                dtype = np.float64,
                             ) / 1000.0
    cam_quatVCv     = np.array(
                                [
                                    float(row[keys["qw_cam"]]),
                                    float(row[keys["qx_cam"]]),
                                    float(row[keys["qy_cam"]]),
                                    float(row[keys["qz_cam"]]),
                                ],
                                dtype = np.float64,
                             )
    Trfm_CvV        = q2rotm(cam_quatVCv)

    T_TvV           = build_transform(Trfm_TvV, soho_VTv)
    T_CvV           = build_transform(Trfm_CvV, cam_VCv)
    T_T_C           = np.asarray(T_CvC, dtype = np.float64) @ (inv_T(T_CvV) @ T_TvV) @ inv_T(np.asarray(T_TvT, dtype = np.float64))

    q_CAM_2_TARGET, r_Co2To_C, q_TARGET_2_CAM = T_T_C_to_pose(T_T_C)
    return q_CAM_2_TARGET, r_Co2To_C, q_TARGET_2_CAM, T_T_C


def sync_measurements(
                        T_T_C_array              : NDArray[np.float64],
                        valid_image_numbers      : list[int],
                        T_CvV_by_image           : dict[int, NDArray[np.float64]],
                        T_TvV_by_image           : dict[int, NDArray[np.float64]],
                        excluded_image_numbers   : list[int] | None = None,
                     ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], list[int], list[int]]:
    image_to_index  = {image_number: idx for idx, image_number in enumerate(valid_image_numbers)}
    common_numbers  = [
                        image_number
                        for image_number in valid_image_numbers
                        if image_number in T_CvV_by_image and image_number in T_TvV_by_image
                      ]
    if len(common_numbers) == 0:
        raise RuntimeError("No overlapping image numbers between ChArUco detections and Vicon data")

    excluded_set    = set([] if excluded_image_numbers is None else excluded_image_numbers)
    kept_numbers    = [image_number for image_number in common_numbers if image_number not in excluded_set]
    dropped_numbers = [image_number for image_number in common_numbers if image_number in excluded_set]
    if len(kept_numbers) == 0:
        raise RuntimeError("All matched image numbers were excluded; nothing remains to solve")

    T_T_C_sync      = np.stack([T_T_C_array[image_to_index[image_number]] for image_number in kept_numbers])
    T_CvV_sync      = np.stack([T_CvV_by_image[image_number] for image_number in kept_numbers])
    T_TvV_sync      = np.stack([T_TvV_by_image[image_number] for image_number in kept_numbers])
    return T_T_C_sync, T_CvV_sync, T_TvV_sync, kept_numbers, dropped_numbers


def rwhe_residuals(
                    params          : NDArray[np.floating],
                    T_T_C_array     : NDArray[np.float64],
                    T_CvV_array     : NDArray[np.float64],
                    T_TvV_array     : NDArray[np.float64],
                  ) -> NDArray[np.float64]:
    T_CvC           = params_to_T(params[:6])
    T_TvT           = params_to_T(params[6:])
    residual_blocks = []

    for T_T_C_obs, T_CvV, T_TvV in zip(T_T_C_array, T_CvV_array, T_TvV_array):
        T_T_C_pred   = T_CvC @ (inv_T(T_CvV) @ T_TvV) @ inv_T(T_TvT)
        diff         = T_T_C_obs[:3, :] - T_T_C_pred[:3, :]
        residual_blocks.append(diff.reshape(-1))

    return np.concatenate(residual_blocks)


def solve_rwhe(
                    T0_CvC         : NDArray[np.float64],
                    T0_TvT         : NDArray[np.float64],
                    T_T_C_array    : NDArray[np.float64],
                    T_CvV_array    : NDArray[np.float64],
                    T_TvV_array    : NDArray[np.float64],
               ) -> tuple[NDArray[np.float64], NDArray[np.float64], object, float]:
    T0_flat         = np.concatenate([T_to_params(T0_CvC), T_to_params(T0_TvT)])
    initial_res     = rwhe_residuals(T0_flat, T_T_C_array, T_CvV_array, T_TvV_array)
    initial_cost    = 0.5 * float(np.sum(initial_res ** 2))

    print(f"Solving RWHE with {len(T_T_C_array)} measurements...")
    print(f"Initial cost: {initial_cost:.6f}")

    result = least_squares(
                            rwhe_residuals,
                            T0_flat,
                            args = (T_T_C_array, T_CvV_array, T_TvV_array),
                            method = "lm",
                            verbose = 1,
                          )

    T_CvC           = params_to_T(result.x[:6])
    T_TvT           = params_to_T(result.x[6:])

    print(f"Final cost:  {result.cost:.6f}")
    print(f"Termination: {result.message}")
    return T_CvC, T_TvT, result, initial_cost


def per_observation_costs(result: object) -> NDArray[np.float64]:
    residual_matrix = np.asarray(result.fun, dtype = np.float64).reshape(-1, 12)
    return np.sum(residual_matrix ** 2, axis = 1)

