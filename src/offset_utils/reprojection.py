from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from numpy.typing import NDArray

from sc_pose.sensors.camera_projections import PoseProjector, draw_uv_points_on_image


def load_target_points(
                            kps_file     : Path,
                            with_origin  : bool = True,
                            units        : str = "mm",
                      ) -> NDArray[np.float64]:
    """ Load target keypoints from json and convert the declared units to meters """
    with Path(kps_file).open("r", encoding = "utf-8") as handle:
        kps_xyz = np.asarray(json.load(handle), dtype = np.float64)

    units_norm = str(units).strip().lower()
    if units_norm in {"mm", "millimeter", "millimeters"}:
        points_m = kps_xyz / 1e3
    elif units_norm in {"m", "meter", "meters"}:
        points_m = kps_xyz
    else:
        raise ValueError(f"Unsupported keypoint units: {units}")

    if with_origin:
        points_m = np.vstack((np.zeros((1, 3), dtype = np.float64), points_m))
    return points_m


def project_points_T_T_C(
                            T_T_C               : NDArray[np.floating],
                            K                   : NDArray[np.floating],
                            dist_coeffs         : NDArray[np.floating],
                            points_xyz_TARGET   : NDArray[np.floating],
                        ) -> NDArray[np.float64]:
    """ Project target-frame points through a homogeneous pose """
    dist_coeffs = np.asarray(dist_coeffs, dtype = np.float64).reshape(5,)
    return PoseProjector.classless_pinhole_project_T4x4_2_uv(
                                                                T_TARGET_CAM       = np.asarray(T_T_C, dtype = np.float64),
                                                                Kmat               = np.asarray(K, dtype = np.float64),
                                                                BC_dist_coeffs     = dist_coeffs,
                                                                points_xyz_TARGET  = np.asarray(points_xyz_TARGET, dtype = np.float64),
                                                             )


def draw_origin(
                    image,
                    uv_points        : NDArray[np.floating],
                    origin_color     : tuple[int, int, int] = (0, 255, 0),
                    origin_radius    : int = 20,
                    origin_thickness : int = 3,
               ):
    """ Draw only the first projected point as the pose origin """
    uv_points = np.asarray(uv_points, dtype = np.float64)
    if uv_points.ndim != 2 or uv_points.shape[0] == 0:
        return image
    if not np.isfinite(uv_points[0]).all():
        return image
    return draw_uv_points_on_image(
                                    img_or_path     = image,
                                    points_uv       = uv_points[:1],
                                    point_color     = origin_color,
                                    point_radius    = origin_radius,
                                    point_thickness = origin_thickness,
                                  )


def draw_pose_overlay(
                        image,
                        uv_points            : NDArray[np.floating],
                        point_color          : tuple[int, int, int],
                        point_radius         : int = 12,
                        point_thickness      : int = 2,
                        origin_color         : tuple[int, int, int] = (0, 255, 0),
                        origin_radius        : int = 20,
                        origin_thickness     : int = 3,
                        text_label           : str | None = None,
                        text_origin          : tuple[int, int] = (30, 50),
                     ):
    """ Draw a projected point cloud plus highlighted origin """
    overlay = draw_uv_points_on_image(
                                        img_or_path     = image,
                                        points_uv       = np.asarray(uv_points, dtype = np.float64),
                                        point_color     = point_color,
                                        point_radius    = point_radius,
                                        point_thickness = point_thickness,
                                     )
    overlay = draw_origin(
                            overlay,
                            uv_points,
                            origin_color = origin_color,
                            origin_radius = origin_radius,
                            origin_thickness = origin_thickness,
                         )
    if text_label:
        cv2.putText(
                        overlay,
                        text_label,
                        text_origin,
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )
    return overlay


def draw_combined_pose_overlay(
                                image_path       : Path | str,
                                uv_opencv        : NDArray[np.floating],
                                uv_truth         : NDArray[np.floating],
                                text_label       : str | None = None,
                             ):
    """ Draw OpenCV and truth reprojections onto the same image """
    overlay = draw_pose_overlay(
                                image = str(image_path),
                                uv_points = uv_opencv,
                                point_color = (0, 0, 255),
                                point_radius = 12,
                                point_thickness = 2,
                                origin_color = (0, 255, 255),
                                origin_radius = 20,
                                origin_thickness = 3,
                              )
    overlay = draw_pose_overlay(
                                image = overlay,
                                uv_points = uv_truth,
                                point_color = (255, 0, 0),
                                point_radius = 10,
                                point_thickness = 2,
                                origin_color = (0, 255, 0),
                                origin_radius = 18,
                                origin_thickness = 3,
                                text_label = text_label,
                              )
    return overlay


def draw_axes_on_image(
                            image,
                            T_T_C           : NDArray[np.floating],
                            K               : NDArray[np.floating],
                            dist_coeffs     : NDArray[np.floating],
                            axis_length_m   : float,
                       ):
    """ Draw the target coordinate frame axes using the supplied T_T_C """
    axis_points_T = np.array(
                            [
                                [0.0, 0.0, 0.0],
                                [axis_length_m, 0.0, 0.0],
                                [0.0, axis_length_m, 0.0],
                                [0.0, 0.0, axis_length_m],
                            ],
                            dtype = np.float64,
                           )

    rvec, _     = cv2.Rodrigues(np.asarray(T_T_C, dtype = np.float64)[:3, :3])
    tvec        = np.asarray(T_T_C, dtype = np.float64)[:3, 3].reshape(3, 1)
    axis_proj, _ = cv2.projectPoints(
                                        axis_points_T,
                                        rvec,
                                        tvec,
                                        np.asarray(K, dtype = np.float64),
                                        np.asarray(dist_coeffs, dtype = np.float64).reshape(5,),
                                     )
    axis_proj       = np.asarray(axis_proj, dtype = np.float64).reshape(-1, 2)
    if not np.all(np.isfinite(axis_proj)):
        return image

    axis_origin     = tuple(np.round(axis_proj[0]).astype(int))
    cv2.line(image, axis_origin, tuple(np.round(axis_proj[1]).astype(int)), (0, 0, 255), 3)
    cv2.line(image, axis_origin, tuple(np.round(axis_proj[2]).astype(int)), (0, 255, 0), 3)
    cv2.line(image, axis_origin, tuple(np.round(axis_proj[3]).astype(int)), (255, 0, 0), 3)
    return image


def _draw_error_line(
                        image,
                        pt_detect : tuple[float, float],
                        pt_proj   : tuple[float, float],
                    ) -> None:
    error_px = float(np.hypot(pt_detect[0] - pt_proj[0], pt_detect[1] - pt_proj[1]))
    if error_px < 0.5:
        color = (0, 255, 0)
    elif error_px < 1.5:
        color = (0, 255, 255)
    else:
        color = (0, 0, 255)

    cv2.line(
                image,
                (int(round(pt_detect[0])), int(round(pt_detect[1]))),
                (int(round(pt_proj[0])), int(round(pt_proj[1]))),
                color,
                1,
            )


def _get_charuco_board_corners(board) -> NDArray[np.float64]:
    if hasattr(board, "chessboardCorners"):
        return np.asarray(board.chessboardCorners, dtype = np.float64)
    if hasattr(board, "getChessboardCorners"):
        return np.asarray(board.getChessboardCorners(), dtype = np.float64)
    raise RuntimeError("CharucoBoard has neither chessboardCorners nor getChessboardCorners")


def write_charuco_reprojection_overlays(
                                            output_dir           : Path,
                                            reprojection_rows    : list[dict[str, object]],
                                            board,
                                            K                   : NDArray[np.floating],
                                            dist_coeffs         : NDArray[np.floating],
                                            axis_length_m       : float,
                                        ) -> list[dict[str, object]]:
    """ Write the solve-image ChArUco reprojection diagnostics """
    output_dir.mkdir(parents = True, exist_ok = True)

    board_corners        = np.asarray(_get_charuco_board_corners(board), dtype = np.float64).reshape(-1, 3)
    marker_object_points = [np.asarray(points, dtype = np.float64).reshape(-1, 3) for points in board.getObjPoints()]
    marker_ids           = np.asarray(board.getIds(), dtype = np.int32).reshape(-1)
    marker_obj_map       = {
                                int(marker_id): marker_points
                                for marker_id, marker_points in zip(marker_ids, marker_object_points)
                            }

    overlay_stats = []

    for row in reprojection_rows:
        image_path       = Path(row["image_path"])
        image_number     = int(row["image_number"])
        rvec             = np.asarray(row["rvec"], dtype = np.float64).reshape(3, 1)
        tvec             = np.asarray(row["tvec"], dtype = np.float64).reshape(3, 1)
        detection        = row["detection"]
        charuco_corners  = np.asarray(detection["charuco_corners"], dtype = np.float64).reshape(-1, 2)
        charuco_ids      = np.asarray(detection["charuco_ids"], dtype = np.int32).reshape(-1)
        marker_corners   = detection["marker_corners"]
        marker_ids_det   = np.asarray(detection["marker_ids"], dtype = np.int32).reshape(-1)

        image            = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            continue
        vis              = image.copy()

        for detected_marker in marker_corners:
            pts = np.asarray(detected_marker, dtype = np.float64).reshape(-1, 2)
            cv2.polylines(
                            vis,
                            [np.round(pts).astype(np.int32).reshape(-1, 1, 2)],
                            True,
                            (0, 255, 0),
                            2,
                         )
            for u, v in pts:
                cv2.circle(vis, (int(round(u)), int(round(v))), 4, (0, 180, 0), -1)

        for marker_id in marker_ids_det:
            obj_points = marker_obj_map.get(int(marker_id))
            if obj_points is None:
                continue
            proj, _ = cv2.projectPoints(
                                        obj_points,
                                        rvec,
                                        tvec,
                                        np.asarray(K, dtype = np.float64),
                                        np.asarray(dist_coeffs, dtype = np.float64).reshape(5,),
                                     )
            proj_pts = np.asarray(proj, dtype = np.float64).reshape(-1, 2)
            cv2.polylines(
                            vis,
                            [np.round(proj_pts).astype(np.int32).reshape(-1, 1, 2)],
                            True,
                            (255, 255, 0),
                            2,
                         )
            for u, v in proj_pts:
                cv2.circle(vis, (int(round(u)), int(round(v))), 4, (255, 255, 0), -1)

        mean_charuco_error_px = float("nan")
        max_charuco_error_px  = float("nan")
        if len(charuco_ids) > 0:
            obj_points   = board_corners[charuco_ids]
            proj, _      = cv2.projectPoints(
                                            obj_points,
                                            rvec,
                                            tvec,
                                            np.asarray(K, dtype = np.float64),
                                            np.asarray(dist_coeffs, dtype = np.float64).reshape(5,),
                                         )
            proj_pts     = np.asarray(proj, dtype = np.float64).reshape(-1, 2)
            err_px       = np.linalg.norm(charuco_corners - proj_pts, axis = 1)
            mean_charuco_error_px = float(np.mean(err_px))
            max_charuco_error_px  = float(np.max(err_px))

            for detected_pt, proj_pt in zip(charuco_corners, proj_pts):
                cv2.circle(vis, (int(round(detected_pt[0])), int(round(detected_pt[1]))), 5, (0, 255, 0), -1)
                cv2.circle(vis, (int(round(proj_pt[0])), int(round(proj_pt[1]))), 8, (0, 0, 255), 2)
                _draw_error_line(
                                    vis,
                                    (float(detected_pt[0]), float(detected_pt[1])),
                                    (float(proj_pt[0]), float(proj_pt[1])),
                                )

        T_T_C          = np.eye(4, dtype = np.float64)
        T_T_C[:3, :3], _ = cv2.Rodrigues(rvec)
        T_T_C[:3, 3]   = tvec.reshape(3,)
        vis            = draw_axes_on_image(vis, T_T_C, K, dist_coeffs, axis_length_m)

        cv2.putText(
                        vis,
                        image_path.name,
                        (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )
        cv2.imwrite(str(output_dir / f"reproj_{image_path.name}"), vis)

        overlay_stats.append(
                            {
                                "image_number"            : image_number,
                                "frame"                   : image_path.name,
                                "num_markers"             : int(len(marker_ids_det)),
                                "num_charuco_corners"     : int(len(charuco_ids)),
                                "mean_charuco_error_px"   : mean_charuco_error_px,
                                "max_charuco_error_px"    : max_charuco_error_px,
                            }
                          )

    return overlay_stats


def write_overlay_stats(
                            output_dir      : Path,
                            overlay_stats   : list[dict[str, object]],
                            basename        : str = "reprojection_metrics",
                        ) -> tuple[Path, Path]:
    stats_df     = pd.DataFrame(overlay_stats)
    csv_path     = output_dir / f"{basename}.csv"
    json_path    = output_dir / f"{basename}.json"
    stats_df.to_csv(csv_path, index = False)
    with json_path.open("w", encoding = "utf-8") as handle:
        json.dump(overlay_stats, handle, indent = 4)
    return csv_path, json_path
