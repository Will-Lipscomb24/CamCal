# src/offset_utils/reprojection.py
""" A set of utilities for loading target keypoints, projecting them through a pose, and visualizing the results """ 

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from numpy.typing import NDArray

# required local import from sc-pose-utils repo
from sc_pose.sensors.camera_projections import PoseProjector, draw_uv_points_on_image

from src.offset_utils.camera_io import ensure_clean_dir, get_charuco_board_corners

def load_target_points(
                            kps_file     : Path,
                            with_origin  : bool = True,
                            units        : str = "mm", #options: "mm", "m", "meter", "meters", "millimeter", "millimeters"
                      ) -> NDArray[np.float64]:
    """ load target keypoints from json and convert the declared units to meters """
    with Path(kps_file).open("r", encoding = "utf-8") as handle:
        kps_xyz     = np.asarray(json.load(handle), dtype = np.float64)

    units_norm      = str(units).strip().lower()
    if units_norm in {"mm", "millimeter", "millimeters"}:
        points_m    = kps_xyz / 1e3
    elif units_norm in {"m", "meter", "meters"}:
        points_m    = kps_xyz
    else:
        raise ValueError(f"Unsupported keypoint units: {units}")

    # add origin if requested
    # add for reprojectin visualization
    if with_origin:
        points_m    = np.vstack((np.zeros((1, 3), dtype = np.float64), points_m))
    return points_m

# project target points onto an image using 4x4 homogeneous pose, camera intrinsics, and distortion coefficients
# return the projected uv coordinates
def project_points_T_T_C(
                            T_T_C               : NDArray[np.floating],
                            K                   : NDArray[np.floating],
                            dist_coeffs         : NDArray[np.floating],
                            points_xyz_TARGET   : NDArray[np.floating],
                        ) -> NDArray[np.float64]:
    """ project target-frame points through a homogeneous pose """
    dist_coeffs = np.asarray(dist_coeffs, dtype = np.float64).reshape(5,)
    return PoseProjector.classless_pinhole_project_T4x4_2_uv(
                                                                T_TARGET_CAM        = np.asarray(T_T_C, dtype = np.float64),
                                                                Kmat                = np.asarray(K, dtype = np.float64),
                                                                BC_dist_coeffs      = dist_coeffs,
                                                                points_xyz_TARGET   = np.asarray(points_xyz_TARGET, dtype = np.float64),
                                                             )


def draw_origin(
                    image,
                    uv_points        : NDArray[np.floating],
                    origin_color     : tuple[int, int, int] = (0, 255, 0),
                    origin_radius    : int = 20,
                    origin_thickness : int = 3,
               ):
    """ draw only the first projected point as the pose origin """
    uv_points   = np.asarray(uv_points, dtype = np.float64)
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
    """ draw a projected point cloud plus highlighted origin """
    # draw all points, then draw the origin on top, then optionally add a text label
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
    """ 
    draw OpenCV and truth reprojections onto the same image

    Colors: OpenCV reprojection in red, truth reprojection in blue, OpenCV origin in yellow, truth origin in green
    """
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
    """ 
    draw the target coordinate frame axes using the supplied T_T_C
    x axis in red, y axis in green, z axis in blue, origin in yellow
    """
    axis_points_T   = np.array(
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
    # use opencv to project the axis points, then draw lines from the origin to each axis endpoint
    axis_proj, _= cv2.projectPoints(
                                        axis_points_T,
                                        rvec,
                                        tvec,
                                        np.asarray(K, dtype = np.float64),
                                        np.asarray(dist_coeffs, dtype = np.float64).reshape(5,),
                                     )
    axis_proj   = np.asarray(axis_proj, dtype = np.float64).reshape(-1, 2)
    if not np.all(np.isfinite(axis_proj)):
        return image

    axis_origin     = tuple(np.round(axis_proj[0]).astype(int))
    cv2.line(image, axis_origin, tuple(np.round(axis_proj[1]).astype(int)), (0, 0, 255), 3) # x axis is red
    cv2.line(image, axis_origin, tuple(np.round(axis_proj[2]).astype(int)), (0, 255, 0), 3) # y axis is green
    cv2.line(image, axis_origin, tuple(np.round(axis_proj[3]).astype(int)), (255, 0, 0), 3) # z axis is blue
    return image


def _draw_error_line(
                        image,
                        pt_detect : tuple[float, float],
                        pt_proj   : tuple[float, float],
                    ) -> None:
    """ draw a line between the detected and projected points, colored by the reprojection error in pixels (green < 0.5px, yellow < 1.5px, red >= 1.5px) """
    error_px    = float(np.hypot(pt_detect[0] - pt_proj[0], pt_detect[1] - pt_proj[1]))
    if error_px < 0.5:
        color   = (0, 255, 0)
    elif error_px < 1.5:
        color   = (0, 255, 255)
    else:
        color   = (0, 0, 255)

    cv2.line(
                image,
                (int(round(pt_detect[0])), int(round(pt_detect[1]))),
                (int(round(pt_proj[0])), int(round(pt_proj[1]))),
                color,
                1,
            )


def draw_truth_charuco_overlay(
                                    image_path       : Path,
                                    detection        : dict[str, object],
                                    board,
                                    T_T_C_truth      : NDArray[np.floating],
                                    K                : NDArray[np.floating],
                                    dist_coeffs      : NDArray[np.floating],
                                    axis_length_m    : float,
                               ) -> NDArray[np.uint8]:
    """
    draw the detected ChArUco corners/markers together with the solved-truth reprojection.
    this is the overlay we use to inspect how the OpenCV solved truth pose sits on the calibration image
    """
    image   = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Failed to read image: {image_path}")
    vis     = image.copy()

    charuco_corners = np.asarray(detection["charuco_corners"], dtype = np.float64).reshape(-1, 2)
    charuco_ids     = np.asarray(detection["charuco_ids"], dtype = np.int32).reshape(-1)
    marker_corners  = [np.asarray(corner, dtype = np.float64).reshape(-1, 2) for corner in detection["marker_corners"]]
    marker_ids      = np.asarray(detection["marker_ids"], dtype = np.int32).reshape(-1)

    # draw the detected board evidence first so we can compare it against the truth reprojection
    for pts in marker_corners:
        cv2.polylines(
                        vis,
                        [np.round(pts).astype(np.int32).reshape(-1, 1, 2)],
                        True,
                        (0, 255, 0),
                        2,
                     )
    for detected_pt in charuco_corners:
        cv2.circle(vis, tuple(np.round(detected_pt).astype(int)), 5, (0, 180, 0), -1)

    board_corners           = np.asarray(get_charuco_board_corners(board), dtype = np.float64).reshape(-1, 3)
    marker_object_points    = [np.asarray(points, dtype = np.float64).reshape(-1, 3) for points in board.getObjPoints()]
    board_marker_ids        = np.asarray(board.getIds(), dtype = np.int32).reshape(-1)
    marker_obj_map          = {
                                int(marker_id): obj_points
                                for marker_id, obj_points in zip(board_marker_ids, marker_object_points)
                              }

    rvec_truth, _   = cv2.Rodrigues(np.asarray(T_T_C_truth, dtype = np.float64)[:3, :3])
    tvec_truth      = np.asarray(T_T_C_truth, dtype = np.float64)[:3, 3].reshape(3, 1)

    # project the detected markers through the solved truth pose so the board outline can be inspected directly
    for marker_id in marker_ids:
        obj_points  = marker_obj_map.get(int(marker_id))
        if obj_points is None:
            continue
        proj, _     = cv2.projectPoints(
                                        obj_points,
                                        rvec_truth,
                                        tvec_truth,
                                        np.asarray(K, dtype = np.float64),
                                        np.asarray(dist_coeffs, dtype = np.float64).reshape(5,),
                                    )  
        proj_pts    = np.asarray(proj, dtype = np.float64).reshape(-1, 2)
        cv2.polylines(
                        vis,
                        [np.round(proj_pts).astype(np.int32).reshape(-1, 1, 2)],
                        True,
                        (255, 255, 0),
                        2,
                     )

    if len(charuco_ids) > 0:
        obj_points = board_corners[charuco_ids]
        proj, _     = cv2.projectPoints(
                                            obj_points,
                                            rvec_truth,
                                            tvec_truth,
                                            np.asarray(K, dtype = np.float64),
                                            np.asarray(dist_coeffs, dtype = np.float64).reshape(5,),
                                        )
        proj_pts    = np.asarray(proj, dtype = np.float64).reshape(-1, 2)
        for proj_pt in proj_pts:
            cv2.circle(vis, tuple(np.round(proj_pt).astype(int)), 7, (0, 0, 255), 2)

    vis = draw_axes_on_image(vis, T_T_C_truth, K, dist_coeffs, axis_length_m)
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
    return vis


def write_sanity_overlays(
                            traj_dir                 : Path,
                            frame_records            : list[dict[str, object]],
                            K                        : NDArray[np.floating],
                            dist_coeffs              : NDArray[np.floating],
                            target_pts_with_origin   : NDArray[np.floating],
                            bboxes_xyxy              : list[list[int] | None] | None = None,
                         ) -> Path:
    """
    Write one projected-pose sanity overlay per trajectory-pack record.
    This is the quick visual check we use for truth packs and refined packs.
    """
    # create the sanity check directory, ensuring that any existing contents are cleared
    traj_dir    = Path(traj_dir)
    sanity_dir  = ensure_clean_dir(traj_dir / "sanity_check")
    # if bboxes are provided, check that the length matches the number of frame records, and if not provided, 
    # create a list of None values to use for consistent processing in the loop
    if bboxes_xyxy is None:
        bboxes_xyxy = [None] * len(frame_records)
    if len(bboxes_xyxy) != len(frame_records):
        raise ValueError(
                            f"bboxes_xyxy length must match frame_records ({len(frame_records)}), got {len(bboxes_xyxy)}."
                        )

    for idx, (record, bbox_xyxy) in enumerate(zip(frame_records, bboxes_xyxy)):
        # create overlays with following format
        token       = f"{idx:05d}"
        image_path  = traj_dir / f"image_{token}.png"
        T_T_C       = np.asarray(record["T_T_C"], dtype = np.float64)
        uv_truth    = project_points_T_T_C(T_T_C, K, dist_coeffs, target_pts_with_origin)
        overlay     = draw_pose_overlay(
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

        if bbox_xyxy is not None and len(bbox_xyxy) == 4:
            xmin, ymin, xmax, ymax  = (int(value) for value in bbox_xyxy)
            cv2.rectangle(overlay, (xmin, ymin), (xmax, ymax), (0, 255, 255), 2)

        cv2.imwrite(str(sanity_dir / f"sanity_{token}.png"), overlay)

    return sanity_dir


def write_charuco_reprojection_overlays(
                                            output_dir           : Path,
                                            reprojection_rows    : list[dict[str, object]],
                                            board,
                                            K                   : NDArray[np.floating],
                                            dist_coeffs         : NDArray[np.floating],
                                            axis_length_m       : float,
                                        ) -> list[dict[str, object]]:
    """ 
    write the solve-image ChArUco reprojection diagnostics 
    This function will 
    1) project the ChArUco board corners and detected marker corners through the estimated T_T_C for each image,
    2) draw the detected and projected corners on the original image
    3) compute reprojection error statistics for the ChArUco corners, and
    4) draw lines between the detected and projected ChArUco corners colored by reprojection error 
    the overlaid images will be saved to the output directory, and a list of reprojection error statistics for each image will be returned
    """
    output_dir.mkdir(parents = True, exist_ok = True)
    
    # precompute the board corner coordinates and a mapping from marker id to marker corner coordinates for efficient lookup during overlay generation
    board_corners           = np.asarray(get_charuco_board_corners(board), dtype = np.float64).reshape(-1, 3)
    marker_object_points    = [np.asarray(points, dtype = np.float64).reshape(-1, 3) for points in board.getObjPoints()]
    marker_ids              = np.asarray(board.getIds(), dtype = np.int32).reshape(-1)
    marker_obj_map          = {
                                int(marker_id): marker_points
                                for marker_id, marker_points in zip(marker_ids, marker_object_points)
                            }

    overlay_stats           = []

    # iterate through the reprojection rows, which contain the image path, estimated pose, and detected corners for each valid ChArUco image, and generate the overlay visualizations and reprojection error statistics
    for row in reprojection_rows:
        image_path          = Path(row["image_path"])
        image_number        = int(row["image_number"])
        rvec                = np.asarray(row["rvec"], dtype = np.float64).reshape(3, 1)
        tvec                = np.asarray(row["tvec"], dtype = np.float64).reshape(3, 1)
        detection           = row["detection"]
        charuco_corners     = np.asarray(detection["charuco_corners"], dtype = np.float64).reshape(-1, 2)
        charuco_ids         = np.asarray(detection["charuco_ids"], dtype = np.int32).reshape(-1)
        marker_corners      = detection["marker_corners"]
        marker_ids_det      = np.asarray(detection["marker_ids"], dtype = np.int32).reshape(-1)

        image               = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            continue
        vis                 = image.copy()

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
        # for each detected marker, look up the corresponding object points using the marker id, project them through the estimated pose, and draw the projected corners on the image in a different color than the detected corners
        for marker_id in marker_ids_det:
            obj_points  = marker_obj_map.get(int(marker_id))
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
        
        # compute reprojection error statistics for the ChArUco corners, and draw lines between the detected and projected ChArUco corners colored by reprojection error
        mean_charuco_error_px   = float("nan")
        max_charuco_error_px    = float("nan")
        if len(charuco_ids) > 0:
            obj_points  = board_corners[charuco_ids]
            proj, _     = cv2.projectPoints(
                                            obj_points,
                                            rvec,
                                            tvec,
                                            np.asarray(K, dtype = np.float64),
                                            np.asarray(dist_coeffs, dtype = np.float64).reshape(5,),
                                         )
            proj_pts    = np.asarray(proj, dtype = np.float64).reshape(-1, 2)
            err_px      = np.linalg.norm(charuco_corners - proj_pts, axis = 1)
            mean_charuco_error_px   = float(np.mean(err_px))
            max_charuco_error_px    = float(np.max(err_px))

            # draw lines between detected and projected ChArUco corners colored by reprojection error
            for detected_pt, proj_pt in zip(charuco_corners, proj_pts):
                cv2.circle(vis, (int(round(detected_pt[0])), int(round(detected_pt[1]))), 5, (0, 255, 0), -1)
                cv2.circle(vis, (int(round(proj_pt[0])), int(round(proj_pt[1]))), 8, (0, 0, 255), 2)
                _draw_error_line(
                                    vis,
                                    (float(detected_pt[0]), float(detected_pt[1])),
                                    (float(proj_pt[0]), float(proj_pt[1])),
                                )

        # draw the target coordinate frame axes using the estimated pose
        T_T_C               = np.eye(4, dtype = np.float64)
        T_T_C[:3, :3], _    = cv2.Rodrigues(rvec)
        T_T_C[:3, 3]        = tvec.reshape(3,)
        vis                 = draw_axes_on_image(vis, T_T_C, K, dist_coeffs, axis_length_m)
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
        # save the overlay image to the output directory with a filename that includes the original image name
        cv2.imwrite(str(output_dir / f"reproj_{image_path.name}"), vis)

        # store the reprojection error statistics for this image in the overlay_stats list, which will be returned at the end of the function and can be written to csv or json for further analysis
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
    """ write the overlay stats to both csv and json in the output directory, using the provided basename for the filename (without extension) """
    stats_df    = pd.DataFrame(overlay_stats)
    csv_path    = output_dir / f"{basename}.csv"
    json_path   = output_dir / f"{basename}.json"
    stats_df.to_csv(csv_path, index = False)
    with json_path.open("w", encoding = "utf-8") as handle:
        json.dump(overlay_stats, handle, indent = 4)
    return csv_path, json_path
