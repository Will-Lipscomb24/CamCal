from __future__ import annotations

import re
import shutil
import warnings
from pathlib import Path

import cv2
import cv2.aruco as aruco
import numpy as np
from numpy.typing import NDArray


def ensure_clean_dir(path: Path) -> Path:
    """ delete and recreate a directory """
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents = True, exist_ok = True)
    return path


def parse_image_number(path_or_name: str | Path) -> int:
    """ parse the integer token from a calibration image name like cal_image_42.png """
    # (\d+) means "match one or more digits and capture them as a group", we return first match only 
    
    match       = re.search(r"(\d+)", Path(path_or_name).stem)
    if match is None:
        raise ValueError(f"Could not parse image number from '{path_or_name}'")
    first_match = int( match.group(1) )
    return first_match


def parse_img_saver_ros_timestamp_v01(path_or_name: str | Path) -> tuple[int, int, float]:
    """ frame_<idx>_<sec>_<nanosec>.png into frame index, stamp_ns, stamp_s """
    match   = re.fullmatch(r"frame_(\d+)_(\d+)_(\d+)", Path(path_or_name).stem)
    if match is None:
        raise ValueError(f"Unsupported rosbag image name format: {Path(path_or_name).name}")

    frame_idx       = int(match.group(1))
    stamp_sec       = int(match.group(2))
    stamp_nanosec   = int(match.group(3))
    stamp_ns        = stamp_sec * 1_000_000_000 + stamp_nanosec
    stamp_s         = stamp_sec + stamp_nanosec / 1_000_000_000
    return frame_idx, stamp_ns, stamp_s


def parse_img_saver_ros_timestamp_v02(path_or_name: str | Path) -> tuple[int, int, float]:
    """ frame_<idx>_<total_nanosec>.png into frame index, stamp_ns, stamp_s """
    match   = re.fullmatch(r"frame_(\d+)_(\d+)", Path(path_or_name).stem)
    if match is None:
        raise ValueError(f"Unsupported rosbag image name format: {Path(path_or_name).name}")

    frame_idx       = int(match.group(1))
    stamp_nanosec   = int(match.group(2))
    stamp_ns        = stamp_nanosec
    stamp_s         = stamp_nanosec / 1_000_000_000
    return frame_idx, stamp_ns, stamp_s


def collect_indxed_image_paths(
                            image_dir    : Path,
                            img_suffix   : str = ".png",
                            rosbag_style : bool = False,
                            img_name_parser : callable[[str | Path], tuple[int, int, float]] = parse_img_saver_ros_timestamp_v01,
                       ) -> list[Path]:
    """ collect and deterministically sort image paths with a parseable index from the filename """
    # grab all image paths with suffix 
    image_paths     = list(image_dir.glob(f"*{img_suffix}"))
    # sort by image number or rosbag timestamp
    if rosbag_style:
        # if rosbag style, use the parsable index from the img filename, which is assumed to contain
        # frame_<idx>_<sec>_<nanosec>.png, and sort by the timestamp (sec + nanosec) portion of the filename
        image_paths = sorted(image_paths, key = lambda path: img_name_parser(path)[0])
    else:
        image_paths = sorted(image_paths, key = parse_image_number)
    if len(image_paths) == 0:
        raise FileNotFoundError(f"No '{img_suffix}' images found under {image_dir}")
    return image_paths


def scale_K_to_image_size(
                            K                       : NDArray[np.floating],
                            calibration_width_px    : int,
                            calibration_height_px   : int,
                            image_width_px          : int,
                            image_height_px         : int,
                         ) -> NDArray[np.float64]:
    """ Scale a loaded camera intrinsic matrix to a new image size """
    K_scaled        = np.asarray(K, dtype = np.float64).copy()
    sx              = float(image_width_px) / float(calibration_width_px)
    sy              = float(image_height_px) / float(calibration_height_px)
    K_scaled[0, 0] *= sx
    K_scaled[0, 2] *= sx
    K_scaled[1, 1] *= sy
    K_scaled[1, 2] *= sy
    return K_scaled


def load_camera_calibration(
                                calibration_yaml_path      : Path,
                                sensor_width_mm            : float,
                                sensor_height_mm           : float,
                                image_width_px             : int,
                                image_height_px            : int,
                                focal_length_mm            : float,
                                square_pixels              : bool = False,
                                scale_loaded_K             : bool = False,
                                calibration_width_px       : int | None = None,
                                calibration_height_px      : int | None = None,
                            ) -> tuple[NDArray[np.float64], NDArray[np.float64], dict[str, float]]:
    """ Load camera calibration through sc_pose and optionally scale K to the requested image size """
    from sc_pose.sensors.camera import PinholeCamera

    calibration_yaml_path   = Path(calibration_yaml_path)
    if not calibration_yaml_path.exists():
        raise FileNotFoundError(f"calibration YAML not found: {calibration_yaml_path}")

    calib_width_px      = int(calibration_width_px) if calibration_width_px is not None else int(image_width_px)
    calib_height_px     = int(calibration_height_px) if calibration_height_px is not None else int(image_height_px)
    # square pixels is needed to avoid warnings when fx != fy, but if the user explicitly sets square_pixels = False
    # we allow non-square pixels and ignore the warning about it
    calib_camera        = PinholeCamera(
                                            sensor_width_mm   = sensor_width_mm,
                                            sensor_height_mm  = sensor_height_mm,
                                            image_width_px    = calib_width_px,
                                            image_height_px   = calib_height_px,
                                            focal_length_mm   = focal_length_mm,
                                            square_pixels     = square_pixels,
                                            dtype             = np.float64,
                                        )

    with warnings.catch_warnings():
        warnings.filterwarnings(
                                    "ignore",
                                    message = "PinholeCamera.set_calibration: setting fx != fy on a non-square pixel camera",
                                    category = RuntimeWarning,
                                )
        calib_camera.set_calibration_yaml(calibration_yaml_path)

    # will return camera instrinics that are loaded b/c fx, fy, cx, cy are all derived from the loaded calibration
    K_loaded            = np.asarray(calib_camera.calc_Kmat(), dtype = np.float64)
    dist_coeffs         = np.asarray(calib_camera._dist_coeffs_as_array(), dtype = np.float64).reshape(5,)

    if scale_loaded_K:
        K_use   = scale_K_to_image_size(
                                            K = K_loaded,
                                            calibration_width_px = calib_width_px,
                                            calibration_height_px = calib_height_px,
                                            image_width_px = image_width_px,
                                        image_height_px = image_height_px,
                                    )
    else:
        K_use = K_loaded

    camera_settings = {
                        "lens"          : float(focal_length_mm),
                        "sensor_width"  : float(sensor_width_mm),
                        "sensor_height" : float(sensor_height_mm),
                    }
    return K_use, dist_coeffs, camera_settings


def build_charuco_board(
                            squares_x       : int,
                            squares_y       : int,
                            square_len_m    : float,
                            marker_len_m    : float,
                            aruco_dict_id   : int,
                        ) -> tuple[aruco.CharucoBoard, aruco.Dictionary]:
    aruco_dict  = cv2.aruco.getPredefinedDictionary(aruco_dict_id)
    board       = cv2.aruco.CharucoBoard(
                                            (int(squares_x), int(squares_y)),
                                            float(square_len_m),
                                            float(marker_len_m),
                                            aruco_dict,
                                        )
    return board, aruco_dict


def _build_transform(passive_rotation_block: NDArray[np.floating], translation: NDArray[np.floating]) -> NDArray[np.float64]:
    # build a 4x4 homogeneous transform from a passive rotation block and translation vector
    transform           = np.eye(4, dtype = np.float64)
    transform[:3, :3]   = np.asarray(passive_rotation_block, dtype = np.float64)
    transform[:3, 3]    = np.asarray(translation, dtype = np.float64).reshape(3,)
    return transform

############################# ChArUco detection and pose estimation functions #############################
#### these  functions handle differences in OpenCV versions for ArUco detection and pose estimation, 
# and are used by the main ChArUco pose estimation function below. We want to support a wide range of OpenCV versions 
# since users may have different versions installed, and the APIs for ArUco detection and pose estimation have changed across versions
def _build_aruco_params():
    # build aruco detection parameters, handling differences between OpenCV versions
    if hasattr(cv2.aruco, "DetectorParameters"):
        return cv2.aruco.DetectorParameters()
    if hasattr(cv2.aruco, "DetectorParameters_create"):
        return cv2.aruco.DetectorParameters_create()
    # TODO: should we raise? 
    return None


def _detect_aruco_markers(gray: NDArray[np.uint8], aruco_dict: aruco.Dictionary):
    # detect aruco markers, handling differences between OpenCV versions. Returns corners, ids, and rejected candidates (if supported)
    aruco_params    = _build_aruco_params()
    # first try to use the new ArucoDetector API if available, which returns corners, ids, and rejected candidates in one call
    if hasattr(cv2.aruco, "ArucoDetector"):
        detector    = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
        return detector.detectMarkers(gray)
    # if not available, fall back to the older detectMarkers API, which only returns corners and ids 
    if hasattr(cv2.aruco, "detectMarkers"):
        if aruco_params is not None:
            return cv2.aruco.detectMarkers(gray, aruco_dict, parameters = aruco_params)
        return cv2.aruco.detectMarkers(gray, aruco_dict)
    # if neither API is available, raise an error
    raise RuntimeError("cv2.aruco has neither ArucoDetector nor detectMarkers")

# difference between aruco and charuco detection is that charuco detection 
# requires an additional interpolation step after aruco marker detection, and the pose estimation function is different
def _detect_charuco_corners(
                                gray        : NDArray[np.uint8],
                                board       : aruco.CharucoBoard,
                                aruco_dict  : aruco.Dictionary,
                           ):
    """
    This function handles the interpolation step and returns the charuco corners and ids if successful, 
    or None if interpolation failed (e.g. not enough markers detected), along with the original marker corners and ids for debugging
    """
    corners, ids, _ = _detect_aruco_markers(gray, aruco_dict)
    if ids is None or len(corners) == 0:
        return None, None, None, None, 0

    if hasattr(cv2.aruco, "interpolateCornersCharuco"):
        ok, charuco_corners, charuco_ids    = cv2.aruco.interpolateCornersCharuco(
                                                                                    corners,
                                                                                    ids,
                                                                                    gray,
                                                                                    board,
                                                                                )
        if not ok or charuco_corners is None or charuco_ids is None:
            return None, None, corners, ids, len(ids)
        return charuco_corners, charuco_ids, corners, ids, len(ids)

    if hasattr(cv2.aruco, "CharucoDetector"):
        detector    = cv2.aruco.CharucoDetector(board)
        charuco_corners, charuco_ids, _, _  = detector.detectBoard(gray)
        if charuco_corners is None or charuco_ids is None:
            return None, None, corners, ids, len(ids)
        return charuco_corners, charuco_ids, corners, ids, len(ids)

    raise RuntimeError("cv2.aruco has neither interpolateCornersCharuco nor CharucoDetector")


def _get_charuco_board_corners(board: aruco.CharucoBoard) -> NDArray[np.float64]:
    if hasattr(board, "chessboardCorners"):
        return np.asarray(board.chessboardCorners, dtype = np.float64)
    if hasattr(board, "getChessboardCorners"):
        return np.asarray(board.getChessboardCorners(), dtype = np.float64)
    raise RuntimeError("CharucoBoard has neither chessboardCorners nor getChessboardCorners")


def _estimate_charuco_pose(
                                charuco_corners : NDArray[np.float32],
                                charuco_ids     : NDArray[np.int32],
                                board           : aruco.CharucoBoard,
                                K               : NDArray[np.float64],
                                dist            : NDArray[np.float64],
                           ) -> tuple[bool, NDArray[np.float64], NDArray[np.float64]]:
    rvec    = np.zeros((3, 1), dtype = np.float64)
    tvec    = np.zeros((3, 1), dtype = np.float64)

    if hasattr(cv2.aruco, "estimatePoseCharucoBoard"):
        success, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(
                                                                    charuco_corners,
                                                                    charuco_ids,
                                                                    board,
                                                                    K,
                                                                    dist,
                                                                    rvec,
                                                                    tvec,
                                                                )
        return bool(success), rvec, tvec

    board_corners   = _get_charuco_board_corners(board)
    ids_flat        = np.asarray(charuco_ids, dtype = np.int32).reshape(-1)
    obj_points      = np.asarray(board_corners[ids_flat], dtype = np.float64).reshape(-1, 3)
    img_points      = np.asarray(charuco_corners, dtype = np.float64).reshape(-1, 2)
    if len(obj_points) < 4:
        return False, rvec, tvec

    success, rvec, tvec = cv2.solvePnP(
                                            obj_points,
                                            img_points,
                                            K,
                                            np.asarray(dist, dtype = np.float64).reshape(-1),
                                            rvec = rvec,
                                            tvec = tvec,
                                            useExtrinsicGuess = False,
                                            flags = cv2.SOLVEPNP_ITERATIVE,
                                        )
    return bool(success), rvec, tvec


def estimate_T_T_C_from_charuco(
                                    image_path   : Path,
                                    K            : NDArray[np.float64],
                                    dist         : NDArray[np.float64],
                                    board        : aruco.CharucoBoard,
                                    aruco_dict   : aruco.Dictionary,
                                 ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], dict[str, object]] | None:
    """ 
    estimate the 4x4 homogeneous transformation matrix T_T_C = ^C T_T from a ChArUco image
    T_T_C is the 4x4 homogeneous transformation matrix from the ChArUco board frame (T) to the camera frame (C)
    T_T_C = [ Trfm_T_C | r_{Co->T_o } ^C ]
            [0 0 0 1                     ]
        where Trfm_T_C is the 3x3 rotation matrix from T to C, and r_{Co->T_o} is the 3x1 translation vector from the 
        camera frame origin to the ChArUco board origin, expressed in the camera frame
    
    The ChArUco board frame is defined such that the origin is at the center of the first chessboard square, 
    1) the x-axis points to the right along the chessboard squares
    2) the y-axis points down along the chessboard squares
    3) the z-axis points out of the board plane according to the right-hand rule 
    """
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Failed to read image: {image_path}")

    gray    = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    charuco_corners, charuco_ids, marker_corners, marker_ids, num_markers = _detect_charuco_corners(
                                                                                                    gray,
                                                                                                    board,
                                                                                                    aruco_dict,
                                                                                                )
    if num_markers < 4:
        print(f"Skipping {image_path.name}: insufficient markers")
        return None
    if charuco_corners is None or charuco_ids is None:
        print(f"Skipping {image_path.name}: insufficient charuco corners")
        return None

    success, rvec, tvec = _estimate_charuco_pose(charuco_corners, charuco_ids, board, K, dist)
    if not success:
        print(f"Skipping {image_path.name}: pose estimation failed")
        return None

    Rotm_C_T, _ = cv2.Rodrigues(rvec) # Rotm_C_T (3x3 activte rotaiton) = Trfm_T_C (3x3 passive rotation) transpose
    # since OpenCV returns the active rotation from C to T
    T_T_C       = _build_transform(Rotm_C_T, tvec.reshape(3,))

    detection_info = {
                        "charuco_corners" : np.asarray(charuco_corners, dtype = np.float64).reshape(-1, 2),
                        "charuco_ids"     : np.asarray(charuco_ids, dtype = np.int32).reshape(-1),
                        "marker_corners"  : [np.asarray(corner, dtype = np.float64).reshape(-1, 2) for corner in marker_corners],
                        "marker_ids"      : np.asarray(marker_ids, dtype = np.int32).reshape(-1),
                     }
    return T_T_C, rvec.reshape(3,), tvec.reshape(3,), detection_info


def get_charuco_T_T_C_series(
                                image_dir      : Path,
                                img_suffix     : str,
                                K              : NDArray[np.float64],
                                dist           : NDArray[np.float64],
                                board          : aruco.CharucoBoard,
                                aruco_dict     : aruco.Dictionary,
                             ) -> tuple[NDArray[np.float64], list[int], list[Path], list[dict[str, object]]]:
    """ estimate T_T_C for all valid ChArUco images """
    image_paths         = collect_indxed_image_paths(image_dir, img_suffix = img_suffix, rosbag_style = False)
    T_T_C_list          = []
    valid_image_numbers = []
    valid_image_paths   = []
    reprojection_rows   = []
    invalid_image_paths = []
    # collect images, initialize lists to store results, which are 
    # valid image paths, their corresponding image numbers parsed from the filename 

    # iterate through images 
    for image_path in image_paths:
        # grab image number
        image_number     = parse_image_number(image_path)
        # estimate pose
        pose_estimate    = estimate_T_T_C_from_charuco(image_path, K, dist, board, aruco_dict)
        # if no pose, continue, otherwise store the T_T_C and corresponding image path and number
        if pose_estimate is None:
            invalid_image_paths.append(image_path)
            continue
        T_T_C, rvec, tvec, detection_info   = pose_estimate
        T_T_C_list.append(T_T_C)
        valid_image_numbers.append(image_number)
        valid_image_paths.append(image_path)
        reprojection_rows.append(
                                {
                                    "image_path" : image_path,
                                    "image_number" : image_number,
                                    "rvec" : np.asarray(rvec, dtype = np.float64),
                                    "tvec" : np.asarray(tvec, dtype = np.float64),
                                    "detection" : detection_info,
                                }
                              )

    if len(T_T_C_list) == 0:
        raise RuntimeError("no valid ChArUco detections were found in the selected image directory")

    print(f"Discovered {len(image_paths)} images")
    print(f"Valid ChArUco detections: {len(T_T_C_list)}")
    print(f"Invalid ChArUco detections: {len(invalid_image_paths)}")
    return np.stack(T_T_C_list), valid_image_numbers, valid_image_paths, reprojection_rows, invalid_image_paths
############################# ChArUco detection and pose estimation functions #############################