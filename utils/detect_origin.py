import cv2
import cv2.aruco as aruco
import numpy as np
import yaml
import os 

IMAGE_DIR = "/home/will/projects/CamCal/data/offset_images"
OUTPUT_DIR = "/home/will/projects/CamCal/data/origin_frame"
CALIBRATION_DIR = "calibration.yaml"
os.makedirs(OUTPUT_DIR, exist_ok=True)



# Import intrinsics
with open(CALIBRATION_DIR) as f:
    data = yaml.safe_load(f)

K = np.array([
    [data['fx'], 0,            data['cx']],
    [0,            data['fy'], data['cy']],
    [0,            0,            1]
], dtype=np.float64)
dist = np.array([
    data['k1'],
    data['k2'],
    data['p1'],
    data['p2'],
    data['k3']
], dtype=np.float64).reshape(1, 5)


# Dictionary
dictionary = aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)
# ARUCO_DICT = cv2.aruco.DICT_6X6_250
# Create board (must match your printed board!)
board = aruco.CharucoBoard(
    (7,5),        # squaresX, squaresY
    20e-3,          # squareLength (meters)
    15e-3,          # markerLength (meters)s
    dictionary
)
count = 1
for img in sorted(os.listdir(IMAGE_DIR)):

    # Load image
    img = cv2.imread(os.path.join(IMAGE_DIR, img))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Detect markers
    corners, ids, _ = aruco.detectMarkers(gray, dictionary)

    # Interpolate charuco corners
    retval, charucoCorners, charucoIds = aruco.interpolateCornersCharuco(
        corners, ids, gray, board
    )
    if charucoCorners is not None and charucoIds is not None and len(charucoIds) >= 4:
        success, rvec, tvec = aruco.estimatePoseCharucoBoard(
            charucoCorners, charucoIds, board, K, dist, None, None
        )
        print(f"Success: {success}, rvec: {rvec}, tvec: {tvec}")
    else:
        print("Skipping pose — insufficient corners")

    cv2.drawFrameAxes(
        img,
        K,
        dist,
        rvec,
        tvec,
        0.05  # axis length in meters
    )
    cv2.imwrite(os.path.join(OUTPUT_DIR, f"origin_frame_{count}.png"), img)
    count += 1
