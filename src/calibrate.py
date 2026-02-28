import cv2
import cv2.aruco as aruco
import numpy as np
import yaml
import os
import glob

# ──────────────────────────────────────────────
# CONFIG — edit these to match your board
# ──────────────────────────────────────────────
IMAGE_PATH    = "/home/will/projects/CamCal/data/offset_images/*.png"  # glob pattern
SQUARES_X     = 7          # columns
SQUARES_Y     = 5          # rows
SQUARE_LEN    = 20e-3      # meters
MARKER_LEN    = 15e-3      # meters
DICTIONARY    = cv2.aruco.DICT_5X5_100
OUTPUT_YAML   = "calibration.yaml"
# ──────────────────────────────────────────────

dictionary = aruco.getPredefinedDictionary(DICTIONARY)

board = aruco.CharucoBoard(
    (SQUARES_X, SQUARES_Y),
    SQUARE_LEN,
    MARKER_LEN,
    dictionary
)

all_charuco_corners = []
all_charuco_ids     = []
image_size          = None

image_files = sorted(glob.glob(IMAGE_PATH))
print(f"Found {len(image_files)} images")

for fpath in image_files:
    img  = cv2.imread(fpath)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if image_size is None:
        image_size = gray.shape[::-1]  # (width, height)

    corners, ids, _ = aruco.detectMarkers(gray, dictionary)

    if ids is None or len(ids) < 4:
        print(f"  SKIP (too few markers): {os.path.basename(fpath)}")
        continue

    retval, charucoCorners, charucoIds = aruco.interpolateCornersCharuco(
        corners, ids, gray, board
    )

    if charucoIds is None or len(charucoIds) < 4:
        print(f"  SKIP (too few charuco corners): {os.path.basename(fpath)}")
        continue

    all_charuco_corners.append(charucoCorners)
    all_charuco_ids.append(charucoIds)
    print(f"  OK ({len(charucoIds)} corners): {os.path.basename(fpath)}")

print(f"\nUsing {len(all_charuco_corners)} valid images for calibration...")

if len(all_charuco_corners) < 5:
    print("ERROR: Not enough valid images (need at least 5). Exiting.")
    exit(1)

ret, K, dist, rvecs, tvecs = aruco.calibrateCameraCharuco(
    all_charuco_corners,
    all_charuco_ids,
    board,
    image_size,
    None,
    None
)

print(f"\nReprojection error: {ret:.4f} px")
print(f"K:\n{K}")
print(f"dist: {dist}")

# Save to YAML
data = {
    "fx": float(K[0, 0]),
    "fy": float(K[1, 1]),
    "cx": float(K[0, 2]),
    "cy": float(K[1, 2]),
    "k1": float(dist[0, 0]),
    "k2": float(dist[0, 1]),
    "p1": float(dist[0, 2]),
    "p2": float(dist[0, 3]),
    "k3": float(dist[0, 4]),
    "reprojection_error": float(ret),
    "image_width":  image_size[0],
    "image_height": image_size[1]
}

with open(OUTPUT_YAML, "w") as f:
    yaml.dump(data, f, default_flow_style=False)

print(f"\nCalibration saved to {OUTPUT_YAML}")
