#Charuco Board Generation and Detection
import cv2 
import os
import json
import shutil


CURRENT_DIR = os.getcwd()
ARUCO_DICTIONARY = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_1000)
IMAGE_DIRECTORY = "calibration_images"
MARKED_DIRECTORY = "Marked Images"
CAMERA = 'XIMEA XiQ'
LENS = '25mm_new'
OUTPUT = f"calibration{LENS}.json"
BOARD_SIZE_SQUARES = (25,15)
BOARD_SIZE_PIXELS = (1936,1216)
SQUARE_LENGTHS = 0.04
MARKER_LENGHTS = 0.03
BORDER_SIZE = 60
SCALE = 1


if os.path.exists(MARKED_DIRECTORY):
    shutil.rmtree(MARKED_DIRECTORY)
os.makedirs(MARKED_DIRECTORY)

def get_calibration_parameters(img_dir, board):
    params = cv2.aruco.CharucoParameters()
    detector = cv2.aruco.CharucoDetector(board, params)



    
    image_files = [os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.endswith(".png")]
    charuco_ids = []
    charuco_corners = []

    for index, image_file in enumerate(image_files):
        image = cv2.imread(image_file)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        imgSize = (image.shape[1], image.shape[0])
        charucoCorners, charucoIds, marker_corners, marker_ids = detector.detectBoard(image)
        marked_images = cv2.aruco.drawDetectedCornersCharuco(image, charucoCorners, charucoIds)
        file_name = f"marked_image{index}.png"
        file_dir = os.path.join(MARKED_DIRECTORY, file_name)
        cv2.imwrite(file_dir, marked_images)
        if marker_ids is not None and len(marker_ids) > 0: 
            if charucoIds is not None and len(charucoCorners) > 3:
                charuco_corners.append(charucoCorners)
                charuco_ids.append(charucoIds)
    
    result, cam_matrix, distortion, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(
        charuco_corners, 
        charuco_ids, 
        board, 
        imgSize, 
        None, 
        None
    )


    return cam_matrix, distortion

   

def charuco_gen(num_squares_wide, num_squares_long, square_length, marker_length):
    board = cv2.aruco.CharucoBoard((num_squares_wide,num_squares_long),
                                    square_length, marker_length, ARUCO_DICTIONARY)

    board_image = board.generateImage(BOARD_SIZE_PIXELS, None, BORDER_SIZE, SCALE)

    output_path = os.path.join(CURRENT_DIR, "Generated_Charuco.jpg")
    cv2.imwrite(output_path, board_image)

    cmat, dist = get_calibration_parameters(IMAGE_DIRECTORY, board)
    return cmat, dist



cmat, dist = charuco_gen(BOARD_SIZE_SQUARES[0], BOARD_SIZE_SQUARES[1], SQUARE_LENGTHS, MARKER_LENGHTS)
data = {"camera": CAMERA, "lens": LENS, "camera_matrix": cmat.tolist(), "distortion": dist.tolist()}
with open(OUTPUT, 'w') as json_file:
    json.dump(data, json_file, indent=2)


