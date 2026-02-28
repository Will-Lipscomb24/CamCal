import numpy as np
import cv2

# ------------------------------
# USER PARAMETERS
ARUCO_DICT = cv2.aruco.DICT_6X6_250

SQUARES_HORIZONTALLY = 7
SQUARES_VERTICALLY   = 5


TOTAL_WIDTH = 16.5e-3
TOTAL_HEIGHT = 11e-3

SQUARE_LENGTH = TOTAL_WIDTH / SQUARES_HORIZONTALLY # 4 inches in meters
MARKER_LENGTH = TOTAL_HEIGHT / SQUARES_VERTICALLY # 3 inches in meters
print(SQUARE_LENGTH)
print(MARKER_LENGTH)
DPI = 300
SAVE_NAME = "Charuco_18mm_300dpi.png"
# ------------------------------

def create_and_save_new_board():

    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)

    # IMPORTANT: (squaresX, squaresY)
    board = cv2.aruco.CharucoBoard(
        (SQUARES_HORIZONTALLY, SQUARES_VERTICALLY),
        SQUARE_LENGTH,
        MARKER_LENGTH,
        dictionary
    )

    # ---- Compute physical board size ----
    board_width_m  = SQUARES_HORIZONTALLY * SQUARE_LENGTH
    board_height_m = SQUARES_VERTICALLY * SQUARE_LENGTH

    # Convert meters -> inches
    board_width_in  = board_width_m  / 0.0254
    board_height_in = board_height_m / 0.0254

    # Convert inches -> pixels (300 DPI)
    width_px  = int(round(board_width_in  * DPI))
    height_px = int(round(board_height_in * DPI))

    print(f"Board physical size: {board_width_m*1000:.1f} mm x {board_height_m*1000:.1f} mm")
    print(f"Image size: {width_px} x {height_px} pixels at {DPI} DPI")

    img = board.generateImage((width_px, height_px))

    cv2.imwrite(SAVE_NAME, img)
    cv2.imshow("Charuco Board", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

create_and_save_new_board()