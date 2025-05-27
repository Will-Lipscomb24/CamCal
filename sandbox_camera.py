# Camera Calibration Script with live feed display

from pypylon import pylon
import keyboard
import os 
import yaml
import cv2
from camera_settings import CameraSettings

COUNTER1 = 0
COUNTER2 = 0
IMG = pylon.PylonImage()
PATH = "test_images"

while os.path.exists(PATH):
    COUNTER1 += 1
    PATH = f"{PATH}_{COUNTER1}"
    
os.makedirs(PATH)


camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
if not camera.IsOpen():
    camera.Open()

with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

camera_settings = CameraSettings(
    exposure_time=config["parameters"].get("exposure"),
    gain=config["parameters"].get("gain"),
    pixel_format=config["parameters"].get("pixel_format"),
    width=config["parameters"].get("image_width"),
    height=config["parameters"].get("image_height")
)
camera_settings.settings(camera)

# Remove trigger mode for continuous grab
nodemap = camera.GetNodeMap()
nodemap.GetNode("TriggerMode").SetValue("Off")
nodemap.GetNode("AcquisitionMode").SetValue("Continuous")

# Prepare converter for OpenCV display
converter = pylon.ImageFormatConverter()
converter.OutputPixelFormat = pylon.PixelType_BGR8packed
converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

try:
    while camera.IsGrabbing():
        # Grab a frame
        result = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
        if result.GrabSucceeded():
            image = converter.Convert(result)
            img_array = image.GetArray()

            # Show live feed
            cv2.imshow("Live Camera Feed", img_array)

            # Save on Enter key press
            if keyboard.is_pressed("enter"):
                COUNTER2 += 1
                filename = f"{PATH}/image_{COUNTER2}.png"
                print(f"Saving image {COUNTER2} to {filename}")
                # Use pylon image to save so metadata preserved
                IMG.AttachGrabResultBuffer(result)
                IMG.Save(pylon.ImageFileFormat_Png, filename)
                IMG.Release()

                # Wait for key release to avoid multiple saves
                while keyboard.is_pressed("enter"):
                    pass

            # Exit on ESC key
            if cv2.waitKey(1) & 0xFF == 27 or keyboard.is_pressed("esc"):
                print("Exiting...")
                break

        result.Release()

finally:
    camera.StopGrabbing()
    camera.Close()
    cv2.destroyAllWindows()
