#Camera Calibration Script
import csv
import os 
from time import sleep
import shutil
import numpy as np
import keyboard
import yaml
import cv2
import pyvicon_datastream as pv
from pyvicon_datastream import tools
from pypylon import pylon
from settings import CameraSettings

#Initialization
OBJECT1 = "calibration_dot"
OBJECT2 = "basler_cam"
IMG = pylon.PylonImage()
VICON_IP = "192.168.1.100"
VICON_PATH = os.path.join("calibration_vicon_data", "vicon_data.csv")
CAMERA_PATH = "calibration_images"
COUNTER = 0


if os.path.exists("calibration_images"):
    shutil.rmtree("calibration_images")
if os.path.exists("calibration_vicon_data"):
    shutil.rmtree("calibration_vicon_data")
#Directory Formatting
os.makedirs("calibration_images", exist_ok=True)
os.makedirs("calibration_vicon_data", exist_ok=True)

#Connect to Vicon System
vicon_client = pv.PyViconDatastream()
ret = vicon_client.connect(VICON_IP)
print("Connecting to Vicon System...")
if ret != pv.Result.Success:
    print(f"\nConnection to {VICON_IP} failed.")
else:
    print(f"\nConnection to {VICON_IP} successful.")
tracker = tools.ObjectTracker(VICON_IP)

#Connect to Basler Camera
camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
if not camera.IsOpen():
    camera.Open()
nodemap = camera.GetNodeMap()
nodemap.GetNode("TriggerMode").SetValue("Off")
nodemap.GetNode("AcquisitionMode").SetValue("Continuous")
print("Software trigger configured.")
converter = pylon.ImageFormatConverter()
converter.OutputPixelFormat = pylon.PixelType_BGR8packed
converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

#Load and Set Camera Settings
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

#Image and Vicon Data Capture
camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
try:
    with open(VICON_PATH, "w", newline='') as f:
        writer = csv.writer(f)
        while True:
            # Grab a frame
            result = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
            if result.GrabSucceeded():
                image = converter.Convert(result)
                img_array = image.GetArray()
                display = cv2.resize(img_array, (1800, 1000), interpolation=cv2.INTER_AREA)
                # Show live feed
                cv2.imshow("Live Camera Feed", display)

                # Save on Enter key press
                if keyboard.is_pressed("enter"):
                    COUNTER += 1
                    img_filename = f"{CAMERA_PATH}/cal_image_{COUNTER}.png"
                    print(f"Saving image {COUNTER} to {img_filename}")
                    # Use pylon image to save so metadata preserved
                    IMG.AttachGrabResultBuffer(result)
                    IMG.Save(pylon.ImageFileFormat_Png, img_filename)
                    IMG.Release()

                    _,_, obj1_pos = tracker.get_position(OBJECT1)
                    _,_, obj2_pos = tracker.get_position(OBJECT2)
                    obj1_data = np.array(obj1_pos[0][2:])
                    obj2_data = np.array(obj2_pos[0][2:])
                    obj_data = np.concatenate((obj1_data, obj2_data))
                    writer.writerow(obj_data.tolist())

                    # Wait for key release to avoid multiple saves
                    while keyboard.is_pressed("enter"):
                        pass

                if cv2.waitKey(1) & 0xFF == 27 or keyboard.is_pressed("esc"):
                    print("Exiting...")
                    break
                
                sleep(0.1)

            result.Release()

finally:
    camera.StopGrabbing()
    camera.Close()
    cv2.destroyAllWindows()
    vicon_client.disconnect()