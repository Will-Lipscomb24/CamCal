
import csv
import os 
from time import sleep
import time
import shutil
import numpy as np
#import keyboard
import yaml
import cv2
import pyvicon_datastream as pv
from pyvicon_datastream import tools
from pypylon import pylon
from src.settings import CameraSettings
from pynput import keyboard 

#Load and Set Camera Settings
with open("configs/config.yaml", "r") as file:
    cfg = yaml.safe_load(file)

#Initialization
OBJECT1 = cfg['vicon']['object1'] 
OBJECT2 = cfg['vicon']['object2'] 
IMG = pylon.PylonImage()
VICON_IP = cfg['vicon']['ip_address']
VICON_PATH = os.path.join('data',cfg['paths']['vicon_path'])
CAMERA_PATH = os.path.join('data',cfg['paths']['image_path'])
COUNTER = 0
HEADER = ['image_number','soho_x','soho_y','soho_z','soho_qw','soho_qx','soho_qy','soho_qz','cam_x','cam_y','cam_z','cam_qw','cam_qx','cam_qy','cam_qz']

os.makedirs(CAMERA_PATH,exist_ok=True)
os.makedirs(VICON_PATH,exist_ok=True)


enter_pressed = False
esc_pressed = False

def on_press(key):
    global enter_pressed, esc_pressed
    if key == keyboard.Key.enter:
        enter_pressed = True
    elif key == keyboard.Key.esc:
        esc_pressed = True
listener = keyboard.Listener(on_press=on_press)
listener.start()    


#Connect to Vicon System
vicon_client = pv.PyViconDatastream()
ret = vicon_client.connect(VICON_IP)
print("Connecting to Vicon System...")
if ret != pv.Result.Success:
    print(f"\nConnection to {VICON_IP} failed.")
else:
    print(f"\nConnection to {VICON_IP} successful.")
tracker = tools.ObjectTracker(VICON_IP)


#Connect to cam Camera
camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
if not camera.IsOpen():
    camera.Open()
nodemap = camera.GetNodeMap()
nodemap.GetNode("TriggerMode").SetValue("Off")
nodemap.GetNode("AcquisitionMode").SetValue("Continuous")
print("Software trigger cfgured.")
converter = pylon.ImageFormatConverter()
converter.OutputPixelFormat = pylon.PixelType_BGR8packed
converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned



camera_settings = CameraSettings(
    exposure_time=cfg["parameters"].get("exposure"),
    gain=cfg["parameters"].get("gain"),
    pixel_format=cfg["parameters"].get("pixel_format"),
    width=cfg["parameters"].get("image_width"),
    height=cfg["parameters"].get("image_height")
)
camera_settings.settings(camera)

#Image and Vicon Data Capture
camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

existing = [f for f in os.listdir(CAMERA_PATH) if f.startswith("cal_image_") and f.endswith(".png")]
COUNTER = len(existing)
print(f"Resuming from image {COUNTER + 1}")

csv_path = os.path.join(VICON_PATH, "vicon_data.csv")
csv_exists = os.path.exists(csv_path)

try:
    with open(csv_path, "a", newline='') as f:
        writer = csv.writer(f)
        if not csv_exists:
            writer.writerow(HEADER)
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
                
                if enter_pressed:
                    enter_pressed = False  # reset immediately

                    COUNTER += 1
                    img_filename = f"{CAMERA_PATH}/cal_image_{COUNTER}.png"

                    # --- Try Vicon data first ---
                    try:
                        _, _, obj1_pos = tracker.get_position(OBJECT1)
                        _, _, obj2_pos = tracker.get_position(OBJECT2)
                        obj1_data = np.array(obj1_pos[0][2:])
                        obj2_data = np.array(obj2_pos[0][2:])
                        obj_data = np.concatenate((obj1_data, obj2_data))
                    except Exception as vicon_err:
                        COUNTER -= 1
                        print(f"Vicon capture failed: {vicon_err}. Skipping, continuing...")
                    else:
                        # --- Vicon succeeded, try CSV ---
                        try:
                            writer.writerow([COUNTER] + obj_data.tolist())
                            f.flush()
                        except Exception as csv_err:
                            COUNTER -= 1
                            print(f"CSV write failed: {csv_err}. Skipping capture, continuing...")
                        else:
                            # --- Both succeeded, now save image ---
                            print(f"Saving image {COUNTER} to {img_filename}")
                            IMG.AttachGrabResultBuffer(result)
                            IMG.Save(pylon.ImageFileFormat_Png, img_filename)
                            IMG.Release()

                if esc_pressed or (cv2.waitKey(1) & 0xFF == 27):
                    print("Exiting...")
                    break
                
                sleep(0.1)

            result.Release()

finally:
    listener.stop()
    camera.StopGrabbing()
    camera.Close()
    cv2.destroyAllWindows()
    vicon_client.disconnect()
# try:
#     with open(csv_path, "a", newline='') as f:
#         writer = csv.writer(f)
#         if not csv_exists:
#             writer.writerow(QUATERNION_HEADER)
#         while True:
#             # Grab a frame
#             result = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
#             if result.GrabSucceeded():
#                 image = converter.Convert(result)
#                 img_array = image.GetArray()
#                 display = cv2.resize(img_array, (1800, 1000), interpolation=cv2.INTER_AREA)
#                 # Show live feed
#                 cv2.imshow("Live Camera Feed", display)

#                 # Save on Enter key press
#                 if keyboard.is_pressed("enter"):
#                     COUNTER += 1
#                     img_filename = f"{CAMERA_PATH}/cal_image_{COUNTER}.png"
#                     print(f"Saving image {COUNTER} to {img_filename}")
#                     # Use pylon image to save so metadata preserved
#                     IMG.AttachGrabResultBuffer(result)
#                     IMG.Save(pylon.ImageFileFormat_Png, img_filename)
#                     IMG.Release()

#                     _,_, obj1_pos = tracker.get_position(OBJECT1)
#                     _,_, obj2_pos = tracker.get_position(OBJECT2)
#                     obj1_data = np.array(obj1_pos[0][2:])
#                     obj2_data = np.array(obj2_pos[0][2:])
#                     obj_data = np.concatenate((obj1_data, obj2_data))
                   
#                     writer.writerow([COUNTER] + obj_data.tolist())

#                     # Wait for key release to avoid multiple saves
#                     while keyboard.is_pressed("enter"):
#                         pass

#                 if cv2.waitKey(1) & 0xFF == 27 or keyboard.is_pressed("esc"):
#                     print("Exiting...")
#                     break
                
#                 sleep(0.1)

#             result.Release()

# finally:
#     camera.StopGrabbing()
#     camera.Close()
#     cv2.destroyAllWindows()
#     vicon_client.disconnect()