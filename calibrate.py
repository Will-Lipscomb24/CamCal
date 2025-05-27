#Camera Calibration Script
import pyvicon_datastream as pv
from pyvicon_datastream import tools
from pypylon import pylon
import numpy as np
import keyboard
import os 


OBJECT1 = "vicon_dot"
OBJECT2 = "basler_cam"
VICON_IP = ""

#File Saving Paths
os.makedirs("calibration_images", exist_ok=True)
os.makedirs("calibration_vicon_data", exist_ok=True)
VICON_PATH = os.path.join("calibration_vicon_data", "vicon_data.csv")
if not os.path.exists(VICON_PATH):
    with open(VICON_PATH, "w") as f:
        f.write(f"id,{OBJECT1},x,y,z,rx,ry,rz,{OBJECT2},x,y,z,rx,ry,rz\n")

CAMERA_PATH = os.path.join("calibration_images", "camera_image.png")


#Connect to Vicon System
vicon_client = pv.PyViconDatastream()
ret = vicon_client.connect(VICON_IP)
print("Connecting to Vicon System...")

if ret != pv.Results.Success:
    print(f"\Connection to {VICON_IP} failed.")
else:
    print(f"\Connection to {VICON_IP} successful.")
tracker = tools.ObjectTracker(VICON_IP)

#Connect to Basler Camera
camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
camera.Open()

# Access the NodeMap to configure software trigger
nodemap = camera.GetNodeMap()

# Set software trigger mode
nodemap.GetNode("TriggerSelector").SetValue("FrameStart")
nodemap.GetNode("TriggerMode").SetValue("On")
nodemap.GetNode("TriggerSource").SetValue("Software")
nodemap.GetNode("AcquisitionMode").SetValue("SingleFrame")  # optional

print("Software trigger configured.")

# Start grabbing
camera.StartGrabbing()

try:
    while True:
        if keyboard.is_pressed("enter"):
            # Wait until trigger is ready
            camera.WaitForFrameTriggerReady(1000, pylon.TimeoutHandling_ThrowException)
            camera.ExecuteSoftwareTrigger()

            dot_data = tracker._get_object_position(OBJECT1)[2:]
            cam_data = tracker._get_object_position(OBJECT2)[2:]



            # Retrieve result
            result = camera.RetrieveResult(2000, pylon.TimeoutHandling_ThrowException)
            if result.GrabSucceeded():
                print(f"Captured: {result.GetWidth()}x{result.GetHeight()}")

            result.Release()

            # Prevent multiple triggers from one key press
            while keyboard.is_pressed("enter"):
                pass  # wait for key release

        elif keyboard.is_pressed("esc"):
            print("Exiting...")
            break

finally:
    camera.StopGrabbing()
    camera.Close()

"""
if basler camera takes an image
    extract the corresponding vicon data
"""
#Exract Vicon Data at Each Image Point 

# Compute Pan and Tilt Angles Based on Vicon Initial Data

#Adjust Camera Pan and Tilt to Align Vicon Dot in Top Left Corner of the Image


#Output: Images and Corresponding Vicon Data