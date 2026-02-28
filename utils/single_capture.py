import cv2
from pypylon import pylon
import os

# Create directory if it doesn't exist
CAMERA_PATH = "./sing_images"
if not os.path.exists(CAMERA_PATH):
    os.makedirs(CAMERA_PATH)

# --- Camera setup ---
camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

converter = pylon.ImageFormatConverter()
converter.OutputPixelFormat = pylon.PixelType_BGR8packed
converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

IMG = pylon.PylonImage()
COUNTER = 0

print("Controls:")
print("  - Press [Enter] to capture an image")
print("  - Press [q] to quit")

while camera.IsGrabbing():
    result = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)

    if result.GrabSucceeded():
        image = converter.Convert(result)
        frame = image.GetArray()

        # Display resized live feed
        display = cv2.resize(frame, (1800, 1000), interpolation=cv2.INTER_AREA)
        cv2.imshow("Live Camera Feed", display)

        # Check for key presses
        key = cv2.waitKey(1) & 0xFF
        
        # 13 is the ASCII code for the Enter key
        if key == 13: 
            COUNTER += 1
            filename = f"{CAMERA_PATH}/cal_image_{COUNTER}.png"
            
            # Save using Pylon's native method
            IMG.AttachGrabResultBuffer(result)
            IMG.Save(pylon.ImageFileFormat_Png, filename)
            IMG.Release()
            
            print(f"Captured: {filename}")

        elif key == ord('q'):
            break

    result.Release()

# --- Cleanup ---
camera.StopGrabbing()
cv2.destroyAllWindows()