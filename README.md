# Basler Camera Intrinsics and Frame Offset Calibration
This repo can be used for the following:
* Vicon and image data collection of defined Vicon objects
* Camera intrinsics calibration using OpenCV functions and Charuco board
* Truth and Vicon frame offset optimization using Scipy non-linear least squares solver


# Setup `sc-pose-utils` dependency 
```bash
git clone https://github.com/aagrawal66/sc-pose-utils.git
cd sc-pose-utils
python -m pip install -e . # installs into "activated" python
```
