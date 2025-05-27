
class CameraSettings:
    def __init__(self, exposure_time, gain, pixel_format, width, height):
        self.exposure_time = exposure_time
        self.gain = gain
        self.pixel_format = pixel_format
        self.width = width
        self.height = height

    def settings(self, camera):
        camera.ExposureTime.SetValue(self.exposure_time)
        camera.Gain.SetValue(self.gain)
        camera.PixelFormat.SetValue(self.pixel_format)
        camera.Width.SetValue(self.width)
        camera.Height.SetValue(self.height)
        print("Camera settings applied.")