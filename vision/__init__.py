from vision.drivers.openni_vision import OpenNIVision
from vision.drivers.realsense_vision import RealSenseVision

class Vision:
    @staticmethod
    def initialize(driver_name):
        """Crea un'istanza di IVision in base al nome del driver."""
        driver_name = driver_name.lower()
        if driver_name == "openni2":
            return OpenNIVision()
        elif driver_name == "realsense":
            return RealSenseVision()
        raise ValueError(f"Driver non supportato: {driver_name}")