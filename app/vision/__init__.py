if __name__ == 'vision':
    from .drivers import *
    from .drivers.ivision import IVision
else:
    from drivers import *
    from drivers.ivision import IVision

class Vision:
    @staticmethod
    def initialize(driver_name, *args) -> IVision:
        """Crea un'istanza di IVision in base al nome del driver."""
        driver_name = driver_name.lower()
        if driver_name == "openni2":
            return OpenNIVision(args)
        elif driver_name == "realsense":
            return RealSenseVision()
        raise ValueError(f"Driver non supportato: {driver_name}")