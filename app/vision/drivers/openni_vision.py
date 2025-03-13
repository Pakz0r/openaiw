from .ivision import IVision
import numpy as np
import cv2
from primesense import openni2

class OpenNIVision(IVision):
    def __init__(self, dll_directories):
        super().__init__("openni2")
        try:
            openni2.initialize(dll_directories) 
            self.dev = openni2.Device.open_any()
            self.depth_stream = self.dev.create_depth_stream()
            self.color_stream = self.dev.create_color_stream()
            self.depth_stream.start()
            self.color_stream.start()
        except Exception as e:
            print(f"Errore nell'inizializzazione di OpenNI: {e}")
            raise

    def get_frames(self):
        try:
            depth_frame = self.depth_stream.read_frame()
            color_frame = self.color_stream.read_frame()

            # Processa il frame di profondità
            depth_image = np.array(depth_frame.get_buffer_as_uint16()).reshape(depth_frame.height, depth_frame.width)
            self.depth_image = depth_image

            # Processa il frame colore
            color_data = np.array(color_frame.get_buffer_as_uint8()).reshape(color_frame.height, color_frame.width, 3)
            color_image = cv2.cvtColor(color_data, cv2.COLOR_RGB2BGR)

            return depth_image, color_image
        except Exception as e:
            print(f"Errore durante la lettura dei frame da OpenNI: {e}")
            return None, None
    
    def get_device_info(self):
        return self.dev.get_device_info()
    
    def convert_point_to_world(self, u: float, v: float):
        depth = self.depth_image[int(v), int(u)] / 1000.0  # Profondità in metri (da mm)
        return openni2.convert_depth_to_world(self.depth_stream, u, v, depth) # Ritorna le coordinate nel sistema right-handed

    def __del__(self):
        if hasattr(self, "depth_stream"):
            self.depth_stream.stop()

        if hasattr(self, "color_stream"):
            self.color_stream.stop()

        openni2.unload()