from .ivision import IVision
import numpy as np
import pyrealsense2 as rs

class RealSenseVision(IVision):
    def __init__(self):
        try:
            self.pipeline = rs.pipeline()
            config = rs.config()
            config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
            config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
            self.pipeline.start(config)
        except Exception as e:
            print(f"Errore nell'inizializzazione di RealSense: {e}")
            raise

    def get_frames(self):
        try:
            frames = self.pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
        
            self.depth_frame = depth_frame
            profile = depth_frame.profile
            self.intrinsics = profile.as_video_stream_profile().get_intrinsics()

            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            return depth_image, color_image
        except Exception as e:
            print(f"Errore durante la lettura dei frame da RealSense: {e}")
            return None, None

    def get_device_info(self):
        return None
    
    def convert_point_to_world(self, u: float, v: float):
        depth = self.depth_frame.get_distance(int(u), int(v))  # Ottieni profondità in metri
        return rs.rs2_deproject_pixel_to_point(self.intrinsics, [u, v], depth)

    def __del__(self):
        if hasattr(self, 'pipeline') and self.pipeline:
            self.pipeline.stop()