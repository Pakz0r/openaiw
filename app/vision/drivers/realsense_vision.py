from .ivision import IVision
import numpy as np
import pyrealsense2 as rs

class RealSenseVision(IVision):
    def __init__(self):
        super().__init__("realsense")
        try:
            self.pipeline = rs.pipeline()
            config = rs.config()
            # Optimal resolution for D435: 848x480.
            config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)
            config.enable_stream(rs.stream.color, 848, 480, rs.format.bgr8, 30)
            pipeline_profile = self.pipeline.start(config)

            self.depth_sensor = pipeline_profile.get_device().first_depth_sensor()
            self.depth_scale = self.depth_sensor.get_depth_scale()

            profile = self.pipeline.get_active_profile()
            depth_profile = rs.video_stream_profile(profile.get_stream(rs.stream.depth))
            self.intrinsics = depth_profile.get_intrinsics()

            self.align = rs.align(rs.stream.color)

            self.spatial_filter = rs.spatial_filter()
            self.spatial_filter.set_option(rs.option.holes_fill, 1)

        except Exception as e:
            print(f"Errore nell'inizializzazione di RealSense: {e}")
            raise

    def get_frames(self):
        try:
            frames = self.pipeline.wait_for_frames()
            aligned_frames = self.align.process(frames)

            color_frame = aligned_frames.first(rs.stream.color)
            depth_frame = aligned_frames.get_depth_frame()

            depth_frame = self.spatial_filter.process(depth_frame).as_depth_frame()
            self.depth_frame = depth_frame

            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            self.depth_image = depth_image
            return depth_image, color_image
        except Exception as e:
            print(f"Errore durante la lettura dei frame da RealSense: {e}")
            return None, None

    def get_device_info(self):
        return None
    
    def convert_point_to_world(self, u: float, v: float):
        u_clamped = np.clip(int(u), 0, self.depth_image.shape[1] - 1)
        v_clamped = np.clip(int(v), 0, self.depth_image.shape[0] - 1)
        depth = self.depth_frame.get_distance(u_clamped, v_clamped) # Profondità in metri
        return rs.rs2_deproject_pixel_to_point(self.intrinsics, [u, v], depth) # Ritorna le coordinate nel sistema right-handed

    def __del__(self):
        if hasattr(self, 'pipeline') and self.pipeline:
            self.pipeline.stop()