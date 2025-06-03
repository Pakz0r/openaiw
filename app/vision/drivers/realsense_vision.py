from .ivision import IVision
import numpy as np
import pyrealsense2 as rs

class RealSenseVision(IVision):
    def __init__(self):
        super().__init__("realsense")
        try:
            self.pipeline = rs.pipeline()
            self.config = rs.config()

            # Optimal resolution for D435: 848x480.
            self.config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)
            self.config.enable_stream(rs.stream.color, 848, 480, rs.format.bgr8, 30)

        except Exception as e:
            print(f"Errore nell'inizializzazione di RealSense: {e}")
            raise

    def start(self):
        try:
            self.pipeline_profile = self.pipeline.start(self.config)
            self.device = self.pipeline_profile.get_device()

            self.depth_sensor = self.device.first_depth_sensor()
            self.depth_scale = self.depth_sensor.get_depth_scale()

            profile = self.pipeline.get_active_profile()
            depth_profile = rs.video_stream_profile(profile.get_stream(rs.stream.depth))
            self.intrinsics = depth_profile.get_intrinsics()

            self.align = rs.align(rs.stream.color)

             # Decimation filter variable 
            self.decimation_magnitude = 2
            
            # Spatial filter variables
            self.spatial_magnitude = 2
            self.spatial_smooth_alpha = 0.5
            self.spatial_smooth_delta = 20
            self.spatial_holes_fill = 0

            # Temporal filter variables
            self.temporal_smooth_alpha = 0.4
            self.temporal_smooth_delta = 20
            self.persistency_index = 7

            # Holes Filling filter variable
            self.hole_filling = 1

            # Filters
            self.depth_to_disparity = rs.disparity_transform(True)
            self.disparity_to_depth = rs.disparity_transform(False)

            self.decimation_filter = rs.decimation_filter()
            self.decimation_filter.set_option(rs.option.filter_magnitude, self.decimation_magnitude)

            self.spatial_filter = rs.spatial_filter()
            self.spatial_filter.set_option(rs.option.filter_magnitude, self.spatial_magnitude)
            self.spatial_filter.set_option(rs.option.filter_smooth_alpha, self.spatial_smooth_alpha)
            self.spatial_filter.set_option(rs.option.filter_smooth_delta, self.spatial_smooth_delta)
            self.spatial_filter.set_option(rs.option.holes_fill, self.spatial_holes_fill)

            self.temporal_filter = rs.temporal_filter()
            self.temporal_filter.set_option(rs.option.holes_fill, self.persistency_index)

            self.hole_filling_filter = rs.hole_filling_filter()
            self.hole_filling_filter.set_option(rs.option.holes_fill, self.hole_filling)

        except Exception as e:
            print(f"Errore nell'avvio del flusso di RealSense: {e}")
            raise

    def start_recording(self, output_path):
        try:
            self.config.enable_record_to_file(output_path)
            self.start()

        except Exception as e:
            print(f"Errore nell'avvio del flusso di RealSense: {e}")
            raise

    def start_playback(self, input_path):
        try:
            self.config.enable_device_from_file(input_path, repeat_playback=False)
            self.start()
            playback = self.device.as_playback()
            playback.set_real_time(False)
            return playback

        except Exception as e:
            print(f"Errore nell'avvio del flusso di RealSense: {e}")
            raise

    def wait_frames(self):
        try:
            self.pipeline.wait_for_frames()

        except Exception as e:
            print(f"Errore durante la lettura dei frame da RealSense: {e}")

    def get_frames(self):
        try:
            frames = self.pipeline.wait_for_frames()
            aligned_frames = self.align.process(frames)

            color_frame = aligned_frames.first(rs.stream.color)
            depth_frame = aligned_frames.get_depth_frame()

            # depth_frame = self.decimation_filter.process(depth_frame)
            depth_frame = self.depth_to_disparity.process(depth_frame)
            depth_frame = self.spatial_filter.process(depth_frame)
            depth_frame = self.temporal_filter.process(depth_frame)
            depth_frame = self.disparity_to_depth.process(depth_frame)
            depth_frame = self.hole_filling_filter.process(depth_frame).as_depth_frame()

            self.depth_frame = depth_frame

            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            self.depth_image = depth_image
            return depth_image, color_image
        
        except Exception as e:
            print(f"Errore durante la lettura dei frame da RealSense: {e}")
            return None, None
        
    def get_color_frame(self):
        try:
            frames = self.pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            color_image = np.asanyarray(color_frame.get_data())
            return color_image
        
        except Exception as e:
            print(f"Errore durante la lettura del frame da RealSense: {e}")
            return None

    def get_device_info(self):
        return None
    
    def convert_point_to_world(self, u: float, v: float):
        u_clamped = np.clip(int(u), 0, self.depth_image.shape[1] - 1)
        v_clamped = np.clip(int(v), 0, self.depth_image.shape[0] - 1)
        depth = self.depth_frame.get_distance(u_clamped, v_clamped) # Profondità in metri

        if depth == 0: # Fallback: cerca un valore di profondità valido in una finestra centrata su (x, y)
            depth, u_clamped, v_clamped = self.get_valid_depth(self.depth_frame, u_clamped, v_clamped)
        
        return rs.rs2_deproject_pixel_to_point(self.intrinsics, [u, v], depth) # Ritorna le coordinate nel sistema right-handed

    def get_valid_depth(self, depth_frame, x, y, window=3):
        w, h = depth_frame.get_width(), depth_frame.get_height()
        for dy in range(-window, window + 1):
            for dx in range(-window, window + 1):
                nx, ny = x + dx, y + dy
                if 0 <= nx < w and 0 <= ny < h:
                    d = depth_frame.get_distance(nx, ny)
                    if d > 0:
                        return d, nx, ny
        return 0, x, y  # Nessun valore valido trovato

    def __del__(self):
        if hasattr(self, 'pipeline') and self.pipeline:
            self.pipeline.stop()