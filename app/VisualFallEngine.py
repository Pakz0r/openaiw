import sys
import datetime
import cv2
import json
from numpy import sum, zeros, expand_dims, linalg, take
from fall_detection.VisualTracker import Tracker
from fall_detection.Model import create_model
from fall_detection.Drawer import *
from vision import Vision
from utils import OPUtils, CoordinateMapper

CONFIG = {
    "vision": {
        "driver" : "realsense",
        "dll_directories": "C:\Program Files\OpenNI2\Redist"
    },
    "openpose" : {
        "model_folder": "/app/models/",
        "model_pose": "BODY_25",
        "net_resolution": "-1x176", # Default "-1x368"; AI_Watch "-1x128"
        "hand": False,
        "hand_net_resolution": "224x224", # "Default "368x368" (multiples of 16)"
        "face": False,
        "face_net_resolution": "224x224", # "Default "368x368" (multiples of 16)"
    },
    "room" : {
        "thing_id" : "digitaltwin:Laboratorio_Corridoio:1"
    },
    "Fall" : {
        "model_root" : '/app/falldetection/models/',
        "device": 'cpu'
    },
    "output_dir" : "output"
}

print("Inizializzazione di OpenPose...")
opUtils = OPUtils(CONFIG["openpose"])

print("Inizializzazione del modulo di Vision...")
vision = Vision.initialize(CONFIG["vision"]["driver"], CONFIG["vision"]["dll_directories"])

print("Inizializzazione del Coordinate Mapper...")
mapper = CoordinateMapper(CONFIG["room"]["thing_id"])

dev_info = vision.get_device_info()
if dev_info is not None:
    print(f"Collegato con {dev_info.vendor.decode()} {dev_info.name.decode()}")

print("Applicazione inizializzata con successo. Premere 'q' per uscire.")


class FallEngine:

    def __init__(self):

        # Data structures
        self.params = dict()
        self.params["model_folder"] = "/app/models" 
        self.fall_threshold = 4
        self.buffer_frames = []
        self.TTL_show_info = 0
        self.iteration = 0
        self.predictions = []

        # Load LSTM-AE model
        self.model = create_model()
        self.model.load_weights("/app/fall_detection/model/model.h5")

        # Initialize Tracker object
        self.tracker = Tracker(counter_threshold=10)


    def _get_centers_of_mass(self, window):
    
        centers_of_mass = []
        # Each frame in the window is a list of subsequent coordinates for each keypoint [x1,y1,x2,y2,x3,y3....]
        for skeleton in window:
            keypoints_for_center_of_mass = [12, 9, 8, 5, 2, 1]
            # delete, from the center of mass calculus, keypoints where x and/or y coordinate is zero 
            for i, idx in enumerate(keypoints_for_center_of_mass):
                if(skeleton[idx*2]==0. and skeleton[(idx*2)+1]==0.):  # IF X OR Y OF THE KEYPOINT IDX IS ZERO
                    keypoints_for_center_of_mass[i] = None      # Set the index to None to be removed later
            
            keypoints_for_center_of_mass = [x for x in keypoints_for_center_of_mass if x!=None]
            # calculate center of mass based on the body bust (shoulders, neck, hip)
            keypoints_x = skeleton[::2]  # x  - start at the beginning at take every second item
            keypoints_y = skeleton[1::2] # y - start at second item and take every second item
            center_of_mass_x = sum(take(keypoints_x, keypoints_for_center_of_mass, axis=0))/len(keypoints_for_center_of_mass)
            center_of_mass_y = sum(take(keypoints_y, keypoints_for_center_of_mass, axis=0))/len(keypoints_for_center_of_mass)
            centers_of_mass.append((int(center_of_mass_x), int(center_of_mass_y)))

        return centers_of_mass


    def _predict_windows(self, windows, unprocessed_windows):
        HAS_FALLEN = False
        # Make inference with the LSTM-AE    
        for i, input_window in enumerate(windows):
            input_window = expand_dims(input_window, axis=0)
            predicted_window = self.model.predict(input_window)
            loss = linalg.norm(predicted_window-input_window)
            # if at the i-th frame at least one window is detected as a fall scene, we return true
            if loss>self.fall_threshold:
                HAS_FALLEN = True
            self.predictions.append((loss, self._get_centers_of_mass(unprocessed_windows[i])))
        
        return HAS_FALLEN


    def _write_output(self, skeletons, image, losses, past_datum):
        for idx, skeleton in enumerate(skeletons):
            # Drawing the center of mass
            cv2.circle(image, (int(skeleton[0][0]), int(skeleton[0][1])), radius=5, color=[0,0,0], thickness=3)
            # Drawing the circle around the center of mass with a pre-calculated radius
            cv2.circle(image, (int(skeleton[0][0]), int(skeleton[0][1])), radius=skeleton[0][2], color=[0,0,0], thickness=3)

            for loss, center_of_mass in losses:
                if loss>self.fall_threshold:
                    color_info = (0,0,200) # Red
                    #cv2.putText(image, "Loss = {:.2f}".format(loss), (center_of_mass[0]+5, center_of_mass[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, color=(255,255,0), thickness=3)
                    cv2.putText(image, "Status = Fallen".format(loss), (center_of_mass[0]+5, center_of_mass[1]+30), cv2.FONT_HERSHEY_SIMPLEX, 1, color=color_info, thickness=3)
                else:
                    color_info = (0,150,0) # Green
                    #cv2.putText(image, "Loss = {:.2f}".format(loss), (center_of_mass[0]+5, center_of_mass[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, color=(255,255,0), thickness=3)
                    cv2.putText(image, "Status = Normal".format(loss), (center_of_mass[0]+5, center_of_mass[1]+30), cv2.FONT_HERSHEY_SIMPLEX, 1, color=color_info, thickness=3)

                cv2.circle(image, (center_of_mass[0], center_of_mass[1]), radius=1, color=[100,200,200], thickness=8)

            for joint in skeleton[1:][0]:
                cv2.circle(image, (int(joint[0]), int(joint[1])), radius=2, color=[0,255,0], thickness=2)
        
        # Drawing in the whole frame the skeletons detected by openpose
        cv2.imshow("OpenPose Output", image)
        # cv2.imwrite("/app/fall_detection/output/{}.jpg".format(str(self.iteration-74)), image)
        # output_json = mapper.generate_json(vision, past_datum, (self.iteration-74), dict(), False)
        # with open("/app/fall_detection/output/{}.json".format(str(self.iteration-74)), "w") as f:
        #     f.write(output_json)
        cv2.waitKey(1)


    def run(self):
        
        try:

            while True:
                # Acquisisci i frame dalla camera selezionata
                depth_image, color_image = vision.get_frames()

                if depth_image is None and color_image is None:
                    print("Frame non validi ricevuti. Continuo...")
                    continue

                # Normalizza il frame di profondità per la visualizzazione
                normalized_depth = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX)
                depth_colormap = cv2.applyColorMap(normalized_depth.astype(np.uint8), cv2.COLORMAP_JET)

                if color_image is not None:
                    # Passa il frame BGR a OpenPose
                    datum = opUtils.process_frame(color_image)
                    output_frame = datum.cvOutputData
                else:
                    output_frame = np.zeros_like(depth_colormap)

                # Get currently generated windows and skeletons in the current frame (for printing/drawing purposes)
                windows, unprocessed_windows, skeletons = self.tracker.run(datum.poseKeypoints)

                # If there are windows to predict
                if len(windows) != 0:
                    HAS_FALLEN = self._predict_windows(windows, unprocessed_windows)

                # Create a buffer of frames/skeletons which holds data waiting to be inputted to the neural model
                self.buffer_frames.append((skeletons, color_image.copy(), datum))

                # Set the delay for printing to 75 frames (3 sec if sampled at 25fps)
                if len(self.buffer_frames) == 75:
                    past_skeletons, past_image, past_datum,  = self.buffer_frames.pop(0)
                    past_predictions = []
 
                    # Gather current predicted skeletons for drawing purposes
                    for past_loss, past_centers_of_mass in self.predictions:
                        past_predictions.append((past_loss, past_centers_of_mass.pop(0)))
                    
                    # Delete empty cells
                    for i, (_,past_centers_of_mass) in reversed(list(enumerate(self.predictions))):
                        if len(past_centers_of_mass)==25:
                            del self.predictions[i]
                    
                    # Draw current frame skeletons
                    self._write_output(past_skeletons, past_image, past_predictions, past_datum)
                
                self.iteration += 1
        
        
        except Exception as e:
            print(e)
            sys.exit(-1)


if __name__ == "__main__":
    engine = FallEngine()
    engine.run()