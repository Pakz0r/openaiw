from numpy import expand_dims, linalg
from .Tracker import Tracker
from .Model import create_model
from .Drawer import *

class FallEngine:
    def __init__(self,  models_root='/fall_detection/models/',):
        self.fall_threshold = 4

        # Load LSTM-AE model
        self.model = create_model()
        self.model.load_weights(f"{models_root}model.h5")

        # Initialize Tracker object
        self.tracker = Tracker(counter_threshold=10)

    def _predict_windows(self, windows):
        HAS_FALLEN = False
        # Make inference with the LSTM-AE    
        for i, input_window in enumerate(windows):
            input_window = expand_dims(input_window, axis=0)
            predicted_window = self.model.predict(input_window)
            loss = linalg.norm(predicted_window-input_window)
            # if at the i-th frame at least one window is detected as a fall scene, we return true
            if loss>self.fall_threshold:
                HAS_FALLEN = True
        
        return HAS_FALLEN


    def run(self, poseKeypoints):
        # Get currently generated windows and skeletons in the current frame (for printing/drawing purposes)
        windows = self.tracker.run(poseKeypoints)
        # If there are windows to predict
        HAS_FALLEN = self._predict_windows(windows)
        return HAS_FALLEN