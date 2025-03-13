import torch
from .FDTransformer import FDTransformer
from .FDTracker import FDTracker
from numpy import linalg, expand_dims

class FDModel:
    def __init__(self, models_root='/falldetection/models/', device='cpu'):
        self.fall_threshold = 0.006
        # Initialize the Transformer model
        self.device = torch.device(device)
        self.model = FDTransformer(embed_dim=38, input_dim=38, num_heads=2, num_encoder_layers=1, num_decoder_layers=1).to(self.device)
        self.model.load_state_dict(torch.load(f"{models_root}model.pth", map_location=self.device))
        self.model.eval()
        # Initialize Tracker object
        self.tracker = FDTracker(counter_threshold=10)

    def _predict_windows(self, windows):
        print("Running FallModel inference...")
        HAS_FALLEN = False

        # Make inference with the LSTM-AE    
        for i, input_window in enumerate(windows):
            input_window = expand_dims(input_window, axis=0)

            input_window = torch.tensor(input_window, dtype=torch.float32).to(self.device)  # Converti in tensor
            tgt = torch.zeros_like(input_window).to(self.device)

            with torch.no_grad():  # Disabilita il gradiente per inferenza
                predicted_window = self.model(input_window, tgt, self.device)

            loss = linalg.norm(predicted_window-input_window)

            # if at the i-th frame at least one window is detected as a fall scene, we return true
            if loss > self.fall_threshold:
                HAS_FALLEN = True
        
        return HAS_FALLEN
    
    def detect_fall(self, datum):
        print("Running fall detection...")
        # Get currently generated windows and skeletons in the current frame (for printing/drawing purposes)
        windows = self.tracker.run(datum.poseKeypoints)
        
        # If there are no windows to predict return false
        if len(windows) == 0:
            return False
        
        return self._predict_windows(windows)