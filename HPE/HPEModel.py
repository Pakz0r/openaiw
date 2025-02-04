from .model import HPEnet
import torch
from tqdm import tqdm
from facenet_pytorch import MTCNN
from torchvision import transforms
from scipy.spatial.transform import Rotation as R
import numpy as np 
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
from .utils import convert_class_cube_to_euler_angles


class HPEModel:
    def __init__(self, model_path='HPE/models/model.pt', device='cpu'):
        self.device = device
        # Initialize the model
        self.model = HPEnet().to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=torch.device(self.device)))
        self.model.eval()
        self.mtcnn = MTCNN(keep_all=True, post_process=False, device='cpu')
        self.mean = torch.load('HPE/mean.pt')
        self.std = torch.load('HPE/std.pt')  

    def detect_faces(self, image):
        print("Running face detection...")
        # Detect face
        boxes, _ = self.mtcnn.detect(image)
        boxes_centroids = []
        faces = []

        # If no boxes have been detected return
        if boxes is None:
            return None, None

        # Add margin to each box, calculate centroids and crop the face image
        for i in range(len(boxes)):
            # Add margin while safe checking
            margin=50
            boxes[i][0] = max(0, boxes[i][0] - margin)
            boxes[i][1] = max(0, boxes[i][1] - margin)
            boxes[i][2] = min(image.width, boxes[i][2] + margin)
            boxes[i][3] = min(image.height, boxes[i][3] + margin)
            boxes_centroids.append([int((boxes[i][0] + boxes[i][2])/2), int((boxes[i][1] + boxes[i][3]) /2)])

            # Crop the face using boxes
            faces.append(image.crop(boxes[0]))

        return faces, boxes_centroids


    def predict(self, face):
        print("Running inference...")

       # Preprocess the image
        transform = transforms.Compose([ 
            transforms.PILToTensor(),
            transforms.Resize((200, 200)),
        ])
        face_tensor = transform(face)
        face_tensor = face_tensor.permute(1, 2, 0)

        # Standardize the tensor
        face_tensor = (face_tensor - self.mean) / self.std
        face_tensor = face_tensor.permute(2, 0, 1)
        face_tensor = face_tensor.type(torch.float32)

        pitch, yaw, roll = None, None, None

        with torch.inference_mode():
            r1, r2, r3, _ = self.model(face_tensor.unsqueeze(0))
            r1, r2, r3 = r1.squeeze().numpy(), r2.squeeze().numpy(), r3.squeeze().numpy()
            print(r1)

            rotation_matrix = np.array([r1, r2, r3])
            pitch, yaw, roll = R.from_matrix(rotation_matrix).as_euler('xzy', degrees=True)
        
        return pitch, yaw, roll
