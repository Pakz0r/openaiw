import cv2
import json
import random
import math
import os
import numpy as np
import time
from PIL import Image
from HPE.utils import plot_rotation_3D, fetch_faces_keypoints, compute_center, find_closest_centroid
from HPE.HPEModel import HPEModel

# Class that wraps some of the AI-WATCH modules in python, 
# such as HPE and FALL detection modules
class AIModels:

    def __init__(self, TODO_CONF=None):
        self.TODO_CONF = TODO_CONF
        self.hpe_model = HPEModel()
        self.fall_model = None
        self.frame_id = 0
        self.__fetch_starting_fid()


    # Function that fetch the lowest frameID from the image folders
    def __fetch_starting_fid(self):

        # Wait for at least one frame written in the folder
        while(not len(os.listdir("build/rs-images/hpe"))):
            continue 
            
        # fetch the filenames as a list
        file_list = os.listdir("build/rs-images/hpe")

        # extract only the ids of the frames 
        id_list = [int(id.split("_")[0]) for id in list(file_list)]

        # sort them ascendingly and fetch the first element 
        init_id = sorted(id_list)[0]

        # assign to the internal attribute 
        self.frame_id = init_id

    
    # Function that fetch the highest frameID from the image folders
    def __fetch_latest_fid(self):
            
        while(True):

            # Wait for at least one frame written in the folder
            while(not len(os.listdir("build/rs-images/hpe"))):
                continue 

            # fetch the filenames as a list
            file_list = os.listdir("build/rs-images/hpe")

            # extract only the ids of the frames 
            id_list = [int(id.split("_")[0]) for id in list(file_list)]

            # sort them ascendingly and fetch the first element 
            last_id = sorted(id_list)[-1]

            # assign to the internal attribute 
            if self.frame_id != last_id:
                self.frame_id = last_id
                break 


    def __run_hpe(self, image):

        # Detect faces
        faces, detected_faces_centroids = self.hpe_model.detect_faces(image)

        # If there are no faces, return
        if faces is None:
            print("No faces detected")
            # Write empty angles in the output file in case there are no faces detected
            with open(f"build/models-output/hpe/{self.frame_id}_poses.json", "w") as f:
                angles = {"angles": [{"pitch":None, "yaw":None, "roll":None}]}
                json.dump(angles, f)
                print(f"File {self.frame_id}_poses.json written to disk with angles")
            return
       
        # Fetch openpose faces centroids in order to match them with detected ones
        openpose_faces_centroids = []
        openpose_faces_keypoints = fetch_faces_keypoints(self.frame_id)
        for face_keypoints in openpose_faces_keypoints:
            openpose_faces_centroids.append(compute_center(face_keypoints))
        
        # Build a list of sorted index such that each face detected by the AIModel is ordered following the openpose ordering schema
        sorted_indices = []
        for openpose_face_centroid in openpose_faces_centroids:
            sorted_indices.append(find_closest_centroid(openpose_face_centroid, detected_faces_centroids))

        # Cycle over the faces, run the HPE and append the output to the angles dictionary
        angles = {"angles": []}
        for index in sorted_indices:
            face = faces[index]
            # run hpe model inference
            pitch, yaw, roll = self.hpe_model.predict(face)
            print(f"Pitch: {pitch}, Yaw:{yaw}, Roll:{roll}")
            angles["angles"].append({"pitch":pitch, "yaw":yaw, "roll":roll})

        # once the hpe outputs are ready, dump them on a json file
        with open(f"build/models-output/hpe/{self.frame_id}_poses.json", "w") as f:
            json.dump(angles, f)
            print(f"File {self.frame_id}_poses.json written to disk")
            

    # main execution loop for ai models
    def run(self):

        # call models predictions on the current frame_id (class attribute)
        while(True):

            # Read current RGB frame
            print(f"Opening image at build/rs-images/hpe/{self.frame_id}_Color.png")

            # Try to read it till it's fully written
            image = None
            while image is None:
                try:
                    image = Image.open(f"build/rs-images/hpe/{self.frame_id}_Color.png").convert("RGB")
                    break
                except:
                    pass

            # HPE prediction
            self.__run_hpe(image)

            # Fetch the latest file id from the input images folder such that it will be predicted next
            self.__fetch_latest_fid()



if __name__ == "__main__":
    models = AIModels()
    models.run()