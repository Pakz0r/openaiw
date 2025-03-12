import cv2
import json
import math
import torch
import numpy as np
from tqdm import tqdm

def find_closest_centroid(point, points_list):
    min_distance = float('inf')
    closest_index = None
    for i, p in enumerate(points_list):
        distance = math.sqrt((point[0] - p[0])**2 + (point[1] - p[1])**2)
        if distance < min_distance:
            min_distance = distance
            closest_index = i
    return closest_index

def fetch_faces_keypoints_from_datum(datum):
    # Important landmarks to extract (face points)
    FACE_KEYPOINTS = [
        0,
        15,
        16,
        17,
        18
    ]
    
    # List to store the facial keypoints
    faces_keypoints = []
    
    # Checks if there are persons into the datum
    if datum.poseKeypoints is not None:
        for person_keypoints in datum.poseKeypoints:
            # Creating an empty face list to append the face keypoins
            face = []
            
            # Extracting for the ith person keypoints converted to tuple only the facial keypoints
            for kp in FACE_KEYPOINTS:
                # Extracting for the ith person keypoints converted to tuple only the facial keypoints
                x, y, conf = person_keypoints[kp]
                # Append the corrispondent keypoint in the faces keypoints list 
                face.append((x, y, conf))

            # Appending the new created face to the faces kps array 
            faces_keypoints.append(face)
            
    # Once finished, return 
    return faces_keypoints

def fetch_faces_keypoints(id):
    json_path = "build/op-output/op/{}_Color_keypoints.json".format(id)
   
    # Try to read the JSON file till it's fully written
    data = None
    while data is None:
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
            # Proceed with data processing
        except Exception as e:
            pass

    people = data["people"]

    # Important landmarks to extract (face points)
    FACE_KEYPOINTS = [
        0,
        15,
        16,
        17,
        18
    ]

    # From each person in people, extract the dictionary containing the facial keypoints 
    # and create new a 3-tuple (x,y,conf) dictionary instead of the flattened one 
    faces_keypoints = []
    for person in people:
        # Extracting the keypoints and making a tuple out of it
        kps = person["pose_keypoints_2d"]
        person_keypoints = [(kps[i], kps[i+1], kps[i+2]) for i in range(0, len(kps), 3)]

        # Creating an empty face list to append the face keypoins
        face = []

        # Extracting for the ith person keypoints converted to tuple only the facial keypoints
        for kp in FACE_KEYPOINTS:
            # Append the corrispondent keypoint in the faces keypoints list 
            face.append(person_keypoints[kp])
        # Appending the new created face to the faces kps array 
        faces_keypoints.append(face)

    # Once finished, return 
    return faces_keypoints


def compute_center(keypoints):
    # For the first (x coord) and the second (y coord) we compute the average to extract the 
    # center of mass of the keypoints 
    return [int(sum(ele) / len(keypoints)) for ele in zip(*keypoints)][:2]


def draw_faces_keypoints(image_id): 
    # Loading the image from the id 
    file_path = "build/rs-images/hpe/{}".format(image_id) + "_Color.png"

    # Load the image 
    image = cv2.imread(file_path)

    # Fetching the facial keypoints if any in the image 
    faces_keypoints = fetch_faces_keypoints(image_id)

    # For the keypoints retrieved, draw some circle on them 
    for face in faces_keypoints:
        for keypoints in face:        
            print(keypoints)
            # Draw the circle on the image 
            cv2.circle(image, (int(keypoints[0]), int(keypoints[1])), 2, (0,255,0), 2)

        # Draw the center of mass of the facial keypoints
        cv2.circle(image, compute_center(face), 2, (0,0,255), 3)

    # Draw the image 
    cv2.imshow("faces kps", image)
    cv2.waitKey(0)

def convert_class_cube_to_euler_angles(class_cube):
    # Convert class cube to Euler angles
    class_labels_list = []
    
    with open("serialized_dataset/classes.json", 'r') as json_file:
        class_labels_list = json.load(json_file)

    print(class_cube)
    return class_labels_list[class_cube]


def calculate_class_weights(dataset, cached=True):
    print("Calculating class weights...")

    if cached:
        # Load class weights from disk
        class_weights = torch.load("serialized_dataset/class_weights.pt")
        return class_weights
    
    num_classes = 0
    with open("serialized_dataset/classes.json", 'r') as json_file:
        class_labels_list = json.load(json_file)
        num_classes = len(class_labels_list)

    # Get the number of samples for each class
    class_samples = np.zeros(num_classes)
    for _, _, class_label in tqdm(dataset):
        class_samples[class_label] += 1
    
    # Calculate the class weights
    class_weights = len(dataset) / class_samples
    # Store class weights to disk as a tensor
    torch.save(torch.tensor(class_weights), "serialized_dataset/class_weights.pt")

    return torch.tensor(class_weights)


def convert_angles_from_AIWatch(angles):
    if angles[0] > 180:
        angles[0] = angles[0] - 360
    angles[0] = -angles[0]
    if angles[1] > 180:
        angles[1] = angles[1] - 360
    angles[1] = -angles[1]
    if angles[2] > 180:
        angles[2] = angles[2] - 360
    angles[2] = -angles[2]

    return angles

    axes_points = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0]
    ], dtype=np.float64)
    axes_points = rotation_matrix @ axes_points
    axes_points = (axes_points[:2, :] * 50).astype(int)
    axes_points[0,0] = -axes_points[0,0] # Invert x axis direction
    axes_points[0, :] = axes_points[0, :] + img.shape[0] / 2
    axes_points[1, :] = axes_points[1, :] + img.shape[1] / 2
    
    new_img = img.copy()
    
    cv2.line(new_img, tuple(axes_points[:, 3].ravel()), tuple(axes_points[:, 0].ravel()), (255, 0, 0), 3)
    cv2.putText(new_img, 'x', tuple(axes_points[:, 0].ravel()), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 0, 0), 1)
    
    cv2.line(new_img, tuple(axes_points[:, 3].ravel()), tuple(axes_points[:, 1].ravel()), (0, 255, 0), 3)
    cv2.putText(new_img, 'y', tuple(axes_points[:, 1].ravel()), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 255, 0), 1)
    
    cv2.line(new_img, tuple(axes_points[:, 3].ravel()), tuple(axes_points[:, 2].ravel()), (0, 0, 255), 3)
    cv2.putText(new_img, 'z', tuple(axes_points[:, 2].ravel()), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 0, 255), 1)

    return new_img