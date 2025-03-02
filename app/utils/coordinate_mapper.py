import cv2
import json
import numpy as np
from vision.drivers.ivision import IVision

class CoordinateMapper:
    def __init__(self, thing_id):
        self.thing_id = thing_id
        self.model_points = np.array([  # Modello 3D della testa
            (0.0, 0.0, 0.0),            # Nose tip
            (0.0, -330.0, -65.0),       # Chin
            (-225.0, 170.0, -135.0),    # Left eye left corner
            (225.0, 170.0, -135.0),     # Right eye right corner
            (-150.0, -150.0, -125.0),   # Left Mouth corner
            (150.0, -150.0, -125.0)     # Right mouth corner
        ])

    # Calcola la rotazione del volto (pitch, roll, yaw)
    def calculate_face_rotation(self, face_keypoints):
        face_keypoints = np.array(face_keypoints).reshape(-1, 3)  # Converti in matrice (N,3)
        
        # Definire punti chiave per il modello della testa
        image_points = np.array([
            face_keypoints[30][:2],  # Nose tip
            face_keypoints[8][:2],   # Chin
            face_keypoints[36][:2],  # Left eye left corner
            face_keypoints[45][:2],  # Right eye right corner
            face_keypoints[48][:2],  # Left mouth corner
            face_keypoints[54][:2]   # Right mouth corner
        ], dtype="double")

        # Parametri della camera
        size = (640, 480)  # Dimensione immagine
        focal_length = size[1]
        center = (size[1] / 2, size[0] / 2)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype="double")

        # Nessuna distorsione
        dist_coeffs = np.zeros((4, 1))

        # Calcolo della rotazione
        success, rotation_vector, translation_vector = cv2.solvePnP(self.model_points, image_points, camera_matrix, dist_coeffs)

        # Convertire la rotazione in angoli (Pitch, Roll, Yaw)
        rmat, _ = cv2.Rodrigues(rotation_vector)
        pitch, yaw, roll = np.degrees(np.arctan2(rmat[2, 1], rmat[2, 2])), np.degrees(np.arctan2(-rmat[2, 0], np.sqrt(rmat[2, 1]**2 + rmat[2, 2]**2))), np.degrees(np.arctan2(rmat[1, 0], rmat[0, 0]))
        return {"pitch": pitch, "roll": roll, "yaw": yaw}

    # Genera il JSON a partire dal datum di OpenPose
    def generate_json(self, vision: IVision, datum, depth_image, frame_id: int):
        result = {
            "ID_Frame": frame_id,
            "People": [],
            "thingId": self.thing_id
        }

        if datum.poseKeypoints is not None:
            for person_id, person_keypoints in enumerate(datum.poseKeypoints):
                person_data = {
                    "face_rotation": self.calculate_face_rotation(datum.faceKeypoints[person_id]),
                    "personID": person_id,
                    "skeleton": []
                }

                for point_id, keypoint in enumerate(person_keypoints):
                    u, v, confidence = keypoint[0], keypoint[1], keypoint[2]
                    x, y, z = vision.convert_point_to_world(u, v, depth_image[int(v), int(u)])
                    person_data["skeleton"].append({
                        "pointID": point_id,
                        "x": float(x),
                        "y": float(y),
                        "z": float(z),
                        "confidence": float(confidence)
                    })

                result["People"].append(person_data)

        return json.dumps(result, indent=4)