import json
from vision.drivers.ivision import IVision

class CoordinateMapper:
    def __init__(self, thing_id):
        self.thing_id = thing_id

    # Genera il JSON a partire dal datum di OpenPose
    def generate_json(self, vision: IVision, datum, frame_id: int, face_rotations: dict, has_any_fallen: bool):
        result = {
            "ID_Frame": frame_id,
            "People": [],
            "thingId": self.thing_id,
            "Has_Fallen": has_any_fallen
        }

        if datum.poseKeypoints is not None:
            for person_id, person_keypoints in enumerate(datum.poseKeypoints):
                face_rotation = face_rotations[person_id] if person_id in face_rotations else { "pitch": 0.0, "yaw": 0.0, "roll": 0.0 }

                person_data = {
                    "face_rotation": face_rotation,
                    "personID": person_id,
                    "skeleton": []
                }

                for point_id, keypoint in enumerate(person_keypoints):
                    u, v, confidence = keypoint[0], keypoint[1], keypoint[2]
                    x, y, z = vision.convert_point_to_world(u, v)

                    # https://github.com/IntelRealSense/librealsense/blob/master/doc/d435i.md#sensor-origin-and-coordinate-system
                    # Converti le coordinate da right-hand a left-hand per Unity
                    # A questa conversione va aggiunto un offset di rotazione di 180° sulla camera Unity
                    # In modo da ottenere che le bone vengano rappresentate correttamente
                    x, y, z = x, -y, z

                    person_data["skeleton"].append({
                        "pointID": point_id,
                        "x": float(x),
                        "y": float(y),
                        "z": float(z),
                        "confidence": float(confidence)
                    })

                result["People"].append(person_data)

        return json.dumps(result, indent=4)