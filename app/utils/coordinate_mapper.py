import json
from vision.drivers.ivision import IVision

class CoordinateMapper:
    def __init__(self, thing_id):
        self.thing_id = thing_id

    # Calcola la rotazione del volto (pitch, roll, yaw)
    def calculate_face_rotation(self):
        pitch, roll, yaw = 0.0, 0.0, 0.0
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
                    "face_rotation": self.calculate_face_rotation(),
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