import json
from vision.drivers.ivision import IVision

class CoordinateMapper:
    def __init__(self, thing_id, room):
        self.thing_id = thing_id
        self.room = room

    def transform_width(self, x = 0.0) -> float:
        env_width = abs(self.room["min_width"]) + abs(self.room["max_width"])
        env_width_WP = abs(self.room["min_width_WP"]) + abs(self.room["max_width_WP"])
        return (((x - self.room["min_width_WP"]) / env_width_WP) * env_width) + self.room["min_width"] + self.room["world_x_origin"]
    
    def transform_height(self, y = 0.0) -> float:
        newY = -y + self.room["max_height_WP"] - self.room["height_offset"]
        return max(0, min(newY, self.room["max_height"])) # clamp height value
    
    def transform_depth(self, z = 0.0) -> float:
        return -(abs(z) + self.room["world_z_origin"] + self.room["backwall_distance"])
    
    def transform(self, x = 0.0, y = 0.0, z = 0.0):
        return self.transform_width(x), self.transform_height(y), self.transform_depth(z)

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

                    if vision.wrapper == "realsense":
                        x, y, z = self.transform(x, y, z)

                    person_data["skeleton"].append({
                        "pointID": point_id,
                        "x": float(x),
                        "y": float(y),
                        "z": float(z),
                        "confidence": float(confidence)
                    })

                result["People"].append(person_data)

        return json.dumps(result, indent=4)