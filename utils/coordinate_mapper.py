import json

#https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/include/openpose/core/datum.hpp

class CoordinateMapper:
    def __init__(self, room):
        self.room = room

    def transform_width(self, value = 0.0):
        env_width = abs(self.room["min_width"]) + abs(self.room["max_width"])
        env_width_WP = abs(self.room["min_width_WP"]) + abs(self.room["max_width_WP"])
        return (((value - self.room["min_width_WP"]) / env_width_WP) * env_width) + self.room["min_width"]
    
    def transform_height(self, value = 0.0):
        newValue = -value + self.room["max_height_WP"] - self.room["height_offset"]
        return max(0, min(newValue, self.room["max_height"])) # clamp height value
    
    def rescale(self, x = 0.0, y = 0.0, z = 0.0):
        return (self.transform_width(x) - abs(self.room["world_x_origin"])), (self.transform_height(y)), (-(abs(z) + abs(self.room["world_z_origin"]) + self.room["backwall_distance"]))

    # Calcola la rotazione del volto (pitch, roll, yaw)
    def calculate_face_rotation(self):
        pitch, roll, yaw = 0.0, 0.0, 0.0
        return {"pitch": pitch, "roll": roll, "yaw": yaw}

    # Genera il JSON a partire dal datum di OpenPose
    def generate_json(self, vision, datum, depth_image, frame_id):
        result = {
            "ID_Frame": frame_id,
            "People": [],
            "thingId": self.room["thing_id"]
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
                    x, y, z = vision.convert_depth_to_world(u, v, depth_image[int(v), int(u)])
                    person_data["skeleton"].append({
                        "pointID": point_id,
                        "x": float(x),
                        "y": float(y),
                        "z": float(z),
                        "confidence": float(confidence)
                    })

                result["People"].append(person_data)

        return json.dumps(result, indent=4)