import json
from openpose import pyopenpose as op

class OPUtils:

    # Configura OpenPose
    def __init__(self, params):
        self.opWrapper = op.WrapperPython()
        self.opWrapper.configure(params)
        self.opWrapper.start()

    # Deinizializza OpenPose
    def __del__(self):
        self.opWrapper.stop()
    
    # Processa i frame con OpenPose
    def process_frame(self, frame):
        datum = op.Datum()
        datum.cvInputData = frame
        self.opWrapper.emplaceAndPop(op.VectorDatum([datum]))
        return datum
    
    # Calcola la rotazione del volto (pitch, roll, yaw)
    def calculate_face_rotation(self):
        pitch, roll, yaw = 0.0, 0.0, 0.0
        return {"pitch": pitch, "roll": roll, "yaw": yaw}

    # Genera il JSON a partire dal datum di OpenPose
    def generate_json(self, datum, frame_id, thing_id="default:thingId"):
        result = {
            "ID_Frame": frame_id,
            "People": [],
            "thingId": thing_id
        }

        if datum.poseKeypoints is not None:
            for person_id, person_keypoints in enumerate(datum.poseKeypoints):
                person_data = {
                    "face_rotation": self.calculate_face_rotation(),
                    "personID": person_id,
                    "skeleton": []
                }

                for point_id, keypoint in enumerate(person_keypoints):
                    x, y, confidence = keypoint[0], keypoint[1], keypoint[2]
                    person_data["skeleton"].append({
                        "pointID": point_id,
                        "x": float(x),
                        "y": float(y),
                        "z": float(0.0),
                        "confidence": float(confidence)
                    })

                result["People"].append(person_data)

        return json.dumps(result, indent=4)