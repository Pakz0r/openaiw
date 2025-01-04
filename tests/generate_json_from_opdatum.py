import os
import json

# https://github.com/dennewbie/AI_Watch_A1/blob/109c5182421960c0a24a0d65d2c63dff996066a4/src/AI_Watch_A1/Managers/OutputManagers/OutputManagerJSON.cpp#L71

def calculate_face_rotation(datum):
    """
    Calcola la rotazione del volto (pitch, roll, yaw) se i dati del volto sono disponibili nel datum.
    """
    # Questo è un esempio generico: sostituisci con la logica corretta per il calcolo delle rotazioni se i dati del volto sono disponibili.
    pitch, roll, yaw = 0.0, 0.0, 0.0
    # Calcolo personalizzato potrebbe essere basato sui keypoints del volto.
    return {"pitch": pitch, "roll": roll, "yaw": yaw}

def generate_json_from_datum(datum, frame_id, thing_id="default:thingId"):
    """
    Genera un JSON nello schema richiesto a partire da un oggetto datum di OpenPose.
    """
    result = {
        "ID_Frame": frame_id,
        "People": [],
        "thingId": thing_id
    }

    if datum.poseKeypoints is not None:
        for person_id, person_keypoints in enumerate(datum.poseKeypoints):
            person_data = {
                "face_rotation": calculate_face_rotation(datum),
                "personID": person_id,
                "skeleton": []
            }

            for point_id, keypoint in enumerate(person_keypoints):
                x, y, confidence = keypoint[0], keypoint[1], keypoint[2]
                person_data["skeleton"].append({
                    "pointID": point_id,
                    "x": x,
                    "y": y,
                    "z": 0.0,  # OpenPose non fornisce Z di default, usa 0.0 o un calcolo personalizzato
                    "confidence": confidence
                })

            result["People"].append(person_data)

    return json.dumps(result, indent=4)

def save_json_to_file(json_data, frame_id, output_dir):
    """
    Salva il JSON generato in un file nel percorso specificato.
    """
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, f"frame{frame_id}_skeletonsPoints3D.json")
    with open(file_path, "w") as f:
        f.write(json_data)
    print(f"File salvato in: {file_path}")

# Esempio di utilizzo con un oggetto datum simulato
class MockDatum:
    def __init__(self):
        self.poseKeypoints = [
            [
                [0.5, 0.6, 0.9],  # Keypoint 0
                [0.4, 0.5, 0.8],  # Keypoint 1
                [0.3, 0.4, 0.7]   # Keypoint 2
                # Aggiungi più keypoints se necessario
            ]
        ]

# Simula un oggetto datum
datum = MockDatum()
frame_id = 0
thing_id = "digitaltwin:Laboratorio_Corridoio:1"
output_directory = "output"  # Percorso personalizzabile per il salvataggio

# Genera il JSON
output_json = generate_json_from_datum(datum, frame_id, thing_id)
save_json_to_file(output_json, frame_id, output_directory)
