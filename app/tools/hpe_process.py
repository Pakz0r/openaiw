from PIL import Image
import pyrealsense2 as rs
import numpy as np
import shutil
import json
import sys
import cv2
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from vision import Vision
from HPE import HPEModel
from HPE.utils import compute_center, fetch_faces_keypoints_from_datum, find_closest_centroid

def serialize(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: serialize(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [serialize(i) for i in obj]
    return obj

def clear_folder(folder_path):
    """Cancella tutto il contenuto della cartella specificata."""
    if os.path.exists(folder_path):
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)  # Rimuovi file o collegamenti simbolici
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)  # Rimuovi directory
            except Exception as e:
                print(f"Errore durante la rimozione di {file_path}: {e}")
        print(f"I contenuti della cartella '{folder_path}' sono stati rimossi.")
    else:
        print(f"La cartella '{folder_path}' non esiste.")

class Datum(object):
    def __init__(self, frame):
        self.frameId = frame['frame']
        self.poseKeypoints = frame['poseKeypoints']

# Programma principale
def main(input_path='input', output_path='processing', bag_path='recording.bag', metadata_path='hpe_metadata.json'):
    try:
        # Pulisci il contenuto della cartella di output
        os.makedirs(output_path, exist_ok=True)
        metadata_path = os.path.join(output_path, metadata_path)

        print("Lettura delle informazioni di OpenPose...")
        openpose_path = os.path.join(output_path, 'openpose_metadata.json')
        with open(openpose_path, 'r') as f:
            frames = json.load(f)

        print("Inizializzazione di HPE...")
        hpe_model = HPEModel('./app/HPE/models/', 'cpu')

        print("Inizializzazione del modulo di Vision...")
        bag_path = os.path.join(input_path, bag_path)
        vision = Vision.initialize("realsense")
        playback = vision.start_playback(bag_path)

        print("Avvio del processo di HPE.")

        frame_count = 0
        metadata = []

        while True:
            ## STEP 1 (FRAMES ACQUISITION)
            # Acquisisci i frame dalla camera selezionata
            color_image = vision.get_color_frame()

            if color_image is None:
                print("Frame non validi ricevuti. Continuo...")
                continue

            ## STEP 2 (GET OPENPOSE DATUM FROM FILE)
            datum = Datum(frames[frame_count])

            ## STEP 3 (RUN HPE MODEL)
            faces, detected_faces_centroids = None, None
            face_rotations = {}

            if color_image is not None:
                # Converti color_image in PIL.Image (RGB) e rileva le facce
                img_rgb = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(img_rgb)
                faces, detected_faces_centroids = hpe_model.detect_faces(image)

            # If there are faces
            if faces is not None:
                print("Faces detected")

                # Fetch openpose faces centroids in order to match them with detected ones
                openpose_faces_centroids = []
                openpose_faces_keypoints = fetch_faces_keypoints_from_datum(datum)
                for face_keypoints in openpose_faces_keypoints:
                    openpose_faces_centroids.append(compute_center(face_keypoints))
                
                # Build a list of sorted index such that each face detected by the AIModel is ordered following the openpose ordering schema
                sorted_indices = []
                for openpose_face_centroid in openpose_faces_centroids:
                    sorted_indices.append(find_closest_centroid(openpose_face_centroid, detected_faces_centroids))

                # Cycle over the faces, run the HPE and append the output to the angles dictionary
                for index in sorted_indices:
                    face = faces[index]
                    # run hpe model inference
                    pitch, yaw, roll = hpe_model.predict(face)
                    face_rotations[index] = {"pitch":pitch, "yaw":yaw, "roll":roll}

            metadata.append({
                "frame": frame_count,
                "face_rotations": serialize(face_rotations)
            })

            print(f"Elaborazione del frame {frame_count} completata.")

            if playback.current_status() != rs.playback_status.playing:
                print("Lettura del file di registrazione terminata.")
                print("Generazione del file di metadati...")
                break

            # Aumenta il contatore dei frame
            frame_count = frame_count + 1

    finally:
        if "vision" in locals():
            del vision

        with open(metadata_path, 'w') as f:
            json.dump(metadata, f)

        print(f"Processo completato.")

if __name__ == "__main__":
    main()