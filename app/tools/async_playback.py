import pyrealsense2 as rs
import numpy as np
import shutil
import json
import sys
import cv2
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from vision import Vision
from utils import CoordinateMapper, SkeletonVisualizer

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

def save_json_to_file(json_data, frame_id, output_dir):
    """Salva il JSON generato in un file nel percorso specificato."""
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, f"frame{frame_id}_skeletonsPoints3D.json")
    with open(file_path, "w") as f:
        f.write(json_data)
    print(f"Nuovo frame salvato in: {file_path}")

class Datum(object):
    def __init__(self, frame):
        self.frameId = frame['frame']
        self.poseKeypoints = frame['poseKeypoints']

# Programma principale
def main(input_path='input', processing_path='processing', output_path='output', bag_path='recording.bag'):
    try:
        # Pulisci il contenuto della cartella di output
        os.makedirs(output_path, exist_ok=True)
        clear_folder(output_path)

        print("Lettura delle informazioni di OpenPose...")
        openpose_path = os.path.join(processing_path, 'openpose_metadata.json')
        with open(openpose_path, 'r') as f:
            frames = json.load(f)

        print("Lettura delle informazioni di HPE...")
        hpe_path = os.path.join(processing_path, 'hpe_metadata.json')
        with open(hpe_path, 'r') as f:
            faces = json.load(f)

        print("Inizializzazione del Coordinate Mapper...")
        mapper = CoordinateMapper("digitaltwin:Laboratorio_Corridoio:1")

        print("Inizializzazione del Visualizzatore dello scheleto...")
        visualizer = SkeletonVisualizer()

        print("Inizializzazione del modulo di Vision...")
        bag_path = os.path.join(input_path, bag_path)
        vision = Vision.initialize("realsense")
        playback = vision.start_playback(bag_path)

        print("Avvio del processo di HPE.")

        frame_count = 0

        while True:
            ## STEP 1 (FRAMES ACQUISITION)
            # Acquisisci i frame dalla camera selezionata
            _, color_image = vision.get_frames()

            if color_image is None:
                print("Frame non validi ricevuti. Continuo...")
                continue

            ## STEP 2 (GET OPENPOSE DATUM FROM FILE)
            datum = Datum(frames[frame_count])

            ## STEP 3 (GET HPE DATA FROM FILE)
            face_rotations = {}
            
            for id in faces[frame_count]['face_rotations']:
                face_rotations[int(id)] = faces[frame_count]['face_rotations'][id]

             ## STEP 4 (FALL DETECTION)
            HAS_FALLEN = False

             ## STEP 5 (IDENTIFICATION)
            ## TO BE ADDED HERE

            ## STEP 6 (GENERATE JSON)
            output_json = mapper.generate_json(vision, datum, frame_count, face_rotations, HAS_FALLEN)
            save_json_to_file(output_json, frame_count, output_path)

            ## STEP 8 (SHOW OUTPUTS)
            cv2.imshow("Input Frame", color_image)

            ## STEP 9 (RENDER RESULTS)
            visualizer.load_data_from_json(output_json)
            visualizer.render_frame()

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

        cv2.destroyAllWindows()

        print(f"Processo completato.")

if __name__ == "__main__":
    main()