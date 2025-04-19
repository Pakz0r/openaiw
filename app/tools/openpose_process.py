import pyrealsense2 as rs
import numpy as np
import shutil
import json
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from vision import Vision
from utils import OPUtils

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

# Programma principale
def main(input_path='input', output_path='processing', bag_path='recording.bag', metadata_path='openpose_metadata.json'):
    try:
        # Pulisci il contenuto della cartella di output
        metadata_path = os.path.join(output_path, metadata_path)
        os.makedirs(output_path, exist_ok=True)
        clear_folder(output_path)

        print("Inizializzazione di OpenPose...")
        opUtils = OPUtils({
            "model_folder": "./app/models/",
            "model_pose": "BODY_25",
            "net_resolution": "-1x368",
        })

        print("Inizializzazione del modulo di Vision...")
        bag_path = os.path.join(input_path, bag_path)
        vision = Vision.initialize("realsense")
        playback = vision.start_playback(bag_path)

        print("Avvio del processo di pose estimation tramite OpenPose.")

        frame_count = 0
        metadata = []

        while True:
            ## STEP 1 (FRAMES ACQUISITION)
            # Acquisisci i frame dalla camera selezionata
            color_image = vision.get_color_frame()

            if color_image is None:
                print("Frame non validi ricevuti. Continuo...")
                continue

            ## STEP 2 (OPENPOSE PROCESS)
            # Processa il frame BGR con OpenPose
            datum = opUtils.process_frame(color_image)
            metadata.append({
                "frame": frame_count,
                "poseKeypoints": serialize(datum.poseKeypoints)
            })

            print(f"Elaborazione del frame {frame_count} completata.")

            if playback.current_status() != rs.playback_status.playing:
                print("Lettura del file di registrazione terminata.")
                print("Generazione del file di metadati...")
                break

            # Aumenta il contatore dei frame
            frame_count = frame_count + 1

    finally:
        if "opUtils" in locals():
            del opUtils

        if "vision" in locals():
            del vision

        with open(metadata_path, 'w') as f:
            json.dump(metadata, f)

        print(f"Processo completato.")

if __name__ == "__main__":
    main()