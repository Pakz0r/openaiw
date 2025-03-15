import os
import shutil
import cv2
import argparse
import numpy as np
import torch
from PIL import Image
from HPE import HPEModel
from HPE.utils import compute_center, fetch_faces_keypoints_from_datum, find_closest_centroid
from falldetection import FDModel
from vision import Vision
from utils import OPUtils, CoordinateMapper

CONFIG = {
    "vision": {
        "driver" : "realsense",
        "dll_directories": "C:\Program Files\OpenNI2\Redist"
    },
    "openpose" : {
        "model_folder": "./app/models/",
        "model_pose": "BODY_25",
        "net_resolution": "-1x368", # Default "-1x368"; AI_Watch "-1x128"
        "hand": False,
        "hand_net_resolution": "224x224", # "Default "368x368" (multiples of 16)"
        "face": False,
        "face_net_resolution": "224x224", # "Default "368x368" (multiples of 16)"
    },
    "room" : {
        "thing_id" : "digitaltwin:Laboratorio_Corridoio:1"
    },
    "HPE" : {
        "model_root" : './app/HPE/models/',
    },
    "Fall" : {
        "model_root" : './app/falldetection/models/',
    },
    "output_dir" : "output"
}

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

def show_results(output_frame, depth_colormap):
    try:
        combined_view = np.hstack((output_frame, depth_colormap))
        cv2.imshow("OpenPose Output + Depth", combined_view)
    except:
        pass

# Programma principale
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("-openni2_dll_directories", type=str, required=False)
    parser.add_argument("-vision_driver", type=str, required=False)
    parser.add_argument("-openpose_model_path", type=str, required=False)
    parser.add_argument("-room_id", type=str, required=False)
    parser.add_argument("-hpe_model_root", type=str, required=False)
    parser.add_argument("-fall_model_root", type=str, required=False)

    args = parser.parse_args()

    if args.vision_driver is not None:
        CONFIG["vision"]["driver"] = args.vision_driver

    if args.openni2_dll_directories is not None and CONFIG["vision"]["driver"] == "openni2":
         CONFIG["vision"]["dll_directories"] = args.openni2_dll_directories

    if args.openpose_model_path is not None:
        CONFIG["openpose"]["model_folder"] = args.openpose_model_path

    if args.room_id is not None:
        CONFIG["room"]["thing_id"] = args.room_id

    if args.hpe_model_root is not None:
        CONFIG["HPE"]["model_root"] = args.hpe_model_root

    if args.fall_model_root is not None:
        CONFIG["Fall"]["model_root"] = args.fall_model_root

    try:
        # Pulisci il contenuto della cartella di output
        clear_folder(CONFIG['output_dir'])

        print("Inizializzazione di OpenPose...")
        opUtils = OPUtils(CONFIG["openpose"])

        print("Inizializzazione del Coordinate Mapper...")
        mapper = CoordinateMapper(CONFIG["room"]["thing_id"])

        print("Recupero informazioni sul Device...")
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        print(f"Inizializzazione del modulo HPE con '{device}'")
        hpe_model = HPEModel(CONFIG["HPE"]['model_root'], device)

        print(f"Inizializzazione del modulo Fall Detection con '{device}'")
        fall_model = FDModel(CONFIG["Fall"]['model_root'], device)

        # Seleziona la camera basandosi sulla configurazione
        print("Inizializzazione del modulo di Vision...")
        vision = Vision.initialize(CONFIG["vision"]["driver"], CONFIG["vision"]["dll_directories"])

        dev_info = vision.get_device_info()
        if dev_info is not None:
            print(f"Collegato con {dev_info.vendor.decode()} {dev_info.name.decode()}")

        print("Applicazione inizializzata con successo. Premere 'q' per uscire.")

        frame_id = -1

        while True:
            ## STEP 1 (FRAMES ACQUISITION)
            # Acquisisci i frame dalla camera selezionata
            depth_image, color_image = vision.get_frames()

            if depth_image is None and color_image is None:
                print("Frame non validi ricevuti. Continuo...")
                continue

            # Aumenta il contatore dei frame
            frame_id = frame_id + 1

            ## STEP 2 (OPENPOSE PROCESS)
            # Normalizza il frame di profondità per la visualizzazione
            normalized_depth = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX)
            depth_colormap = cv2.applyColorMap(normalized_depth.astype(np.uint8), cv2.COLORMAP_JET)

            if color_image is not None:
                # Passa il frame BGR a OpenPose
                datum = opUtils.process_frame(color_image)
                output_frame = datum.cvOutputData
            else:
                output_frame = np.zeros_like(depth_colormap)

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

            ## STEP 4 (FALL DETECTION)
            HAS_FALLEN = False

            try: 
                HAS_FALLEN = fall_model.detect_fall(datum)
                
                if HAS_FALLEN:
                    print("Fall person detected")
            except Exception as e:
                print(f"Errore durante esecuzione del modulo di fall detection: {e}")
                pass

            ## STEP 5 (IDENTIFICATION)
            ## TO BE ADDED HERE

            ## STEP 6 (GENERATE JSON)
            output_json = mapper.generate_json(vision, datum, frame_id, face_rotations, HAS_FALLEN)
            save_json_to_file(output_json, frame_id, CONFIG['output_dir'])

            ## STEP 7 (SHOW RESULTS)
            show_results(output_frame, depth_colormap)
            
            ## STEP 8 (CHECK FOR QUIT KEY)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except Exception as e:
        print(f"Errore durante esecuzione dell'applicazione: {e}")
        raise e
    finally:
        print("Arresto dei flussi...")

        if "opUtils" in locals():
            del opUtils

        if "vision" in locals():
            del vision

        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
