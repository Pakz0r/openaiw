from PIL import Image
import pyrealsense2 as rs
import numpy as np
import shutil
import time
import sys
import cv2
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from vision import Vision
from HPE import HPEModel
from HPE.utils import compute_center, fetch_faces_keypoints_from_datum, find_closest_centroid
from utils import CoordinateMapper, SkeletonVisualizer, OPUtils

CONFIG = {
    "vision": {
        "input_path" : 'input',
        "bag_path" : 'recording.bag'
    },
    "openpose" : {
        "model_folder": "./app/models/",
        "model_pose": "BODY_25",
        "net_resolution": "-1x256", # Default "-1x368"; AI_Watch "-1x128"; Max "-1x176"
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
        "device": 'cpu'
    },
    "Fall" : {
        "model_root" : './app/fall_detection/models/',
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

def show_outputs(output_frame, depth_colormap):
    try:
        cv2.imshow("OpenPose Output", output_frame)
        cv2.imshow("Depth Output", depth_colormap)
    except:
        pass

# Programma principale
def main():

    try:
        # Pulisci il contenuto della cartella di output
        clear_folder(CONFIG['output_dir'])

        print("Inizializzazione di OpenPose...")
        opUtils = OPUtils(CONFIG["openpose"])

        print("Inizializzazione del Coordinate Mapper...")
        mapper = CoordinateMapper(CONFIG["room"]["thing_id"])

        print("Inizializzazione del Visualizzatore dello scheleto...")
        visualizer = SkeletonVisualizer()

        device = CONFIG["HPE"]["device"]
        print(f"Inizializzazione del modulo HPE con '{device}'...")
        hpe_model = HPEModel(CONFIG["HPE"]['model_root'], device)

        print("Inizializzazione del modulo di Vision...")
        bag_path = os.path.join(CONFIG["vision"]["input_path"], CONFIG["vision"]["bag_path"])
        vision = Vision.initialize("realsense")
        playback = vision.start_playback(bag_path)

        dev_info = vision.get_device_info()
        if dev_info is not None:
            print(f"Collegato con {dev_info.vendor.decode()} {dev_info.name.decode()}")

        print("Applicazione inizializzata con successo. Premere 'q' per uscire.")

        frame_id = -1
        prev_frame_time = 0
        new_frame_time = 0

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

            ## STEP 5 (IDENTIFICATION)
            ## TO BE ADDED HERE

            ## STEP 6 (GENERATE JSON)
            output_json = mapper.generate_json(vision, datum, frame_id, face_rotations, HAS_FALLEN)
            save_json_to_file(output_json, frame_id, CONFIG['output_dir'])

            ## STEP 7 (PROFILER INFO)
            new_frame_time = time.time()
            fps = int(1 / (new_frame_time - prev_frame_time))
            prev_frame_time = new_frame_time
            cv2.putText(output_frame, f"FPS: {fps}", (7, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 255, 0), 2)

            ## STEP 8 (SHOW OUTPUTS)
            show_outputs(output_frame, depth_colormap)

            # Checks for quit key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            ## Checks for pause key
            if cv2.waitKey(1) & 0xFF == ord('p'):
                cv2.waitKey(0) & 0xFF == ord('p') 

            ## STEP 9 (RENDER RESULTS)
            visualizer.load_data_from_json(output_json)
            visualizer.render_frame()

            # Checks for quit key
            if visualizer.get_key_pressed() == b'q':
                break

            if playback.current_status() != rs.playback_status.playing:
                print("Lettura del file di registrazione terminata.")
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
