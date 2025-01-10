import os
import cv2
import numpy as np
from vision import Vision
from utils import OPUtils, CoordinateMapper
from rooms.home import HOME as ROOM

CONFIG = {
    "vision": {
        "driver" : "openni2"  # Opzioni: "realsense" oppure "openni2"
    },
    "openpose" : {
        "model_folder": "./openpose/models/",  # Path ai modelli di OpenPose
        "model_pose": "BODY_25",
        "net_resolution": "-1x96", # Default "-1x368"; AI_Watch "-1x128"
        "hand": False,
        "hand_net_resolution": "224x224", # "Default "368x368" (multiples of 16)"
        "face": False,
        "face_net_resolution": "224x224", # "Default "368x368" (multiples of 16)"
    },
    "output_dir" : "output"
}

# Salva il JSON generato in un file nel percorso specificato.
def save_json_to_file(json_data, frame_id, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, f"frame{frame_id}_skeletonsPoints3D.json")
    with open(file_path, "w") as f:
        f.write(json_data)
    print(f"Nuovo frame salvato in: {file_path}")

# Programma principale
def main():    
    try:
        # Seleziona la camera basandosi sulla configurazione
        vision = Vision.initialize(CONFIG["vision"]["driver"])

        dev_info = vision.get_device_info()
        if dev_info is not None:
            print(f"Collegato con {dev_info.vendor.decode()} {dev_info.name.decode()}")

        print("Inizializzazione di OpenPose...")
        opUtils = OPUtils(CONFIG["openpose"])

        print("Inizializzazione del Mapper...")
        mapper = CoordinateMapper(ROOM)

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
                # Passa il frame RGB a OpenPose
                datum = opUtils.process_frame(color_image)
                output_frame = datum.cvOutputData
            else:
                output_frame = np.zeros_like(depth_colormap)

            ## STEP 3 (GENERATE JSON)
            output_json = mapper.generate_json(vision, datum, depth_image, frame_id)
            save_json_to_file(output_json, frame_id, CONFIG['output_dir'])

            ## STEP 4 (SHOW RESULTS)
            # Mostra il risultato combinato
            combined_view = np.hstack((output_frame, depth_colormap))
            cv2.imshow("OpenPose Output + Depth", combined_view)

            # Premi 'q' per uscire
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except Exception as e:
        print(f"Errore durante esecuzione dell'applicazione: {e}")
        raise e
    finally:
        print("Arresto dei flussi...")
        del opUtils
        del vision
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
