import cv2
import numpy as np
from openpose import pyopenpose as op
from vision import Vision

CONFIG = {
    "vision": "openni2"  # Opzioni: "realsense" oppure "openni2"
}

# Configura OpenPose
def initialize_openpose():
    try:
        params = {
            "model_folder": "openpose/models/",  # Path ai modelli di OpenPose
            "model_pose": "BODY_25",
            "net_resolution": "-1x96", # Default "-1x368"; AI_Watch "-1x128"
            "hand": False,
            "hand_net_resolution": "224x224", # "Default "368x368" (multiples of 16)"
            "face": False,
            "face_net_resolution": "224x224", # "Default "368x368" (multiples of 16)"
        }
        opWrapper = op.WrapperPython()
        opWrapper.configure(params)
        opWrapper.start()
        return opWrapper
    except Exception as e:
        print(f"Errore nell'inizializzazione di OpenPose: {e}")
        return None

# Processa i frame con OpenPose
def process_frame_with_openpose(opWrapper, frame):
    datum = op.Datum()
    datum.cvInputData = frame
    opWrapper.emplaceAndPop(op.VectorDatum([datum]))
    return datum

# Programma principale
def main():    
    print("Inizializzazione di OpenPose...")
    opWrapper = initialize_openpose()

    try:
        # Seleziona la camera basandosi sulla configurazione
        vision = Vision.initialize(CONFIG["vision"])

        dev_info = vision.get_device_info()
        if dev_info is not None:
            print(f"Collegato con {dev_info.vendor.decode()} {dev_info.name.decode()}")

        print("Applicazione inizializzata con successo. Premere 'q' per uscire.")

        while True:
            # Acquisisci i frame dalla camera selezionata
            depth_frame, color_frame = vision.get_frames()

            if depth_frame is None and color_frame is None:
                print("Frame non validi ricevuti. Continuo...")
                continue

            # Normalizza il frame di profondità per la visualizzazione
            normalized_depth = cv2.normalize(depth_frame, None, 0, 255, cv2.NORM_MINMAX)
            depth_colormap = cv2.applyColorMap(normalized_depth.astype(np.uint8), cv2.COLORMAP_JET)

            if color_frame is not None:
                # Passa il frame RGB a OpenPose
                datum = process_frame_with_openpose(opWrapper, color_frame)
                output_frame = datum.cvOutputData
            else:
                output_frame = np.zeros_like(depth_colormap)

            # Mostra il risultato combinato
            combined_view = np.hstack((output_frame, depth_colormap))
            cv2.imshow("OpenPose Output + Depth", combined_view)

            # Premi 'q' per uscire
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except Exception as e:
        print(f"Errore durante esecuzione dell'applicazione: {e}")
    finally:
        print("Arresto dei flussi...")
        opWrapper.stop()
        del vision
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
