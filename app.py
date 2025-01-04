import cv2
import traceback
import numpy as np
import pyrealsense2 as rs
from primesense import openni2
from openpose import pyopenpose as op

# Configurazione per la selezione della camera
CONFIG = {
    "camera": "astra"  # Opzioni: "realsense" oppure "astra"
}

# Inizializza la depth camera (Intel RealSense D435)
def initialize_realsense():
    try:
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        pipeline.start(config)
        return pipeline
    except Exception as e:
        print(f"Errore nell'inizializzazione della depth camera: {e}")
        if hasattr(e, '__traceback__'):
            traceback.print_tb(e.__traceback__)
        return None, None

# Inizializza la depth camera (Orbbec Astra)
def initialize_astra():
    try:
        openni2.initialize()  # Carica la libreria OpenNI
        dev = openni2.Device.open_any()
        depth_stream = dev.create_depth_stream()
        color_stream = dev.create_color_stream()
        depth_stream.start()
        color_stream.start()
        return depth_stream, color_stream
    except Exception as e:
        print(f"Errore nell'inizializzazione della depth camera: {e}")
        if hasattr(e, '__traceback__'):
            traceback.print_tb(e.__traceback__)
        return None, None

# Leggi i frame dalla RealSense camera
def get_realsense_frames(pipeline):
    try:
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        if not depth_frame or not color_frame:
            return None, None

        # Converti i frame in array numpy
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        return depth_image, color_image
    except Exception as e:
        print(f"Errore durante la lettura dei frame: {e}")
        if hasattr(e, '__traceback__'):
            traceback.print_tb(e.__traceback__)
        return None, None

# Leggi i frame dalla Astra camera
def get_astra_frames(depth_stream, color_stream):
    try:
        depth_frame = depth_stream.read_frame()
        color_frame = color_stream.read_frame()

        # Processa il frame di profondità
        depth_image = np.array(depth_frame.get_buffer_as_uint16()).reshape(depth_frame.height, depth_frame.width)

        # Processa il frame colore
        color_data = np.array(color_frame.get_buffer_as_uint8()).reshape(color_frame.height, color_frame.width, 3)
        color_image = cv2.cvtColor(color_data, cv2.COLOR_RGB2BGR)

        return depth_image, color_image
    except Exception as e:
        print(f"Errore durante la lettura dei frame: {e}")
        if hasattr(e, '__traceback__'):
            traceback.print_tb(e.__traceback__)
        return None, None

# Configura OpenPose
def initialize_openpose():
    try:
        params = {
            "model_folder": "models/",  # Path ai modelli di OpenPose
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
        if hasattr(e, '__traceback__'):
            traceback.print_tb(e.__traceback__)
        return None

# Processa i frame con OpenPose
def process_frame_with_openpose(opWrapper, frame):
    datum = op.Datum()
    datum.cvInputData = frame
    opWrapper.emplaceAndPop(op.VectorDatum([datum]))
    return datum

# Programma principale
def main():
    # Seleziona la camera basandosi sulla configurazione
    if CONFIG["camera"] == "realsense":
        print("Inizializzazione della Intel RealSense D435...")
        pipeline = initialize_realsense()
        get_frames = lambda: get_realsense_frames(pipeline)
        stop_camera = lambda: (pipeline.stop())
        print("Utilizzando Intel RealSense D435")
    elif CONFIG["camera"] == "astra":
        print("Inizializzazione della Orbbec Astra...")
        astra_depth, astra_color = initialize_astra()
        get_frames = lambda: get_astra_frames(astra_depth, astra_color)
        stop_camera = lambda: (astra_depth.stop(), astra_color.stop(), openni2.unload())
        print("Utilizzando Orbbec Astra")
    else:
        print("Configurazione della camera non valida. Uscita.")
        return

    print("Inizializzazione di OpenPose...")
    opWrapper = initialize_openpose()

    print("Applicazione inizializzata con successo. Premere 'q' per uscire.")

    try:
        while True:
            # Acquisisci i frame dalla camera selezionata
            depth_frame, color_frame = get_frames()

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
        if hasattr(e, '__traceback__'):
            traceback.print_tb(e.__traceback__)
    finally:
        print("Arresto dei flussi...")
        opWrapper.stop()
        stop_camera()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
