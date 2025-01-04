import cv2
import numpy as np
from __init__ import Vision

def main():
    driver_name = "openni2" # input("Inserisci il nome del driver (openni2/realsense): ")

    try:
        vision = Vision.initialize(driver_name)
        print("Camera inizializzata. Premi 'q' per uscire.")

        dev_info = vision.get_device_info()
        if dev_info is not None:
            print(f"Collegato con {dev_info.vendor.decode()} {dev_info.name.decode()}")

        while True:
            depth_frame, color_frame = vision.get_frames()

            if depth_frame is None or color_frame is None:
                print("Frame non validi ricevuti. Continuo...")
                continue

            # Normalizza il frame di profondità per visualizzazione
            normalized_depth = cv2.normalize(depth_frame, None, 0, 255, cv2.NORM_MINMAX)
            depth_colormap = cv2.applyColorMap(normalized_depth.astype(np.uint8), cv2.COLORMAP_JET)

            # Mostra i frame
            combined_view = np.hstack((color_frame, depth_colormap))
            cv2.imshow("Vision - Color + Depth", combined_view)

            # Premi 'q' per uscire
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except Exception as e:
        print(f"Errore durante esecuzione dell'applicazione: {e}")
    finally:
        print("Arresto dei flussi...")
        del vision  # Chiamata esplicita al distruttore per rilasciare risorse
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
