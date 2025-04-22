import pyrealsense2 as rs
import cv2
import sys
import os

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from vision import Vision

# Programma principale
def main(input_path='input', bag_path='recording.bag'):
    try:
        print("Inizializzazione del modulo di Vision...")
        bag_path = os.path.join(input_path, bag_path)
        vision = Vision.initialize("realsense")
        playback = vision.start_playback(bag_path)

        print("Avvio lettura del file di registrazione.")

        frame_count = 0

        while True:
            depth_image, color_image = vision.get_frames()

            if depth_image is None and color_image is None:
                print("Frame non validi ricevuti. Continuo...")
                continue

            # Normalizza il frame di profondità per la visualizzazione
            normalized_depth = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX)
            depth_colormap = cv2.applyColorMap(normalized_depth.astype(np.uint8), cv2.COLORMAP_JET)
            
            cv2.imshow("RGB Frame", color_image) 
            cv2.imshow("Depth Frame", depth_colormap)

            print(f"Lettura del frame {frame_count} completata.")

            if playback.current_status() != rs.playback_status.playing:
                print("Lettura del file di registrazione terminata.")
                print("Generazione del file di metadati...")
                break

            # Checks for quit key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # Aumenta il contatore dei frame
            frame_count = frame_count + 1

    finally:
        if "vision" in locals():
            del vision

        print(f"Registrazione completata.")

if __name__ == "__main__":
    main()