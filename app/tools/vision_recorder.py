import keyboard
import shutil
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from vision import Vision

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
def main(output_path='input', bag_path='recording.bag'):
    try:
        # Pulisci il contenuto della cartella di output
        os.makedirs(output_path, exist_ok=True)
        clear_folder(output_path)

        print("Inizializzazione del modulo di Vision...")
        bag_path = os.path.join(output_path, bag_path)
        vision = Vision.initialize("realsense")
        vision.start_recording(bag_path)

        print("Avvio della registrazione. Premere 'q' per uscire.")

        while True:
            vision.wait_frames()

            if keyboard.is_pressed('q'):
                break

    finally:
        if "vision" in locals():
            del vision

        print(f"Registrazione completata.")

if __name__ == "__main__":
    main()