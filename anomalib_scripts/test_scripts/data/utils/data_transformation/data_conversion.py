import os
import cv2


def from_video_to_image(src_directory, dest_directory, max_frame_rate):
    # Controlla se la directory di destinazione esiste, altrimenti creala
    if not os.path.exists(dest_directory):
        os.makedirs(dest_directory)

    # Cicla attraverso i file nella directory sorgente
    for filename in os.listdir(src_directory):
        if filename.endswith('.mp4') or filename.endswith('.avi'):
            video_path = os.path.join(src_directory, filename)
            video_capture = cv2.VideoCapture(video_path)
            fps = video_capture.get(cv2.CAP_PROP_FPS)
            frame_interval = int(fps / max_frame_rate)
            success, image = video_capture.read()
            count = 0

            # Estrapola e salva i frame come immagini PNG
            while success:
                if count % frame_interval == 0:
                    frame_path = os.path.join(dest_directory, f"{filename.split('.')[0]}_frame{count}.png")
                    cv2.imwrite(frame_path, image)  # Salva il frame come PNG
                success, image = video_capture.read()
                count += 1

            video_capture.release()


if __name__ == "__main__":
    # Directory sorgente e destinazione
    cwd = os.getcwd()
    cwd = f"{cwd.split('anomalib_project')[0]}/anomalib_project"

    src_directory = f"{cwd}/custom_datasets/cabinet_surface_metallic-white/video_sources/white_surface_v6"
    dest_directory = f"{cwd}/custom_datasets/cabinet_surface_metallic-white/output_frame/white_surface_v6"
    max_frame_rate = 1  # Imposta il massimo numero di frame al secondo desiderato

    # Esegui la funzione per estrarre i frame dai video
    from_video_to_image(src_directory, dest_directory, max_frame_rate)