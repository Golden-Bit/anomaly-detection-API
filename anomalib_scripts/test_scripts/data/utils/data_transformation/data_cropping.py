import csv
import os
from pathlib import Path

import cv2


def load_csv_to_list(csv_path):
    """
    Carica un file CSV e restituisce una lista di liste.
    """
    data = []
    with open(csv_path, newline='') as csvfile:
        csv_reader = csv.reader(csvfile)
        for row in csv_reader:
            data.append(row)
    return data


def csv_to_dict_list(csv_data):
    """
    Converte i dati CSV in una lista di dizionari, con chiavi corrispondenti agli header del CSV.
    """
    headers = csv_data[0]  # Prima riga contiene gli header
    dict_list = []
    for row in csv_data[1:]:  # Ogni riga dopo la prima contiene i dati
        row_dict = dict()
        headers[0] = "index"
        for i, value in enumerate(row):
            row_dict[headers[i]] = value
        dict_list.append(row_dict)

    return dict_list


def crop_image(percorso_immagine, origine_x, origine_y, larghezza_crop, altezza_crop, percorso_output):
    # Carica l'immagine utilizzando OpenCV
    immagine = cv2.imread(percorso_immagine)
    # Esegui il crop dell'immagine
    immagine_croppata = immagine[origine_y:origine_y + altezza_crop, origine_x:origine_x + larghezza_crop]

    # Salva l'immagine croppata nella directory di output
    cv2.imwrite(percorso_output, immagine_croppata)


def main():
    csv_path = f"/home/cyberneid/Desktop/anomalib_project/custom_datasets/cabinet_surface_metallic-white/rects_to_crop/v1.csv" #input("Inserisci il percorso del file CSV: ")
    csv_data = load_csv_to_list(csv_path)
    dict_list = csv_to_dict_list(csv_data)
    print("Lista di dizionari:")
    for item in dict_list:
        print(item)


def extract_tiles(percorso_immagine, larghezza_crop, altezza_crop, traslazione_x, traslazione_y, nuova_larghezza_output, nuova_altezza_output, percorso_output):
    # Carica l'immagine utilizzando OpenCV
    immagine = cv2.imread(percorso_immagine)
    altezza_immagine, larghezza_immagine, _ = immagine.shape

    # Estrai tutti i possibili crop
    for y in range(0, altezza_immagine - altezza_crop + 1, traslazione_y):
        for x in range(0, larghezza_immagine - larghezza_crop + 1, traslazione_x):
            crop = immagine[y:y + altezza_crop, x:x + larghezza_crop]
            nome_output = f"{os.path.splitext(percorso_output)[0]}/crop_x{x}_y{y}.png" #{os.path.splitext(percorso_output)[1]}"
            crop_redimensionato = cv2.resize(crop, (nuova_larghezza_output, nuova_altezza_output))
            print(nome_output)
            cv2.imwrite(nome_output, crop_redimensionato)


if __name__ == "__main__" and False:

    wd = "/home/cyberneid/Desktop/anomalib_project/custom_datasets/cabinet_surface_metallic-white"

    csv_path = f"{wd}/rects_to_crop/v6.csv" #input("Inserisci il percorso del file CSV: ")
    csv_data = load_csv_to_list(csv_path)
    dict_list = csv_to_dict_list(csv_data)

    for i, item in enumerate(dict_list):
        print(item)

        images_dir = f"{wd}/output_frame/white_surface_v6"

        if not Path(f"{wd}/output_cropped_frame/{images_dir.split('/')[-1]}").is_dir():
            os.mkdir(f"{wd}/output_cropped_frame/{images_dir.split('/')[-1]}")

        for img_file_name in os.listdir(images_dir):

            if not Path(f"{wd}/output_cropped_frame/{images_dir.split('/')[-1]}/crop_index_{i}").is_dir():
                os.mkdir(f"{wd}/output_cropped_frame/{images_dir.split('/')[-1]}/crop_index_{i}")

            # Esempio di utilizzo dello script
            percorso_immagine = f"{images_dir}/{img_file_name}"  # Sostituisci con il percorso dell'immagine
            origin_x_key = item["originX"]
            origin_x = int(item[origin_x_key])  # Coordinate x del vertice in alto a sinistra
            origin_y_key = item["originY"]
            origin_y = int(item[origin_y_key])  # Coordinate y del vertice in alto a sinistra
            larghezza_crop = int(item["width"])  # Larghezza del crop
            altezza_crop = int(item["height"])  # Altezza del crop
            percorso_output = f"{wd}/output_cropped_frame/{images_dir.split('/')[-1]}/crop_index_{i}/{img_file_name}"  # Sostituisci con il percorso di output desiderato

            # Esegui il crop dell'immagine utilizzando OpenCV
            crop_image(percorso_immagine, origin_x, origin_y, larghezza_crop, altezza_crop, percorso_output)


if __name__ == "__main__":

    for index in range(0,1):
        wd = f"/home/cyberneid/Desktop/anomalib_project/custom_datasets/cabinet_surface_metallic-white/output_cropped_frame/normal_white"

        output_dir = f"{wd}_tiles_256-256_8-8"

        if not Path(output_dir).is_dir():
            os.mkdir(output_dir)

        sampling_rate = 1

        cnt = 0
        for img_file_name in os.listdir(wd):

            if not (cnt % sampling_rate):

                # Esempio di utilizzo dello script
                percorso_immagine = f"{wd}/{img_file_name}"  # Sostituisci con il percorso dell'immagine
                larghezza_crop = 256  # Larghezza del crop in pixel
                altezza_crop = 256  # Altezza del crop in pixel
                traslazione_x = 8  # Valore di traslazione lungo l'asse x in pixel
                traslazione_y = 8  # Valore di traslazione lungo l'asse y in pixel
                nuova_altezza_output = 256
                nuova_larghezza_output = 256
                percorso_output = output_dir  # Sostituisci con il percorso di output desiderato

                # Estrai tutti i crop
                extract_tiles(percorso_immagine, larghezza_crop, altezza_crop, traslazione_x, traslazione_y, nuova_altezza_output, nuova_larghezza_output, percorso_output)

            cnt += 1
