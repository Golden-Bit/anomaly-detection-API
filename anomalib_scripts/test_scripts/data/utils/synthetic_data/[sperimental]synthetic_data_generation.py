import random
from pathlib import Path

import numpy as np
from PIL import Image


def paste_random_anomaly(percorso_immagine_a, percorso_immagine_b, anomaly_relative_size, percorso_output, percorso_output_maschera):

    # Carica le immagini
    immagine_a = Image.open(Path(percorso_immagine_a))
    immagine_b = Image.open(Path(percorso_immagine_b))

    # Ridimensiona l'immagine
    immagine_b_resized = immagine_b.resize((int(immagine_a.height * anomaly_relative_size * immagine_a.height/immagine_b.height), int(immagine_a.width * anomaly_relative_size * immagine_a.width/immagine_b.width)))

    x = random.randint(0, immagine_a.width - immagine_b_resized.width)
    y = random.randint(0, immagine_a.height - immagine_b_resized.height)

    # Crea la maschera in bianco e nero dell'immagine b rispetto all'immagine a
    maschera = Image.new('L', immagine_a.size, 0)
    regione_di_interesse = (x, y, x + immagine_b_resized.width, y + immagine_b_resized.height)
    maschera.paste(255, regione_di_interesse)

    # Sovrappone l'immagine b sull'immagine a alle coordinate specificate
    immagine_a.paste(immagine_b_resized, (x, y)) #, immagine_b_resized)

    # Salva l'immagine risultante nella directory di output
    immagine_a.save(percorso_output)
    maschera.save(percorso_output_maschera)


if __name__ == "__main__":
    # Esempio di utilizzo della funzione
    percorso_immagine_a = "/home/cyberneid/Desktop/anomalib_project/custom_datasets/cabinet_surface_metallic-white/dataset/normal_white_tiles_256-256_8-8/crop_x0_y0.png"  # Sostituisci con il percorso dell'immagine a
    percorso_immagine_b = "/home/cyberneid/Desktop/anomalib_project/custom_datasets/cabinet_surface_metallic-white/anomalies_source_images/anomaly_0.jpg"  # Sostituisci con il percorso dell'immagine b
    anomaly_relative_size = 0.1
    percorso_output = "/home/cyberneid/Desktop/anomalib_project/custom_datasets/cabinet_surface_metallic-white/dataset/synthetic_abnormal_white/crop_x0_y0.png"  # Sostituisci con il percorso di output desiderato
    percorso_output_maschera = "/home/cyberneid/Desktop/anomalib_project/custom_datasets/cabinet_surface_metallic-white/dataset/ground_truth/synthetic_abnormal_white/crop_x0_y0.png"

    # Sovrappone le immagini e salva il risultato
    paste_random_anomaly(percorso_immagine_a, percorso_immagine_b, anomaly_relative_size, percorso_output, percorso_output_maschera)
