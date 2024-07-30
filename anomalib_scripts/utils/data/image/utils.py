import os
from PIL import Image
import numpy as np
from pathlib import Path


def get_anomaly_dimension(mask_path):

    mask_image = Image.open(mask_path)
    mask_array = np.array(mask_image)

    return sum(sum(mask_array))


def split_by_anomaly_dimension(mask_paths, abnormal_image_paths, delimit_values=None):

    anomaly_info_list = list()
    anomaly_dim_values = list()

    for mask_path, abnormal_image_path in zip(mask_paths, abnormal_image_paths):
        anomaly_dimension = get_anomaly_dimension(mask_path)

        anomaly_dim_values.append(anomaly_dimension)
        anomaly_info_list.append(
            {
                "anomaly_dim": anomaly_dimension,
                "mask_path": mask_path,
                "abnormal_image_path": abnormal_image_path,
            }
        )

    min_dim = min(anomaly_dim_values)
    max_dim = max(anomaly_dim_values)
    average_dim = sum(anomaly_dim_values)/len(anomaly_dim_values)

    if not delimit_values:
        delimit_values = list()
        delimit_value = min_dim
        while True:
            delimit_value *= 2
            delimit_values.append(delimit_value)
            if delimit_value > max_dim:
                break

    anomalies_by_dim = dict()
    for delimit_value in delimit_values:
        key = f"{int(delimit_value/2)}-{delimit_value}"
        anomalies_by_dim[key] = list()

    for anomaly_info in anomaly_info_list:
        anomaly_dim = anomaly_info["anomaly_dim"]
        for dim_range in anomalies_by_dim:
            low_value = int(dim_range.split("-")[0])
            up_value = int(dim_range.split("-")[1])
            if up_value >= anomaly_dim >= low_value:
                anomalies_by_dim[dim_range].append(anomaly_info)

    #print(min_dim, max_dim, average_dim)
    #print(delimit_values)
    #print(anomalies_by_dim.keys())
    #for dim_range in anomalies_by_dim:
    #    print(len(anomalies_by_dim[dim_range]))

    return anomalies_by_dim


def generate_empty_masks_from_dir(
        input_dir: str,
        output_dir: str,
):

    input_image_paths = [f"{input_dir}/{image_name}" for image_name in os.listdir(input_dir)]

    for input_image_path in input_image_paths:
        input_image = Image.open(input_image_path)

        output_image = Image.new(mode='L', size=input_image.size, color=0)

        output_image.save(f"{output_dir}/{input_image_path.split('/')[-1]}")

    return


def resize_images_from_dir(
        input_dir: str,
        size: int | tuple[int, int],
        output_dir: str,
):

    input_image_paths = [f"{input_dir}/{image_name}" for image_name in os.listdir(input_dir)]

    os.makedirs(output_dir, exist_ok=True)

    for input_image_path in input_image_paths:

        input_image = Image.open(input_image_path)

        output_image = input_image.resize(size=size)

        output_image.save(f"{output_dir}/{input_image_path.split('/')[-1]}")

    return


if __name__ == "__main__":

    resize_images_from_dir(
        input_dir="/home/cyberneid/Desktop/anomalib_project/custom_datasets/cabinet_metallic_surface/blue/2-3-5_without-background/mask/test_anomalies",
        size=(512, 512),
        output_dir="/home/cyberneid/Desktop/anomalib_project/custom_datasets/cabinet_metallic_surface/blue/2-3-5_without-background/mask/resized_test_anomalies",
    )


    #generate_empty_masks_from_dir(
    #    input_dir="/home/cyberneid/Desktop/anomalib_project/custom_datasets/cabinet_metallic_surface/blue/2-3-5_without-background/abnormal/original_anomalies",
    #    output_dir="/home/cyberneid/Desktop/anomalib_project/custom_datasets/cabinet_metallic_surface/blue/2-3-5_without-background/mask/original_anomalies",
    #)
    input("...")
    abnormal_root = "./custom_datasets/cabinet_metallic_surface/blue/2-3-5_without-background/abnormal"
    abnormal_dir = "type_1___2___without_background___abnormal_tiles___512-512_32-32"
    abnormal_dir_path = f"{abnormal_root}/{abnormal_dir}"
    mask_root = "./custom_datasets/cabinet_metallic_surface/blue/2-3-5_without-background/mask"
    mask_dir = "type_1___2___without_background___abnormal_tiles___512-512_32-32"
    mask_dir_path = Path(f"{mask_root}/{mask_dir}")

    mask_paths = [f"{mask_dir_path}/{mask_name}" for mask_name in os.listdir(mask_dir_path)]
    abnormal_paths = [f"{abnormal_dir_path}/{abnormal_name}" for abnormal_name in os.listdir(abnormal_dir_path)]

    delimit_values = [668, 1336, 2672, 5344, 10688, 21376, 42752]

    anomalies_by_dim = split_by_anomaly_dimension(mask_paths, abnormal_paths, delimit_values)

    for dim_range in anomalies_by_dim:

        output_abnormal_dir_path = f"{abnormal_dir_path}_{dim_range}"
        #output_abnormal_dir_path = f"{abnormal_root}/{output_abnormal_dir}"
        os.makedirs(output_abnormal_dir_path, exist_ok=True)

        output_mask_dir_path = f"{mask_dir_path}_{dim_range}"
        #output_mask_dir_path = Path(f"{mask_root}/{output_mask_dir}")
        os.makedirs(output_mask_dir_path, exist_ok=True)

        anomalies_subgroup = anomalies_by_dim[dim_range]

        for anomaly_info in anomalies_subgroup:
            src_mask_path = anomaly_info["mask_path"]
            src_mask_name = src_mask_path.split('/')[-1].split("\\")[-1]
            output_mask_path = f"{output_mask_dir_path}/{src_mask_name}"
            src_abnormal_image_path = anomaly_info["abnormal_image_path"]
            src_abnormal_image_name = src_abnormal_image_path.split('/')[-1].split("\\")[-1]
            output_abnormal_image_path = f"{output_abnormal_dir_path}/{src_abnormal_image_name}"

            Image.open(src_mask_path).save(output_mask_path)
            Image.open(src_abnormal_image_path).save(output_abnormal_image_path)

