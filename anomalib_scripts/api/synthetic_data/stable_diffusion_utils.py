
import random
import uuid
from datetime import datetime
import urllib.request
import base64
import json
import time
import os
from pathlib import Path
from typing import Any, List, Dict

import numpy as np
from PIL import Image

webui_server_url = 'http://127.0.0.1:7860'

out_dir = 'api_out'
out_dir_t2i = os.path.join(out_dir, 'txt2img')
out_dir_i2i = os.path.join(out_dir, 'img2img')
os.makedirs(out_dir_t2i, exist_ok=True)
os.makedirs(out_dir_i2i, exist_ok=True)


def timestamp():
    return datetime.fromtimestamp(time.time()).strftime("%Y%m%d-%H%M%S")


def encode_file_to_base64(path):
    with open(path, 'rb') as file:
        return base64.b64encode(file.read()).decode('utf-8')


def decode_and_save_base64(base64_str, save_path):
    with open(save_path, "wb") as file:
        file.write(base64.b64decode(base64_str))


def call_api(api_endpoint, **payload):
    data = json.dumps(payload).encode('utf-8')
    request = urllib.request.Request(
        f'{webui_server_url}/{api_endpoint}',
        headers={'Content-Type': 'application/json'},
        data=data,
    )
    response = urllib.request.urlopen(request)
    return json.loads(response.read().decode('utf-8'))


def call_txt2img_api(**payload):
    response = call_api('sdapi/v1/txt2img', **payload)
    for index, image in enumerate(response.get('images')):
        save_path = os.path.join(out_dir_t2i, f'txt2img-{timestamp()}-{index}.png')
        decode_and_save_base64(image, save_path)


def call_img2img_api(**payload):
    response = call_api('sdapi/v1/img2img', **payload)
    for index, image in enumerate(response.get('images')):
        save_path = os.path.join(out_dir_i2i, f'img2img-{timestamp()}-{index}.png')
        decode_and_save_base64(image, save_path)


def anomaly_mask_from_dict(
        base_image_size: tuple[int, int],
        anomaly_dict: Dict[str, Any],
):

    anomaly_mask_origin = anomaly_dict["mask"]["origin"]
    anomaly_mask_width = anomaly_dict["mask"]["width"]
    anomaly_mask_height = anomaly_dict["mask"]["height"]

    left, upper = anomaly_mask_origin
    right, lower = list(np.array(anomaly_mask_origin) + np.array([anomaly_mask_width, anomaly_mask_height]))

    anomaly_box = (left, upper, right, lower)

    anomaly_mask = Image.new('L', base_image_size, 0)
    anomaly_mask.paste(255, anomaly_box)

    return {
        "mask_image": anomaly_mask,
        "mask_box": anomaly_box,
    }


def get_sd_prompt(
        anomaly_type: str,
        anomaly_description: str,
):

    # use LLM model to generate optimal prompt to use with Stable-Diffusion models
    sd_prompt = str()

    return sd_prompt


def synthetize_abnormal_data(
        src_dir_path: str,
        src_img_name: str,
        anomalies: List[Dict[str, Any]]=None,
        stable_diffusion_kwargs: Dict[str, Any] = None,
):

    src_img_path = f"{src_dir_path}/{src_img_name}"
    src_img = Image.open(src_img_path)
    src_img_size = src_img.size

    ecoded_src_img = encode_file_to_base64(rf"{src_img_path}")

    anomalies_mask = Image.new('L', src_img_size, 0)

    for index, anomaly in enumerate(anomalies):

        anomaly_type = anomaly["type"]
        anomaly_description = anomaly["description"]
        anomaly_prompt = anomaly["prompt"]

        if not anomaly_prompt:
            anomaly_prompt = get_sd_prompt(
                anomaly_type=anomaly_type,
                anomaly_description=anomaly_description
            )

        anomaly_mask = anomaly_mask_from_dict(
            base_image_size=src_img_size,
            anomaly_dict=anomaly,
        )

        anomaly_mask_img = anomaly_mask["mask_image"]
        anomaly_mask_box = anomaly_mask["mask_box"]

        # implement parameters to select output directory in which store data generate in intermediate steps
        temp_dir = "."
        temp_image_name = f"{uuid.uuid4()}.png"
        temp_mask_path = f"{temp_dir}/{temp_image_name}"
        anomaly_mask_img.save(temp_mask_path)
        ecoded_mask = encode_file_to_base64(rf"{temp_mask_path}")
        os.remove(temp_mask_path)

        ####################################################################################################
        # call SD model to generate anomaly on source image
        ####################################################################################################
        #
        # {WRITE YOUR CODE HERE}
        #
        ####################################################################################################

        anomalies_mask.paste(255, anomaly_mask_box)

    return {
        "abnormal_data": None,
        "anomalies_mask": anomalies_mask
    }


def synthetic_dataset_from_dir(
        src_dir_path: str,
        output_dir_path: str,
        output_images_dir: str,
        output_masks_dir: str,
        min_n_anomalies_per_image: int = 1,
        max_n_anomalies_per_image: int = 10,
        min_mask_relative_dim: tuple[float, float] = (0.01, 0.01),
        max_mask_relative_dim: tuple[float, float] = (0.2, 0.2),
        anomaly_type: str = None,
        anomaly_description: str = None,
        anomaly_prompt: str = None,
        stable_diffusion_kwargs: Dict[str, Any] = None,
):

    if not Path(output_dir_path).is_dir():
        os.mkdir(output_dir_path)

    output_images_dir_path = f"{output_dir_path}/{output_images_dir}"
    output_masks_dir_path = f"{output_dir_path}/{output_masks_dir}"

    if not Path(output_images_dir_path).is_dir():
        os.mkdir(output_images_dir_path)

    if not Path(output_masks_dir_path).is_dir():
        os.mkdir(output_masks_dir_path)

    for src_img_name in os.listdir(src_dir_path):

        src_img_path = f"{src_dir_path}/{src_img_name}"
        src_img = Image.open(src_img_path)
        src_img_width = src_img.width
        src_img_height = src_img.height

        anomalies = list()
        n_anomalies = random.randint(min_n_anomalies_per_image, max_n_anomalies_per_image)
        for _ in range(n_anomalies):

            # while di controllo su compatibilità
            while True:

                x_origin = random.randint(0, src_img_width)
                y_origin = random.randint(0, src_img_height)

                min_mask_width = int(min_mask_relative_dim[0] * src_img_width)
                max_mask_width = int(max_mask_relative_dim[0] * src_img_width)
                min_mask_height = int(min_mask_relative_dim[1] * src_img_height)
                max_mask_height = int(max_mask_relative_dim[1] * src_img_height)

                mask_width = random.randint(min_mask_width, max_mask_width)
                mask_height = random.randint(min_mask_height, max_mask_height)

                if x_origin + mask_width < src_img_width and y_origin + mask_height < src_img_height:
                    break

            anomalies.append({
                "type": anomaly_type,
                "description": anomaly_description,
                "prompt": anomaly_prompt,
                "mask": {
                    "origin": [x_origin, y_origin],
                    "width": mask_width,
                    "height": mask_height,
                },
            })

        synthetization_result = synthetize_abnormal_data(
            src_dir_path=src_img_path,
            src_img_name=src_img_name,
            anomalies=anomalies,
            stable_diffusion_kwargs=stable_diffusion_kwargs,
        )

        # do something...
        print(synthetization_result)


if __name__ == "__main__":

    """        
    paylod = {
        "prompt": "",
        "negative_prompt": "",
        "styles": [
            "string"
        ],
        "seed": -1,
        "subseed": -1,
        "subseed_strength": 0,
        "seed_resize_from_h": -1,
        "seed_resize_from_w": -1,
        "sampler_name": "string",
        "scheduler": "string",
        "batch_size": 1,
        "n_iter": 1,
        "steps": 50,
        "cfg_scale": 7,
        "width": 512,
        "height": 512,
        "restore_faces": true,
        "tiling": true,
        "do_not_save_samples": false,
        "do_not_save_grid": false,
        "eta": 0,
        "denoising_strength": 0.75,
        "s_min_uncond": 0,
        "s_churn": 0,
        "s_tmax": 0,
        "s_tmin": 0,
        "s_noise": 0,
        "override_settings": {},
        "override_settings_restore_afterwards": true,
        "refiner_checkpoint": "string",
        "refiner_switch_at": 0,
        "disable_extra_networks": false,
        "firstpass_image": "string",
        "comments": {},
        "init_images": init_images,
        "resize_mode": 0,
        "image_cfg_scale": 0,
        "mask": "string",
        "mask_blur_x": 4,
        "mask_blur_y": 4,
        "mask_blur": 0,
        "mask_round": true,
        "inpainting_fill": 0,
        "inpaint_full_res": true,
        "inpaint_full_res_padding": 0,
        "inpainting_mask_invert": 0,
        "initial_noise_multiplier": 0,
        "latent_mask": "string",
        "force_task_id": "string",
        "sampler_index": "Euler",
        "include_init_images": false,
        "script_name": "string",
        "script_args": [],
        "send_images": true,
        "save_images": false,
        "alwayson_scripts": {},
        "infotext": "string"
    }
    """

    batch_size = 2
    payload = {
        "prompt": "write here your prompt",
        "seed": 1,  # random.randint(0, 999),
        "steps": 20,
        "width": 512,
        "height": 512,
        "denoising_strength": 0.9,
        "n_iter": 1,
        "init_images": None,  # init_images,
        "batch_size": batch_size,  # batch_size if len(init_images) == 1 else len(init_images),
        "mask": None,  # encode_file_to_base64(r"./input_masks/3.jpeg")
    }

    # if len(init_images) > 1 then batch_size should be == len(init_images)
    # else if len(init_images) == 1 then batch_size can be any value int >= 1
    call_img2img_api(**payload)

    # there exist a useful extension that allows converting of webui calls to api payload
    # particularly useful when you wish setup arguments of extensions and scripts
    # https://github.com/huchenlei/sd-webui-api-payload-display

    pass

