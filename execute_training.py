import requests

abnormal_subsets = ['abnormal_tile_0', 'abnormal_tile_0', 'abnormal_tile_0', 'abnormal_tile_0', 'abnormal_tile_0', 'abnormal_tile_0', 'abnormal_tile_0', 'abnormal_tile_0', 'abnormal_tile_0', 'abnormal_tile_0', 'abnormal_tile_0', 'abnormal_tile_0', 'abnormal_tile_0', 'abnormal_tile_0', 'abnormal_tile_0', 'abnormal_tile_0', 'abnormal_tile_0', 'abnormal_tile_0', 'abnormal_tile_0', 'abnormal_tile_0', 'abnormal_tile_0', 'abnormal_tile_0', 'abnormal_tile_0', 'abnormal_tile_0', 'abnormal_tile_0', 'abnormal_tile_0', 'abnormal_tile_0', 'abnormal_tile_0', 'abnormal_tile_0', 'abnormal_tile_0', 'abnormal_tile_0', 'abnormal_tile_0', 'abnormal_tile_0', 'abnormal_tile_0', 'abnormal_tile_0', 'abnormal_tile_0', 'abnormal_tile_0', 'abnormal_tile_0', 'abnormal_tile_0', 'abnormal_tile_0', 'abnormal_tile_0', 'abnormal_tile_0', 'abnormal_tile_0', 'abnormal_tile_0', 'abnormal_tile_0', 'abnormal_tile_0', 'abnormal_tile_0', 'abnormal_tile_0', 'abnormal_tile_0', 'abnormal_tile_0', 'abnormal_tile_0', 'abnormal_tile_0', 'abnormal_tile_0', 'abnormal_tile_0', 'abnormal_tile_0', 'abnormal_tile_0', 'abnormal_tile_0', 'abnormal_tile_0', 'abnormal_tile_0', 'abnormal_tile_0', 'abnormal_tile_0', 'abnormal_tile_0', 'abnormal_tile_0', 'abnormal_tile_0', 'abnormal_tile_0', 'abnormal_tile_0', 'abnormal_tile_0', 'abnormal_tile_0', 'abnormal_tile_0', 'abnormal_tile_0']

normal_subsets = [['tile_0_0', 'normal_tile_0'], ['tile_1_0', 'normal_tile_0'], ['tile_2_0', 'normal_tile_0'], ['tile_3_0', 'normal_tile_0'], ['tile_4_0', 'normal_tile_0'], ['tile_5_0', 'normal_tile_0'], ['tile_6_0', 'normal_tile_0'], ['tile_7_0', 'normal_tile_0'], ['tile_8_0', 'normal_tile_0'], ['tile_9_0', 'normal_tile_0'], ['tile_10_0', 'normal_tile_0'], ['tile_11_0', 'normal_tile_0'], ['tile_12_0', 'normal_tile_0'], ['tile_13_0', 'normal_tile_0'], ['tile_0_1', 'normal_tile_0'], ['tile_1_1', 'normal_tile_0'], ['tile_2_1', 'normal_tile_0'], ['tile_3_1', 'normal_tile_0'], ['tile_4_1', 'normal_tile_0'], ['tile_5_1', 'normal_tile_0'], ['tile_6_1', 'normal_tile_0'], ['tile_7_1', 'normal_tile_0'], ['tile_8_1', 'normal_tile_0'], ['tile_9_1', 'normal_tile_0'], ['tile_10_1', 'normal_tile_0'], ['tile_11_1', 'normal_tile_0'], ['tile_12_1', 'normal_tile_0'], ['tile_13_1', 'normal_tile_0'], ['tile_0_2', 'normal_tile_0'], ['tile_1_2', 'normal_tile_0'], ['tile_2_2', 'normal_tile_0'], ['tile_3_2', 'normal_tile_0'], ['tile_4_2', 'normal_tile_0'], ['tile_5_2', 'normal_tile_0'], ['tile_6_2', 'normal_tile_0'], ['tile_7_2', 'normal_tile_0'], ['tile_8_2', 'normal_tile_0'], ['tile_9_2', 'normal_tile_0'], ['tile_10_2', 'normal_tile_0'], ['tile_11_2', 'normal_tile_0'], ['tile_12_2', 'normal_tile_0'], ['tile_13_2', 'normal_tile_0'], ['tile_0_3', 'normal_tile_0'], ['tile_1_3', 'normal_tile_0'], ['tile_2_3', 'normal_tile_0'], ['tile_3_3', 'normal_tile_0'], ['tile_4_3', 'normal_tile_0'], ['tile_5_3', 'normal_tile_0'], ['tile_6_3', 'normal_tile_0'], ['tile_7_3', 'normal_tile_0'], ['tile_8_3', 'normal_tile_0'], ['tile_9_3', 'normal_tile_0'], ['tile_10_3', 'normal_tile_0'], ['tile_11_3', 'normal_tile_0'], ['tile_12_3', 'normal_tile_0'], ['tile_13_3', 'normal_tile_0'], ['tile_0_4', 'normal_tile_0'], ['tile_1_4', 'normal_tile_0'], ['tile_2_4', 'normal_tile_0'], ['tile_3_4', 'normal_tile_0'], ['tile_4_4', 'normal_tile_0'], ['tile_5_4', 'normal_tile_0'], ['tile_6_4', 'normal_tile_0'], ['tile_7_4', 'normal_tile_0'], ['tile_8_4', 'normal_tile_0'], ['tile_9_4', 'normal_tile_0'], ['tile_10_4', 'normal_tile_0'], ['tile_11_4', 'normal_tile_0'], ['tile_12_4', 'normal_tile_0'], ['tile_13_4', 'normal_tile_0']]

dataset_dir_name = "white_metallic_surface_v2"

for normal_subset, abnormal_subset in zip(normal_subsets, abnormal_subsets):
    pyload = {
      "name": f"{dataset_dir_name}-{'_'.join(normal_subset)}-{'_'.join([abnormal_subset])}",
      "task_type": "SEGMENTATION",
      "dataset_root": f"datasets/{dataset_dir_name}",
      "normal_dir": normal_subset,
      "abnormal_dir": abnormal_subset,
      "mask_dir": "mask",
      "image_size": [
        256,
        256
      ],
      "num_workers": 0,
      "model_kwargs": {
        "backbone": "wide_resnet50_2"
      },
      "output_file_path": f"train_outputs/{dataset_dir_name}-{'_'.join(normal_subset)}-{'_'.join([abnormal_subset])}.pkl"
    }

    response = requests.post(url="http://127.0.0.1:8102/train", json=pyload)

    print(response.json())