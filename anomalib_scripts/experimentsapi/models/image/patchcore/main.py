import os
from typing import Any

import numpy as np
from anomalib.data.utils import ValSplitMode, TestSplitMode
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from matplotlib import pyplot as plt
from PIL import Image
from torch.utils.data import DataLoader

from anomalib.data import PredictDataset, MVTec, Folder
from anomalib.engine import Engine
from anomalib.models import Patchcore
from anomalib.utils.post_processing import superimpose_anomaly_map
from anomalib import TaskType
from pathlib import Path

from utils import (
    train,
    test
)


if __name__ == "__main__":

    """
    Setting up the Dataset Directory

    This cell is to ensure we change the directory to have access to the datasets.
    """

    # NOTE: Provide the path to the dataset root directory.
    #   If the datasets is not downloaded, it will be downloaded
    #   to this directory.
    dataset_root = Path.cwd() / "custom_datasets" / "cabinet_metallic_surface" / "blue"

    """
    Data Module

    To train the model end-to-end, we do need to have a dataset. In our previous notebooks, we demonstrate how to initialize 
    benchmark- and custom datasets. In this tutorial, we will use MVTec AD DataModule. We assume that datasets directory 
    is created in the anomalib root directory and MVTec dataset is located in datasets directory.

    Before creating the dataset, let's define the task type that we will be working on. In this notebook, we will be working 
    on a segmentation task. Therefore the task variable would be:
    """

    #mvtec_categories = [category_name for category_name in os.listdir(dataset_root) if Path(f"{dataset_root}/{category_name}").is_dir()]

    task = TaskType.SEGMENTATION
    abnormal_subdirs = ["abnormal"]
    normal_subdirs = [#"normal/type_1___1___with_background___normal_tiles___384-384_32-32",
                      "normal/type_1___1___with_background___normal_tiles___512-512_32-32",
                      #"normal/type_1___3___with_background___normal_tiles___384-384_32-32",
                      "normal/type_1___3___with_background___normal_tiles___512-512_32-32",
                      #"normal/type_1___9___with_background___normal_tiles___384-384_32-32",
                      #"normal/type_1___9___with_background___normal_tiles___512-512_32-32",
                      #"normal/type_1___10___with_background___normal_tiles___384-384_32-32",
                      #"normal/type_1___10___with_background___normal_tiles___512-512_32-32",
                      #"normal/type_1___12___with_background___normal_tiles___384-384_32-32",
                      #"normal/type_1___12___with_background___normal_tiles___512-512_32-32",
                      #"normal/type_1___13___with_background___normal_tiles___384-384_32-32",
                      "normal/type_1___13___with_background___normal_tiles___512-512_32-32",
                      #"normal/type_1___14___with_background___normal_tiles___384-384_32-32",
                      #"normal/type_1___14___with_background___normal_tiles___512-512_32-32",
                      #"normal/type_1___15___with_background___normal_tiles___384-384_32-32",
                      "normal/type_1___15___with_background___normal_tiles___512-512_32-32",
                      #"normal/type_1___16___with_background___normal_tiles___384-384_32-32",
                      "normal/type_1___16___with_background___normal_tiles___512-512_32-32",
                      #"normal/type_1___1___without_background___normal_tiles___256-256_32-32",
                      #"normal/type_1___1___without_background___normal_tiles___384-384_32-32",
                      #"normal/type_1___1___without_background___normal_tiles___512-512_32-32",
                      #"normal/type_1___3___without_background___normal_tiles___256-256_32-32",
                      #"normal/type_1___3___without_background___normal_tiles___384-384_32-32",
                      #"normal/type_1___3___without_background___normal_tiles___512-512_32-32",
                      #"normal/type_1___4___without_background___normal_tiles___384-384_32-32",
                      #"normal/type_1___4___without_background___normal_tiles___512-512_32-32",
                      #"normal/type_1___11___without_background___normal_tiles___384-384_32-32",
                      #"normal/type_1___11___without_background___normal_tiles___512-512_32-32",
                      #"normal/type_1___16___without_background___normal_tiles___256-256_32-32",
                      #"normal/type_1___16___without_background___normal_tiles___384-384_32-32",
                      #"normal/type_1___16___without_background___normal_tiles___512-512_32-32"
                      ]

    for abnormal_subdir in abnormal_subdirs:

        datamodule = Folder(
            name="cabinet_metallic_surface-blue_test_0",
            root=dataset_root,
            normal_dir=normal_subdirs,
            abnormal_dir=abnormal_subdir,
            task=task,
            mask_dir=dataset_root / "mask" / abnormal_subdir,
            image_size=None, #(256, 256),
            test_split_mode=TestSplitMode.FROM_DIR,
            test_split_ratio=0.2,
            #test_split_mode=TestSplitMode.SYNTHETIC,
            val_split_mode=ValSplitMode.SYNTHETIC,
            #val_split_mode=ValSplitMode.NONE,
            val_split_ratio=0.02,
        )

        datamodule.setup()

        train_result = train(
            datamodule=datamodule,
            task_type=task,
            model_kwargs=dict(),
            engine_kwargs=dict(),
            train_kwargs=dict(),
        )

        model = train_result["model"]
        engine = train_result["engine"]

        test_result = test(
            datamodule=datamodule,
            model=model,
            engine=engine,
            model_kwargs=dict(),
            engine_kwargs=dict(),
            test_kwargs=dict(),
        )

        print(test_result)

