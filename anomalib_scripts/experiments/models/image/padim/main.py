import os
from typing import Any

import numpy as np
from anomalib.data.utils import ValSplitMode, TestSplitMode
from anomalib.utils.normalization import NormalizationMethod
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from matplotlib import pyplot as plt
from PIL import Image
from torch.utils.data import DataLoader

from anomalib.data import PredictDataset, MVTec, Folder
from anomalib.data.image.folder import FolderDataset
from anomalib.engine import Engine
from anomalib.models import Padim
from anomalib.utils.post_processing import superimpose_anomaly_map
from anomalib import TaskType

from torchvision import transforms

from pathlib import Path

from utils import (
    train,
    test,
    predict
)


if __name__ == "__main__":

    dataset_root = Path.cwd() / "custom_datasets" / "cabinet_metallic_surface" / "blue" / "2-3-5_without-background"

    normal_dirs = [
        dataset_root / "normal" / "type_1___2___without_background___normal_tiles___512-512_32-32",
        dataset_root / "normal" / "type_1___3___without_background___normal_tiles___512-512_32-32",
        dataset_root / "normal" / "type_1___5___without_background___normal_tiles___512-512_32-32"
    ]

    abnormal_dirs = [
        dataset_root / "abnormal" / "original_anomalies",
        dataset_root / "abnormal" / "cropped_whitebackground_original_anomalies",
        dataset_root / "abnormal" / "cropped_original_anomalies",
        dataset_root / "abnormal" / "resized_test_anomalies",

        dataset_root / "abnormal" / "type_1___2___without_background___abnormal_tiles___512-512_32-32_1336-2672",
        dataset_root / "abnormal" / "type_1___2___without_background___abnormal_tiles___512-512_32-32_2672-5344",
        dataset_root / "abnormal" / "type_1___2___without_background___abnormal_tiles___512-512_32-32_5344-10688",
        dataset_root / "abnormal" / "type_1___2___without_background___abnormal_tiles___512-512_32-32_10688-21376",
        dataset_root / "abnormal" / "type_1___2___without_background___abnormal_tiles___512-512_32-32_21376-42752",

        dataset_root / "abnormal" / "type_1___3___without_background___abnormal_tiles___512-512_32-32_1336-2672",
        dataset_root / "abnormal" / "type_1___3___without_background___abnormal_tiles___512-512_32-32_2672-5344",
        dataset_root / "abnormal" / "type_1___3___without_background___abnormal_tiles___512-512_32-32_5344-10688",
        dataset_root / "abnormal" / "type_1___3___without_background___abnormal_tiles___512-512_32-32_10688-21376",
        dataset_root / "abnormal" / "type_1___3___without_background___abnormal_tiles___512-512_32-32_21376-42752",

        dataset_root / "abnormal" / "type_1___5___without_background___abnormal_tiles___512-512_32-32_334-668",
        dataset_root / "abnormal" / "type_1___5___without_background___abnormal_tiles___512-512_32-32_668-1336",
        dataset_root / "abnormal" / "type_1___5___without_background___abnormal_tiles___512-512_32-32_1336-2672",
        dataset_root / "abnormal" / "type_1___5___without_background___abnormal_tiles___512-512_32-32_2672-5344",
        dataset_root / "abnormal" / "type_1___5___without_background___abnormal_tiles___512-512_32-32_5344-10688",
        dataset_root / "abnormal" / "type_1___5___without_background___abnormal_tiles___512-512_32-32_10688-21376",
        dataset_root / "abnormal" / "type_1___5___without_background___abnormal_tiles___512-512_32-32_21376-42752",

        dataset_root / "abnormal" / "resized_test_anomalies",
    ]

    mask_dirs = [
        dataset_root / "mask" / "original_anomalies",
        dataset_root / "mask" / "cropped_whitebackground_original_anomalies",
        dataset_root / "mask" / "cropped_original_anomalies",
        dataset_root / "mask" / "resized_test_anomalies",

        dataset_root / "mask" / "type_1___2___without_background___abnormal_tiles___512-512_32-32_1336-2672",
        dataset_root / "mask" / "type_1___2___without_background___abnormal_tiles___512-512_32-32_2672-5344",
        dataset_root / "mask" / "type_1___2___without_background___abnormal_tiles___512-512_32-32_5344-10688",
        dataset_root / "mask" / "type_1___2___without_background___abnormal_tiles___512-512_32-32_10688-21376",
        dataset_root / "mask" / "type_1___2___without_background___abnormal_tiles___512-512_32-32_21376-42752",

        dataset_root / "mask" / "type_1___3___without_background___abnormal_tiles___512-512_32-32_1336-2672",
        dataset_root / "mask" / "type_1___3___without_background___abnormal_tiles___512-512_32-32_2672-5344",
        dataset_root / "mask" / "type_1___3___without_background___abnormal_tiles___512-512_32-32_5344-10688",
        dataset_root / "mask" / "type_1___3___without_background___abnormal_tiles___512-512_32-32_10688-21376",
        dataset_root / "mask" / "type_1___3___without_background___abnormal_tiles___512-512_32-32_21376-42752",

        dataset_root / "mask" / "type_1___5___without_background___abnormal_tiles___512-512_32-32_334-668",
        dataset_root / "mask" / "type_1___5___without_background___abnormal_tiles___512-512_32-32_668-1336",
        dataset_root / "mask" / "type_1___5___without_background___abnormal_tiles___512-512_32-32_1336-2672",
        dataset_root / "mask" / "type_1___5___without_background___abnormal_tiles___512-512_32-32_2672-5344",
        dataset_root / "mask" / "type_1___5___without_background___abnormal_tiles___512-512_32-32_5344-10688",
        dataset_root / "mask" / "type_1___5___without_background___abnormal_tiles___512-512_32-32_10688-21376",
        dataset_root / "mask" / "type_1___5___without_background___abnormal_tiles___512-512_32-32_21376-42752",
    ]

    normal_train_dirs = normal_dirs[:] + [abnormal_dirs[3] for i in range(3)]
    abnormal_train_dirs = None # abnormal_dirs#[1:3]
    mask_train_dirs = None #mask_dirs#[1:3]

    normal_eval_dirs = normal_dirs[:]
    abnormal_eval_dirs = abnormal_dirs[16:19]
    mask_eval_dirs = mask_dirs[16:19]

    normal_test_dirs = abnormal_dirs[3:4]
    abnormal_test_dirs = abnormal_dirs[3:4]
    mask_test_dirs = mask_dirs[3:4]

    task = TaskType.SEGMENTATION

    normalization = NormalizationMethod.MIN_MAX

    # Folder Segmentation Train Set

    folder_dataset_train = FolderDataset(
        name="cabinet-metallic_blue-surface_without-background_512-512_32-32_v1.0",
        root=dataset_root,
        normal_dir=normal_train_dirs,
        abnormal_dir=None,
        #split='train',
        #transform=None,
        mask_dir=None,
        task=TaskType.SEGMENTATION,
    )

    _train_dataloader = DataLoader(
        dataset=folder_dataset_train,
        batch_size=32,
        shuffle=False,
        num_workers=8,
    )

    print(folder_dataset_train.samples.head())

    folder_dataset_eval = FolderDataset(
        name="cabinet-metallic_blue-surface_without-background_512-512_32-32_v1.0",
        root=dataset_root,
        normal_dir=normal_eval_dirs,
        abnormal_dir=abnormal_eval_dirs,
        #split='val',
        #transform=None,
        mask_dir=mask_eval_dirs,
        task=TaskType.SEGMENTATION,
    )

    print(folder_dataset_eval.samples.head())

    _val_dataloader = DataLoader(
        dataset=folder_dataset_eval,
        batch_size=32,
        shuffle=False,
        num_workers=8,
    )

    folder_dataset_test = FolderDataset(
        name="cabinet-metallic_blue-surface_without-background_512-512_32-32_v1.0",
        root=dataset_root,
        normal_dir=normal_test_dirs,
        abnormal_dir=abnormal_test_dirs,
        #split='test',
        #transform=None,
        mask_dir=mask_test_dirs,
        task=TaskType.SEGMENTATION,
    )

    print(folder_dataset_eval.samples.head())

    _test_dataloader = DataLoader(
        dataset=folder_dataset_test,
        batch_size=32,
        shuffle=False,
        num_workers=8,
    )

    engine_kwargs = dict()
    engine_kwargs["callbacks"]: []
    engine_kwargs["normalization"] = normalization
    #engine_kwargs["threshold"] = "F1AdaptiveThreshold"
    engine_kwargs["task"] = task
    #engine_kwargs["image_metrics"] = None
    engine_kwargs["pixel_metrics"] = "AUROC"
    engine_kwargs["accelerator"] = "gpu"  # \<"cpu", "gpu", "tpu", "ipu", "hpu", "auto">,
    engine_kwargs["devices"] = [0]
    engine_kwargs["logger"] = False
    engine_kwargs["default_root_dir"] = "results/cabinet-metallic_blue-surface_without-background_512-512_32-32_v1.0"

    model_kwargs = dict()
    model_kwargs["backbone"] = "resnet18"  # "wide_resnet50_2"
    model_kwargs["layers"] = ["layer1", "layer2", "layer3"]

    train_result = train(
        datamodule=None,
        train_dataloader=_train_dataloader,
        eval_dataloader=_val_dataloader,
        model_kwargs=model_kwargs,
        engine_kwargs=engine_kwargs,
        train_kwargs=dict(),
    )

    model = train_result["model"]
    engine = train_result["engine"]

    test_result = test(
        datamodule=None,
        test_dataloader=_test_dataloader,
        model=model,
        engine=engine,
        engine_kwargs=engine_kwargs,
        test_kwargs=dict(),
    )

    """    
    engine_kwargs = dict()
    engine_kwargs["callbacks"]: []
    engine_kwargs["normalization"] = normalization
    #engine_kwargs["threshold"] = "F1AdaptiveThreshold"
    engine_kwargs["task"] = task
    #engine_kwargs["image_metrics"] = None
    engine_kwargs["pixel_metrics"] = "AUROC"
    engine_kwargs["accelerator"] = "auto"  # \<"cpu", "gpu", "tpu", "ipu", "hpu", "auto">,
    engine_kwargs["devices"] = 1
    engine_kwargs["logger"] = False
    engine_kwargs["default_root_dir"] = "results/cabinet-metallic_blue-surface_without-background_512-512_32-32_v1.0"

    predictions_result = predict(
        dataset_path=abnormal_dirs[-1],
        #engine=engine,
        engine_kwargs=engine_kwargs,
        model=model,
        output_dir_path="results/cabinet-metallic_blue-surface_without-background_512-512_32-32_v1.0/predictions"
    )

    print(test_result)
    print(predictions_result)"""

