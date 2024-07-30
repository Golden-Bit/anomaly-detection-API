import os
from typing import Any, Dict, List

import numpy as np
from anomalib.data.utils import ValSplitMode, TestSplitMode
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from matplotlib import pyplot as plt
from PIL import Image
from torch.utils.data import DataLoader

from anomalib.data import PredictDataset, MVTec, Folder
from anomalib.data.image.folder import FolderDataset
from anomalib.engine import Engine
from anomalib.models import Padim, AnomalyModule
from anomalib.utils.post_processing import superimpose_anomaly_map
from anomalib.utils.normalization import NormalizationMethod
from anomalib import TaskType
from pathlib import Path


# TODO:
#  1. load models from path
#  2.


def train(
        datamodule: Folder = None,
        train_dataloader: Any = None,
        eval_dataloader: Any = None,
        model: Padim = None,
        model_kwargs: Dict[str, Any] = None,
        engine: Engine = None,
        engine_kwargs: Dict[str, Any] = None,
        train_kwargs: Dict[str, Any] = None,
):

    if not model_kwargs and not model:
        model_kwargs = {
            "backbone": "resnet18", #"wide_resnet50_2"
            "layers": ["layer1", "layer2", "layer3"],
        }

    if not engine_kwargs and not engine:
        engine_kwargs = {

        }

    if not model:
        model = Padim(**model_kwargs)

    if not engine:
        engine = Engine(**engine_kwargs)

    engine.fit(
        datamodule=datamodule,
        train_dataloaders=train_dataloader,
        val_dataloaders=eval_dataloader,
        model=model,
        **train_kwargs,
    )

    return {
        "engine": engine,
        "model": model
    }


def test(
        datamodule: Folder = None,
        test_dataloader: Any = None,
        model: Padim = None,
        engine: Engine = None,
        engine_kwargs: Any = None,
        test_kwargs: Any = None
):

    if not engine_kwargs and not engine:
        engine_kwargs = {

        }

    if not engine:
        engine = Engine(**engine_kwargs)

    test_result = engine.test(datamodule=datamodule,
                              model=model,
                              dataloaders=test_dataloader,
                              **test_kwargs)

    return {"result": test_result}


def predict(
        dataset_path: str | Path | List[str | Path] = None,
        data_path: str | Path = None,
        model: AnomalyModule | str = None,
        model_ckpt_path: str | Path = None,
        engine: Engine = None,
        engine_kwargs: Dict[str, Any] = None,
        output_dir_path: str | Path = None,
):

    if not engine_kwargs and not engine:
        engine_kwargs = dict()

    if output_dir_path:
        os.makedirs(output_dir_path, exist_ok=True)
        engine_kwargs["default_root_dir"] = output_dir_path
    elif "output_dir_path" in engine_kwargs and engine_kwargs.get("output_dir_path", None):
        os.makedirs(engine_kwargs["default_root_dir"], exist_ok=True)

    if not engine:
        engine = Engine(**engine_kwargs)

    data_paths = [dataset_path / data for data in os.listdir(dataset_path)] if dataset_path else [data_path]

    for data_path in data_paths:

        print(data_path)

        inference_dataset = PredictDataset(path=data_path)
                                           #image_size=(512, 512))

        inference_dataloader = DataLoader(dataset=inference_dataset)#t,
                                          #num_workers=8)

        predictions = engine.predict(model=model,
                                     dataloaders=inference_dataloader,
                                     #dataloaders=inference_dataloader,
                                     #ckpt_path=model_ckpt_path,
                                     #return_predictions=None
        )[0]

        return predictions

