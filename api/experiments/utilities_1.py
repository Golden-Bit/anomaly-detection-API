import json
import os
from pathlib import Path
from typing import Any, Dict, List

#import numpy as np
#from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
#from matplotlib import pyplot as plt
#from PIL import Image
from torch.utils.data import DataLoader

from anomalib.data import PredictDataset, MVTec, Folder
#from anomalib.data.image.folder import FolderDataset
from anomalib.engine import Engine
from anomalib.models import Padim, AnomalyModule
#from anomalib.utils.post_processing import superimpose_anomaly_map
#from anomalib.utils.normalization import NormalizationMethod
from anomalib import TaskType


def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, 'r') as file:
        config = json.load(file)
    return config


def train(config: Dict[str, Any]):
    dataset_config = config['dataset']
    model_config = config['model']
    engine_config = config['engine']
    train_config = config['train']

    # Create DataModule
    datamodule = Folder(
        name=dataset_config['name'],
        root=Path(dataset_config['root']),
        normal_dir=dataset_config['normal_train_dirs'],
        abnormal_dir=dataset_config.get('abnormal_train_dirs'),
        mask_dir=dataset_config.get('mask_train_dirs'),
        task=TaskType[dataset_config['task']],
        image_size=tuple(dataset_config['image_size']),
    )

    datamodule.setup()

    # Create Model
    model = Padim(**model_config)

    # Create Engine
    engine = Engine(**engine_config)

    engine.fit(
        model=model,
        train_dataloaders=datamodule.train_dataloader(),
        val_dataloaders=datamodule.val_dataloader(),
        **train_config
    )

    return engine, model


def test(config: Dict[str, Any],
         engine: Engine,
         model: AnomalyModule):

    dataset_config = config['dataset']
    test_config = config['test']

    datamodule = Folder(
        name=dataset_config['name'],
        root=Path(dataset_config['root']),
        normal_dir=dataset_config['normal_eval_dirs'],
        abnormal_dir=dataset_config.get('abnormal_eval_dirs'),
        mask_dir=dataset_config.get('mask_eval_dirs'),
        task=TaskType[dataset_config['task']],
        image_size=tuple(dataset_config['image_size']),
    )

    datamodule.setup()

    test_results = engine.test(
        datamodule=datamodule,
        model=model,
        dataloaders=datamodule.test_dataloader(),
        **test_config
    )

    return test_results


def benchmark(config: Dict[str, Any], engine: Engine, model: AnomalyModule):
    dataset_config = config['dataset']
    benchmark_config = config['benchmark']

    datamodule = Folder(
        name=dataset_config['name'],
        root=Path(dataset_config['root']),
        normal_dir=dataset_config['normal_eval_dirs'],
        abnormal_dir=dataset_config.get('abnormal_eval_dirs'),
        mask_dir=dataset_config.get('mask_eval_dirs'),
        task=TaskType[dataset_config['task']],
        image_size=tuple(dataset_config['image_size']),
    )

    datamodule.setup()

    benchmark_results = engine.test(
        model=model,
        dataloaders=datamodule.test_dataloader(),
        **benchmark_config
    )

    return benchmark_results


def infer(config: Dict[str, Any], engine: Engine, model: AnomalyModule):
    dataset_config = config['dataset']
    infer_config = config['infer']

    data_paths = [Path(dataset_config['root']) / path for path in infer_config['data_paths']]
    results = []

    for data_path in data_paths:
        inference_dataset = PredictDataset(path=data_path)
        inference_dataloader = DataLoader(dataset=inference_dataset, batch_size=infer_config['batch_size'])

        predictions = engine.predict(
            model=model,
            dataloaders=inference_dataloader,
            **infer_config
        )

        results.append(predictions)

    return results


if __name__ == "__main__":
    config_path = "config.json"
    config = load_config(config_path)

    # Train the model
    engine, model = train(config)

    # Test the model
    test_results = test(config, engine, model)
    print("Test Results:", test_results)

    # Benchmark the model
    benchmark_results = benchmark(config, engine, model)
    print("Benchmark Results:", benchmark_results)

    # Perform inference
    infer_results = infer(config, engine, model)
    print("Inference Results:", infer_results)
