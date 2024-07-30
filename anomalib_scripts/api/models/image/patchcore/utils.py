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
from anomalib.utils.normalization import NormalizationMethod
from anomalib import TaskType
from pathlib import Path


def train(
        datamodule: Folder,
        task_type: TaskType,
        model_kwargs: Any=None,
        engine_kwargs: Any=None,
        train_kwargs: Any=None
):

    model = Patchcore()

    """
    Callbacks

    To train the model properly, we will to add some other "non-essential" logic such as saving the weights, early-stopping, 
    normalizing the anomaly scores and visualizing the input/output images. To achieve these we use Callbacks. Anomalib has 
    its own callbacks and also supports PyTorch Lightning's native callbacks. So, let's create the list of callbacks we want 
    to execute during the training.
    """

    callbacks = [
        ModelCheckpoint(
            mode="max",
            monitor="pixel_AUROC",
        ),
        EarlyStopping(
            monitor="pixel_AUROC",
            mode="max",
            patience=3,
        ),
    ]

    """
    Training

    Now that we set up the datamodule, model, optimizer and the callbacks, we could now train the model.

    The final component to train the model is Engine object, which handles train/test/predict pipeline. 
    Let's create the engine object to train the model.
    """

    engine = Engine(
        #callbacks=callbacks,
        #normalization=NormalizationMethod.MIN_MAX, #.NONE
        #threshold="F1AdaptiveThreshold",
        task=task_type,
        #image_metrics=None,
        #pixel_metrics="AUROC",
        accelerator="cpu",  # \<"cpu", "gpu", "tpu", "ipu", "hpu", "auto">,
        #devices=[0],
        logger=False,
        default_root_dir="results"
    )

    engine.fit(datamodule=datamodule, model=model)

    return {
        "engine": engine,
        "model": model
    }


def test(
        datamodule: Folder,
        model: Patchcore,
        engine: Engine,
        model_kwargs: Any=None,
        engine_kwargs: Any=None,
        test_kwargs: Any=None
):

    test_result = engine.test(datamodule=datamodule, model=model)

    #print(engine.__dict__)

    """
    Inference

    Since we have a trained model, we could infer the model on an individual image or folder of images. Anomalib has an 
    PredictDataset to let you create an inference dataset. So let's try it.
    """

    # inference_dataset = PredictDataset(path=dataset_root / "bottle/test/broken_large/000.png")
    # inference_dataloader = DataLoader(dataset=inference_dataset,
    #                                  num_workers=8)

    """
    We could utilize Trainer's predict method to infer, and get the outputs to visualize
    """

    # predictions = engine.predict(model=model, dataloaders=inference_dataloader)[0]

    """
    predictions contain image, anomaly maps, predicted scores, labels and mask. These are all stored in a dictionary. 
    We could check this by printing the prediction keys.
    """

    # print(
    #    f'Output description:\n'
    #    f'- Image Shape: {predictions["image"].shape},\n'
    #    f'- Anomaly Map Shape: {predictions["anomaly_maps"].shape}, \n'
    #    f'- Predicted Mask Shape: {predictions["pred_masks"].shape}\n',
    # )

    """
    Visualization

    To properly visualize the predictions, we will need to perform some post-processing operations.

    Let's first show the input image. To do so, we will use image_path key from the predictions dictionary, and read the 
    image from path. Note that predictions dictionary already contains image. However, this is the normalized image with 
    pixel values between 0 and 1. We will use the original image to visualize the input image.
    """

    # image_path = predictions["image_path"][0]
    # image_size = predictions["image"].shape[-2:]
    # image = np.array(Image.open(image_path).resize(image_size))

    """
    The first output of the predictions is the anomaly map. As can be seen above, it's also a torch tensor and of size 
    torch.Size([1, 1, 256, 256]). We therefore need to convert it to numpy and squeeze the dimensions to make it 256x256 
    output to visualize.
    """

    # anomaly_map = predictions["anomaly_maps"][0]
    # anomaly_map = anomaly_map.cpu().numpy().squeeze()
    # plt.imshow(anomaly_map)

    """
    We could superimpose (overlay) the anomaly map on top of the original image to get a heat map. Anomalib has a built-in 
    function to achieve this. Let's try it.
    """

    # heat_map = superimpose_anomaly_map(anomaly_map=anomaly_map, image=image, normalize=True)
    # plt.imshow(heat_map)

    """
    predictions also contains prediction scores and labels.
    """

    # pred_score = predictions["pred_scores"][0]
    # pred_labels = predictions["pred_labels"][0]

    # print(
    #    f"- Example predicted score: {pred_score}\n"
    #    f"- Example predicted label: {pred_labels}\n")

    """
    The last part of the predictions is the mask that is predicted by the model. This is a boolean mask containing 
    True/False for the abnormal_white/normal_white pixels, respectively.
    """

    # pred_masks = predictions["pred_masks"][0].squeeze().cpu().numpy()
    # plt.imshow(pred_masks)

    return {"result": test_result}

