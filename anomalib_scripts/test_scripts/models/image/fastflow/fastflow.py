import os
from typing import Any

import numpy as np
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from matplotlib import pyplot as plt
from PIL import Image
from torch.utils.data import DataLoader

from anomalib.data import PredictDataset, MVTec
from anomalib.engine import Engine
from anomalib.models import Fastflow
from anomalib.utils.post_processing import superimpose_anomaly_map
from anomalib import TaskType


def fastflow_train_and_test(
        datamodule: Any
):


    """
    FastFlow Model

    Now that we have created the MVTec datamodule, we could create the FastFlow model.
    We could start with printing its docstring.
    """

    # Fastflow??

    model = Fastflow(backbone="resnet18", flow_steps=8)

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
        callbacks=callbacks,
        pixel_metrics="AUROC",
        accelerator="gpu",  # \<"cpu", "gpu", "tpu", "ipu", "hpu", "auto">,
        devices=[0],
        logger=False,
    )

    """
    Trainer object has number of options that suit all specific needs. For more details, refer to Lightning Documentation 
    to see how it could be tweaked to your needs.
    
    Let's train the model now.
    """

    engine.fit(datamodule=datamodule, model=model)

    """
    The training has finished after 12 epochs. This is because, we set the EarlyStopping criteria with a patience of 3, 
    which terminated the training after pixel_AUROC stopped improving. If we increased the patience, the training would 
    continue further.
    """

    """
    Testing
    
    Now that we trained the model, we could test the model to check the overall performance on the test set. We will also 
    be writing the output of the test images to a file since we set VisualizerCallback in callbacks.
    """

    test_result = engine.test(datamodule=datamodule, model=model)

    """
    Inference
    
    Since we have a trained model, we could infer the model on an individual image or folder of images. Anomalib has an 
    PredictDataset to let you create an inference dataset. So let's try it.
    """

    inference_dataset = PredictDataset(path=dataset_root / "bottle/test/broken_large/000.png")
    inference_dataloader = DataLoader(dataset=inference_dataset,
                                      num_workers=8)

    """
    We could utilize Trainer's predict method to infer, and get the outputs to visualize
    """

    predictions = engine.predict(model=model, dataloaders=inference_dataloader)[0]

    """
    predictions contain image, anomaly maps, predicted scores, labels and mask. These are all stored in a dictionary. 
    We could check this by printing the prediction keys.
    """

    print(
        f'Output description:\n'
        f'- Image Shape: {predictions["image"].shape},\n'
        f'- Anomaly Map Shape: {predictions["anomaly_maps"].shape}, \n'
        f'- Predicted Mask Shape: {predictions["pred_masks"].shape}\n',
    )

    """
    Visualization
    
    To properly visualize the predictions, we will need to perform some post-processing operations.
    
    Let's first show the input image. To do so, we will use image_path key from the predictions dictionary, and read the 
    image from path. Note that predictions dictionary already contains image. However, this is the normalized image with 
    pixel values between 0 and 1. We will use the original image to visualize the input image.
    """

    image_path = predictions["image_path"][0]
    image_size = predictions["image"].shape[-2:]
    image = np.array(Image.open(image_path).resize(image_size))

    """
    The first output of the predictions is the anomaly map. As can be seen above, it's also a torch tensor and of size 
    torch.Size([1, 1, 256, 256]). We therefore need to convert it to numpy and squeeze the dimensions to make it 256x256 
    output to visualize.
    """

    anomaly_map = predictions["anomaly_maps"][0]
    anomaly_map = anomaly_map.cpu().numpy().squeeze()
    #plt.imshow(anomaly_map)

    """
    We could superimpose (overlay) the anomaly map on top of the original image to get a heat map. Anomalib has a built-in 
    function to achieve this. Let's try it.
    """

    heat_map = superimpose_anomaly_map(anomaly_map=anomaly_map, image=image, normalize=True)
    #plt.imshow(heat_map)

    """
    predictions also contains prediction scores and labels.
    """

    pred_score = predictions["pred_scores"][0]
    pred_labels = predictions["pred_labels"][0]

    print(
        f"- Example predicted score: {pred_score}\n"
        f"- Example predicted label: {pred_labels}\n")

    """
    The last part of the predictions is the mask that is predicted by the model. This is a boolean mask containing 
    True/False for the abnormal_white/normal_white pixels, respectively.
    """

    #pred_masks = predictions["pred_masks"][0].squeeze().cpu().numpy()
    #plt.imshow(pred_masks)

    return test_result


if __name__ == "__main__":

    """
    Setting up the Dataset Directory

    This cell is to ensure we change the directory to have access to the datasets.
    """

    from pathlib import Path

    # NOTE: Provide the path to the dataset root directory.
    #   If the datasets is not downloaded, it will be downloaded
    #   to this directory.
    dataset_root = Path.cwd() / "datasets" / "MVTec"

    """
    Data Module

    To train the model end-to-end, we do need to have a dataset. In our previous notebooks, we demonstrate how to initialize 
    benchmark- and custom datasets. In this tutorial, we will use MVTec AD DataModule. We assume that datasets directory 
    is created in the anomalib root directory and MVTec dataset is located in datasets directory.

    Before creating the dataset, let's define the task type that we will be working on. In this notebook, we will be working 
    on a segmentation task. Therefore the task variable would be:
    """

    mvtec_categories = [category_name for category_name in os.listdir(dataset_root) if Path(f"{dataset_root}/{category_name}").is_dir()]

    task = TaskType.SEGMENTATION

    for category in mvtec_categories:

        datamodule = MVTec(
            root=dataset_root,
            category=category,
            image_size=(256, 256),
            train_batch_size=32,
            eval_batch_size=32,
            num_workers=8,
            task=task,
        )

        test_result = fastflow_train_and_test(
            datamodule=datamodule
        )

        print(test_result)

