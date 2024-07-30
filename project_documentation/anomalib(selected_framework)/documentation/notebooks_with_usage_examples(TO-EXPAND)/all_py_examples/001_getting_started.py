from typing import Any

import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
from torchvision.transforms import ToPILImage

from anomalib import TaskType
from anomalib.data import MVTec
from anomalib.data.utils import read_image
from anomalib.deploy import ExportType, OpenVINOInferencer
from anomalib.engine import Engine
from anomalib.models import Padim

from pathlib import Path

"""
Model

Currently, there are 13 anomaly detection models available in anomalib library. Namely,

    CFA
    CS-Flow
    CFlow
    DFKDE
    DFM
    DRAEM
    FastFlow
    Ganomaly
    Padim
    Patchcore
    Reverse Distillation
    R-KDE
    STFPM

In this tutorial, we'll be using Padim.
"""

"""
Dataset: MVTec AD

MVTec AD is a dataset for benchmarking anomaly detection methods with a focus on industrial inspection. It contains over 5000 high-resolution images divided into 15 different object and texture categories. Each category comprises a set of defect-free training images and a test set of images with various kinds of defects as well as images without defects. If the dataset is not located in the root datasets directory, anomalib will automatically install the dataset.

We could now import the MVtec AD dataset using its specific datamodule implemented in anomalib.
"""


# We could now import the MVtec AD dataset using its specific datamodule implemented in anomalib.

datamodule = MVTec(root="./datasets/MVTec",
                   num_workers=0)

datamodule.prepare_data()  # Downloads the dataset if it's not in the specified `root` directory
datamodule.setup()  # Create train/val/test/prediction sets.

i, data = next(enumerate(datamodule.val_dataloader()))
print(data.keys())


# Let's check the shapes of the input images and masks.

print(data["image"].shape, data["mask"].shape)


# We could now visualize a normal and abnormal sample from the validation set.

def show_image_and_mask(sample: dict[str, Any], index: int) -> Image:
    """Show an image with a mask.

    Args:
        sample (dict[str, Any]): Sample from the dataset.
        index (int): Index of the sample.

    Returns:
        Image: Output image with a mask.
    """
    # Load the image from the path
    image = Image.open(sample["image_path"][index])

    # Load the mask and convert it to RGB
    mask = ToPILImage()(sample["mask"][index]).convert("RGB")

    # Resize mask to match image size, if they differ
    if image.size != mask.size:
        mask = mask.resize(image.size)

    return Image.fromarray(np.hstack((np.array(image), np.array(mask))))


# Visualize an image with a mask
show_image_and_mask(data, index=0)


"""
Prepare Model

Let's create the Padim and train it.
"""

# Get the model and datamodule
model = Padim()
datamodule = MVTec(num_workers=0)


# start training
engine = Engine(task=TaskType.SEGMENTATION)
engine.fit(model=model, datamodule=datamodule)


"""
Validation
"""

# load best model from checkpoint before evaluating
test_results = engine.test(
    model=model,
    datamodule=datamodule,
    ckpt_path=engine.trainer.checkpoint_callback.best_model_path,
)

print(test_results)


"""
OpenVINO Inference

Now that we trained and tested a model, we could check a single inference result using OpenVINO inferencer object. This will demonstrate how a trained model could be used for inference.
"""


# Before we can use OpenVINO inference, let's export the model to OpenVINO format first.

engine.export(
    model=model,
    export_type=ExportType.OPENVINO,
)


# Load a Test Image
# Let's read an image from the test set and perform inference using OpenVINO inferencer.

image_path = "datasets/MVTec/bottle/test/broken_large/000.png"
image = read_image(path="./datasets/MVTec/bottle/test/broken_large/000.png")
plt.imshow(image)


# Load the OpenVINO Model
# By default, the output files are saved into results directory. Let's check where the OpenVINO model is stored.

output_path = Path(engine.trainer.default_root_dir)
print(output_path)

openvino_model_path = output_path / "weights" / "openvino" / "model.bin"
metadata = output_path / "weights" / "openvino" / "metadata.json"
print(openvino_model_path.exists(), metadata.exists())

inferencer = OpenVINOInferencer(
    path=openvino_model_path,  # Path to the OpenVINO IR model.
    metadata=metadata,  # Path to the metadata file.
    device="CPU",  # We would like to run it on an Intel CPU.
)


# Perform Inference
# Predicting an image using OpenVINO inferencer is as simple as calling predict method.

predictions = inferencer.predict(image=image_path)

# where predictions contain any relevant information regarding the task type.
# For example, predictions for a segmentation model could contain image, anomaly maps, predicted scores, labels or masks.


# Visualizing Inference Results

print(predictions.pred_score, predictions.pred_label)


# Visualize the original image
plt.imshow(predictions.image)


# Visualize the raw anomaly maps predicted by the model.
plt.imshow(predictions.anomaly_map)


# Visualize the heatmaps, on which raw anomaly map is overlayed on the original image.
plt.imshow(predictions.heat_map)


# Visualize the segmentation mask.
plt.imshow(predictions.pred_mask)


# Visualize the segmentation mask with the original image.
plt.imshow(predictions.segmentations)

