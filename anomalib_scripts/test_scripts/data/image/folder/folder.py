# flake8: noqa
import numpy as np
from PIL import Image
from anomalib.data.utils import TestSplitMode, ValSplitMode
from torchvision.transforms.v2 import Resize
from torchvision.transforms.v2.functional import to_pil_image

from anomalib.data.image.folder import Folder, FolderDataset
from anomalib import TaskType

from pathlib import Path

"""
Setting up the Dataset Directory

This cell is to ensure we change the directory to have access to the datasets.
"""

# NOTE: Provide the path to the dataset root directory.
#   If the datasets is not downloaded, it will be downloaded
#   to this directory.
dataset_root = Path.cwd() / "custom_datasets" / "hazelnut_toy"

"""
Use Folder Dataset (for Custom Datasets) via API

Here we show how one can utilize custom datasets to train anomalib models. A custom dataset in this model can be of the 
following types:

A dataset with good and bad images.
A dataset with good and bad images as well as mask ground-truths for pixel-wise evaluation.
A dataset with good and bad images that is already split into training and testing sets.
To experiment this setting we provide a toy dataset that could be downloaded from the 
following https://github.com/openvinotoolkit/anomalib/blob/main/docs/source/data/hazelnut_toy.zip. 
For the rest of the tutorial, we assume that the dataset is downloaded and extracted to ../datasets, 
located in the anomalib directory.
"""

"""
DataModule

Similar to how we created the datamodules for existing benchmarking datasets in the previous tutorials, we can also 
create an Anomalib datamodule for our custom hazelnut dataset.

In addition to the root folder of the dataset, we now also specify which folder contains the normal_white images, which folder 
contains the anomalous images, and which folder contains the ground truth mask for the anomalous images.
"""

folder_datamodule = Folder(
    name="hazelnut_toy",
    root=dataset_root,
    normal_dir="good",
    abnormal_dir=["crack", "colour"],
    task=TaskType.SEGMENTATION,
    mask_dir=[dataset_root / "mask" / "crack", dataset_root / "mask" / "colour"],
    image_size=(256, 256),
)
folder_datamodule.setup()

# Train images
i, data = next(enumerate(folder_datamodule.train_dataloader()))
print(data.keys(), data["image"].shape)

print(folder_datamodule.train_dataloader().sampler.data_source.__dict__)

# Test images
i, data = next(enumerate(folder_datamodule.test_dataloader()))
print(data.keys(), data["image"].shape, data["mask"].shape)

print(folder_datamodule.test_dataloader().sampler.data_source.__dict__)

"""
As can be seen above, creating the dataloaders are pretty straghtforward, which could be directly used 
for training/testing/inference. We could visualize samples from the dataloaders as well.
"""

img = to_pil_image(data["image"][0].clone())
msk = to_pil_image(data["mask"][0]).convert("RGB")

Image.fromarray(np.hstack((np.array(img), np.array(msk))))

"""
Folder data module offers much more flexibility cater all different sorts of needs. Please refer to the documentation 
for more details.
"""

"""
Torch Dataset

As in earlier examples, we can also create a standalone PyTorch dataset instance.
"""

# FolderDataset??

"""
We can add some transforms that will be applied to the images using torchvision. Let's add a transform that resizes 
the input image to 256x256 pixels.
"""

image_size = (256, 256)
transform = Resize(image_size, antialias=True)

"""
Classification Task
"""

# MVTec Classification Train Set
folder_dataset_classification_train = FolderDataset(
    name="hazelnut_toy",
    normal_dir=dataset_root / "good",
    abnormal_dir=dataset_root / "crack",
    split="train",
    transform=transform,
    task=TaskType.CLASSIFICATION,
)
folder_dataset_classification_train.samples.head()

"""
Let's look at the first sample in the dataset.
"""

data = folder_dataset_classification_train[0]
print(data.keys(), data["image"].shape)

"""
As can be seen above, when we choose classification task and train split, the dataset only returns image. 
This is mainly because training only requires normal_white images and no labels. Now let's try test split 
for the classification task
"""

# Folder Classification Test Set
folder_dataset_classification_test = FolderDataset(
    name="hazelnut_toy",
    normal_dir=dataset_root / "good",
    abnormal_dir=dataset_root / "crack",
    split="test",
    transform=transform,
    task=TaskType.CLASSIFICATION,
)
folder_dataset_classification_test.samples.head()

data = folder_dataset_classification_test[0]
print(data.keys(), data["image"].shape, data["image_path"], data["label"])

"""
Segmentation Task

It is also possible to configure the Folder dataset for the segmentation task, where the dataset object returns 
image and ground-truth mask.
"""

# Folder Segmentation Train Set
folder_dataset_segmentation_train = FolderDataset(
    name="hazelnut_toy",
    normal_dir=dataset_root / "good",
    abnormal_dir=dataset_root / "crack",
    split="train",
    transform=transform,
    mask_dir=dataset_root / "mask" / "crack",
    task=TaskType.SEGMENTATION,
)

print(folder_dataset_segmentation_train.samples.head(100))

# Folder Segmentation Test Set
folder_dataset_segmentation_test = FolderDataset(
    name="hazelnut_toy",
    normal_dir=dataset_root / "good",
    abnormal_dir=dataset_root / "crack",
    split="test",
    transform=transform,
    mask_dir=dataset_root / "mask" / "crack",
    task=TaskType.SEGMENTATION,
)

print(folder_dataset_segmentation_test.samples.head(100))

data = folder_dataset_segmentation_test[3]
print(data.keys(), data["image"].shape, data["mask"].shape)

"""
Let's visualize the image and the mask...
"""

img = to_pil_image(data["image"].clone())
msk = to_pil_image(data["mask"]).convert("RGB")

Image.fromarray(np.hstack((np.array(img), np.array(msk))))
