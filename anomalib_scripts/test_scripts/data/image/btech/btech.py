# flake8: noqa
import numpy as np
from PIL import Image
from torchvision.transforms.v2 import Resize
from torchvision.transforms.v2.functional import to_pil_image

from anomalib.data.image.btech import BTech, BTechDataset
from anomalib import TaskType

"""
Setting up the Dataset Directory
This cell is to ensure we change the directory to have access to the datasets.
"""

from pathlib import Path

# NOTE: Provide the path to the dataset root directory.
#   If the datasets is not downloaded, it will be downloaded
#   to this directory.
dataset_root = Path.cwd() / "datasets" / "BTech"
output_dir = "test_scripts/data/image/btech/output_dir"
"""
Use BeanTech Dataset via API
"""

"""
DataModule

Anomalib data modules are based on PyTorch Lightning (PL)'s LightningDataModule class. This class handles all the 
boilerplate code related to subset splitting, and creating the dataset and dataloader instances. A datamodule instance 
can be directly passed to a PL Trainer which is responsible for carrying out Anomalib's training/testing/inference pipelines.

In the current example, we will show how an Anomalib data module can be created for the BeanTech Dataset, and how we can 
obtain training and testing dataloaders from it.

To create a datamodule, we simply pass the path to the root folder of the dataset on the file system, together with some 
basic parameters related to pre-processing and image loading:
"""

btech_datamodule = BTech(
    root=dataset_root,
    category="01",
    image_size=256,
    train_batch_size=32,
    eval_batch_size=32,
    num_workers=8,
    task=TaskType.SEGMENTATION,
)

"""
For the illustrative purposes of the current example, we need to manually call the prepare_data and setup methods. 
Normally it is not necessary to call these methods explicitly, as the PL Trainer would call these automatically under 
the hood.

prepare_data checks if the dataset files can be found at the specified file system location. If not, it will download 
the dataset and place it in the folder.

setup applies the subset splitting and prepares the PyTorch dataset objects for each of the train/val/test subsets.
"""

btech_datamodule.prepare_data()
btech_datamodule.setup()

"""
After the datamodule has been set up, we can use it to obtain the dataloaders of the different subsets.
"""

# Train images
#i, data = next(enumerate(btech_datamodule.train_dataloader()))
#print(data.keys(), data["image"].shape)

# Test images
#i, data = next(enumerate(btech_datamodule.test_dataloader()))
#print(data.keys(), data["image"].shape, data["mask"].shape)

"""
As can be seen above, creating the dataloaders is pretty straghtforward, which could be directly used for 
training/testing/inference. We could visualize samples from the dataloaders as well.
"""

#img = to_pil_image(data["image"][0].clone())
#msk = to_pil_image(data["mask"][0]).convert("RGB")

#Image.fromarray(np.hstack((np.array(img), np.array(msk) * 255))).show()
#Image.fromarray(np.hstack((np.array(img), np.array(msk) * 255))).save(f"{output_dir}/1.png")

"""
Torch Dataset

In some cases it might be desirable to create a standalone PyTorch dataset without a PL data module. For example, this 
could be useful for training a PyTorch model outside Anomalib, so without the use of a PL Trainer instance. 
In such cases, the PyTorch Dataset instance can be instantiated directly.
"""

# BTechDataset??

"""
We can add some transforms that will be applied to the images using torchvision. Let's add a transform that resizes 
the input image to 256x256 pixels.
"""

image_size = (256, 256)
transform = Resize(image_size, antialias=True)

"""
Classification Task
"""

# BTechDataset Classification Train Set
btech_dataset_classification_train = BTechDataset(
    root=dataset_root,
    category="01",
    transform=transform,
    split="train",
    task=TaskType.CLASSIFICATION,
)

#print()
#print(btech_dataset_classification_train.samples.head())
#print()

sample = btech_dataset_classification_train[0]
print(sample.keys(), sample["image"].shape)

"""
As can be seen above, when we choose classification task and train split, the dataset only returns image. 
This is mainly because training only requires normal_white images and no labels. Now let's try test split for the 
classification task
"""

# BTech Classification Test Set
btech_dataset_classification_test = BTechDataset(
    root=dataset_root,
    category="01",
    transform=transform,
    split="test",
    task=TaskType.CLASSIFICATION,
)
sample = btech_dataset_classification_test[0]
print(sample.keys(), sample["image"].shape, sample["image_path"], sample["label"])

"""
where a classification test sample returns image, image_path and label. image_path is used to extract the filename 
when saving images.
"""

"""
Segmentation Task

It is also possible to configure the BTech dataset for the segmentation task, where the dataset object returns image 
and ground-truth mask.
"""

print("\n\n#---SEGMENTATION TASK---#\n")

# BTech Segmentation Train Set
btech_dataset_segmentation_train = BTechDataset(
    root=dataset_root,
    category="01",
    transform=transform,
    split="train",
    task=TaskType.SEGMENTATION,
)

print("Train samples:")
print(f"{btech_dataset_segmentation_train.samples.head()}\n")


"""
The above dataframe stores all the necessary information regarding the dataset. __getitem__ method returns the 
corresponding information depending on the task type or train/test split.
"""

# BTech Segmentation Test Set
btech_dataset_segmentation_test = BTechDataset(
    root=dataset_root,
    category="01",
    transform=transform,
    split="test",
    task=TaskType.SEGMENTATION,
)

print("Test samples:")
print(f"{btech_dataset_segmentation_test.samples.head()}\n")

sample_index = 20
sample = btech_dataset_segmentation_test[sample_index]

print(f"""Test sample n.{sample_index}:
- sample's keys: {list(sample.keys())}
- image path: {sample["image_path"]}
- image shape: {sample["image"].shape}
- mask shape: {sample["mask"].shape}""")

"""
Let's visualize the image and the mask...
"""

# img = Image.open(sample["image_path"]).resize(image_size)
img = to_pil_image(sample["image"].clone())
msk = to_pil_image(sample["mask"]).convert("RGB")

Image.fromarray(np.hstack((np.array(img), np.array(msk) * 255))).show()
Image.fromarray(np.hstack((np.array(img), np.array(msk) * 255))).save(f"{output_dir}/test_sample_{sample_index}_with_mask.png")

print(f"- saved to: {output_dir}/test_sample_{sample_index}_with_mask.png\n")




