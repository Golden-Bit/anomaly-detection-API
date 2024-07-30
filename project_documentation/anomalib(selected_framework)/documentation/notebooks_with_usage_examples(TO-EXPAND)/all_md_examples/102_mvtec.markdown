---
jupyter:
  kernelspec:
    display_name: anomalib
    language: python
    name: python3
  language_info:
    codemirror_mode:
      name: ipython
      version: 3
    file_extension: .py
    mimetype: text/x-python
    name: python
    nbconvert_exporter: python
    pygments_lexer: ipython3
    version: 3.10.13
  nbformat: 4
  nbformat_minor: 2
  orig_nbformat: 4
  vscode:
    interpreter:
      hash: ae223df28f60859a2f400fae8b3a1034248e0a469f5599fd9a89c32908ed7a84
---

::: {.cell .markdown}
# Use MVTec AD Dataset via API

# Installing Anomalib

The easiest way to install anomalib is to use pip. You can install it
from the command line using the following command:
:::

::: {.cell .code}
``` python
%pip install anomalib
```
:::

::: {.cell .code execution_count="1"}
``` python
# flake8: noqa
import numpy as np
from PIL import Image
from torchvision.transforms.v2 import Resize
from torchvision.transforms.v2.functional import to_pil_image

from anomalib.data.image.mvtec import MVTec, MVTecDataset
from anomalib import TaskType
```
:::

::: {.cell .markdown}
## Setting up the Dataset Directory

This cell is to ensure we change the directory to have access to the
datasets.
:::

::: {.cell .code execution_count="2"}
``` python
from pathlib import Path

# NOTE: Provide the path to the dataset root directory.
#   If the datasets is not downloaded, it will be downloaded
#   to this directory.
dataset_root = Path.cwd().parent / "datasets" / "MVTec"
```
:::

::: {.cell .markdown}
### DataModule

Anomalib data modules are based on PyTorch Lightning (PL)\'s
`LightningDataModule` class. This class handles all the boilerplate code
related to subset splitting, and creating the dataset and dataloader
instances. A datamodule instance can be directly passed to a PL Trainer
which is responsible for carrying out Anomalib\'s
training/testing/inference pipelines.

In the current example, we will show how an Anomalib data module can be
created for the MVTec Dataset, and how we can obtain training and
testing dataloaders from it.

To create a datamodule, we simply pass the path to the root folder of
the dataset on the file system, together with some basic parameters
related to pre-processing and image loading:
:::

::: {.cell .code execution_count="3"}
``` python
mvtec_datamodule = MVTec(
    root=dataset_root,
    category="bottle",
    image_size=256,
    train_batch_size=32,
    eval_batch_size=32,
    num_workers=0,
    task=TaskType.SEGMENTATION,
)
```
:::

::: {.cell .markdown}
For the illustrative purposes of the current example, we need to
manually call the `prepare_data` and `setup` methods. Normally it is not
necessary to call these methods explicitly, as the PL Trainer would call
these automatically under the hood.

`prepare_data` checks if the dataset files can be found at the specified
file system location. If not, it will download the dataset and place it
in the folder.

`setup` applies the subset splitting and prepares the PyTorch dataset
objects for each of the train/val/test subsets.
:::

::: {.cell .code execution_count="4"}
``` python
mvtec_datamodule.prepare_data()
mvtec_datamodule.setup()
```
:::

::: {.cell .markdown}
After the datamodule has been set up, we can use it to obtain the
dataloaders of the different subsets.
:::

::: {.cell .code execution_count="5"}
``` python
# Train images
i, data = next(enumerate(mvtec_datamodule.train_dataloader()))
print(data.keys(), data["image"].shape)
```

::: {.output .stream .stdout}
    dict_keys(['image_path', 'label', 'image', 'mask']) torch.Size([32, 3, 900, 900])
:::
:::

::: {.cell .code execution_count="6"}
``` python
# Test images
i, data = next(enumerate(mvtec_datamodule.test_dataloader()))
print(data.keys(), data["image"].shape, data["mask"].shape)
```

::: {.output .stream .stdout}
    dict_keys(['image_path', 'label', 'image', 'mask']) torch.Size([32, 3, 900, 900]) torch.Size([32, 900, 900])
:::
:::

::: {.cell .markdown}
As can be seen above, creating the dataloaders are pretty
straghtforward, which could be directly used for
training/testing/inference. We could visualize samples from the
dataloaders as well.
:::

::: {.cell .code execution_count="7"}
``` python
img = to_pil_image(data["image"][0].clone())
msk = to_pil_image(data["mask"][0]).convert("RGB")

Image.fromarray(np.hstack((np.array(img), np.array(msk))))
```

::: {.output .execute_result execution_count="7"}
![](vertopal_9d55f35058144f1cbb8f798ff90f49c8/dd6721f3eb7c2e83ade5da32759143bdd9a085fe.png)
:::
:::

::: {.cell .markdown}
### Torch Dataset

In some cases it might be desirable to create a standalone PyTorch
dataset without a PL data module. For example, this could be useful for
training a PyTorch model outside Anomalib, so without the use of a PL
Trainer instance. In such cases, the PyTorch Dataset instance can be
instantiated directly.
:::

::: {.cell .code}
``` python
MVTecDataset??
```
:::

::: {.cell .markdown}
We can add some transforms that will be applied to the images using
torchvision. Let\'s add a transform that resizes the input image to
256x256 pixels.
:::

::: {.cell .code execution_count="8"}
``` python
image_size = (256, 256)
transform = Resize(image_size, antialias=True)
```
:::

::: {.cell .markdown}
#### Classification Task
:::

::: {.cell .code execution_count="9"}
``` python
# MVTec Classification Train Set
mvtec_dataset_classification_train = MVTecDataset(
    root=dataset_root,
    category="bottle",
    transform=transform,
    split="train",
    task="classification",
)
mvtec_dataset_classification_train.samples.head()
```

::: {.output .execute_result execution_count="9"}
```{=html}
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>path</th>
      <th>split</th>
      <th>label</th>
      <th>image_path</th>
      <th>label_index</th>
      <th>mask_path</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>/home/djameln/datasets/MVTec/bottle</td>
      <td>train</td>
      <td>good</td>
      <td>/home/djameln/datasets/MVTec/bottle/train/good...</td>
      <td>0</td>
      <td></td>
    </tr>
    <tr>
      <th>1</th>
      <td>/home/djameln/datasets/MVTec/bottle</td>
      <td>train</td>
      <td>good</td>
      <td>/home/djameln/datasets/MVTec/bottle/train/good...</td>
      <td>0</td>
      <td></td>
    </tr>
    <tr>
      <th>2</th>
      <td>/home/djameln/datasets/MVTec/bottle</td>
      <td>train</td>
      <td>good</td>
      <td>/home/djameln/datasets/MVTec/bottle/train/good...</td>
      <td>0</td>
      <td></td>
    </tr>
    <tr>
      <th>3</th>
      <td>/home/djameln/datasets/MVTec/bottle</td>
      <td>train</td>
      <td>good</td>
      <td>/home/djameln/datasets/MVTec/bottle/train/good...</td>
      <td>0</td>
      <td></td>
    </tr>
    <tr>
      <th>4</th>
      <td>/home/djameln/datasets/MVTec/bottle</td>
      <td>train</td>
      <td>good</td>
      <td>/home/djameln/datasets/MVTec/bottle/train/good...</td>
      <td>0</td>
      <td></td>
    </tr>
  </tbody>
</table>
</div>
```
:::
:::

::: {.cell .code execution_count="10"}
``` python
sample = mvtec_dataset_classification_train[0]
print(sample.keys(), sample["image"].shape)
```

::: {.output .stream .stdout}
    dict_keys(['image_path', 'label', 'image']) torch.Size([3, 256, 256])
:::
:::

::: {.cell .markdown}
As can be seen above, when we choose `classification` task and `train`
split, the dataset only returns `image`. This is mainly because training
only requires normal images and no labels. Now let\'s try `test` split
for the `classification` task
:::

::: {.cell .code execution_count="11"}
``` python
# MVTec Classification Test Set
mvtec_dataset_classification_test = MVTecDataset(
    root=dataset_root,
    category="bottle",
    transform=transform,
    split="test",
    task="classification",
)
sample = mvtec_dataset_classification_test[0]
print(sample.keys(), sample["image"].shape, sample["image_path"], sample["label"])
```

::: {.output .stream .stdout}
    dict_keys(['image_path', 'label', 'image']) torch.Size([3, 256, 256]) /home/djameln/datasets/MVTec/bottle/test/broken_large/000.png 1
:::
:::

::: {.cell .markdown}
#### Segmentation Task

It is also possible to configure the MVTec dataset for the segmentation
task, where the dataset object returns image and ground-truth mask.
:::

::: {.cell .code execution_count="12"}
``` python
# MVTec Segmentation Train Set
mvtec_dataset_segmentation_train = MVTecDataset(
    root=dataset_root,
    category="bottle",
    transform=transform,
    split="train",
    task="segmentation",
)
mvtec_dataset_segmentation_train.samples.head()
```

::: {.output .execute_result execution_count="12"}
```{=html}
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>path</th>
      <th>split</th>
      <th>label</th>
      <th>image_path</th>
      <th>label_index</th>
      <th>mask_path</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>/home/djameln/datasets/MVTec/bottle</td>
      <td>train</td>
      <td>good</td>
      <td>/home/djameln/datasets/MVTec/bottle/train/good...</td>
      <td>0</td>
      <td></td>
    </tr>
    <tr>
      <th>1</th>
      <td>/home/djameln/datasets/MVTec/bottle</td>
      <td>train</td>
      <td>good</td>
      <td>/home/djameln/datasets/MVTec/bottle/train/good...</td>
      <td>0</td>
      <td></td>
    </tr>
    <tr>
      <th>2</th>
      <td>/home/djameln/datasets/MVTec/bottle</td>
      <td>train</td>
      <td>good</td>
      <td>/home/djameln/datasets/MVTec/bottle/train/good...</td>
      <td>0</td>
      <td></td>
    </tr>
    <tr>
      <th>3</th>
      <td>/home/djameln/datasets/MVTec/bottle</td>
      <td>train</td>
      <td>good</td>
      <td>/home/djameln/datasets/MVTec/bottle/train/good...</td>
      <td>0</td>
      <td></td>
    </tr>
    <tr>
      <th>4</th>
      <td>/home/djameln/datasets/MVTec/bottle</td>
      <td>train</td>
      <td>good</td>
      <td>/home/djameln/datasets/MVTec/bottle/train/good...</td>
      <td>0</td>
      <td></td>
    </tr>
  </tbody>
</table>
</div>
```
:::
:::

::: {.cell .code execution_count="13"}
``` python
# MVTec Segmentation Test Set
mvtec_dataset_segmentation_test = MVTecDataset(
    root=dataset_root,
    category="bottle",
    transform=transform,
    split="test",
    task="segmentation",
)
sample = mvtec_dataset_segmentation_test[20]
print(sample.keys(), sample["image"].shape, sample["mask"].shape)
```

::: {.output .stream .stdout}
    dict_keys(['image_path', 'label', 'image', 'mask']) torch.Size([3, 256, 256]) torch.Size([256, 256])
:::
:::

::: {.cell .markdown}
Let\'s visualize the image and the mask\...
:::

::: {.cell .code execution_count="14"}
``` python
img = to_pil_image(sample["image"].clone())
msk = to_pil_image(sample["mask"]).convert("RGB")

Image.fromarray(np.hstack((np.array(img), np.array(msk))))
```

::: {.output .execute_result execution_count="14"}
![](vertopal_9d55f35058144f1cbb8f798ff90f49c8/0c9e66f0afd8b70a298c402e9603084086b917bd.png)
:::
:::
