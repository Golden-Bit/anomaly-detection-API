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
# Use `Folder` for Customs Datasets

# Installing Anomalib

The easiest way to install anomalib is to use pip. You can install it
from the command line using the following command:
:::

::: {.cell .code}
``` python
%pip install anomalib
```
:::

::: {.cell .markdown}
## Setting up the Dataset Directory

This cell is to ensure we change the directory to have access to the
datasets.
:::

::: {.cell .code execution_count="1"}
``` python
from pathlib import Path

# NOTE: Provide the path to the dataset root directory.
#   If the datasets is not downloaded, it will be downloaded
#   to this directory.
dataset_root = Path.cwd().parent / "datasets" / "hazelnut_toy"
```
:::

::: {.cell .markdown}
## Use Folder Dataset (for Custom Datasets) via API

Here we show how one can utilize custom datasets to train anomalib
models. A custom dataset in this model can be of the following types:

-   A dataset with good and bad images.
-   A dataset with good and bad images as well as mask ground-truths for
    pixel-wise evaluation.
-   A dataset with good and bad images that is already split into
    training and testing sets.

To experiment this setting we provide a toy dataset that could be
downloaded from the following
[https://github.com/openvinotoolkit/anomalib/blob/main/docs/source/data/hazelnut_toy.zip](link).
For the rest of the tutorial, we assume that the dataset is downloaded
and extracted to `../datasets`, located in the `anomalib` directory.
:::

::: {.cell .code execution_count="2"}
``` python
# flake8: noqa
import numpy as np
from PIL import Image
from torchvision.transforms.v2 import Resize
from torchvision.transforms.v2.functional import to_pil_image

from anomalib.data.image.folder import Folder, FolderDataset
from anomalib import TaskType
```
:::

::: {.cell .markdown}
### DataModule

Similar to how we created the datamodules for existing benchmarking
datasets in the previous tutorials, we can also create an Anomalib
datamodule for our custom hazelnut dataset.

In addition to the root folder of the dataset, we now also specify which
folder contains the normal images, which folder contains the anomalous
images, and which folder contains the ground truth masks for the
anomalous images.
:::

::: {.cell .code execution_count="3"}
``` python
folder_datamodule = Folder(
    name="hazelnut_toy",
    root=dataset_root,
    normal_dir="good",
    abnormal_dir="crack",
    task=TaskType.SEGMENTATION,
    mask_dir=dataset_root / "mask" / "crack",
    image_size=(256, 256),
)
folder_datamodule.setup()
```
:::

::: {.cell .code execution_count="4"}
``` python
# Train images
i, data = next(enumerate(folder_datamodule.train_dataloader()))
print(data.keys(), data["image"].shape)
```

::: {.output .stream .stdout}
    dict_keys(['image_path', 'label', 'image', 'mask']) torch.Size([28, 3, 256, 256])
:::
:::

::: {.cell .code execution_count="5"}
``` python
# Test images
i, data = next(enumerate(folder_datamodule.test_dataloader()))
print(data.keys(), data["image"].shape, data["mask"].shape)
```

::: {.output .stream .stdout}
    dict_keys(['image_path', 'label', 'image', 'mask']) torch.Size([6, 3, 256, 256]) torch.Size([6, 256, 256])
:::
:::

::: {.cell .markdown}
As can be seen above, creating the dataloaders are pretty
straghtforward, which could be directly used for
training/testing/inference. We could visualize samples from the
dataloaders as well.
:::

::: {.cell .code execution_count="6"}
``` python
img = to_pil_image(data["image"][0].clone())
msk = to_pil_image(data["mask"][0]).convert("RGB")

Image.fromarray(np.hstack((np.array(img), np.array(msk))))
```

::: {.output .execute_result execution_count="6"}
![](vertopal_d33c91b4ee04414eadbe224ac2ee8bfd/81bef3f0a3281ff411329c6f31fe11012997a429.png)
:::
:::

::: {.cell .markdown}
`Folder` data module offers much more flexibility cater all different
sorts of needs. Please refer to the documentation for more details.
:::

::: {.cell .markdown}
### Torch Dataset

As in earlier examples, we can also create a standalone PyTorch dataset
instance.
:::

::: {.cell .code}
``` python
FolderDataset??
```
:::

::: {.cell .markdown}
We can add some transforms that will be applied to the images using
torchvision. Let\'s add a transform that resizes the input image to
256x256 pixels.
:::

::: {.cell .code execution_count="7"}
``` python
image_size = (256, 256)
transform = Resize(image_size, antialias=True)
```
:::

::: {.cell .markdown}
#### Classification Task
:::

::: {.cell .code execution_count="8"}
``` python
folder_dataset_classification_train = FolderDataset(
    name="hazelnut_toy",
    normal_dir=dataset_root / "good",
    abnormal_dir=dataset_root / "crack",
    split="train",
    transform=transform,
    task=TaskType.CLASSIFICATION,
)
folder_dataset_classification_train.samples.head()
```

::: {.output .execute_result execution_count="8"}
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
      <th>image_path</th>
      <th>label</th>
      <th>label_index</th>
      <th>mask_path</th>
      <th>split</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>/home/djameln/datasets/hazelnut_toy/good/00.jpg</td>
      <td>DirType.NORMAL</td>
      <td>0</td>
      <td></td>
      <td>Split.TRAIN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>/home/djameln/datasets/hazelnut_toy/good/01.jpg</td>
      <td>DirType.NORMAL</td>
      <td>0</td>
      <td></td>
      <td>Split.TRAIN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>/home/djameln/datasets/hazelnut_toy/good/02.jpg</td>
      <td>DirType.NORMAL</td>
      <td>0</td>
      <td></td>
      <td>Split.TRAIN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>/home/djameln/datasets/hazelnut_toy/good/03.jpg</td>
      <td>DirType.NORMAL</td>
      <td>0</td>
      <td></td>
      <td>Split.TRAIN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>/home/djameln/datasets/hazelnut_toy/good/04.jpg</td>
      <td>DirType.NORMAL</td>
      <td>0</td>
      <td></td>
      <td>Split.TRAIN</td>
    </tr>
  </tbody>
</table>
</div>
```
:::
:::

::: {.cell .markdown}
Let\'s look at the first sample in the dataset.
:::

::: {.cell .code execution_count="9"}
``` python
data = folder_dataset_classification_train[0]
print(data.keys(), data["image"].shape)
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

::: {.cell .code execution_count="10"}
``` python
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
```

::: {.output .execute_result execution_count="10"}
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
      <th>image_path</th>
      <th>label</th>
      <th>label_index</th>
      <th>mask_path</th>
      <th>split</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>/home/djameln/datasets/hazelnut_toy/crack/01.jpg</td>
      <td>DirType.ABNORMAL</td>
      <td>1</td>
      <td></td>
      <td>Split.TEST</td>
    </tr>
    <tr>
      <th>1</th>
      <td>/home/djameln/datasets/hazelnut_toy/crack/02.jpg</td>
      <td>DirType.ABNORMAL</td>
      <td>1</td>
      <td></td>
      <td>Split.TEST</td>
    </tr>
    <tr>
      <th>2</th>
      <td>/home/djameln/datasets/hazelnut_toy/crack/03.jpg</td>
      <td>DirType.ABNORMAL</td>
      <td>1</td>
      <td></td>
      <td>Split.TEST</td>
    </tr>
    <tr>
      <th>3</th>
      <td>/home/djameln/datasets/hazelnut_toy/crack/04.jpg</td>
      <td>DirType.ABNORMAL</td>
      <td>1</td>
      <td></td>
      <td>Split.TEST</td>
    </tr>
    <tr>
      <th>4</th>
      <td>/home/djameln/datasets/hazelnut_toy/crack/05.jpg</td>
      <td>DirType.ABNORMAL</td>
      <td>1</td>
      <td></td>
      <td>Split.TEST</td>
    </tr>
  </tbody>
</table>
</div>
```
:::
:::

::: {.cell .code execution_count="11"}
``` python
data = folder_dataset_classification_test[0]
print(data.keys(), data["image"].shape, data["image_path"], data["label"])
```

::: {.output .stream .stdout}
    dict_keys(['image_path', 'label', 'image']) torch.Size([3, 256, 256]) /home/djameln/datasets/hazelnut_toy/crack/01.jpg 1
:::
:::

::: {.cell .markdown}
#### Segmentation Task

It is also possible to configure the Folder dataset for the segmentation
task, where the dataset object returns image and ground-truth mask.
:::

::: {.cell .code execution_count="12"}
``` python
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
folder_dataset_segmentation_train.samples.head()
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
      <th>image_path</th>
      <th>label</th>
      <th>label_index</th>
      <th>mask_path</th>
      <th>split</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>/home/djameln/datasets/hazelnut_toy/good/00.jpg</td>
      <td>DirType.NORMAL</td>
      <td>0</td>
      <td></td>
      <td>Split.TRAIN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>/home/djameln/datasets/hazelnut_toy/good/01.jpg</td>
      <td>DirType.NORMAL</td>
      <td>0</td>
      <td></td>
      <td>Split.TRAIN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>/home/djameln/datasets/hazelnut_toy/good/02.jpg</td>
      <td>DirType.NORMAL</td>
      <td>0</td>
      <td></td>
      <td>Split.TRAIN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>/home/djameln/datasets/hazelnut_toy/good/03.jpg</td>
      <td>DirType.NORMAL</td>
      <td>0</td>
      <td></td>
      <td>Split.TRAIN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>/home/djameln/datasets/hazelnut_toy/good/04.jpg</td>
      <td>DirType.NORMAL</td>
      <td>0</td>
      <td></td>
      <td>Split.TRAIN</td>
    </tr>
  </tbody>
</table>
</div>
```
:::
:::

::: {.cell .code execution_count="13"}
``` python
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
folder_dataset_segmentation_test.samples.head(10)
```

::: {.output .execute_result execution_count="13"}
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
      <th>image_path</th>
      <th>label</th>
      <th>label_index</th>
      <th>mask_path</th>
      <th>split</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>/home/djameln/datasets/hazelnut_toy/crack/01.jpg</td>
      <td>DirType.ABNORMAL</td>
      <td>1</td>
      <td>/home/djameln/datasets/hazelnut_toy/mask/crack...</td>
      <td>Split.TEST</td>
    </tr>
    <tr>
      <th>1</th>
      <td>/home/djameln/datasets/hazelnut_toy/crack/02.jpg</td>
      <td>DirType.ABNORMAL</td>
      <td>1</td>
      <td>/home/djameln/datasets/hazelnut_toy/mask/crack...</td>
      <td>Split.TEST</td>
    </tr>
    <tr>
      <th>2</th>
      <td>/home/djameln/datasets/hazelnut_toy/crack/03.jpg</td>
      <td>DirType.ABNORMAL</td>
      <td>1</td>
      <td>/home/djameln/datasets/hazelnut_toy/mask/crack...</td>
      <td>Split.TEST</td>
    </tr>
    <tr>
      <th>3</th>
      <td>/home/djameln/datasets/hazelnut_toy/crack/04.jpg</td>
      <td>DirType.ABNORMAL</td>
      <td>1</td>
      <td>/home/djameln/datasets/hazelnut_toy/mask/crack...</td>
      <td>Split.TEST</td>
    </tr>
    <tr>
      <th>4</th>
      <td>/home/djameln/datasets/hazelnut_toy/crack/05.jpg</td>
      <td>DirType.ABNORMAL</td>
      <td>1</td>
      <td>/home/djameln/datasets/hazelnut_toy/mask/crack...</td>
      <td>Split.TEST</td>
    </tr>
  </tbody>
</table>
</div>
```
:::
:::

::: {.cell .code execution_count="14"}
``` python
data = folder_dataset_segmentation_test[3]
print(data.keys(), data["image"].shape, data["mask"].shape)
```

::: {.output .stream .stdout}
    dict_keys(['image_path', 'label', 'image', 'mask']) torch.Size([3, 256, 256]) torch.Size([256, 256])
:::
:::

::: {.cell .markdown}
Let\'s visualize the image and the mask\...
:::

::: {.cell .code execution_count="15"}
``` python
img = to_pil_image(data["image"].clone())
msk = to_pil_image(data["mask"]).convert("RGB")

Image.fromarray(np.hstack((np.array(img), np.array(msk))))
```

::: {.output .execute_result execution_count="15"}
![](vertopal_d33c91b4ee04414eadbe224ac2ee8bfd/783e244cdfc96ee22201f8a2f8076721fa567dc0.png)
:::
:::
