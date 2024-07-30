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
```{=html}
<center><img src="https://raw.githubusercontent.com/openvinotoolkit/anomalib/main/docs/source/_static/images/logos/anomalib-wide-blue.png" alt="Paris" class="center"></center>
```
```{=html}
<center>💙 A library for benchmarking, developing and deploying deep learning anomaly detection algorithms</center>
```

------------------------------------------------------------------------

> NOTE: This notebook is originally created by \@innat on
> [Kaggle](https://www.kaggle.com/code/ipythonx/mvtec-ad-anomaly-detection-with-anomalib-library/notebook).

[Anomalib](https://github.com/openvinotoolkit/anomalib): Anomalib is a
deep learning library that aims to collect state-of-the-art anomaly
detection algorithms for benchmarking on both public and private
datasets. Anomalib provides several ready-to-use implementations of
anomaly detection algorithms described in the recent literature, as well
as a set of tools that facilitate the development and implementation of
custom models. The library has a strong focus on image-based anomaly
detection, where the goal of the algorithm is to identify anomalous
images, or anomalous pixel regions within images in a dataset.

The library supports a number of image and video datasets for
**benchmarking** and custom dataset support for **training/inference**.
In this notebook, we will explore `anomalib` training a PADIM model on
the `MVTec AD` bottle dataset and evaluating the model\'s performance.
:::

::: {.cell .markdown}
## Installing Anomalib
:::

::: {.cell .markdown}
Installation can be done in two ways: (i) install via PyPI, or (ii)
installing from sourc, both of which are shown below:
:::

::: {.cell .markdown}
### I. Install via PyPI {#i-install-via-pypi}
:::

::: {.cell .code ExecuteTime="{\"end_time\":\"2024-01-26T12:18:56.096138098Z\",\"start_time\":\"2024-01-26T12:18:56.046631009Z\"}"}
``` python
# Option - I: Uncomment the next line if you want to install via pip.
# %pip install anomalib
# %anomalib install -v
```
:::

::: {.cell .markdown}
> NOTE:
>
> Although v1.0.0 is on PyPI, it may not be stable and may have bugs. It
> is therefore recommended to install from source.
:::

::: {.cell .markdown}
### II. Install from Source {#ii-install-from-source}

This option would initially download anomalib repository from github and
manually install `anomalib` from source, which is shown below:
:::

::: {.cell .code ExecuteTime="{\"end_time\":\"2024-01-26T12:18:56.101357180Z\",\"start_time\":\"2024-01-26T12:18:56.098734268Z\"}"}
``` python
# Option - II: Uncomment the next three lines if you want to install from the source.
# !git clone https://github.com/openvinotoolkit/anomalib.git
# %cd anomalib
# %pip install .
# %anomalib install -v
```
:::

::: {.cell .markdown}
Now let\'s verify the working directory. This is to access the datasets
and configs when the notebook is run from different platforms such as
local or Google Colab.
:::

::: {.cell .code ExecuteTime="{\"end_time\":\"2024-01-26T12:18:56.112607883Z\",\"start_time\":\"2024-01-26T12:18:56.104276975Z\"}"}
``` python
import os
from pathlib import Path

from git.repo import Repo

current_directory = Path.cwd()
if current_directory.name == "000_getting_started":
    # On the assumption that, the notebook is located in
    #   ~/anomalib/notebooks/000_getting_started/
    root_directory = current_directory.parent.parent
elif current_directory.name == "anomalib":
    # This means that the notebook is run from the main anomalib directory.
    root_directory = current_directory
else:
    # Otherwise, we'll need to clone the anomalib repo to the `current_directory`
    repo = Repo.clone_from(
        url="https://github.com/openvinotoolkit/anomalib.git",
        to_path=current_directory,
    )
    root_directory = current_directory / "anomalib"

os.chdir(root_directory)
```
:::

::: {.cell .markdown}
## Imports
:::

::: {.cell .code ExecuteTime="{\"end_time\":\"2024-01-26T12:18:56.112848196Z\",\"start_time\":\"2024-01-26T12:18:56.110866741Z\"}"}
``` python
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
```
:::

::: {.cell .markdown}
## Model

Currently, there are **13** anomaly detection models available in
`anomalib` library. Namely,

-   [CFA](https://arxiv.org/abs/2206.04325)
-   [CS-Flow](https://arxiv.org/abs/2110.02855v1)
-   [CFlow](https://arxiv.org/pdf/2107.12571v1.pdf)
-   [DFKDE](https://github.com/openvinotoolkit/anomalib/tree/main/anomalib/models/dfkde)
-   [DFM](https://arxiv.org/pdf/1909.11786.pdf)
-   [DRAEM](https://arxiv.org/abs/2108.07610)
-   [FastFlow](https://arxiv.org/abs/2111.07677)
-   [Ganomaly](https://arxiv.org/abs/1805.06725)
-   [Padim](https://arxiv.org/pdf/2011.08785.pdf)
-   [Patchcore](https://arxiv.org/pdf/2106.08265.pdf)
-   [Reverse Distillation](https://arxiv.org/abs/2201.10703)
-   [R-KDE](https://ieeexplore.ieee.org/document/8999287)
-   [STFPM](https://arxiv.org/pdf/2103.04257.pdf)

In this tutorial, we\'ll be using
[Padim](https://arxiv.org/pdf/2011.08785.pdf).
:::

::: {.cell .markdown}
## Dataset: MVTec AD

**MVTec AD** is a dataset for benchmarking anomaly detection methods
with a focus on industrial inspection. It contains over **5000**
high-resolution images divided into **15** different object and texture
categories. Each category comprises a set of defect-free training images
and a test set of images with various kinds of defects as well as images
without defects. If the dataset is not located in the root datasets
directory, anomalib will automatically install the dataset.

We could now import the MVtec AD dataset using its specific datamodule
implemented in anomalib.
:::

::: {.cell .code execution_count="81" ExecuteTime="{\"end_time\":\"2024-01-26T12:18:57.203133970Z\",\"start_time\":\"2024-01-26T12:18:56.111365813Z\"}"}
``` python
datamodule = MVTec(num_workers=0)
datamodule.prepare_data()  # Downloads the dataset if it's not in the specified `root` directory
datamodule.setup()  # Create train/val/test/prediction sets.

i, data = next(enumerate(datamodule.val_dataloader()))
print(data.keys())
```

::: {.output .stream .stdout}
    dict_keys(['image_path', 'label', 'image', 'mask'])
:::
:::

::: {.cell .markdown}
Let\'s check the shapes of the input images and masks.
:::

::: {.cell .code execution_count="80" ExecuteTime="{\"end_time\":\"2024-01-26T12:18:57.203997320Z\",\"start_time\":\"2024-01-26T12:18:57.202960908Z\"}"}
``` python
print(data["image"].shape, data["mask"].shape)
```

::: {.output .stream .stdout}
    torch.Size([32, 3, 900, 900]) torch.Size([32, 900, 900])
:::
:::

::: {.cell .markdown}
We could now visualize a normal and abnormal sample from the validation
set.
:::

::: {.cell .code execution_count="79" ExecuteTime="{\"end_time\":\"2024-01-26T12:18:57.312944404Z\",\"start_time\":\"2024-01-26T12:18:57.203237964Z\"}"}
``` python
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
```

::: {.output .execute_result execution_count="79"}
![](vertopal_351f90a78836466fb8cb17da4d17eb99/f33cf934faf7c33401f60e9d7b14ca6e987c4a30.jpg)
:::
:::

::: {.cell .markdown}
## Prepare Model

Let\'s create the Padim and train it.
:::

::: {.cell .code execution_count="78" ExecuteTime="{\"end_time\":\"2024-01-26T12:18:57.633634551Z\",\"start_time\":\"2024-01-26T12:18:57.312301960Z\"}"}
``` python
# Get the model and datamodule
model = Padim()
datamodule = MVTec(num_workers=0)
```
:::

::: {.cell .code execution_count="77" ExecuteTime="{\"end_time\":\"2024-01-26T12:19:03.278278808Z\",\"start_time\":\"2024-01-26T12:18:57.635288644Z\"}"}
``` python
# start training
engine = Engine(task=TaskType.SEGMENTATION)
engine.fit(model=model, datamodule=datamodule)
```

::: {.output .stream .stderr}
    Trainer will use only 1 of 2 GPUs because it is running inside an interactive / notebook environment. You may try to set `Trainer(devices=2)` but please note that multi-GPU inside interactive / notebook environments is considered experimental and unstable. Your mileage may vary.
    GPU available: True (cuda), used: True
    TPU available: False, using: 0 TPU cores
    IPU available: False, using: 0 IPUs
    HPU available: False, using: 0 HPUs
    `Trainer(val_check_interval=1.0)` was configured so validation will run at the end of the training epoch..
    LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0,1]

      | Name                  | Type                     | Params
    -------------------------------------------------------------------
    0 | model                 | PadimModel               | 2.8 M 
    1 | _transform            | Compose                  | 0     
    2 | normalization_metrics | MinMax                   | 0     
    3 | image_threshold       | F1AdaptiveThreshold      | 0     
    4 | pixel_threshold       | F1AdaptiveThreshold      | 0     
    5 | image_metrics         | AnomalibMetricCollection | 0     
    6 | pixel_metrics         | AnomalibMetricCollection | 0     
    -------------------------------------------------------------------
    2.8 M     Trainable params
    0         Non-trainable params
    2.8 M     Total params
    11.131    Total estimated model params size (MB)
:::

::: {.output .display_data}
``` json
{"model_id":"a9db1de73dc6413e844d3d3bd5d2d2c1","version_major":2,"version_minor":0}
```
:::

::: {.output .display_data}
``` json
{"model_id":"40c9ef72ce3e4f4195e58e5877fb2fb5","version_major":2,"version_minor":0}
```
:::

::: {.output .stream .stderr}
    `Trainer.fit` stopped: `max_epochs=1` reached.
:::
:::

::: {.cell .markdown}
## Validation
:::

::: {.cell .code execution_count="76" ExecuteTime="{\"end_time\":\"2024-01-26T12:19:05.567521337Z\",\"start_time\":\"2024-01-26T12:19:03.280992538Z\"}"}
``` python
# load best model from checkpoint before evaluating
test_results = engine.test(
    model=model,
    datamodule=datamodule,
    ckpt_path=engine.trainer.checkpoint_callback.best_model_path,
)
```

::: {.output .stream .stderr}
    Restoring states from the checkpoint path at /home/djameln/anomalib/lightning_logs/version_144/checkpoints/epoch=0-step=7.ckpt
    LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0,1]
    Loaded model weights from the checkpoint at /home/djameln/anomalib/lightning_logs/version_144/checkpoints/epoch=0-step=7.ckpt
:::

::: {.output .display_data}
``` json
{"model_id":"c88160bbf6344547803bd1ca8aa4035a","version_major":2,"version_minor":0}
```
:::

::: {.output .display_data}
```{=html}
<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">┏━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃<span style="font-weight: bold">        Test metric        </span>┃<span style="font-weight: bold">       DataLoader 0        </span>┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│<span style="color: #008080; text-decoration-color: #008080">        image_AUROC        </span>│<span style="color: #800080; text-decoration-color: #800080">    0.9992063641548157     </span>│
│<span style="color: #008080; text-decoration-color: #008080">       image_F1Score       </span>│<span style="color: #800080; text-decoration-color: #800080">    0.9921259880065918     </span>│
│<span style="color: #008080; text-decoration-color: #008080">        pixel_AUROC        </span>│<span style="color: #800080; text-decoration-color: #800080">    0.9842503070831299     </span>│
│<span style="color: #008080; text-decoration-color: #008080">       pixel_F1Score       </span>│<span style="color: #800080; text-decoration-color: #800080">    0.7291697859764099     </span>│
└───────────────────────────┴───────────────────────────┘
</pre>
```
:::
:::

::: {.cell .code execution_count="75"}
``` python
print(test_results)
```

::: {.output .stream .stdout}
    [{'pixel_AUROC': 0.9842503070831299, 'pixel_F1Score': 0.7291697859764099, 'image_AUROC': 0.9992063641548157, 'image_F1Score': 0.9921259880065918}]
:::
:::

::: {.cell .markdown}
## OpenVINO Inference

Now that we trained and tested a model, we could check a single
inference result using OpenVINO inferencer object. This will demonstrate
how a trained model could be used for inference.
:::

::: {.cell .markdown}
Before we can use OpenVINO inference, let\'s export the model to
OpenVINO format first.
:::

::: {.cell .code execution_count="74" ExecuteTime="{\"end_time\":\"2024-01-26T12:19:06.645604243Z\",\"start_time\":\"2024-01-26T12:19:05.569089932Z\"}"}
``` python
engine.export(
    model=model,
    export_type=ExportType.OPENVINO,
)
```

::: {.output .execute_result execution_count="74"}
    PosixPath('/home/djameln/anomalib/weights/openvino/model.xml')
:::
:::

::: {.cell .markdown}
### Load a Test Image

Let\'s read an image from the test set and perform inference using
OpenVINO inferencer.
:::

::: {.cell .code execution_count="73" ExecuteTime="{\"end_time\":\"2024-01-26T12:19:06.867644218Z\",\"start_time\":\"2024-01-26T12:19:06.646217079Z\"}"}
``` python
image_path = root_directory / "datasets/MVTec/bottle/test/broken_large/000.png"
image = read_image(path="./datasets/MVTec/bottle/test/broken_large/000.png")
plt.imshow(image)
```

::: {.output .execute_result execution_count="73"}
    <matplotlib.image.AxesImage at 0x7fd239fcc460>
:::

::: {.output .display_data}
![](vertopal_351f90a78836466fb8cb17da4d17eb99/e164d4a4fa9daafc863db549f972269a5532c10a.png)
:::
:::

::: {.cell .markdown}
### Load the OpenVINO Model

By default, the output files are saved into `results` directory. Let\'s
check where the OpenVINO model is stored.
:::

::: {.cell .code execution_count="72" ExecuteTime="{\"end_time\":\"2024-01-26T12:19:06.869599561Z\",\"start_time\":\"2024-01-26T12:19:06.866628785Z\"}"}
``` python
output_path = Path(engine.trainer.default_root_dir)
print(output_path)
```

::: {.output .stream .stdout}
    /home/djameln/anomalib
:::
:::

::: {.cell .code execution_count="71" ExecuteTime="{\"end_time\":\"2024-01-26T12:19:06.880794392Z\",\"start_time\":\"2024-01-26T12:19:06.868965582Z\"}"}
``` python
openvino_model_path = output_path / "weights" / "openvino" / "model.bin"
metadata = output_path / "weights" / "openvino" / "metadata.json"
print(openvino_model_path.exists(), metadata.exists())
```

::: {.output .stream .stdout}
    True True
:::
:::

::: {.cell .code execution_count="70" ExecuteTime="{\"end_time\":\"2024-01-26T12:19:07.127278601Z\",\"start_time\":\"2024-01-26T12:19:06.879785016Z\"}"}
``` python
inferencer = OpenVINOInferencer(
    path=openvino_model_path,  # Path to the OpenVINO IR model.
    metadata=metadata,  # Path to the metadata file.
    device="CPU",  # We would like to run it on an Intel CPU.
)
```
:::

::: {.cell .markdown}
### Perform Inference

Predicting an image using OpenVINO inferencer is as simple as calling
`predict` method.
:::

::: {.cell .code execution_count="69" ExecuteTime="{\"end_time\":\"2024-01-26T12:19:07.221219176Z\",\"start_time\":\"2024-01-26T12:19:07.170939341Z\"}"}
``` python
predictions = inferencer.predict(image=image_path)
```
:::

::: {.cell .markdown}
where `predictions` contain any relevant information regarding the task
type. For example, predictions for a segmentation model could contain
image, anomaly maps, predicted scores, labels or masks.
:::

::: {.cell .markdown}
### Visualizing Inference Results
:::

::: {.cell .code execution_count="68" ExecuteTime="{\"end_time\":\"2024-01-26T12:19:07.222396309Z\",\"start_time\":\"2024-01-26T12:19:07.214650568Z\"}"}
``` python
print(predictions.pred_score, predictions.pred_label)
```

::: {.output .stream .stdout}
    0.8962510235051898 True
:::
:::

::: {.cell .code execution_count="67" ExecuteTime="{\"end_time\":\"2024-01-26T12:19:07.347717385Z\",\"start_time\":\"2024-01-26T12:19:07.214884777Z\"}"}
``` python
# Visualize the original image
plt.imshow(predictions.image)
```

::: {.output .execute_result execution_count="67"}
    <matplotlib.image.AxesImage at 0x7fd239f02080>
:::

::: {.output .display_data}
![](vertopal_351f90a78836466fb8cb17da4d17eb99/1a77a8176162c2f149f2afe999b326d78a458244.png)
:::
:::

::: {.cell .code execution_count="66" ExecuteTime="{\"end_time\":\"2024-01-26T12:19:07.471919621Z\",\"start_time\":\"2024-01-26T12:19:07.346789142Z\"}"}
``` python
# Visualize the raw anomaly maps predicted by the model.
plt.imshow(predictions.anomaly_map)
```

::: {.output .execute_result execution_count="66"}
    <matplotlib.image.AxesImage at 0x7fd39c2540a0>
:::

::: {.output .display_data}
![](vertopal_351f90a78836466fb8cb17da4d17eb99/82163c3646d4fdc818b11b968a7a43d2537d73b8.png)
:::
:::

::: {.cell .code execution_count="65" ExecuteTime="{\"end_time\":\"2024-01-26T12:19:07.644440308Z\",\"start_time\":\"2024-01-26T12:19:07.479955777Z\"}"}
``` python
# Visualize the heatmaps, on which raw anomaly map is overlayed on the original image.
plt.imshow(predictions.heat_map)
```

::: {.output .execute_result execution_count="65"}
    <matplotlib.image.AxesImage at 0x7fd2a3a0ae30>
:::

::: {.output .display_data}
![](vertopal_351f90a78836466fb8cb17da4d17eb99/51272f2a90216829a2b36261fc62b0cee6060ea2.png)
:::
:::

::: {.cell .code execution_count="64" ExecuteTime="{\"end_time\":\"2024-01-26T12:19:07.759913041Z\",\"start_time\":\"2024-01-26T12:19:07.644757570Z\"}"}
``` python
# Visualize the segmentation mask.
plt.imshow(predictions.pred_mask)
```

::: {.output .execute_result execution_count="64"}
    <matplotlib.image.AxesImage at 0x7fd223f42020>
:::

::: {.output .display_data}
![](vertopal_351f90a78836466fb8cb17da4d17eb99/9d8bb1a792d023b549ee5c8d34021182c3ba2762.png)
:::
:::

::: {.cell .code execution_count="63" ExecuteTime="{\"end_time\":\"2024-01-26T12:19:07.925019564Z\",\"start_time\":\"2024-01-26T12:19:07.762215888Z\"}"}
``` python
# Visualize the segmentation mask with the original image.
plt.imshow(predictions.segmentations)
```

::: {.output .execute_result execution_count="63"}
    <matplotlib.image.AxesImage at 0x7fd1f83fb280>
:::

::: {.output .display_data}
![](vertopal_351f90a78836466fb8cb17da4d17eb99/00392a5320be55e32cdaf518f12eeedb5ef1597e.png)
:::
:::

::: {.cell .markdown}
This wraps the `getting_started` notebook. There are a lot more
functionalities that could be explored in the library. Please refer to
the [documentation](https://openvinotoolkit.github.io/anomalib/) for
more details.
:::
