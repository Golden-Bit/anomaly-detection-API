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
  vscode:
    interpreter:
      hash: ae223df28f60859a2f400fae8b3a1034248e0a469f5599fd9a89c32908ed7a84
---

::: {.cell .markdown}
# Simulation of production line with defects
:::

::: {.cell .markdown}
In this notebook we will train a Anomalib model using the Anomalib API
and our own dataset. This notebook is also part of the Dobot series
notebooks.

### Use case

Using the [Dobot
Magician](https://www.dobot.cc/dobot-magician/product-overview.html) we
could simulate a production line system. Imagine we have a cubes factory
and they need to know when a defect piece appear in the process. We know
very well what is the aspecto of the normal cubes. Defects are coming no
often and we need to put those defect cubes out of the production line.

`<img src="https://user-images.githubusercontent.com/10940214/174126337-b344bbdc-6343-4d85-93e8-0cb1bf39a4e3.png" alt="drawing" style="width:400px;"/>`{=html}

  ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
  Class      Yellow cube                                                                                                                                                      Red cube                                                                                                                                                         Green cube                                                                                                                                                       Inferencing using Anomalib
  ---------- ---------------------------------------------------------------------------------------------------------------------------------------------------------------- ---------------------------------------------------------------------------------------------------------------------------------------------------------------- ---------------------------------------------------------------------------------------------------------------------------------------------------------------- ----------------------------------------------------------------------------------------------------------------------------------------------------------------
  Normal     `<img src="https://user-images.githubusercontent.com/10940214/174083561-38eec918-efc2-4ceb-99b1-bbb4c91396b2.jpg" alt="drawing" style="width:150px;"/>`{=html}   `<img src="https://user-images.githubusercontent.com/10940214/174083638-85ff889c-6222-4428-9c7d-9ad62bd15afe.jpg" alt="drawing" style="width:150px;"/>`{=html}   `<img src="https://user-images.githubusercontent.com/10940214/174083707-364177d4-373b-4891-96ce-3e5ea923e440.jpg" alt="drawing" style="width:150px;"/>`{=html}   `<img src="https://user-images.githubusercontent.com/10940214/174129305-03d9b71c-dfd9-492f-b42e-01c5c24171cc.jpg" alt="drawing" style="width:150px;"/>`{=html}

  Abnormal   `<img src="https://user-images.githubusercontent.com/10940214/174083805-df0a0b03-58c7-4ba8-af50-fd94d3a13e58.jpg" alt="drawing" style="width:150px;"/>`{=html}   `<img src="https://user-images.githubusercontent.com/10940214/174083873-22699523-22b4-4a55-a3da-6520095af8af.jpg" alt="drawing" style="width:150px;"/>`{=html}   `<img src="https://user-images.githubusercontent.com/10940214/174083944-38d5a6f4-f647-455b-ba4e-69482dfa3562.jpg" alt="drawing" style="width:150px;"/>`{=html}   `<img src="https://user-images.githubusercontent.com/10940214/174129253-f7a567d0-84f7-4050-8065-f00ba8bb973d.jpg" alt="drawing" style="width:150px;"/>`{=html}
  ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Using Anomalib we are expecting to see this result.
:::

::: {.cell .markdown}
# Installing Anomalib

To install anomalib with the required dependencies, please follow the
steps under `Install from source` [on
GitHub](https://github.com/openvinotoolkit/anomalib?tab=readme-ov-file#-installation).
:::

::: {.cell .markdown}
## Imports
:::

::: {.cell .code execution_count="1" ExecuteTime="{\"end_time\":\"2024-01-12T16:55:35.855912923Z\",\"start_time\":\"2024-01-12T16:55:30.140865729Z\"}"}
``` python
"""501a_training_a_model_with_cubes_from_a_robotic_arm.ipynb."""

from pathlib import Path

from anomalib.data.utils import read_image
from anomalib.deploy import OpenVINOInferencer
```
:::

::: {.cell .markdown}
## Download dataset and Robot API/Driver

We should prepare the folder to save the dataset and the Dobot API and
drivers. To download the dataset and the Dobot API and drivers we will
use anomalib\'s `download_and_extract` utility function.
:::

::: {.cell .code execution_count="2" ExecuteTime="{\"end_time\":\"2024-01-12T16:55:40.293464182Z\",\"start_time\":\"2024-01-12T16:55:35.850186281Z\"}"}
``` python
from anomalib.data.utils import DownloadInfo, download_and_extract

dataset_download_info = DownloadInfo(
    name="cubes.zip",
    url="https://github.com/openvinotoolkit/anomalib/releases/download/dobot/cubes.zip",
    hashsum="182ce0a48dabf452bf9a6aeb83132466088e30ed7a5c35d7d3a10a9fc11daac4",
)
api_download_info = DownloadInfo(
    name="dobot_api.zip",
    url="https://github.com/openvinotoolkit/anomalib/releases/download/dobot/dobot_api.zip",
    hashsum="eb79bb9c6346be1628a0fe5e1196420dcc4e122ab1aa0d5abbc82f63236f0527",
)
download_and_extract(root=Path.cwd(), info=dataset_download_info)
download_and_extract(root=Path.cwd(), info=api_download_info)
```

::: {.output .stream .stderr}
    cubes.zip: 6.99MB [00:01, 5.86MB/s]                            
    dobot_api.zip: 3.69MB [00:00, 5.43MB/s]                            
:::
:::

::: {.cell .markdown}
### Dataset: Cubes

Prepare your own dataset for normal and defect pieces.
:::

::: {.cell .code execution_count="3" ExecuteTime="{\"end_time\":\"2024-01-12T16:55:40.725983993Z\",\"start_time\":\"2024-01-12T16:55:40.274675101Z\"}"}
``` python
from anomalib.data import Folder
from anomalib import TaskType

datamodule = Folder(
    name="cubes",
    root=Path.cwd() / "cubes",
    normal_dir="normal",
    abnormal_dir="abnormal",
    normal_split_ratio=0.2,
    image_size=(256, 256),
    train_batch_size=32,
    eval_batch_size=32,
    task=TaskType.CLASSIFICATION,
)
datamodule.setup()

i, data = next(enumerate(datamodule.val_dataloader()))
print(data.keys())
```

::: {.output .stream .stdout}
    dict_keys(['image_path', 'label', 'image'])
:::
:::

::: {.cell .code execution_count="4" ExecuteTime="{\"end_time\":\"2024-01-12T16:55:40.734861023Z\",\"start_time\":\"2024-01-12T16:55:40.727834331Z\"}"}
``` python
# Check image size
print(data["image"].shape)
```

::: {.output .stream .stdout}
    torch.Size([32, 3, 256, 256])
:::
:::

::: {.cell .markdown}
## Model

`anomalib` supports a wide range of unsupervised anomaly detection
models. The table in this
[link](https://anomalib.readthedocs.io/en/latest/markdown/guides/reference/models/image/index.html)
shows the list of models currently supported by `anomalib` library.
:::

::: {.cell .markdown}
### Prepare the Model

We will use Padim model for this use case, which could be imported from
`anomalib.models`.
:::

::: {.cell .code execution_count="5" ExecuteTime="{\"end_time\":\"2024-01-12T16:55:41.184026691Z\",\"start_time\":\"2024-01-12T16:55:40.731669374Z\"}"}
``` python
from anomalib.models import Padim

model = Padim(
    backbone="resnet18",
    layers=["layer1", "layer2", "layer3"],
)
```
:::

::: {.cell .markdown}
## Training

Now that we set up the datamodule and model, we could now train the
model.

The final component to train the model is `Engine` object, which handles
train/test/predict/export pipeline. Let\'s create the engine object to
train the model.
:::

::: {.cell .code execution_count="6" ExecuteTime="{\"end_time\":\"2024-01-12T16:55:45.425314142Z\",\"start_time\":\"2024-01-12T16:55:41.180954949Z\"}" scrolled="true"}
``` python
from anomalib.engine import Engine
from anomalib.utils.normalization import NormalizationMethod

engine = Engine(
    normalization=NormalizationMethod.MIN_MAX,
    threshold="F1AdaptiveThreshold",
    task=TaskType.CLASSIFICATION,
    image_metrics=["AUROC"],
    accelerator="auto",
    check_val_every_n_epoch=1,
    devices=1,
    max_epochs=1,
    num_sanity_val_steps=0,
    val_check_interval=1.0,
)

engine.fit(model=model, datamodule=datamodule)
```

::: {.output .stream .stderr}
    /home/djameln/miniconda3/envs/anomalibv1source/lib/python3.10/site-packages/torchmetrics/utilities/prints.py:36: UserWarning: Metric `PrecisionRecallCurve` will save all targets and predictions in buffer. For large datasets this may lead to large memory footprint.
      warnings.warn(*args, **kwargs)
    GPU available: True (cuda), used: True
    TPU available: False, using: 0 TPU cores
    IPU available: False, using: 0 IPUs
    HPU available: False, using: 0 HPUs
    `Trainer(val_check_interval=1.0)` was configured so validation will run at the end of the training epoch..
    You are using a CUDA device ('NVIDIA GeForce RTX 3090') that has Tensor Cores. To properly utilize them, you should set `torch.set_float32_matmul_precision('medium' | 'high')` which will trade-off precision for performance. For more details, read https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html#torch.set_float32_matmul_precision
    /home/djameln/miniconda3/envs/anomalibv1source/lib/python3.10/site-packages/torchmetrics/utilities/prints.py:36: UserWarning: Metric `ROC` will save all targets and predictions in buffer. For large datasets this may lead to large memory footprint.
      warnings.warn(*args, **kwargs)
    LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0,1]
    /home/djameln/miniconda3/envs/anomalibv1source/lib/python3.10/site-packages/lightning/pytorch/core/optimizer.py:180: `LightningModule.configure_optimizers` returned `None`, this fit will run with no optimizer

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
{"model_id":"cd7082c9791c4745a28c995a0f50abc8","version_major":2,"version_minor":0}
```
:::

::: {.output .stream .stderr}
    /home/djameln/miniconda3/envs/anomalibv1source/lib/python3.10/site-packages/lightning/pytorch/loops/optimization/automatic.py:129: `training_step` returned `None`. If this was on purpose, ignore this warning...
:::

::: {.output .display_data}
``` json
{"model_id":"5bf5b899ecb340109d4fca28c67da82e","version_major":2,"version_minor":0}
```
:::

::: {.output .stream .stderr}
    `Trainer.fit` stopped: `max_epochs=1` reached.
:::
:::

::: {.cell .code execution_count="7" ExecuteTime="{\"end_time\":\"2024-01-12T16:55:46.739592200Z\",\"start_time\":\"2024-01-12T16:55:45.426593728Z\"}"}
``` python
# Validation
test_results = engine.test(model=model, datamodule=datamodule)
```

::: {.output .stream .stderr}
    /home/djameln/miniconda3/envs/anomalibv1source/lib/python3.10/site-packages/torchmetrics/utilities/prints.py:36: UserWarning: Metric `PrecisionRecallCurve` will save all targets and predictions in buffer. For large datasets this may lead to large memory footprint.
      warnings.warn(*args, **kwargs)
    LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0,1]
:::

::: {.output .display_data}
``` json
{"model_id":"6d530fe5f50d41dab51d8f2fefdcdb75","version_major":2,"version_minor":0}
```
:::

::: {.output .display_data}
```{=html}
<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">┏━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃<span style="font-weight: bold">        Test metric        </span>┃<span style="font-weight: bold">       DataLoader 0        </span>┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│<span style="color: #008080; text-decoration-color: #008080">        image_AUROC        </span>│<span style="color: #800080; text-decoration-color: #800080">            1.0            </span>│
└───────────────────────────┴───────────────────────────┘
</pre>
```
:::
:::

::: {.cell .code execution_count="8" ExecuteTime="{\"end_time\":\"2024-01-12T16:55:48.906878137Z\",\"start_time\":\"2024-01-12T16:55:46.673514722Z\"}" collapsed="false"}
``` python
from anomalib.deploy import ExportType

# Exporting model to OpenVINO
openvino_model_path = engine.export(
    model=model,
    export_type=ExportType.OPENVINO,
    export_root=str(Path.cwd()),
)
```

::: {.output .stream .stderr}
    /home/djameln/miniconda3/envs/anomalibv1source/lib/python3.10/site-packages/torch/onnx/_internal/jit_utils.py:307: UserWarning: Constant folding - Only steps=1 can be constant folded for opset >= 10 onnx::Slice op. Constant folding not applied. (Triggered internally at ../torch/csrc/jit/passes/onnx/constant_fold.cpp:179.)
      _C._jit_pass_onnx_node_shape_type_inference(node, params_dict, opset_version)
    /home/djameln/miniconda3/envs/anomalibv1source/lib/python3.10/site-packages/torch/onnx/utils.py:702: UserWarning: Constant folding - Only steps=1 can be constant folded for opset >= 10 onnx::Slice op. Constant folding not applied. (Triggered internally at ../torch/csrc/jit/passes/onnx/constant_fold.cpp:179.)
      _C._jit_pass_onnx_graph_shape_type_inference(
    /home/djameln/miniconda3/envs/anomalibv1source/lib/python3.10/site-packages/torch/onnx/utils.py:1209: UserWarning: Constant folding - Only steps=1 can be constant folded for opset >= 10 onnx::Slice op. Constant folding not applied. (Triggered internally at ../torch/csrc/jit/passes/onnx/constant_fold.cpp:179.)
      _C._jit_pass_onnx_graph_shape_type_inference(
:::
:::

::: {.cell .markdown}
## OpenVINO Inference

Now that we trained and tested a model, we could check a single
inference result using OpenVINO inferencer object. This will demonstrate
how a trained model could be used for inference.
:::

::: {.cell .markdown}
### Load a Test Image

Let\'s read an image from the test set and perform inference using
OpenVINO inferencer.
:::

::: {.cell .code execution_count="9" ExecuteTime="{\"end_time\":\"2024-01-12T16:55:49.107125004Z\",\"start_time\":\"2024-01-12T16:55:48.908620452Z\"}" scrolled="true"}
``` python
from matplotlib import pyplot as plt

image_path = "./cubes/abnormal/input_20230210134059.jpg"
image = read_image(path="./cubes/abnormal/input_20230210134059.jpg")
plt.imshow(image)
```

::: {.output .execute_result execution_count="9"}
    <matplotlib.image.AxesImage at 0x7f5648255420>
:::
:::

::: {.cell .markdown}
### Load the OpenVINO Model

By default, the output files are saved into `results` directory. Let\'s
check where the OpenVINO model is stored.
:::

::: {.cell .code execution_count="10" ExecuteTime="{\"end_time\":\"2024-01-12T16:55:49.109896590Z\",\"start_time\":\"2024-01-12T16:55:49.107381700Z\"}"}
``` python
metadata_path = openvino_model_path.parent / "metadata.json"
print(openvino_model_path.exists(), metadata_path.exists())
```

::: {.output .stream .stdout}
    True True
:::
:::

::: {.cell .code execution_count="11" ExecuteTime="{\"end_time\":\"2024-01-12T16:55:49.447048687Z\",\"start_time\":\"2024-01-12T16:55:49.110849785Z\"}"}
``` python
inferencer = OpenVINOInferencer(
    path=openvino_model_path,  # Path to the OpenVINO IR model.
    metadata=metadata_path,  # Path to the metadata file.
    device="CPU",  # We would like to run it on an Intel CPU.
)
```
:::

::: {.cell .markdown}
### Perform Inference

Predicting an image using OpenVINO inferencer is as simple as calling
`predict` method.
:::

::: {.cell .code execution_count="12" ExecuteTime="{\"end_time\":\"2024-01-12T16:55:49.489524314Z\",\"start_time\":\"2024-01-12T16:55:49.447882511Z\"}"}
``` python
print(image.shape)
predictions = inferencer.predict(image=image)
```

::: {.output .stream .stdout}
    (480, 640, 3)
:::
:::

::: {.cell .markdown}
where `predictions` contain any relevant information regarding the task
type. For example, predictions for a segmentation model could contain
image, anomaly maps, predicted scores, labels or masks.

### Visualizing Inference Results

`anomalib` provides a number of tools to visualize the inference
results. Let\'s visualize the inference results using the `Visualizer`
method.
:::

::: {.cell .code execution_count="13" ExecuteTime="{\"end_time\":\"2024-01-12T16:55:49.577270470Z\",\"start_time\":\"2024-01-12T16:55:49.489434931Z\"}"}
``` python
from anomalib.utils.visualization.image import ImageVisualizer, VisualizationMode
from PIL import Image

visualizer = ImageVisualizer(mode=VisualizationMode.FULL, task=TaskType.CLASSIFICATION)
output_image = visualizer.visualize_image(predictions)
Image.fromarray(output_image)
```

::: {.output .execute_result execution_count="13"}
![](vertopal_3a00677eeddb4e0aaf4d0c07ed6ab559/c5dc05aa628ba372239cc3949d23584323b06f9e.jpg)
:::
:::

::: {.cell .markdown}
Since `predictions` contain a number of information, we could specify
which information we want to visualize. For example, if we want to
visualize the predicted mask and the segmentation results, we could
specify the task type as `TaskType.SEGMENTATION`, which would produce
the following visualization.
:::

::: {.cell .code execution_count="14" ExecuteTime="{\"end_time\":\"2024-01-12T16:55:49.691819697Z\",\"start_time\":\"2024-01-12T16:55:49.583066408Z\"}"}
``` python
visualizer = ImageVisualizer(mode=VisualizationMode.FULL, task=TaskType.SEGMENTATION)
output_image = visualizer.visualize_image(predictions)
Image.fromarray(output_image)
```

::: {.output .execute_result execution_count="14"}
![](vertopal_3a00677eeddb4e0aaf4d0c07ed6ab559/ea9682872d9600b2cbad1bd164cbe5ce9383c08a.jpg)
:::
:::
