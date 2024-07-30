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
      hash: f26beec5b578f06009232863ae217b956681fd13da2e828fa5a0ecf8cf2ccd29
---

::: {.cell .markdown}
# Train a Model via API

This notebook demonstrates how to train, test and infer the FastFlow
model via Anomalib API. Compared to the CLI entrypoints such as
\`tools/\<train, test, inference\>.py, the API offers more flexibility
such as modifying the existing model or designing custom approaches.

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

::: {.cell .code}
``` python
from pathlib import Path

# NOTE: Provide the path to the dataset root directory.
#   If the datasets is not downloaded, it will be downloaded
#   to this directory.
dataset_root = Path.cwd().parent / "datasets" / "MVTec"
```
:::

::: {.cell .markdown pycharm="{\"name\":\"#%% md\\n\"}"}
## Imports
:::

::: {.cell .code pycharm="{\"name\":\"#%%\\n\"}"}
``` python
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
```
:::

::: {.cell .markdown pycharm="{\"name\":\"#%% md\\n\"}"}
## Data Module

To train the model end-to-end, we do need to have a dataset. In our
[previous
notebooks](https://github.com/openvinotoolkit/anomalib/tree/main/notebooks/100_datamodules),
we demonstrate how to initialize benchmark- and custom datasets. In this
tutorial, we will use MVTec AD DataModule. We assume that `datasets`
directory is created in the `anomalib` root directory and `MVTec`
dataset is located in `datasets` directory.

Before creating the dataset, let\'s define the task type that we will be
working on. In this notebook, we will be working on a segmentation task.
Therefore the `task` variable would be:
:::

::: {.cell .code}
``` python
task = TaskType.SEGMENTATION
```
:::

::: {.cell .code pycharm="{\"name\":\"#%%\\n\"}"}
``` python
datamodule = MVTec(
    root=dataset_root,
    category="bottle",
    image_size=256,
    train_batch_size=32,
    eval_batch_size=32,
    num_workers=0,
    task=task,
)
```
:::

::: {.cell .markdown pycharm="{\"name\":\"#%% md\\n\"}"}
## FastFlow Model

Now that we have created the MVTec datamodule, we could create the
FastFlow model. We could start with printing its docstring.
:::

::: {.cell .code execution_count="30" pycharm="{\"name\":\"#%%\\n\"}"}
``` python
Fastflow??
```

::: {.output .stream .stdout}
    Init signature:
    Fastflow(
        backbone: str = 'resnet18',
        pre_trained: bool = True,
        flow_steps: int = 8,
        conv3x3_only: bool = False,
        hidden_ratio: float = 1.0,
    ) -> None
    Source:        
    class Fastflow(AnomalyModule):
        """PL Lightning Module for the FastFlow algorithm.

        Args:
            input_size (tuple[int, int]): Model input size.
                Defaults to ``(256, 256)``.
            backbone (str): Backbone CNN network
                Defaults to ``resnet18``.
            pre_trained (bool, optional): Boolean to check whether to use a pre_trained backbone.
                Defaults to ``True``.
            flow_steps (int, optional): Flow steps.
                Defaults to ``8``.
            conv3x3_only (bool, optinoal): Use only conv3x3 in fast_flow model.
                Defaults to ``False``.
            hidden_ratio (float, optional): Ratio to calculate hidden var channels.
                Defaults to ``1.0`.
        """

        def __init__(
            self,
            backbone: str = "resnet18",
            pre_trained: bool = True,
            flow_steps: int = 8,
            conv3x3_only: bool = False,
            hidden_ratio: float = 1.0,
        ) -> None:
            super().__init__()

            self.backbone = backbone
            self.pre_trained = pre_trained
            self.flow_steps = flow_steps
            self.conv3x3_only = conv3x3_only
            self.hidden_ratio = hidden_ratio

            self.loss = FastflowLoss()

            self.model: FastflowModel

        def _setup(self) -> None:
            assert self.input_size is not None, "Fastflow needs input size to build torch model."
            self.model = FastflowModel(
                input_size=self.input_size,
                backbone=self.backbone,
                pre_trained=self.pre_trained,
                flow_steps=self.flow_steps,
                conv3x3_only=self.conv3x3_only,
                hidden_ratio=self.hidden_ratio,
            )

        def training_step(self, batch: dict[str, str | torch.Tensor], *args, **kwargs) -> STEP_OUTPUT:
            """Perform the training step input and return the loss.

            Args:
                batch (batch: dict[str, str | torch.Tensor]): Input batch
                args: Additional arguments.
                kwargs: Additional keyword arguments.

            Returns:
                STEP_OUTPUT: Dictionary containing the loss value.
            """
            del args, kwargs  # These variables are not used.

            hidden_variables, jacobians = self.model(batch["image"])
            loss = self.loss(hidden_variables, jacobians)
            self.log("train_loss", loss.item(), on_epoch=True, prog_bar=True, logger=True)
            return {"loss": loss}

        def validation_step(self, batch: dict[str, str | torch.Tensor], *args, **kwargs) -> STEP_OUTPUT:
            """Perform the validation step and return the anomaly map.

            Args:
                batch (dict[str, str | torch.Tensor]): Input batch
                args: Additional arguments.
                kwargs: Additional keyword arguments.

            Returns:
                STEP_OUTPUT | None: batch dictionary containing anomaly-maps.
            """
            del args, kwargs  # These variables are not used.

            anomaly_maps = self.model(batch["image"])
            batch["anomaly_maps"] = anomaly_maps
            return batch

        @property
        def trainer_arguments(self) -> dict[str, Any]:
            """Return FastFlow trainer arguments."""
            return {"gradient_clip_val": 0, "num_sanity_val_steps": 0}

        def configure_optimizers(self) -> torch.optim.Optimizer:
            """Configure optimizers for each decoder.

            Returns:
                Optimizer: Adam optimizer for each decoder
            """
            return optim.Adam(
                params=self.model.parameters(),
                lr=0.001,
                weight_decay=0.00001,
            )

        @property
        def learning_type(self) -> LearningType:
            """Return the learning type of the model.

            Returns:
                LearningType: Learning type of the model.
            """
            return LearningType.ONE_CLASS
    File:           ~/anomalib/src/anomalib/models/image/fastflow/lightning_model.py
    Type:           ABCMeta
    Subclasses:     
:::
:::

::: {.cell .code execution_count="29" pycharm="{\"name\":\"#%%\\n\"}"}
``` python
model = Fastflow(backbone="resnet18", flow_steps=8)
```
:::

::: {.cell .markdown pycharm="{\"name\":\"#%% md\\n\"}"}
## Callbacks

To train the model properly, we will to add some other \"non-essential\"
logic such as saving the weights, early-stopping, normalizing the
anomaly scores and visualizing the input/output images. To achieve these
we use `Callbacks`. Anomalib has its own callbacks and also supports
PyTorch Lightning\'s native callbacks. So, let\'s create the list of
callbacks we want to execute during the training.
:::

::: {.cell .code execution_count="28" pycharm="{\"name\":\"#%%\\n\"}"}
``` python
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
```
:::

::: {.cell .markdown pycharm="{\"name\":\"#%% md\\n\"}"}
## Training

Now that we set up the datamodule, model, optimizer and the callbacks,
we could now train the model.

The final component to train the model is `Engine` object, which handles
train/test/predict pipeline. Let\'s create the engine object to train
the model.
:::

::: {.cell .code execution_count="27" pycharm="{\"name\":\"#%%\\n\"}"}
``` python
engine = Engine(
    callbacks=callbacks,
    pixel_metrics="AUROC",
    accelerator="auto",  # \<"cpu", "gpu", "tpu", "ipu", "hpu", "auto">,
    devices=1,
    logger=False,
)
```
:::

::: {.cell .markdown pycharm="{\"name\":\"#%% md\\n\"}"}
`Trainer` object has number of options that suit all specific needs. For
more details, refer to [Lightning
Documentation](https://pytorch-lightning.readthedocs.io/en/stable/common/engine.html)
to see how it could be tweaked to your needs.

Let\'s train the model now.
:::

::: {.cell .code pycharm="{\"name\":\"#%%\\n\"}"}
``` python
engine.fit(datamodule=datamodule, model=model)
```
:::

::: {.cell .markdown pycharm="{\"name\":\"#%% md\\n\"}"}
The training has finished after 12 epochs. This is because, we set the
`EarlyStopping` criteria with a patience of 3, which terminated the
training after `pixel_AUROC` stopped improving. If we increased the
`patience`, the training would continue further.

## Testing

Now that we trained the model, we could test the model to check the
overall performance on the test set. We will also be writing the output
of the test images to a file since we set `VisualizerCallback` in
`callbacks`.
:::

::: {.cell .code execution_count="26" pycharm="{\"name\":\"#%%\\n\"}"}
``` python
engine.test(datamodule=datamodule, model=model)
```

::: {.output .stream .stderr}
    LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0,1]
:::

::: {.output .display_data}
``` json
{"model_id":"4b40cd5a1e094248b521f07ef14291de","version_major":2,"version_minor":0}
```
:::

::: {.output .display_data}
```{=html}
<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">┏━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃<span style="font-weight: bold">        Test metric        </span>┃<span style="font-weight: bold">       DataLoader 0        </span>┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│<span style="color: #008080; text-decoration-color: #008080">        image_AUROC        </span>│<span style="color: #800080; text-decoration-color: #800080">            1.0            </span>│
│<span style="color: #008080; text-decoration-color: #008080">       image_F1Score       </span>│<span style="color: #800080; text-decoration-color: #800080">            1.0            </span>│
│<span style="color: #008080; text-decoration-color: #008080">        pixel_AUROC        </span>│<span style="color: #800080; text-decoration-color: #800080">    0.9769068956375122     </span>│
└───────────────────────────┴───────────────────────────┘
</pre>
```
:::

::: {.output .execute_result execution_count="26"}
    [{'pixel_AUROC': 0.9769068956375122, 'image_AUROC': 1.0, 'image_F1Score': 1.0}]
:::
:::

::: {.cell .markdown pycharm="{\"name\":\"#%% md\\n\"}"}
## Inference

Since we have a trained model, we could infer the model on an individual
image or folder of images. Anomalib has an `PredictDataset` to let you
create an inference dataset. So let\'s try it.
:::

::: {.cell .code execution_count="25" pycharm="{\"name\":\"#%%\\n\"}"}
``` python
inference_dataset = PredictDataset(path=dataset_root / "bottle/test/broken_large/000.png")
inference_dataloader = DataLoader(dataset=inference_dataset)
```
:::

::: {.cell .markdown pycharm="{\"name\":\"#%% md\\n\"}"}
We could utilize `Trainer`\'s `predict` method to infer, and get the
outputs to visualize
:::

::: {.cell .code pycharm="{\"name\":\"#%%\\n\"}"}
``` python
predictions = engine.predict(model=model, dataloaders=inference_dataloader)[0]
```
:::

::: {.cell .markdown pycharm="{\"name\":\"#%% md\\n\"}"}
`predictions` contain image, anomaly maps, predicted scores, labels and
masks. These are all stored in a dictionary. We could check this by
printing the `prediction` keys.
:::

::: {.cell .code execution_count="24" pycharm="{\"name\":\"#%%\\n\"}"}
``` python
print(
    f'Image Shape: {predictions["image"].shape},\n'
    'Anomaly Map Shape: {predictions["anomaly_maps"].shape}, \n'
    'Predicted Mask Shape: {predictions["pred_masks"].shape}',
)
```

::: {.output .stream .stdout}
    Image Shape: torch.Size([1, 3, 256, 256]),
    Anomaly Map Shape: {predictions["anomaly_maps"].shape}, 
    Predicted Mask Shape: {predictions["pred_masks"].shape}
:::
:::

::: {.cell .markdown pycharm="{\"name\":\"#%% md\\n\"}"}
## Visualization
:::

::: {.cell .markdown pycharm="{\"name\":\"#%% md\\n\"}"}
To properly visualize the predictions, we will need to perform some
post-processing operations.
:::

::: {.cell .markdown}
Let\'s first show the input image. To do so, we will use `image_path`
key from the `predictions` dictionary, and read the image from path.
Note that `predictions` dictionary already contains `image`. However,
this is the normalized image with pixel values between 0 and 1. We will
use the original image to visualize the input image.
:::

::: {.cell .code execution_count="23"}
``` python
image_path = predictions["image_path"][0]
image_size = predictions["image"].shape[-2:]
image = np.array(Image.open(image_path).resize(image_size))
```
:::

::: {.cell .markdown pycharm="{\"name\":\"#%% md\\n\"}"}
The first output of the predictions is the anomaly map. As can be seen
above, it\'s also a torch tensor and of size
`torch.Size([1, 1, 256, 256])`. We therefore need to convert it to numpy
and squeeze the dimensions to make it `256x256` output to visualize.
:::

::: {.cell .code execution_count="22" pycharm="{\"name\":\"#%%\\n\"}"}
``` python
anomaly_map = predictions["anomaly_maps"][0]
anomaly_map = anomaly_map.cpu().numpy().squeeze()
plt.imshow(anomaly_map)
```

::: {.output .execute_result execution_count="22"}
    <matplotlib.image.AxesImage at 0x7fa298ad6fb0>
:::

::: {.output .display_data}
![](vertopal_157f981f05684299bac5668fdb188f67/8cfb334a24a17410f28ac61fc57b1d4dbdda84b5.png)
:::
:::

::: {.cell .markdown pycharm="{\"name\":\"#%% md\\n\"}"}
We could superimpose (overlay) the anomaly map on top of the original
image to get a heat map. Anomalib has a built-in function to achieve
this. Let\'s try it.
:::

::: {.cell .code execution_count="21" pycharm="{\"name\":\"#%%\\n\"}"}
``` python
heat_map = superimpose_anomaly_map(anomaly_map=anomaly_map, image=image, normalize=True)
plt.imshow(heat_map)
```

::: {.output .execute_result execution_count="21"}
    <matplotlib.image.AxesImage at 0x7fa298a71f00>
:::

::: {.output .display_data}
![](vertopal_157f981f05684299bac5668fdb188f67/d76fd2320477a508dd30665b3887fd3b0ce0552a.png)
:::
:::

::: {.cell .markdown pycharm="{\"name\":\"#%% md\\n\"}"}
`predictions` also contains prediction scores and labels.
:::

::: {.cell .code execution_count="20" pycharm="{\"name\":\"#%%\\n\"}"}
``` python
pred_score = predictions["pred_scores"][0]
pred_labels = predictions["pred_labels"][0]
print(pred_score, pred_labels)
```

::: {.output .stream .stdout}
    tensor(0.6486) tensor(True)
:::
:::

::: {.cell .markdown pycharm="{\"name\":\"#%% md\\n\"}"}
The last part of the predictions is the mask that is predicted by the
model. This is a boolean mask containing True/False for the
abnormal/normal pixels, respectively.
:::

::: {.cell .code execution_count="19" pycharm="{\"name\":\"#%%\\n\"}"}
``` python
pred_masks = predictions["pred_masks"][0].squeeze().cpu().numpy()
plt.imshow(pred_masks)
```

::: {.output .execute_result execution_count="19"}
    <matplotlib.image.AxesImage at 0x7fa298a016f0>
:::

::: {.output .display_data}
![](vertopal_157f981f05684299bac5668fdb188f67/bd9277646b2749565fc7a4b96ceb9fe15c7f00b4.png)
:::
:::

::: {.cell .markdown pycharm="{\"name\":\"#%% md\\n\"}"}
That wraps it! In this notebook, we show how we could train, test and
finally infer a FastFlow model using Anomalib API.
:::
