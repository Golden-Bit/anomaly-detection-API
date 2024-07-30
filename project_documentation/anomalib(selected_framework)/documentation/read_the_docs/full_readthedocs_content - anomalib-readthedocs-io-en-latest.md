# Anomalib

## Release 2022

## Intel OpenVINO

**Mar 28, 2024**



## GET STARTED

- 1 Installation
- 2 Get Started
- 3 Guides
   - 3.1 Anomalib in 15 Minutes
   - 3.2 Migrating from 0.* to 1.0
   - 3.3 Reference Guide
   - 3.4 How to Guide
   - 3.5 Topic Guide
   - 3.6 Developer Guide
   - 3.7 Awards and Recognition
   - 3.8 Community Engagement
- Python Module Index
- Index


**ii**


Anomalib is a deep learning library that aims to collect state-of-the-art anomaly detection algorithms for benchmarking
on both public and private datasets. Anomalib provides several ready-to-use implementations of anomaly detection
algorithms described in the recent literature, as well as a set of tools that facilitate the development and implementation
of custom models. The library has a strong focus on visual anomaly detection, where the goal of the algorithm is
to detect and/or localize anomalies within images or videos in a dataset. Anomalib is constantly updated with new
algorithms and training/inference extensions, so keep checking!

```
Key Features
```
- Simple and modular API and CLI for training, inference, benchmarking, and hyperparameter optimization.
- The largest public collection of ready-to-use deep learning anomaly detection algorithms and benchmark datasets.
- Lightning based model implementations to reduce boilerplate code and limit the implementation efforts to the
    bare essentials.
- All models can be exported to OpenVINO Intermediate Representation (IR) for accelerated inference on intel
    hardware.
- A set of inference tools for quick and easy deployment of the standard or custom anomaly detection models.

### GET STARTED 1


## 2 Get Started


### CHAPTER

**ONE**

**INSTALLATION**

Anomalib provides two ways to install the library. The first is through PyPI, and the second is through a local installa-
tion. PyPI installation is recommended if you want to use the library without making any changes to the source code.
If you want to make changes to the library, then a local installation is recommended.

**PyPI**

pip install anomalib

**Source**

# Use of virtual environment is highy recommended
# Using conda
yes | conda create -n anomalib_env python=3.
conda activate anomalib_env

# Or using your favorite virtual environment
# ...

# Clone the repository and install in editable mode
git clone https://github.com/openvinotoolkit/anomalib.git
cd anomalib
pip install -e.

## 3 Guides


**4 Chapter 1. Installation**


### CHAPTER

**TWO**

**GET STARTED**

Anomalib in 15 minutes Get started with anomalib in 15 minutes.

### 5


**6 Chapter 2. Get Started**


### CHAPTER

**THREE**

**GUIDES**

Reference Guide Learn more about anomalib API and CLI.

```
How-To Guide Learn how to use anomalib for your anomaly detection tasks.
Topic Guide Learn more about the internals of anomalib.
Developer Guide Learn how to develop and contribute to anomalib.
```
**3.1 Anomalib in 15 Minutes**

This section will walk you through the steps to train a model and use it to detect anomalies in a dataset.

**3.1.1 Installation**

Installation is simple and can be done in two ways. The first is through PyPI, and the second is through a local installa-
tion. PyPI installation is recommended if you want to use the library without making any changes to the source code.
If you want to make changes to the library, then a local installation is recommended.

**Installing the Installer**

Anomalib comes with a CLI installer that can be used to install the full package. The installer can be installed using
the following commands:

### API

pip install anomalib

### 7


**Source**

# Use of virtual environment is highy recommended
# Using conda
yes | conda create -n anomalib_env python=3.
conda activate anomalib_env

# Or using your favorite virtual environment
# ...

# Clone the repository and install in editable mode
git clone https://github.com/openvinotoolkit/anomalib.git
cd anomalib
pip install -e.

The main reason why PyPI and source installer does not install the full package is to keep the installation wheel small.
The CLI installer also automates the installation such as finding the torch version with the right CUDA/CUDNN version.

The next section demonstrates how to install the full package using the CLI installer.

**Installing the Full Package**

After installing anomalib, you can install the full package using the following commands:

anomalib -h
To use other subcommand using`anomalib install`
To use any logger install it using`anomalib install -v`
Arguments
Usage: anomalib [-h] [-c CONFIG] [--print_config [=flags]] {install} ...

```
Options:
-h, --help Show this help message and exit.
-c, --config CONFIG Path to a configuration fileinjson or yaml format.
--print_config [=flags]
Print the configuration after applying all other
arguments and exit. The optional flags customizes the
output and are one or more keywords separated b comma.
The supported flags are: comments, skip_default,
skip_null.
```
```
Subcommands:
For more details of each subcommand, add it as an argument followed by
--help.
```
```
Available subcommands:
install Install the full-package foranomalib.
```
As can be seen above, the only available sub-command isinstallat the moment. Theinstallsub-command has
options to install either the full package or the specific components of the package.

**8 Chapter 3. Guides**


anomalib install -h
To use other subcommand using`anomalib install`
To use any logger install it using`anomalib install -v`
Arguments
Usage: anomalib [options] install [-h]
[--option {full,core,dev,loggers,notebooks,openvino}]
[-v]

```
Options:
-h, --help Show this help message and exit.
--option {full,core,dev,loggers,notebooks,openvino}
Install the full or optional-dependencies.
(type: None, default: full)
-v, --verbose Set Logger level to INFO (default: False)
```
By default theinstallsub-command installs the full package. If you want to install only the specific components of
the package, you can use the--optionflag.

# Get help for the installation arguments
anomalib install -h

# Install the full package
anomalib install

# Install with verbose output
anomalib install -v

# Install the core package option only to train and evaluate models via Torch and␣
˓→Lightning
anomalib install --option core

# Install with OpenVINO option only. This is useful for edge deployment as the wheel␣
˓→size is smaller.
anomalib install --option openvino

After following these steps, your environment will be ready to use anomalib!

**3.1. Anomalib in 15 Minutes 9**


**3.1.2 Training**

Anomalib supports both API and CLI-based training. The API is more flexible and allows for more customization,
while the CLI training utilizes command line interfaces, and might be easier for those who would like to use anomalib
off-the-shelf.

### API

# Import the required modules
from anomalib.data importMVTec
from anomalib.models import Patchcore
from anomalib.engine import Engine

# Initialize the datamodule, model and engine
datamodule = MVTec()
model = Patchcore()
engine = Engine()

# Train the model
engine.fit(datamodule=datamodule, model=model)

### CLI

# Get help about the training arguments, run:
anomalib train -h

# Train by using the default values.
anomalib train --model Patchcore --data anomalib.data.MVTec

# Train by overriding arguments.
anomalib train --model Patchcore --data anomalib.data.MVTec --data.category transistor

# Train by using a config file.
anomalib train --config <path/to/config>

**10 Chapter 3. Guides**


**3.1.3 Inference**

Anomalib includes multiple inferencing scripts, including Torch, Lightning, Gradio, and OpenVINO inferencers to
perform inference using the trained/exported model. Here we show an inference example using the Lightning inferencer.

**Lightning Inference**

### API

# Assuming the datamodule, model and engine is initialized from the previous step,
# a prediction via a checkpoint file can be performed as follows:
predictions = engine.predict(
datamodule=datamodule,
model=model,
ckpt_path="path/to/checkpoint.ckpt",
)

### CLI

# To get help about the arguments, run:
anomalib predict -h

# Predict by using the default values.
anomalib predict --model anomalib.models.Patchcore\
--data anomalib.data.MVTec\
--ckpt_path <path/to/model.ckpt>

# Predict by overriding arguments.
anomalib predict --model anomalib.models.Patchcore\
--data anomalib.data.MVTec\
--ckpt_path <path/to/model.ckpt>
--return_predictions

# Predict by using a config file.
anomalib predict --config <path/to/config> --return_predictions

**Torch Inference**

**API**

Python code here.

**3.1. Anomalib in 15 Minutes 11**


### CLI

CLI command here.

**OpenVINO Inference**

**API**

Python code here.

### CLI

CLI command here.

**Gradio Inference**

### API

Python code here.

### CLI

CLI command here.

**3.1.4 Hyper-Parameter Optimization**

Anomalib supports hyper-parameter optimization using wandb and comet.ml. Here we show an example of hyper-
parameter optimization using both comet and wandb.

### CLI

# To perform hpo using wandb sweep
anomalib hpo --backend WANDB --sweep_config tools/hpo/configs/wandb.yaml

# To perform hpo using comet.ml sweep
anomalib hpo --backend COMET --sweep_config tools/hpo/configs/comet.yaml

**12 Chapter 3. Guides**


### API

# To be enabled in v1.

**3.1.5 Experiment Management**

Anomalib is integrated with various libraries for experiment tracking such as comet, tensorboard, and wandb through
lighting loggers.

### CLI

To run a training experiment with experiment tracking, you will need the following configuration file:

# Place the experiment management config here.

By using the configuration file above, you can run the experiment with the following command:

# Place the Experiment Management CLI command here.

### API

# To be enabled in v1.

**3.1.6 Benchmarking**

Anomalib provides a benchmarking tool to evaluate the performance of the anomaly detection models on a given
dataset. The benchmarking tool can be used to evaluate the performance of the models on a given dataset, or to
compare the performance of multiple models on a given dataset.

Each model in anomalib is benchmarked on a set of datasets, and the results are available insrc/anomalib/models/
<model_name>README.md. For example, the MVTec AD results for the Patchcore model are available in the corre-
sponding README.md file.

### CLI

To run the benchmarking tool, run the following command:

anomalib benchmark --config tools/benchmarking/benchmark_params.yaml

**3.1. Anomalib in 15 Minutes 13**


### API

# To be enabled in v1.

**3.1.7 Reference**

If you use this library and love it, use this to cite it:

@inproceedings{akcay2022anomalib,
title={Anomalib: A deep learning library for anomaly detection},
author={
Akcay, Samet and
Ameln, Dick and
Vaidya, Ashwin and
Lakshmanan, Barath
and Ahuja, Nilesh
and Genc, Utku
},
booktitle={2022 IEEE International Conference on Image Processing (ICIP)},
pages={1706--1710},
year={2022},
organization={IEEE}
}

**3.2 Migrating from 0.* to 1.**

**3.2.1 Overview**

The 1.0 release of the Anomaly Detection Library (AnomalyLib) introduces several changes to the library. This guide
provides an overview of the changes and how to migrate from 0.* to 1.0.

**3.2.2 Installation**

For installation instructions, refer to the _installation guide_.

**3.2.3 Changes to the CLI**

**Upgrading the Configuration**

There are several changes to the configuration of Anomalib. The configuration file has been updated to include new
parameters and remove deprecated parameters. In addition, some parameters have been moved to different sections of
the configuration.

Anomalib provides a python script to update the configuration file from 0.* to 1.0. To update the configuration file, run
the following command:

python tools/upgrade/config.py\
--input_config <path_to_0.*_config>\
--output_config <path_to_1.0_config>

**14 Chapter 3. Guides**


This script will ensure that the configuration file is updated to the 1.0 format.

In the following sections, we will discuss the changes to the configuration file in more detail.

**Changes to the Configuration File**

**Data**

Thedatasection of the configuration file has been updated such that the args can be directly used to instantiate the data
object. Below are the differences between the old and new configuration files highlighted in a markdown diff format.

-dataset:
+data:

- name: mvtec
- format: mvtec
+ class_path: anomalib.data.MVTec
+ init_args:
- path: ./datasets/MVTec
+ root: ./datasets/MVTec
    category: bottle
    image_size: 256
    center_crop: null
    normalization: imagenet
    train_batch_size: 72
    eval_batch_size: 32
    num_workers: 8
    task: segmentation
    test_split_mode: from_dir # options: [from_dir, synthetic]
    test_split_ratio: 0.2 # fraction of train images held out testing (usage depends on␣
˓→test_split_mode)
val_split_mode: same_as_test # options: [same_as_test, from_test, synthetic]
val_split_ratio: 0.5 # fraction of train/test images held out for validation (usage␣
˓→depends on val_split_mode)
seed: null
- transform_config:
- train: null
- eval: null
+ transform_config_train: null
+ transform_config_eval: null
- tiling:
- apply: false
- tile_size: null
- stride: null
- remove_border_count: 0
- use_random_tiling: False
- random_tile_count: 16+data:

Here is the summary of the changes to the configuration file:

- Thenameandformat keysfrom the old configuration are absent in the new configuration, possibly integrated
    into the design of the class atclass_path.
- Introduction of aclass_pathkey in the new configuration specifies the Python class path for data handling.

**3.2. Migrating from 0.* to 1.0 15**


- The structure has been streamlined in the new configuration, moving everything underdataandinit_args
    keys, simplifying the hierarchy.
- transform_configkeys were split intotransform_config_trainandtransform_config_evalto clearly
    separate training and evaluation configurations.
- Thetilingsection present in the old configuration has been completely removed in the new configuration.
    v1.0.0 does not support tiling. This feature will be added back in a future release.

**Model**

Similar to data configuration, themodelsection of the configuration file has been updated such that the args can be
directly used to instantiate the model object. Below are the differences between the old and new configuration files
highlighted in a markdown diff format.

```
model:
```
- name: patchcore
- backbone: wide_resnet50_
- pre_trained: true
- layers:
+ class_path: anomalib.models.Patchcore
+ init_args:
+ backbone: wide_resnet50_
+ pre_trained: true
+ layers:
    - layer
    - layer
- coreset_sampling_ratio: 0.
- num_neighbors: 9
+ coreset_sampling_ratio: 0.
+ num_neighbors: 9
- normalization_method: min_max # options: [null, min_max, cdf]
+normalization:
+ normalization_method: min_max

Here is the summary of the changes to the configuration file:

- Model Identification: Transition fromnametoclass_pathfor specifying the model, indicating a more explicit
    reference to the model’s implementation.
- Initialization Structure: Introduction ofinit_argsto encapsulate model initialization parameters, suggesting
    a move towards a more structured and possibly dynamically loaded configuration system.
- Normalization Method: Thenormalization_methodkey is removed from themodelsection and moved to a
    separatenormalizationsection in the new configuration.

**16 Chapter 3. Guides**


**Metrics**

Themetricssection of the configuration file has been updated such that the args can be directly used to instantiate
the metrics object. Below are the differences between the old and new configuration files highlighted in a markdown
diff format.

metrics:
image:

- F1Score
- AUROC
pixel:
- F1Score
- AUROC
threshold:
- method: adaptive #options: [adaptive, manual]
- manual_image: null
- manual_pixel: null
+ class_path: anomalib.metrics.F1AdaptiveThreshold
+ init_args:
+ default_value: 0.5

Here is the summary of the changes to the configuration file:

- Metric Identification: Transition frommethodtoclass_pathfor specifying the metric, indicating a more ex-
    plicit reference to the metric’s implementation.
- Initialization Structure: Introduction ofinit_argsto encapsulate metric initialization parameters, suggesting
    a move towards a more structured and possibly dynamically loaded configuration system.
- Threshold Method: Themethodkey is removed from thethresholdsection and moved to a separate
    class_pathsection in the new configuration.

**3.3 Reference Guide**

This section contains the API and CLI reference for anomalib.

Data Learn more about anomalib datamodules.

```
Models Learn more about image and video models.
Engine Learn more about anomalib Engine.
Metrics Learn more about anomalib metrics
Loggers Learn more about anomalib loggers
Callbacks Learn more about anomalib callbacks
CLI Learn more about anomalib CLI
Deployment Learn more about anomalib CLI
Pipelines Learn more about anomalib hpo, sweep and benchmarking pipelines
```
**3.3. Reference Guide 17**


**3.3.1 Data**

Anomalib data can be categorized into four main types: base, image, video, and depth. Image, video and depth datasets
are based on the base dataset and datamodule implementations.

Base Classes Learn more about base anomalib data interfaces.

```
Image Learn more about anomalib image datasets.
Video Learn more about anomalib video datasets.
Depth Learn more about anomalib depth datasets.
```
**Base Data**

Base Dataset Learn more about base anomalib dataset

```
Base Datamodule Learn more about base anomalib datamodule
Video Learn more about base anomalib video data
Depth Learn more about base anomalib depth data
```
**Base Dataset**

Anomalib dataset base class.

classanomalib.data.base.dataset.AnomalibDataset( _task_ , _transform=None_ )

```
Bases:Dataset,ABC
Anomalib dataset.
The dataset is based on a dataframe that contains the information needed by the dataloader to load each of the
dataset items into memory.
The samples dataframe must be set from the subclass using the setter of the samples property.
The DataFrame must, at least, include the following columns:
```
- _split_ (str): The subset to which the dataset item is assigned (e.g., ‘train’, ‘test’).
- _image_path_ (str): Path to the file system location where the image is stored.
- _label_index_ (int): Index of the anomaly label, typically 0 for ‘normal’ and 1 for ‘anomalous’.
- _mask_path_ (str, optional): Path to the ground truth masks (for the anomalous images only).
Required if task is ‘segmentation’.
**Example DataFrame:**

```
image_path label label_index mask_path split
0 path/to/image.png anomalous 1 path/to/mask.png train
```
```
Note: The example above is illustrative and may need to be adjusted based on the specific dataset structure.
```
**18 Chapter 3. Guides**


```
Parameters
```
- task(str) – Task type, either ‘classification’ or ‘segmentation’
- transform(Transform, optional) – Transforms that should be applied to the input im-
    ages. Defaults toNone.

```
property category: str | None
Get the category of the dataset.
property has_anomalous: bool
Check if the dataset contains any anomalous samples.
property has_normal: bool
Check if the dataset contains any normal samples.
property name: str
Name of the dataset.
property samples: DataFrame
Get the samples dataframe.
subsample( indices , inplace=False )
Subsamples the dataset at the provided indices.
Parameters
```
- indices(Sequence[int]) – Indices at which the dataset is to be subsampled.
- inplace(bool) – When true, the subsampling will be performed on the instance itself.
    Defaults toFalse.
**Return type**
AnomalibDataset

**Base Datamodules**

Anomalib datamodule base class.

classanomalib.data.base.datamodule.AnomalibDataModule( _train_batch_size_ , _eval_batch_size_ ,
_num_workers_ , _val_split_mode_ ,
_val_split_ratio_ , _test_split_mode=None_ ,
_test_split_ratio=None_ , _image_size=None_ ,
_transform=None_ , _train_transform=None_ ,
_eval_transform=None_ , _seed=None_ )
Bases:LightningDataModule,ABC
Base Anomalib data module.
**Parameters**

- train_batch_size(int) – Batch size used by the train dataloader.
- eval_batch_size(int) – Batch size used by the val and test dataloaders.
- num_workers(int) – Number of workers used by the train, val and test dataloaders.
- val_split_mode(ValSplitMode) – Determines how the validation split is obtained. Op-
    tions: [none, same_as_test, from_test, synthetic]
- val_split_ratio(float) – Fraction of the train or test images held our for validation.

**3.3. Reference Guide 19**


- test_split_mode(Optional[TestSplitMode], optional) – Determines how the test
    split is obtained. Options: [none, from_dir, synthetic]. Defaults toNone.
- test_split_ratio(float) – Fraction of the train images held out for testing. Defaults to
    None.
- image_size(tuple[int, int], optional) – Size to which input images should be re-
    sized. Defaults toNone.
- transform(Transform, optional) – Transforms that should be applied to the input im-
    ages. Defaults toNone.
- train_transform(Transform, optional) – Transforms that should be applied to the
    input images during training. Defaults toNone.
- eval_transform(Transform, optional) – Transforms that should be applied to the
    input images during evaluation. Defaults toNone.
- seed(int | None, optional) – Seed used during random subset splitting. Defaults to
    None.
property category: str
Get the category of the datamodule.
property eval_transform: Transform
Get the transform that will be passed to the val/test/predict datasets.
If the eval_transform is not set, the engine will request the transform from the model.
property name: str
Name of the datamodule.
predict_dataloader()
Use the test dataloader for inference unless overridden.
**Return type**
Any
setup( _stage=None_ )
Set up train, validation and test data.
**Parameters**
stage(str|None) – str | None: Train/Val/Test stages. Defaults toNone.
**Return type**
None
test_dataloader()
Get test dataloader.
**Return type**
Any
train_dataloader()
Get train dataloader.
**Return type**
Any
property train_transform: Transform
Get the transforms that will be passed to the train dataset.
If the train_transform is not set, the engine will request the transform from the model.

**20 Chapter 3. Guides**


```
property transform: Transform
Property that returns the user-specified transform for the datamodule, if any.
This property is accessed by the engine to set the transform for the model. The eval_transform takes prece-
dence over the train_transform, because the transform that we store in the model is the one that should be
used during inference.
val_dataloader()
Get validation dataloader.
Return type
Any
```
anomalib.data.base.datamodule.collate_fn( _batch_ )

```
Collate bounding boxes as lists.
Bounding boxes are collated as a list of tensors, while the default collate function is used for all other entries.
Parameters
batch(List) – list of items in the batch where len(batch) is equal to the batch size.
Returns
Dictionary containing the collated batch information.
Return type
dict[str, Any]
```
**Base Depth Data**

Base Depth Dataset.

classanomalib.data.base.depth.AnomalibDepthDataset( _task_ , _transform=None_ )

```
Bases:AnomalibDataset,ABC
Base depth anomalib dataset class.
Parameters
```
- task(str) – Task type, either ‘classification’ or ‘segmentation’
- transform(Transform, optional) – Transforms that should be applied to the input im-
    ages. Defaults toNone.

**Base Video Data**

Base Video Dataset.

classanomalib.data.base.video.AnomalibVideoDataModule( _train_batch_size_ , _eval_batch_size_ ,
_num_workers_ , _val_split_mode_ ,
_val_split_ratio_ , _test_split_mode=None_ ,
_test_split_ratio=None_ , _image_size=None_ ,
_transform=None_ , _train_transform=None_ ,
_eval_transform=None_ , _seed=None_ )
Bases:AnomalibDataModule
Base class for video data modules.

**3.3. Reference Guide 21**


classanomalib.data.base.video.AnomalibVideoDataset( _task_ , _clip_length_in_frames_ ,
_frames_between_clips_ , _transform=None_ ,
_target_frame=VideoTargetFrame.LAST_ )
Bases:AnomalibDataset,ABC
Base video anomalib dataset class.
**Parameters**

- task(str) – Task type, either ‘classification’ or ‘segmentation’
- clip_length_in_frames(int) – Number of video frames in each clip.
- frames_between_clips(int) – Number of frames between each consecutive video clip.
- transform(Transform, optional) – Transforms that should be applied to the input
    clips. Defaults toNone.
- target_frame(VideoTargetFrame) – Specifies the target frame in the video clip, used
    for ground truth retrieval. Defaults toVideoTargetFrame.LAST.
property samples: DataFrame
Get the samples dataframe.

classanomalib.data.base.video.VideoTargetFrame( _value_ , _names=None_ , _*_ , _module=None_ ,
_qualname=None_ , _type=None_ , _start=1_ ,
_boundary=None_ )
Bases:str,Enum
Target frame for a video-clip.
Used in multi-frame models to determine which frame’s ground truth information will be used.

**Image Data**

BTech Learn more about BTech dataset.

```
Folder Learn more about custom folder dataset.
Kolektor Learn more about Kolektor dataset.
MVTec 2D Learn more about MVTec 2D dataset
Visa Learn more about Visa dataset.
```
**BTech Data**

BTech Dataset.

This script contains PyTorch Lightning DataModule for the BTech dataset.

If the dataset is not on the file system, the script downloads and extracts the dataset and create PyTorch data objects.

classanomalib.data.image.btech.BTech( _root='./datasets/BTech'_ , _category=' 01 '_ , _train_batch_size=32_ ,
_eval_batch_size=32_ , _num_workers=8_ ,
_task=TaskType.SEGMENTATION_ , _image_size=None_ ,
_transform=None_ , _train_transform=None_ , _eval_transform=None_ ,
_test_split_mode=TestSplitMode.FROM_DIR_ , _test_split_ratio=0.2_ ,
_val_split_mode=ValSplitMode.SAME_AS_TEST_ ,
_val_split_ratio=0.5_ , _seed=None_ )

**22 Chapter 3. Guides**


```
Bases:AnomalibDataModule
BTech Lightning Data Module.
Parameters
```
- root(Path | str) – Path to the BTech dataset. Defaults to"./datasets/BTech".
- category(str) – Name of the BTech category. Defaults to"01".
- train_batch_size(int, optional) – Training batch size. Defaults to 32.
- eval_batch_size(int, optional) – Eval batch size. Defaults to 32.
- num_workers(int, optional) – Number of workers. Defaults to 8.
- task(TaskType, optional) – Task type. Defaults toTaskType.SEGMENTATION.
- image_size(tuple[int, int], optional) – Size to which input images should be re-
    sized. Defaults toNone.
- transform(Transform, optional) – Transforms that should be applied to the input im-
    ages. Defaults toNone.
- train_transform(Transform, optional) – Transforms that should be applied to the
    input images during training. Defaults toNone.
- eval_transform(Transform, optional) – Transforms that should be applied to the
    input images during evaluation. Defaults toNone.
- test_split_mode(TestSplitMode, optional) – Setting that determines how the test-
    ing subset is obtained. Defaults toTestSplitMode.FROM_DIR.
- test_split_ratio(float, optional) – Fraction of images from the train set that will
    be reserved for testing. Defaults to0.2.
- val_split_mode(ValSplitMode, optional) – Setting that determines how the valida-
    tion subset is obtained. Defaults toValSplitMode.SAME_AS_TEST.
- val_split_ratio(float, optional) – Fraction of train or test images that will be re-
    served for validation. Defaults to0.5.
- seed(int | None, optional) – Seed which may be set to a fixed value for reproducibil-
    ity. Defaults toNone.

```
Examples
```
```
To create the BTech datamodule, we need to instantiate the class, and call thesetupmethod.
```
```
>>> from anomalib.data importBTech
>>>datamodule = BTech(
... root="./datasets/BTech",
... category="01",
... image_size=256,
... train_batch_size=32,
... eval_batch_size=32,
... num_workers=8,
... transform_config_train=None,
... transform_config_eval=None,
(continues on next page)
```
**3.3. Reference Guide 23**


```
(continued from previous page)
...)
>>>datamodule.setup()
```
```
To get the train dataloader and the first batch of data:
```
```
>>>i, data = next(enumerate(datamodule.train_dataloader()))
>>>data.keys()
dict_keys(['image'])
>>>data["image"].shape
torch.Size([32, 3, 256, 256])
```
```
To access the validation dataloader and the first batch of data:
```
```
>>>i, data = next(enumerate(datamodule.val_dataloader()))
>>>data.keys()
dict_keys(['image_path', 'label', 'mask_path', 'image', 'mask'])
>>>data["image"].shape, data["mask"].shape
(torch.Size([32, 3, 256, 256]), torch.Size([32, 256, 256]))
```
```
Similarly, to access the test dataloader and the first batch of data:
```
```
>>>i, data = next(enumerate(datamodule.test_dataloader()))
>>>data.keys()
dict_keys(['image_path', 'label', 'mask_path', 'image', 'mask'])
>>>data["image"].shape, data["mask"].shape
(torch.Size([32, 3, 256, 256]), torch.Size([32, 256, 256]))
```
```
prepare_data()
Download the dataset if not available.
This method checks if the specified dataset is available in the file system. If not, it downloads and extracts
the dataset into the appropriate directory.
Return type
None
```
```
Example
```
```
Assume the dataset is not available on the file system. Here’s how the directory structure looks before and
after calling the prepare_data method:
Before:
```
```
$ tree datasets
datasets
dataset1
dataset2
```
```
Calling the method:
```
```
>> datamodule = BTech(root="./datasets/BTech", category="01")
>> datamodule.prepare_data()
```
```
After:
```
**24 Chapter 3. Guides**


```
$ tree datasets
datasets
dataset1
dataset2
BTech
01
02
03
```
classanomalib.data.image.btech.BTechDataset( _root_ , _category_ , _transform=None_ , _split=None_ ,
_task=TaskType.SEGMENTATION_ )
Bases:AnomalibDataset
Btech Dataset class.
**Parameters**

- root(str|Path) – Path to the BTech dataset
- category(str) – Name of the BTech category.
- transform(Transform, optional) – Transforms that should be applied to the input im-
    ages. Defaults toNone.
- split(str|Split|None) – ‘train’, ‘val’ or ‘test’
- task(TaskType|str) –classification,detectionorsegmentation
- create_validation_set– Create a validation subset in addition to the train and test sub-
    sets

```
Examples
```
```
>>> from anomalib.data.image.btech importBTechDataset
>>> from anomalib.data.utils.transforms importget_transforms
>>>transform = get_transforms(image_size=256)
>>>dataset = BTechDataset(
... task="classification",
... transform=transform,
... root='./datasets/BTech',
... category=' 01 ',
...)
>>>dataset[0].keys()
>>>dataset.setup()
dict_keys(['image'])
```
```
>>>dataset.split = "test"
>>>dataset[0].keys()
dict_keys(['image', 'image_path', 'label'])
```
```
>>>dataset.task = "segmentation"
>>>dataset.split = "train"
>>>dataset[0].keys()
dict_keys(['image'])
```
**3.3. Reference Guide 25**


```
>>>dataset.split = "test"
>>>dataset[0].keys()
dict_keys(['image_path', 'label', 'mask_path', 'image', 'mask'])
```
```
>>>dataset[0]["image"].shape, dataset[0]["mask"].shape
(torch.Size([3, 256, 256]), torch.Size([256, 256]))
```
anomalib.data.image.btech.make_btech_dataset( _path_ , _split=None_ )

```
Create BTech samples by parsing the BTech data file structure.
The files are expected to follow the structure:
```
```
path/to/dataset/split/category/image_filename.png
path/to/dataset/ground_truth/category/mask_filename.png
```
```
Parameters
```
- path(Path) – Path to dataset
- split(str |Split| None, optional) – Dataset split (ie., either train or test). De-
    faults toNone.

```
Example
```
```
The following example shows how to get training samples from BTech 01 category:
```
```
>>>root = Path('./BTech')
>>>category =' 01 '
>>>path = root / category
>>>path
PosixPath('BTech/01')
```
```
>>>samples = make_btech_dataset(path, split='train')
>>>samples.head()
path split label image_path mask_path ␣
˓→label_index
0 BTech/01 train 01 BTech/01/train/ok/105.bmp BTech/01/ground_truth/ok/105.png ␣
˓→ 0
1 BTech/01 train 01 BTech/01/train/ok/017.bmp BTech/01/ground_truth/ok/017.png ␣
˓→ 0
...
```
```
Returns
an output dataframe containing samples for the requested split (ie., train or test)
Return type
DataFrame
```
**26 Chapter 3. Guides**


**Folder Data**

Custom Folder Dataset.

This script creates a custom dataset from a folder.

classanomalib.data.image.folder.Folder( _name_ , _normal_dir_ , _root=None_ , _abnormal_dir=None_ ,
_normal_test_dir=None_ , _mask_dir=None_ ,
_normal_split_ratio=0.2_ , _extensions=None_ , _train_batch_size=32_ ,
_eval_batch_size=32_ , _num_workers=8_ ,
_task=TaskType.SEGMENTATION_ , _image_size=None_ ,
_transform=None_ , _train_transform=None_ , _eval_transform=None_ ,
_test_split_mode=TestSplitMode.FROM_DIR_ ,
_test_split_ratio=0.2_ ,
_val_split_mode=ValSplitMode.FROM_TEST_ ,
_val_split_ratio=0.5_ , _seed=None_ )
Bases:AnomalibDataModule
Folder DataModule.
**Parameters**

- name(str) – Name of the dataset. This is used to name the datamodule, especially when
    logging/saving.
- normal_dir(str | Path | Sequence) – Name of the directory containing normal im-
    ages.
- root(str | Path | None) – Path to the root folder containing normal and abnormal dirs.
    Defaults toNone.
- abnormal_dir(str | Path | None | Sequence) – Name of the directory containing
    abnormal images. Defaults toNone.
- normal_test_dir(str | Path | Sequence | None, optional) – Path to the direc-
    tory containing normal images for the test dataset. Defaults toNone.
- mask_dir(str | Path | Sequence | None, optional) – Path to the directory con-
    taining the mask annotations. Defaults toNone.
- normal_split_ratio(float, optional) – Ratio to split normal training images and
    add to the test set in case test set doesn’t contain any normal images. Defaults to 0.2.
- extensions(tuple[str, ...] | None, optional) – Type of the image extensions to
    read from the directory. Defaults toNone.
- train_batch_size(int, optional) – Training batch size. Defaults to 32.
- eval_batch_size(int, optional) – Validation, test and predict batch size. Defaults to
    32.
- num_workers(int, optional) – Number of workers. Defaults to 8.
- task(TaskType, optional) – Task type. Could beclassification,detectionor
    segmentation. Defaults tosegmentation.
- image_size(tuple[int, int], optional) – Size to which input images should be re-
    sized. Defaults toNone.
- transform(Transform, optional) – Transforms that should be applied to the input im-
    ages. Defaults toNone.

**3.3. Reference Guide 27**


- train_transform(Transform, optional) – Transforms that should be applied to the
    input images during training. Defaults toNone.
- eval_transform(Transform, optional) – Transforms that should be applied to the
    input images during evaluation. Defaults toNone.
- test_split_mode(TestSplitMode) – Setting that determines how the testing subset is
    obtained. Defaults toTestSplitMode.FROM_DIR.
- test_split_ratio(float) – Fraction of images from the train set that will be reserved
    for testing. Defaults to0.2.
- val_split_mode(ValSplitMode) – Setting that determines how the validation subset is
    obtained. Defaults toValSplitMode.FROM_TEST.
- val_split_ratio(float) – Fraction of train or test images that will be reserved for vali-
    dation. Defaults to0.5.
- seed(int | None, optional) – Seed used during random subset splitting. Defaults to
    None.

```
Examples
```
```
The following code demonstrates how to use theFolderdatamodule. Assume that the dataset is structured as
follows:
```
```
$ tree sample_dataset
sample_dataset
colour
00.jpg
...
x.jpg
crack
00.jpg
...
y.jpg
good
...
z.jpg
LICENSE
mask
colour
...
x.jpg
crack
...
y.jpg
```
```
folder_datamodule = Folder(
root=dataset_root,
normal_dir="good",
abnormal_dir="crack",
task=TaskType.SEGMENTATION,
mask_dir=dataset_root / "mask" / "crack",
image_size=256,
(continues on next page)
```
**28 Chapter 3. Guides**


```
(continued from previous page)
normalization=InputNormalizationMethod.NONE,
)
folder_datamodule.setup()
```
```
To access the training images,
```
```
>> i, data = next(enumerate(folder_datamodule.train_dataloader()))
>> print(data.keys(), data["image"].shape)
```
```
To access the test images,
```
```
>> i, data = next(enumerate(folder_datamodule.test_dataloader()))
>> print(data.keys(), data["image"].shape)
```
```
property name: str
Name of the datamodule.
Folder datamodule overrides the name property to provide a custom name.
```
classanomalib.data.image.folder.FolderDataset( _name_ , _task_ , _normal_dir_ , _transform=None_ , _root=None_ ,
_abnormal_dir=None_ , _normal_test_dir=None_ ,
_mask_dir=None_ , _split=None_ , _extensions=None_ )
Bases:AnomalibDataset
Folder dataset.
This class is used to create a dataset from a folder. The class utilizes the Torch Dataset class.
**Parameters**

- name(str) – Name of the dataset. This is used to name the datamodule, especially when
    logging/saving.
- task(TaskType) – Task type. (classification,detectionorsegmentation).
- transform(Transform, optional) – Transforms that should be applied to the input im-
    ages. Defaults toNone.
- normal_dir(str | Path | Sequence) – Path to the directory containing normal im-
    ages.
- root(str | Path | None) – Root folder of the dataset. Defaults toNone.
- abnormal_dir(str | Path | Sequence | None, optional) – Path to the directory
    containing abnormal images. Defaults toNone.
- normal_test_dir(str | Path | Sequence | None, optional) – Path to the direc-
    tory containing normal images for the test dataset. Defaults toNone.
- mask_dir(str | Path | Sequence | None, optional) – Path to the directory con-
    taining the mask annotations. Defaults toNone.
- split(str |Split| None) – Fixed subset split that follows from folder structure on
    file system. Choose from [Split.FULL, Split.TRAIN, Split.TEST] Defaults toNone.
- extensions(tuple[str, ...] | None, optional) – Type of the image extensions to
    read from the directory. Defaults toNone.
**Raises**
ValueError– When task is set to classification and _mask_dir_ is provided. When _mask_dir_ is
provided, _task_ should be set to _segmentation_.

**3.3. Reference Guide 29**


```
Examples
```
```
Assume that we would like to use thisFolderDatasetto create a dataset from a folder for a classification task.
We could first create the transforms,
```
```
>>> from anomalib.data.utils importInputNormalizationMethod, get_transforms
>>>transform = get_transforms(image_size=256,␣
˓→normalization=InputNormalizationMethod.NONE)
```
```
We could then create the dataset as follows,
```
```
folder_dataset_classification_train = FolderDataset(
normal_dir=dataset_root / "good",
abnormal_dir=dataset_root / "crack",
split="train",
transform=transform,
task=TaskType.CLASSIFICATION,
)
```
```
property name: str
Name of the dataset.
Folder dataset overrides the name property to provide a custom name.
```
anomalib.data.image.folder.make_folder_dataset( _normal_dir_ , _root=None_ , _abnormal_dir=None_ ,
_normal_test_dir=None_ , _mask_dir=None_ , _split=None_ ,
_extensions=None_ )
Make Folder Dataset.
**Parameters**

- normal_dir(str | Path | Sequence) – Path to the directory containing normal im-
    ages.
- root(str | Path | None) – Path to the root directory of the dataset. Defaults toNone.
- abnormal_dir(str | Path | Sequence | None, optional) – Path to the directory
    containing abnormal images. Defaults toNone.
- normal_test_dir(str | Path | Sequence | None, optional) – Path to the direc-
    tory containing normal images for the test dataset. Normal test images will be a split of
    _normal_dir_ if _None_. Defaults toNone.
- mask_dir(str | Path | Sequence | None, optional) – Path to the directory con-
    taining the mask annotations. Defaults toNone.
- split(str |Split| None, optional) – Dataset split (ie., Split.FULL, Split.TRAIN
    or Split.TEST). Defaults toNone.
- extensions(tuple[str, ...] | None, optional) – Type of the image extensions to
    read from the directory. Defaults toNone.
**Returns**
an output dataframe containing samples for the requested split (ie., train or test).
**Return type**
DataFrame

**30 Chapter 3. Guides**


```
Examples
```
```
Assume that we would like to use thismake_folder_datasetto create a dataset from a folder. We could then
create the dataset as follows,
```
```
folder_df = make_folder_dataset(
normal_dir=dataset_root / "good",
abnormal_dir=dataset_root / "crack",
split="train",
)
folder_df.head()
```
```
image_path label label_index mask_path split
0 ./toy/good/00.jpg DirType.NORMAL 0 Split.TRAIN
1 ./toy/good/01.jpg DirType.NORMAL 0 Split.TRAIN
2 ./toy/good/02.jpg DirType.NORMAL 0 Split.TRAIN
3 ./toy/good/03.jpg DirType.NORMAL 0 Split.TRAIN
4 ./toy/good/04.jpg DirType.NORMAL 0 Split.TRAIN
```
**Kolektor Data**

Kolektor Surface-Defect Dataset (CC BY-NC-SA 4.0).

**Description:**
This script provides a PyTorch Dataset, DataLoader, and PyTorch Lightning DataModule for the Kolektor
Surface-Defect dataset. The dataset can be accessed at Kolektor Surface-Defect Dataset.

**License:**
The Kolektor Surface-Defect dataset is released under the Creative Commons Attribution-NonCommercial-
ShareAlike 4.0 International License (CC BY-NC-SA 4.0). For more details, visit Creative Commons License.

**Reference:**
Tabernik, Domen, Samo Šela, Jure Skvarč, and Danijel Skočaj. “Segmentation-based deep-learning approach
for surface-defect detection.” Journal of Intelligent Manufacturing 31, no. 3 (2020): 759-776.

classanomalib.data.image.kolektor.Kolektor( _root='./datasets/kolektor'_ , _train_batch_size=32_ ,
_eval_batch_size=32_ , _num_workers=8_ ,
_task=TaskType.SEGMENTATION_ , _image_size=None_ ,
_transform=None_ , _train_transform=None_ ,
_eval_transform=None_ ,
_test_split_mode=TestSplitMode.FROM_DIR_ ,
_test_split_ratio=0.2_ ,
_val_split_mode=ValSplitMode.SAME_AS_TEST_ ,
_val_split_ratio=0.5_ , _seed=None_ )
Bases:AnomalibDataModule
Kolektor Datamodule.
**Parameters**

- root(Path | str) – Path to the root of the dataset
- train_batch_size(int, optional) – Training batch size. Defaults to 32.
- eval_batch_size(int, optional) – Test batch size. Defaults to 32.
- num_workers(int, optional) – Number of workers. Defaults to 8.

**3.3. Reference Guide 31**


- TaskType)(task) – Task type, ‘classification’, ‘detection’ or ‘segmentation’ Defaults to
    TaskType.SEGMENTATION.
- image_size(tuple[int, int], optional) – Size to which input images should be re-
    sized. Defaults toNone.
- transform(Transform, optional) – Transforms that should be applied to the input im-
    ages. Defaults toNone.
- train_transform(Transform, optional) – Transforms that should be applied to the
    input images during training. Defaults toNone.
- eval_transform(Transform, optional) – Transforms that should be applied to the
    input images during evaluation. Defaults toNone.
- test_split_mode(TestSplitMode) – Setting that determines how the testing subset is
    obtained. Defaults toTestSplitMode.FROM_DIR
- test_split_ratio(float) – Fraction of images from the train set that will be reserved
    for testing. Defaults to0.2
- val_split_mode(ValSplitMode) – Setting that determines how the validation subset is
    obtained. Defaults toValSplitMode.SAME_AS_TEST
- val_split_ratio(float) – Fraction of train or test images that will be reserved for vali-
    dation. Defaults to0.5
- seed(int | None, optional) – Seed which may be set to a fixed value for reproducibil-
    ity. Defaults toNone.
prepare_data()
Download the dataset if not available.
This method checks if the specified dataset is available in the file system. If not, it downloads and extracts
the dataset into the appropriate directory.
**Return type**
None

```
Example
```
```
Assume the dataset is not available on the file system. Here’s how the directory structure looks before and
after calling the prepare_data method:
Before:
```
```
$ tree datasets
datasets
dataset1
dataset2
```
```
Calling the method:
```
```
>> datamodule = Kolektor(root="./datasets/kolektor")
>> datamodule.prepare_data()
```
```
After:
```
**32 Chapter 3. Guides**


```
$ tree datasets
datasets
dataset1
dataset2
kolektor
kolektorsdd
kos01
...
kos50
Part0.jpg
Part0_label.bmp
...
```
classanomalib.data.image.kolektor.KolektorDataset( _task_ , _root='./datasets/kolektor'_ , _transform=None_ ,
_split=None_ )
Bases:AnomalibDataset
Kolektor dataset class.
**Parameters**

- task(TaskType) – Task type,classification,detectionorsegmentation
- root(Path | str) – Path to the root of the dataset Defaults to./datasets/kolektor.
- transform(Transform, optional) – Transforms that should be applied to the input im-
    ages. Defaults toNone.
- split(str |Split | None) – Split of the dataset, usually Split.TRAIN or Split.TEST
    Defaults toNone.

anomalib.data.image.kolektor.make_kolektor_dataset( _root_ , _train_split_ratio=0.8_ , _split=None_ )

```
Create Kolektor samples by parsing the Kolektor data file structure.
The files are expected to follow this structure: - Image files: path/to/dataset/item/image_filename.jpg ,
path/to/dataset/kos01/Part0.jpg - Mask files: path/to/dataset/item/mask_filename.bmp ,
path/to/dataset/kos01/Part0_label.bmp
This function creates a DataFrame to store the parsed information in the following format:
```
```
path item split label image_path mask_path label_index
0 KolektorSDD kos01 test Bad /path/to/image_file /path/to/mask_file 1
```
```
Parameters
```
- root(Path) – Path to the dataset.
- train_split_ratio(float, optional) – Ratio for splitting good images into train/test
    sets. Defaults to0.8.
- split(str |Split| None, optional) – Dataset split (either ‘train’ or ‘test’). De-
    faults toNone.
**Returns**
An output DataFrame containing the samples of the dataset.
**Return type**
pandas.DataFrame

**3.3. Reference Guide 33**


```
Example
```
```
The following example shows how to get training samples from the Kolektor Dataset:
```
```
>>> from pathlib importPath
>>>root = Path('./KolektorSDD/')
>>>samples = create_kolektor_samples(root, train_split_ratio=0.8)
>>>samples.head()
path item split label image_path mask_path ␣
˓→ label_index
0 KolektorSDD kos01 train Good KolektorSDD/kos01/Part0.jpg KolektorSDD/
˓→kos01/Part0_label.bmp 0
1 KolektorSDD kos01 train Good KolektorSDD/kos01/Part1.jpg KolektorSDD/
˓→kos01/Part1_label.bmp 0
2 KolektorSDD kos01 train Good KolektorSDD/kos01/Part2.jpg KolektorSDD/
˓→kos01/Part2_label.bmp 0
3 KolektorSDD kos01 test Good KolektorSDD/kos01/Part3.jpg KolektorSDD/
˓→kos01/Part3_label.bmp 0
4 KolektorSDD kos01 train Good KolektorSDD/kos01/Part4.jpg KolektorSDD/
˓→kos01/Part4_label.bmp 0
```
**MVTec Data**

MVTec AD Dataset (CC BY-NC-SA 4.0).

**Description:**
This script contains PyTorch Dataset, Dataloader and PyTorch Lightning DataModule for the MVTec AD dataset.
If the dataset is not on the file system, the script downloads and extracts the dataset and create PyTorch data
objects.

**License:**
MVTec AD dataset is released under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 Inter-
national License (CC BY-NC-SA 4.0)(https://creativecommons.org/licenses/by-nc-sa/4.0/).

**References**

- Paul Bergmann, Kilian Batzner, Michael Fauser, David Sattlegger, Carsten Steger: The MVTec Anomaly De-
    tection Dataset: A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection; in: International
    Journal of Computer Vision 129(4):1038-1059, 2021, DOI: 10.1007/s11263-020-01400-4.
- Paul Bergmann, Michael Fauser, David Sattlegger, Carsten Steger: MVTec AD — A Comprehensive Real-
    World Dataset for Unsupervised Anomaly Detection; in: IEEE/CVF Conference on Computer Vision and Pattern
    Recognition (CVPR), 9584-9592, 2019, DOI: 10.1109/CVPR.2019.00982.

classanomalib.data.image.mvtec.MVTec( _root='./datasets/MVTec'_ , _category='bottle'_ , _train_batch_size=32_ ,
_eval_batch_size=32_ , _num_workers=8_ ,
_task=TaskType.SEGMENTATION_ , _image_size=None_ ,
_transform=None_ , _train_transform=None_ , _eval_transform=None_ ,
_test_split_mode=TestSplitMode.FROM_DIR_ , _test_split_ratio=0.2_ ,
_val_split_mode=ValSplitMode.SAME_AS_TEST_ ,
_val_split_ratio=0.5_ , _seed=None_ )
Bases:AnomalibDataModule
MVTec Datamodule.

**34 Chapter 3. Guides**


```
Parameters
```
- root(Path | str) – Path to the root of the dataset. Defaults to"./datasets/MVTec".
- category(str) – Category of the MVTec dataset (e.g. “bottle” or “cable”). Defaults to
    "bottle".
- train_batch_size(int, optional) – Training batch size. Defaults to 32.
- eval_batch_size(int, optional) – Test batch size. Defaults to 32.
- num_workers(int, optional) – Number of workers. Defaults to 8.
- TaskType)(task) – Task type, ‘classification’, ‘detection’ or ‘segmentation’ Defaults to
    TaskType.SEGMENTATION.
- image_size(tuple[int, int], optional) – Size to which input images should be re-
    sized. Defaults toNone.
- transform(Transform, optional) – Transforms that should be applied to the input im-
    ages. Defaults toNone.
- train_transform(Transform, optional) – Transforms that should be applied to the
    input images during training. Defaults toNone.
- eval_transform(Transform, optional) – Transforms that should be applied to the
    input images during evaluation. Defaults toNone.
- test_split_mode(TestSplitMode) – Setting that determines how the testing subset is
    obtained. Defaults toTestSplitMode.FROM_DIR.
- test_split_ratio(float) – Fraction of images from the train set that will be reserved
    for testing. Defaults to0.2.
- val_split_mode(ValSplitMode) – Setting that determines how the validation subset is
    obtained. Defaults toValSplitMode.SAME_AS_TEST.
- val_split_ratio(float) – Fraction of train or test images that will be reserved for vali-
    dation. Defaults to0.5.
- seed(int | None, optional) – Seed which may be set to a fixed value for reproducibil-
    ity. Defualts toNone.

```
Examples
```
```
To create an MVTec AD datamodule with default settings:
```
```
>>>datamodule = MVTec()
>>>datamodule.setup()
>>>i, data = next(enumerate(datamodule.train_dataloader()))
>>>data.keys()
dict_keys(['image_path', 'label', 'image', 'mask_path', 'mask'])
```
```
>>>data["image"].shape
torch.Size([32, 3, 256, 256])
```
```
To change the category of the dataset:
```
```
>>>datamodule = MVTec(category="cable")
```
```
To change the image and batch size:
```
**3.3. Reference Guide 35**


```
>>>datamodule = MVTec(image_size=(512, 512), train_batch_size=16, eval_batch_
˓→size=8)
```
```
MVTec AD dataset does not provide a validation set. If you would like to use a separate validation set, you can
use theval_split_modeandval_split_ratioarguments to create a validation set.
```
```
>>>datamodule = MVTec(val_split_mode=ValSplitMode.FROM_TEST, val_split_ratio=0.1)
```
```
This will subsample the test set by 10% and use it as the validation set. If you would like to create a validation
set synthetically that would not change the test set, you can use theValSplitMode.SYNTHETICoption.
```
```
>>>datamodule = MVTec(val_split_mode=ValSplitMode.SYNTHETIC, val_split_ratio=0.2)
```
```
prepare_data()
Download the dataset if not available.
This method checks if the specified dataset is available in the file system. If not, it downloads and extracts
the dataset into the appropriate directory.
Return type
None
```
```
Example
```
```
Assume the dataset is not available on the file system. Here’s how the directory structure looks before and
after calling the prepare_data method:
Before:
```
```
$ tree datasets
datasets
dataset1
dataset2
```
```
Calling the method:
```
```
>> datamodule = MVTec(root="./datasets/MVTec", category="bottle")
>> datamodule.prepare_data()
```
```
After:
```
```
$ tree datasets
datasets
dataset1
dataset2
MVTec
bottle
...
zipper
```
classanomalib.data.image.mvtec.MVTecDataset( _task_ , _root='./datasets/MVTec'_ , _category='bottle'_ ,
_transform=None_ , _split=None_ )
Bases:AnomalibDataset
MVTec dataset class.

**36 Chapter 3. Guides**


```
Parameters
```
- task(TaskType) – Task type,classification,detectionorsegmentation.
- root(Path | str) – Path to the root of the dataset. Defaults to./datasets/MVTec.
- category(str) – Sub-category of the dataset, e.g. ‘bottle’ Defaults tobottle.
- transform(Transform, optional) – Transforms that should be applied to the input im-
    ages. Defaults toNone.
- split(str |Split | None) – Split of the dataset, usually Split.TRAIN or Split.TEST
    Defaults toNone.

```
Examples
```
```
from anomalib.data.image.mvtec importMVTecDataset
from anomalib.data.utils.transforms importget_transforms
```
```
transform = get_transforms(image_size=256)
dataset = MVTecDataset(
task="classification",
transform=transform,
root='./datasets/MVTec',
category='zipper',
)
dataset.setup()
print(dataset[0].keys())
# Output: dict_keys(['image_path','label','image'])
```
```
When the task is segmentation, the dataset will also contain the mask:
```
```
dataset.task = "segmentation"
dataset.setup()
print(dataset[0].keys())
# Output: dict_keys(['image_path','label','image','mask_path', 'mask'])
```
```
The image is a torch tensor of shape (C, H, W) and the mask is a torch tensor of shape (H, W).
```
```
print(dataset[0]["image"].shape, dataset[0]["mask"].shape)
# Output: (torch.Size([3, 256, 256]), torch.Size([256, 256]))
```
anomalib.data.image.mvtec.make_mvtec_dataset( _root_ , _split=None_ , _extensions=None_ )

```
Create MVTec AD samples by parsing the MVTec AD data file structure.
The files are expected to follow the structure:
path/to/dataset/split/category/image_filename.png path/to/dataset/ground_truth/category/mask_filename.png
This function creates a dataframe to store the parsed information based on the following format:
```
```
path split label im-
age_path
```
```
mask_path la-
bel_index
0 datasets/name test de-
fect
```
```
file-
name.png
```
```
ground_truth/defect/filename_mask.png 1
```
**3.3. Reference Guide 37**


```
Parameters
```
- root(Path) – Path to dataset
- split(str |Split| None, optional) – Dataset split (ie., either train or test). De-
    faults toNone.
- extensions(Sequence[str] | None, optional) – List of file extensions to be in-
    cluded in the dataset. Defaults toNone.

```
Examples
```
```
The following example shows how to get training samples from MVTec AD bottle category:
```
```
>>>root = Path('./MVTec')
>>>category ='bottle'
>>>path = root / category
>>>path
PosixPath('MVTec/bottle')
```
```
>>>samples = make_mvtec_dataset(path, split='train', split_ratio=0.1, seed=0)
>>>samples.head()
path split label image_path mask_path ␣
˓→ label_index
0 MVTec/bottle train good MVTec/bottle/train/good/105.png MVTec/bottle/ground_
˓→truth/good/105_mask.png 0
1 MVTec/bottle train good MVTec/bottle/train/good/017.png MVTec/bottle/ground_
˓→truth/good/017_mask.png 0
2 MVTec/bottle train good MVTec/bottle/train/good/137.png MVTec/bottle/ground_
˓→truth/good/137_mask.png 0
3 MVTec/bottle train good MVTec/bottle/train/good/152.png MVTec/bottle/ground_
˓→truth/good/152_mask.png 0
4 MVTec/bottle train good MVTec/bottle/train/good/109.png MVTec/bottle/ground_
˓→truth/good/109_mask.png 0
```
```
Returns
an output dataframe containing the samples of the dataset.
Return type
DataFrame
```
**Visa Data**

Visual Anomaly (VisA) Dataset (CC BY-NC-SA 4.0).

**Description:**

```
This script contains PyTorch Dataset, Dataloader and PyTorch
Lightning DataModule for the Visual Anomal (VisA) dataset.
If the dataset is not on the file system, the script downloads and
extracts the dataset and create PyTorch data objects.
```
**License:**
The VisA dataset is released under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 Interna-
tional License (CC BY-NC-SA 4.0)(https://creativecommons.org/licenses/by-nc-sa/4.0/).

**38 Chapter 3. Guides**


**Reference:**

- Zou, Y., Jeong, J., Pemula, L., Zhang, D., & Dabeer, O. (2022). SPot-the-Difference Self-supervised Pre-
    training for Anomaly Detection and Segmentation. In European Conference on Computer Vision (pp.
    392-408). Springer, Cham.

classanomalib.data.image.visa.Visa( _root='./datasets/visa'_ , _category='capsules'_ , _train_batch_size=32_ ,
_eval_batch_size=32_ , _num_workers=8_ ,
_task=TaskType.SEGMENTATION_ , _image_size=None_ ,
_transform=None_ , _train_transform=None_ , _eval_transform=None_ ,
_test_split_mode=TestSplitMode.FROM_DIR_ , _test_split_ratio=0.2_ ,
_val_split_mode=ValSplitMode.SAME_AS_TEST_ , _val_split_ratio=0.5_ ,
_seed=None_ )
Bases:AnomalibDataModule
VisA Datamodule.
**Parameters**

- root(Path | str) – Path to the root of the dataset Defaults to"./datasets/visa".
- category(str) – Category of the Visa dataset such ascandle. Defaults to"candle".
- train_batch_size(int, optional) – Training batch size. Defaults to 32.
- eval_batch_size(int, optional) – Test batch size. Defaults to 32.
- num_workers(int, optional) – Number of workers. Defaults to 8.
- task(TaskType) – Task type, ‘classification’, ‘detection’ or ‘segmentation’ Defaults to
    TaskType.SEGMENTATION.
- image_size(tuple[int, int], optional) – Size to which input images should be re-
    sized. Defaults toNone.
- transform(Transform, optional) – Transforms that should be applied to the input im-
    ages. Defaults toNone.
- train_transform(Transform, optional) – Transforms that should be applied to the
    input images during training. Defaults toNone.
- eval_transform(Transform, optional) – Transforms that should be applied to the
    input images during evaluation. Defaults toNone.
- test_split_mode(TestSplitMode) – Setting that determines how the testing subset is
    obtained. Defaults toTestSplitMode.FROM_DIR.
- test_split_ratio(float) – Fraction of images from the train set that will be reserved
    for testing. Defaults to0.2.
- val_split_mode(ValSplitMode) – Setting that determines how the validation subset is
    obtained. Defaults toValSplitMode.SAME_AS_TEST.
- val_split_ratio(float) – Fraction of train or test images that will be reserved for vali-
    dation. Defatuls to0.5.
- seed(int | None, optional) – Seed which may be set to a fixed value for reproducibil-
    ity. Defaults toNone.
apply_cls1_split()
Apply the 1-class subset splitting using the fixed split in the csv file.
adapted from https://github.com/amazon-science/spot-diff

**3.3. Reference Guide 39**


```
Return type
None
prepare_data()
Download the dataset if not available.
This method checks if the specified dataset is available in the file system. If not, it downloads and extracts
the dataset into the appropriate directory.
Return type
None
```
```
Example
```
```
Assume the dataset is not available on the file system. Here’s how the directory structure looks before and
after calling the prepare_data method:
Before:
```
```
$ tree datasets
datasets
dataset1
dataset2
```
```
Calling the method:
```
```
>> datamodule = Visa()
>> datamodule.prepare_data()
```
```
After:
```
```
$ tree datasets
datasets
dataset1
dataset2
visa
candle
...
pipe_fryum
Data
image_anno.csv
split_csv
1cls.csv
2cls_fewshot.csv
2cls_highshot.csv
VisA_20220922.tar
visa_pytorch
candle
...
pcb4
pipe_fryum
```
```
prepare_dataensures that the dataset is converted to MVTec format. visa_pytorchis the directory
that contains the dataset in the MVTec format.visais the directory that contains the original dataset.
```
**40 Chapter 3. Guides**


classanomalib.data.image.visa.VisaDataset( _task_ , _root_ , _category_ , _transform=None_ , _split=None_ )

```
Bases:AnomalibDataset
VisA dataset class.
Parameters
```
- task(TaskType) – Task type,classification,detectionorsegmentation
- root(str | Path) – Path to the root of the dataset
- category(str) – Sub-category of the dataset, e.g. ‘candle’
- transform(Transform, optional) – Transforms that should be applied to the input im-
    ages. Defaults toNone.
- split(str |Split | None) – Split of the dataset, usually Split.TRAIN or Split.TEST
    Defaults toNone.

```
Examples
```
```
To create a Visa dataset for classification:
```
```
from anomalib.data.image.visa importVisaDataset
from anomalib.data.utils.transforms importget_transforms
```
```
transform = get_transforms(image_size=256)
dataset = VisaDataset(
task="classification",
transform=transform,
split="train",
root="./datasets/visa/visa_pytorch/",
category="candle",
)
dataset.setup()
dataset[0].keys()
```
```
# Output
dict_keys(['image_path', 'label', 'image'])
```
```
If you want to use the dataset for segmentation, you can use the same code as above, with the task set to
segmentation. The dataset will then have amaskkey in the output dictionary.
```
```
from anomalib.data.image.visa importVisaDataset
from anomalib.data.utils.transforms importget_transforms
```
```
transform = get_transforms(image_size=256)
dataset = VisaDataset(
task="segmentation",
transform=transform,
split="train",
root="./datasets/visa/visa_pytorch/",
category="candle",
)
dataset.setup()
dataset[0].keys()
(continues on next page)
```
**3.3. Reference Guide 41**


```
(continued from previous page)
```
```
# Output
dict_keys(['image_path', 'label', 'image', 'mask_path', 'mask'])
```
**Video Data**

Avenue Learn more about Avenue dataset.

```
Shanghai Tech Learn more about Shanghai Tech dataset.
UCSD Learn more about UCSD Ped1 and Ped2 datasets.
```
**Avenue Data**

CUHK Avenue Dataset.

**Description:**
This module provides a PyTorch Dataset and PyTorch Lightning DataModule for the CUHK Avenue dataset. If
the dataset is not already present on the file system, the DataModule class will download and extract the dataset,
converting the .mat mask files to .png format.

**Reference:**

- Lu, Cewu, Jianping Shi, and Jiaya Jia. “Abnormal event detection at 150 fps in Matlab.” In Proceedings of
    the IEEE International Conference on Computer Vision, 2013.

classanomalib.data.video.avenue.Avenue( _root='./datasets/avenue'_ ,
_gt_dir='./datasets/avenue/ground_truth_demo'_ ,
_clip_length_in_frames=2_ , _frames_between_clips=1_ ,
_target_frame=VideoTargetFrame.LAST_ ,
_task=TaskType.SEGMENTATION_ , _image_size=None_ ,
_transform=None_ , _train_transform=None_ , _eval_transform=None_ ,
_train_batch_size=32_ , _eval_batch_size=32_ , _num_workers=8_ ,
_val_split_mode=ValSplitMode.SAME_AS_TEST_ ,
_val_split_ratio=0.5_ , _seed=None_ )
Bases:AnomalibVideoDataModule
Avenue DataModule class.
**Parameters**

- root(Path | str) – Path to the root of the dataset Defaults to./datasets/avenue.
- gt_dir(Path | str) – Path to the ground truth files Defaults to./datasets/avenue/
    ground_truth_demo.
- clip_length_in_frames(int, optional) – Number of video frames in each clip. De-
    faults to 2.
- frames_between_clips(int, optional) – Number of frames between each consecu-
    tive video clip. Defaults to 1.
- target_frame(VideoTargetFrame) – Specifies the target frame in the video clip, used
    for ground truth retrieval Defaults toVideoTargetFrame.LAST.

**42 Chapter 3. Guides**


- task(TaskType) – Task type, ‘classification’, ‘detection’ or ‘segmentation’ Defaults to
    TaskType.SEGMENTATION.
- image_size(tuple[int, int], optional) – Size to which input images should be re-
    sized. Defaults toNone.
- transform(Transform, optional) – Transforms that should be applied to the input im-
    ages. Defaults toNone.
- train_transform(Transform, optional) – Transforms that should be applied to the
    input images during training. Defaults toNone.
- eval_transform(Transform, optional) – Transforms that should be applied to the
    input images during evaluation. Defaults toNone.
- train_batch_size(int, optional) – Training batch size. Defaults to 32.
- eval_batch_size(int, optional) – Test batch size. Defaults to 32.
- num_workers(int, optional) – Number of workers. Defaults to 8.
- val_split_mode(ValSplitMode) – Setting that determines how the validation subset is
    obtained. Defaults toValSplitMode.FROM_TEST.
- val_split_ratio(float) – Fraction of train or test images that will be reserved for vali-
    dation. Defaults to0.5.
- seed(int | None, optional) – Seed which may be set to a fixed value for reproducibil-
    ity. Defaults toNone.

```
Examples
```
```
To create a DataModule for Avenue dataset with default parameters:
```
```
datamodule = Avenue()
datamodule.setup()
```
```
i, data = next(enumerate(datamodule.train_dataloader()))
data.keys()
# Output: dict_keys(['image', 'video_path','frames', 'last_frame','original_image'])
```
```
i, data = next(enumerate(datamodule.test_dataloader()))
data.keys()
# Output: dict_keys(['image', 'mask', 'video_path','frames', 'last_frame','original_image
˓→', 'label'])
```
```
data["image"].shape
# Output: torch.Size([32, 2, 3, 256, 256])
```
```
Note that the default task type is segmentation and the dataloader returns a mask in addition to the input. Also, it is
important to note that the dataloader returns a batch of clips, where each clip is a sequence of frames. The number
of frames in each clip is determined by theclip_length_in_framesparameter. Theframes_between_clips
parameter determines the number of frames between each consecutive clip. Thetarget_frameparameter deter-
mines which frame in the clip is used for ground truth retrieval. For example, ifclip_length_in_frames=2,
frames_between_clips=1andtarget_frame=VideoTargetFrame.LAST, then the dataloader will return a
batch of clips where each clip contains two consecutive frames from the video. The second frame in each clip will
be used as the ground truth for the first frame in the clip. The following code shows how to create a dataloader
for classification:
```
**3.3. Reference Guide 43**


```
datamodule = Avenue(
task="classification",
clip_length_in_frames=2,
frames_between_clips=1,
target_frame=VideoTargetFrame.LAST
)
datamodule.setup()
```
```
i, data = next(enumerate(datamodule.train_dataloader()))
data.keys()
# Output: dict_keys(['image', 'video_path','frames', 'last_frame','original_image'])
```
```
data["image"].shape
# Output: torch.Size([32, 2, 3, 256, 256])
```
```
prepare_data()
Download the dataset if not available.
This method checks if the specified dataset is available in the file system. If not, it downloads and extracts
the dataset into the appropriate directory.
Return type
None
```
```
Example
```
```
Assume the dataset is not available on the file system. Here’s how the directory structure looks before and
after calling the prepare_data method:
Before:
```
```
$ tree datasets
datasets
dataset1
dataset2
```
```
Calling the method:
```
```
>> datamodule = Avenue()
>> datamodule.prepare_data()
```
```
After:
```
```
$ tree datasets
datasets
dataset1
dataset2
avenue
ground_truth_demo
ground_truth_show.m
Readme.txt
testing_label_mask
testing_videos
testing_videos
(continues on next page)
```
**44 Chapter 3. Guides**


```
(continued from previous page)
...
21.avi
testing_vol
...
vol21.mat
training_videos
...
16.avi
training_vol
...
vol16.mat
```
classanomalib.data.video.avenue.AvenueDataset( _task_ , _split_ , _root='./datasets/avenue'_ ,
_gt_dir='./datasets/avenue/ground_truth_demo'_ ,
_clip_length_in_frames=2_ , _frames_between_clips=1_ ,
_transform=None_ ,
_target_frame=VideoTargetFrame.LAST_ )
Bases:AnomalibVideoDataset
Avenue Dataset class.
**Parameters**

- task(TaskType) – Task type, ‘classification’, ‘detection’ or ‘segmentation’
- split(Split) – Split of the dataset, usually Split.TRAIN or Split.TEST
- root(Path | str) – Path to the root of the dataset Defaults to./datasets/avenue.
- gt_dir(Path | str) – Path to the ground truth files Defaults to./datasets/avenue/
    ground_truth_demo.
- clip_length_in_frames(int, optional) – Number of video frames in each clip. De-
    faults to 2.
- frames_between_clips(int, optional) – Number of frames between each consecu-
    tive video clip. Defaults to 1.
- target_frame(VideoTargetFrame) – Specifies the target frame in the video clip, used
    for ground truth retrieval. Defaults toVideoTargetFrame.LAST.
- transform(Transform, optional) – Transforms that should be applied to the input im-
    ages. Defaults toNone.

```
Examples
```
```
To create an Avenue dataset to train a classification model:
```
```
transform = A.Compose([A.Resize(256, 256), A.pytorch.ToTensorV2()])
dataset = AvenueDataset(
task="classification",
transform=transform,
split="train",
root="./datasets/avenue/",
)
```
```
(continues on next page)
```
**3.3. Reference Guide 45**


```
(continued from previous page)
dataset.setup()
dataset[0].keys()
```
```
# Output: dict_keys(['image', 'video_path','frames', 'last_frame','original_image'])
```
```
If you would like to test a segmentation model, you can use the following code:
```
```
dataset = AvenueDataset(
task="segmentation",
transform=transform,
split="test",
root="./datasets/avenue/",
)
```
```
dataset.setup()
dataset[0].keys()
```
```
# Output: dict_keys(['image', 'mask', 'video_path','frames', 'last_frame','original_image
˓→', 'label'])
```
```
Avenue video dataset can also be used as an image dataset if you set the clip length to 1. This means that each
video frame will be treated as a separate sample. This is useful for training a classification model on the Avenue
dataset. The following code shows how to create an image dataset for classification:
```
```
dataset = AvenueDataset(
task="classification",
transform=transform,
split="test",
root="./datasets/avenue/",
clip_length_in_frames=1,
)
```
```
dataset.setup()
dataset[0].keys()
# Output: dict_keys(['image', 'video_path','frames', 'last_frame','original_image',
˓→'label'])
```
```
dataset[0]["image"].shape
# Output: torch.Size([3, 256, 256])
```
anomalib.data.video.avenue.make_avenue_dataset( _root_ , _gt_dir_ , _split=None_ )

```
Create CUHK Avenue dataset by parsing the file structure.
The files are expected to follow the structure:
```
- path/to/dataset/[training_videos|testing_videos]/video_filename.avi
- path/to/ground_truth/mask_filename.mat

```
Parameters
```
- root(Path) – Path to dataset
- gt_dir(Path) – Path to the ground truth

**46 Chapter 3. Guides**


- split(Split| str | None = None, optional) – Dataset split (ie., either train or
    test). Defaults toNone.

```
Example
```
```
The following example shows how to get testing samples from Avenue dataset:
```
```
>>>root = Path('./avenue')
>>>gt_dir = Path('./avenue/masks')
>>>samples = make_avenue_dataset(path, gt_dir, split='test')
>>>samples.head()
root folder image_path mask_path ␣
˓→ split
0 ./avenue testing_videos ./avenue/training_videos/01.avi ./avenue/masks/01_label.
˓→mat test
1 ./avenue testing_videos ./avenue/training_videos/02.avi ./avenue/masks/01_label.
˓→mat test
...
```
```
Returns
an output dataframe containing samples for the requested split (ie., train or test)
Return type
DataFrame
```
**Shanghai Tech Data**

ShanghaiTech Campus Dataset.

**Description:**
This module contains PyTorch Dataset and PyTorch Lightning DataModule for the ShanghaiTech Campus
dataset. If the dataset is not on the file system, the DataModule class downloads and extracts the dataset and
converts video files to a format that is readable by pyav.

**License:**
ShanghaiTech Campus Dataset is released under the BSD 2-Clause License.

**Reference:**

- W. Liu and W. Luo, D. Lian and S. Gao. “Future Frame Prediction for Anomaly Detection – A New
    Baseline.” IEEE Conference on Computer Vision and Pattern Recognition (CVPR). 2018.

classanomalib.data.video.shanghaitech.ShanghaiTech( _root='./datasets/shanghaitech'_ , _scene=1_ ,
_clip_length_in_frames=2_ ,
_frames_between_clips=1_ ,
_target_frame=VideoTargetFrame.LAST_ ,
_task=TaskType.SEGMENTATION_ ,
_image_size=None_ , _transform=None_ ,
_train_transform=None_ , _eval_transform=None_ ,
_train_batch_size=32_ , _eval_batch_size=32_ ,
_num_workers=8_ ,
_val_split_mode=ValSplitMode.SAME_AS_TEST_ ,
_val_split_ratio=0.5_ , _seed=None_ )
Bases:AnomalibVideoDataModule

**3.3. Reference Guide 47**


```
ShanghaiTech DataModule class.
Parameters
```
- root(Path | str) – Path to the root of the dataset
- scene(int) – Index of the dataset scene (category) in range [1, 13]
- clip_length_in_frames(int, optional) – Number of video frames in each clip.
- frames_between_clips(int, optional) – Number of frames between each consecu-
    tive video clip.
- target_frame(VideoTargetFrame) – Specifies the target frame in the video clip, used
    for ground truth retrieval
- TaskType)(task) – Task type, ‘classification’, ‘detection’ or ‘segmentation’
- image_size(tuple[int, int], optional) – Size to which input images should be re-
    sized. Defaults toNone.
- transform(Transform, optional) – Transforms that should be applied to the input im-
    ages. Defaults toNone.
- train_transform(Transform, optional) – Transforms that should be applied to the
    input images during training. Defaults toNone.
- eval_transform(Transform, optional) – Transforms that should be applied to the
    input images during evaluation. Defaults toNone.
- train_batch_size(int, optional) – Training batch size. Defaults to 32.
- eval_batch_size(int, optional) – Test batch size. Defaults to 32.
- num_workers(int, optional) – Number of workers. Defaults to 8.
- val_split_mode(ValSplitMode) – Setting that determines how the validation subset is
    obtained.
- val_split_ratio(float) – Fraction of train or test images that will be reserved for vali-
    dation.
- seed(int | None, optional) – Seed which may be set to a fixed value for reproducibil-
    ity.
prepare_data()
Download the dataset and convert video files.
**Return type**
None

classanomalib.data.video.shanghaitech.ShanghaiTechDataset( _task_ , _split_ ,
_root='./datasets/shanghaitech'_ ,
_scene=1_ , _clip_length_in_frames=2_ ,
_frames_between_clips=1_ , _tar-
get_frame=VideoTargetFrame.LAST_ ,
_transform=None_ )
Bases:AnomalibVideoDataset
ShanghaiTech Dataset class.
**Parameters**

- task(TaskType) – Task type, ‘classification’, ‘detection’ or ‘segmentation’

**48 Chapter 3. Guides**


- split(Split) – Split of the dataset, usually Split.TRAIN or Split.TEST
- root(Path | str) – Path to the root of the dataset
- scene(int) – Index of the dataset scene (category) in range [1, 13]
- clip_length_in_frames(int, optional) – Number of video frames in each clip.
- frames_between_clips(int, optional) – Number of frames between each consecu-
    tive video clip.
- target_frame(VideoTargetFrame) – Specifies the target frame in the video clip, used
    for ground truth retrieval.
- transform(Transform, optional) – Transforms that should be applied to the input im-
    ages. Defaults toNone.

classanomalib.data.video.shanghaitech.ShanghaiTechTestClipsIndexer( _video_paths_ , _mask_paths_ ,
_clip_length_in_frames=2_ ,
_frames_between_clips=1_ )
Bases:ClipsIndexer
Clips indexer for the test set of the ShanghaiTech Campus dataset.
The train and test subsets of the ShanghaiTech dataset use different file formats, so separate clips indexer imple-
mentations are needed.
get_clip( _idx_ )
Get a subclip from a list of videos.
**Parameters**
idx(int) – index of the subclip. Must be between 0 and num_clips().
**Return type**
tuple[Tensor,Tensor,dict[str,Any],int]
**Returns**
video (torch.Tensor) audio (torch.Tensor) info (Dict) video_idx (int): index of the video in
_video_paths_
get_mask( _idx_ )
Retrieve the masks from the file system.
**Return type**
Tensor|None

classanomalib.data.video.shanghaitech.ShanghaiTechTrainClipsIndexer( _video_paths_ , _mask_paths_ ,
_clip_length_in_frames=2_ ,
_frames_between_clips=1_ )
Bases:ClipsIndexer
Clips indexer for ShanghaiTech dataset.
The train and test subsets of the ShanghaiTech dataset use different file formats, so separate clips indexer imple-
mentations are needed.
get_mask( _idx_ )
No masks available for training set.
**Return type**
Tensor|None

**3.3. Reference Guide 49**


anomalib.data.video.shanghaitech.make_shanghaitech_dataset( _root_ , _scene_ , _split=None_ )

```
Create ShanghaiTech dataset by parsing the file structure.
The files are expected to follow the structure:
path/to/dataset/[training_videos|testing_videos]/video_filename.avi path/to/ground_truth/mask_filename.mat
```
```
Parameters
```
- root(Path) – Path to dataset
- scene(int) – Index of the dataset scene (category) in range [1, 13]
- split(Split| str | None, optional) – Dataset split (ie., either train or test). De-
    faults to None.

```
Example
```
```
The following example shows how to get testing samples from ShanghaiTech dataset:
```
```
>>>root = Path('./shanghaiTech')
>>>scene = 1
>>>samples = make_avenue_dataset(path, scene, split='test')
>>>samples.head()
root image_path split mask_path
0 shanghaitech shanghaitech/testing/frames/01_0014 test ␣
˓→shanghaitech/testing/test_pixel_mask/01_0014.npy
1 shanghaitech shanghaitech/testing/frames/01_0015 test ␣
˓→shanghaitech/testing/test_pixel_mask/01_0015.npy
...
```
```
Returns
an output dataframe containing samples for the requested split (ie., train or test)
Return type
DataFrame
```
**UCSD Data**

UCSD Pedestrian dataset.

classanomalib.data.video.ucsd_ped.UCSDped( _root='./datasets/ucsd'_ , _category='UCSDped2'_ ,
_clip_length_in_frames=2_ , _frames_between_clips=10_ ,
_target_frame=VideoTargetFrame.LAST_ ,
_task=TaskType.SEGMENTATION_ , _image_size=None_ ,
_transform=None_ , _train_transform=None_ ,
_eval_transform=None_ , _train_batch_size=8_ ,
_eval_batch_size=8_ , _num_workers=8_ ,
_val_split_mode=ValSplitMode.SAME_AS_TEST_ ,
_val_split_ratio=0.5_ , _seed=None_ )
Bases:AnomalibVideoDataModule
UCSDped DataModule class.
**Parameters**

- root(Path | str) – Path to the root of the dataset

**50 Chapter 3. Guides**


- category(str) – Sub-category of the dataset, e.g. “UCSDped1” or “UCSDped2”
- clip_length_in_frames(int, optional) – Number of video frames in each clip.
- frames_between_clips(int, optional) – Number of frames between each consecu-
    tive video clip.
- target_frame(VideoTargetFrame) – Specifies the target frame in the video clip, used
    for ground truth retrieval
- task(TaskType) – Task type, ‘classification’, ‘detection’ or ‘segmentation’
- image_size(tuple[int, int], optional) – Size to which input images should be re-
    sized. Defaults toNone.
- transform(Transform, optional) – Transforms that should be applied to the input im-
    ages. Defaults toNone.
- train_transform(Transform, optional) – Transforms that should be applied to the
    input images during training. Defaults toNone.
- eval_transform(Transform, optional) – Transforms that should be applied to the
    input images during evaluation. Defaults toNone.
- train_batch_size(int, optional) – Training batch size. Defaults to 32.
- eval_batch_size(int, optional) – Test batch size. Defaults to 32.
- num_workers(int, optional) – Number of workers. Defaults to 8.
- val_split_mode(ValSplitMode) – Setting that determines how the validation subset is
    obtained.
- val_split_ratio(float) – Fraction of train or test images that will be reserved for vali-
    dation.
- seed(int | None, optional) – Seed which may be set to a fixed value for reproducibil-
    ity.
prepare_data()
Download the dataset if not available.
**Return type**
None

classanomalib.data.video.ucsd_ped.UCSDpedClipsIndexer( _video_paths_ , _mask_paths_ ,
_clip_length_in_frames=2_ ,
_frames_between_clips=1_ )
Bases:ClipsIndexer
Clips class for UCSDped dataset.
get_clip( _idx_ )
Get a subclip from a list of videos.
**Parameters**
idx(int) – index of the subclip. Must be between 0 and num_clips().
**Return type**
tuple[Tensor,Tensor,dict[str,Any],int]
**Returns**
video (torch.Tensor) audio (torch.Tensor) info (dict) video_idx (int): index of the video in
_video_paths_

**3.3. Reference Guide 51**


```
get_mask( idx )
Retrieve the masks from the file system.
Return type
ndarray|None
```
classanomalib.data.video.ucsd_ped.UCSDpedDataset( _task_ , _root_ , _category_ , _split_ ,
_clip_length_in_frames=2_ ,
_frames_between_clips=10_ ,
_target_frame=VideoTargetFrame.LAST_ ,
_transform=None_ )
Bases:AnomalibVideoDataset
UCSDped Dataset class.
**Parameters**

- task(TaskType) – Task type, ‘classification’, ‘detection’ or ‘segmentation’
- root(Path | str) – Path to the root of the dataset
- category(str) – Sub-category of the dataset, e.g. “UCSDped1” or “UCSDped2”
- split(str |Split | None) – Split of the dataset, usually Split.TRAIN or Split.TEST
- clip_length_in_frames(int, optional) – Number of video frames in each clip.
- frames_between_clips(int, optional) – Number of frames between each consecu-
    tive video clip.
- target_frame(VideoTargetFrame) – Specifies the target frame in the video clip, used
    for ground truth retrieval.
- transform(Transform, optional) – Transforms that should be applied to the input im-
    ages. Defaults toNone.

anomalib.data.video.ucsd_ped.make_ucsd_dataset( _path_ , _split=None_ )

```
Create UCSD Pedestrian dataset by parsing the file structure.
The files are expected to follow the structure:
path/to/dataset/category/split/video_id/image_filename.tif path/to/dataset/category/split/video_id_gt/mask_filename.bmp
```
```
Parameters
```
- path(Path) – Path to dataset
- split(str |Split| None, optional) – Dataset split (ie., either train or test). De-
    faults to None.

```
Example
```
```
The following example shows how to get testing samples from UCSDped2 category:
```
```
>>>root = Path('./UCSDped')
>>>category ='UCSDped2'
>>>path = root / category
>>>path
PosixPath('UCSDped/UCSDped2')
```
**52 Chapter 3. Guides**


```
>>>samples = make_ucsd_dataset(path, split='test')
>>>samples.head()
root folder image_path mask_path ␣
˓→ split
0 UCSDped/UCSDped2 Test UCSDped/UCSDped2/Test/Test001 UCSDped/UCSDped2/Test/
˓→Test001_gt test
1 UCSDped/UCSDped2 Test UCSDped/UCSDped2/Test/Test002 UCSDped/UCSDped2/Test/
˓→Test002_gt test
...
```
```
Returns
an output dataframe containing samples for the requested split (ie., train or test)
Return type
DataFrame
```
**Depth Data**

Folder 3D Learn more about custom folder 3D dataset.

```
MVTec 3D Learn more about MVTec 3D dataset
```
**Folder 3D Data**

Custom Folder Dataset.

This script creates a custom dataset from a folder.

classanomalib.data.depth.folder_3d.Folder3D( _name_ , _normal_dir_ , _root_ , _abnormal_dir=None_ ,
_normal_test_dir=None_ , _mask_dir=None_ ,
_normal_depth_dir=None_ , _abnormal_depth_dir=None_ ,
_normal_test_depth_dir=None_ , _extensions=None_ ,
_train_batch_size=32_ , _eval_batch_size=32_ ,
_num_workers=8_ , _task=TaskType.SEGMENTATION_ ,
_image_size=None_ , _transform=None_ ,
_train_transform=None_ , _eval_transform=None_ ,
_test_split_mode=TestSplitMode.FROM_DIR_ ,
_test_split_ratio=0.2_ ,
_val_split_mode=ValSplitMode.FROM_TEST_ ,
_val_split_ratio=0.5_ , _seed=None_ )
Bases:AnomalibDataModule
Folder DataModule.
**Parameters**

- name(str) – Name of the dataset. This is used to name the datamodule, especially when
    logging/saving.
- normal_dir(str | Path) – Name of the directory containing normal images.
- root(str | Path | None) – Path to the root folder containing normal and abnormal dirs.
    Defaults toNone.

**3.3. Reference Guide 53**


- abnormal_dir(str | Path | None) – Name of the directory containing abnormal im-
    ages. Defaults toabnormal.
- normal_test_dir(str | Path | None, optional) – Path to the directory containing
    normal images for the test dataset. Defaults toNone.
- mask_dir(str | Path | None, optional) – Path to the directory containing the mask
    annotations. Defaults toNone.
- normal_depth_dir(str | Path | None, optional) – Path to the directory contain-
    ing normal depth images for the test dataset. Normal test depth images will be a split of
    _normal_dir_
- abnormal_depth_dir(str | Path | None, optional) – Path to the directory con-
    taining abnormal depth images for the test dataset.
- normal_test_depth_dir(str | Path | None, optional) – Path to the directory
    containing normal depth images for the test dataset. Normal test images will be a split of
    _normal_dir_ if _None_. Defaults to None.
- normal_split_ratio(float, optional) – Ratio to split normal training images and
    add to the test set in case test set doesn’t contain any normal images. Defaults to 0.2.
- extensions(tuple[str, ...] | None, optional) – Type of the image extensions to
    read from the directory. Defaults to None.
- train_batch_size(int, optional) – Training batch size. Defaults to 32.
- eval_batch_size(int, optional) – Test batch size. Defaults to 32.
- num_workers(int, optional) – Number of workers. Defaults to 8.
- task(TaskType, optional) – Task type. Could beclassification,detectionor
    segmentation. Defaults toTaskType.SEGMENTATION.
- image_size(tuple[int, int], optional) – Size to which input images should be re-
    sized. Defaults toNone.
- transform(Transform, optional) – Transforms that should be applied to the input im-
    ages. Defaults toNone.
- train_transform(Transform, optional) – Transforms that should be applied to the
    input images during training. Defaults toNone.
- eval_transform(Transform, optional) – Transforms that should be applied to the
    input images during evaluation. Defaults toNone.
- test_split_mode(TestSplitMode) – Setting that determines how the testing subset is
    obtained. Defaults toTestSplitMode.FROM_DIR.
- test_split_ratio(float) – Fraction of images from the train set that will be reserved
    for testing. Defaults to0.2.
- val_split_mode(ValSplitMode) – Setting that determines how the validation subset is
    obtained. Defaults toValSplitMode.FROM_TEST.
- val_split_ratio(float) – Fraction of train or test images that will be reserved for vali-
    dation. Defaults to0.5.
- seed(int | None, optional) – Seed used during random subset splitting. Defaults to
    None.

**54 Chapter 3. Guides**


```
property name: str
Name of the datamodule.
Folder3D datamodule overrides the name property to provide a custom name.
```
classanomalib.data.depth.folder_3d.Folder3DDataset( _name_ , _task_ , _normal_dir_ , _root=None_ ,
_abnormal_dir=None_ , _normal_test_dir=None_ ,
_mask_dir=None_ , _normal_depth_dir=None_ ,
_abnormal_depth_dir=None_ ,
_normal_test_depth_dir=None_ , _transform=None_ ,
_split=None_ , _extensions=None_ )
Bases:AnomalibDepthDataset
Folder dataset.
**Parameters**

- name(str) – Name of the dataset.
- task(TaskType) – Task type. (classification,detectionorsegmentation).
- transform(Transform, optional) – Transforms that should be applied to the input im-
    ages.
- normal_dir(str | Path) – Path to the directory containing normal images.
- root(str | Path | None) – Root folder of the dataset. Defaults toNone.
- abnormal_dir(str | Path | None, optional) – Path to the directory containing ab-
    normal images. Defaults toNone.
- normal_test_dir(str | Path | None, optional) – Path to the directory containing
    normal images for the test dataset. Defaults toNone.
- mask_dir(str | Path | None, optional) – Path to the directory containing the mask
    annotations. Defaults toNone.
- normal_depth_dir(str | Path | None, optional) – Path to the directory contain-
    ing normal depth images for the test dataset. Normal test depth images will be a split of
    _normal_dir_ Defaults toNone.
- abnormal_depth_dir(str | Path | None, optional) – Path to the directory con-
    taining abnormal depth images for the test dataset. Defaults toNone.
- normal_test_depth_dir(str | Path | None, optional) – Path to the directory
    containing normal depth images for the test dataset. Normal test images will be a split of
    _normal_dir_ if _None_. Defaults toNone.
- transform– Transforms that should be applied to the input images. Defaults toNone.
- split(str |Split| None) – Fixed subset split that follows from folder structure on
    file system. Choose from [Split.FULL, Split.TRAIN, Split.TEST] Defaults toNone.
- extensions(tuple[str, ...] | None, optional) – Type of the image extensions to
    read from the directory. Defaults toNone.
**Raises**
ValueError– When task is set to classification and _mask_dir_ is provided. When _mask_dir_ is
provided, _task_ should be set to _segmentation_.
property name: str
Name of the dataset.
Folder3D dataset overrides the name property to provide a custom name.

**3.3. Reference Guide 55**


anomalib.data.depth.folder_3d.make_folder3d_dataset( _normal_dir_ , _root=None_ , _abnormal_dir=None_ ,
_normal_test_dir=None_ , _mask_dir=None_ ,
_normal_depth_dir=None_ ,
_abnormal_depth_dir=None_ ,
_normal_test_depth_dir=None_ , _split=None_ ,
_extensions=None_ )
Make Folder Dataset.
**Parameters**

- normal_dir(str | Path) – Path to the directory containing normal images.
- root(str | Path | None) – Path to the root directory of the dataset. Defaults toNone.
- abnormal_dir(str | Path | None, optional) – Path to the directory containing ab-
    normal images. Defaults toNone.
- normal_test_dir(str | Path | None, optional) – Path to the directory containing
    normal images for the test
- None.(dataset. Normal test images will be a split of normal_dir if) –
    Defaults toNone.
- mask_dir(str | Path | None, optional) – Path to the directory containing the mask
    annotations. Defaults toNone.
- normal_depth_dir(str | Path | None, optional) – Path to the directory contain-
    ing normal depth images for the test dataset. Normal test depth images will be a split of
    _normal_dir_ Defaults toNone.
- abnormal_depth_dir(str | Path | None, optional) – Path to the directory con-
    taining abnormal depth images for the test dataset. Defaults toNone.
- normal_test_depth_dir(str | Path | None, optional) – Path to the directory
    containing normal depth images for the test dataset. Normal test images will be a split of
    _normal_dir_ if _None_. Defaults toNone.
- split(str |Split| None, optional) – Dataset split (ie., Split.FULL, Split.TRAIN
    or Split.TEST). Defaults toNone.
- extensions(tuple[str, ...] | None, optional) – Type of the image extensions to
    read from the directory. Defaults toNone.
**Returns**
an output dataframe containing samples for the requested split (ie., train or test)
**Return type**
DataFrame

**MVTec 3D Data**

MVTec 3D-AD Dataset (CC BY-NC-SA 4.0).

**Description:**
This script contains PyTorch Dataset, Dataloader and PyTorch Lightning DataModule for the MVTec 3D-AD
dataset. If the dataset is not on the file system, the script downloads and extracts the dataset and create PyTorch
data objects.

**License:**

**56 Chapter 3. Guides**


```
MVTec 3D-AD dataset is released under the Creative Commons
Attribution-NonCommercial-ShareAlike 4.0 International
License (CC BY-NC-SA 4.0)(https://creativecommons.org/licenses/by-nc-sa/4.0/).
```
**Reference:**

- **Paul Bergmann, Xin Jin, David Sattlegger, Carsten Steger: The MVTec 3D-AD Dataset for**
    **Unsupervised 3D Anomaly**
       Detection and Localization in: Proceedings of the 17th International Joint Conference on Computer
       Vision, Imaging and Computer Graphics Theory and Applications - Volume 5: VISAPP, 202-213,
       2022, DOI: 10.5220/ 0010865000003124.

classanomalib.data.depth.mvtec_3d.MVTec3D( _root='./datasets/MVTec3D'_ , _category='bagel'_ ,
_train_batch_size=32_ , _eval_batch_size=32_ , _num_workers=8_ ,
_task=TaskType.SEGMENTATION_ , _image_size=None_ ,
_transform=None_ , _train_transform=None_ ,
_eval_transform=None_ ,
_test_split_mode=TestSplitMode.FROM_DIR_ ,
_test_split_ratio=0.2_ ,
_val_split_mode=ValSplitMode.SAME_AS_TEST_ ,
_val_split_ratio=0.5_ , _seed=None_ )
Bases:AnomalibDataModule
MVTec Datamodule.
**Parameters**

- root(Path | str) – Path to the root of the dataset Defaults to"./datasets/MVTec3D".
- category(str) – Category of the MVTec dataset (e.g. “bottle” or “cable”). Defaults to
    bagel.
- train_batch_size(int, optional) – Training batch size. Defaults to 32.
- eval_batch_size(int, optional) – Test batch size. Defaults to 32.
- num_workers(int, optional) – Number of workers. Defaults to 8.
- task(TaskType) – Task type, ‘classification’, ‘detection’ or ‘segmentation’ Defaults to
    TaskType.SEGMENTATION.
- image_size(tuple[int, int], optional) – Size to which input images should be re-
    sized. Defaults toNone.
- transform(Transform, optional) – Transforms that should be applied to the input im-
    ages. Defaults toNone.
- train_transform(Transform, optional) – Transforms that should be applied to the
    input images during training. Defaults toNone.
- eval_transform(Transform, optional) – Transforms that should be applied to the
    input images during evaluation. Defaults toNone.
- test_split_mode(TestSplitMode) – Setting that determines how the testing subset is
    obtained. Defaults toTestSplitMode.FROM_DIR.
- test_split_ratio(float) – Fraction of images from the train set that will be reserved
    for testing. Defaults to0.2.
- val_split_mode(ValSplitMode) – Setting that determines how the validation subset is
    obtained. Defaults toValSplitMode.SAME_AS_TEST.
- val_split_ratio(float) – Fraction of train or test images that will be reserved for vali-
    dation. Defaults to0.5.

**3.3. Reference Guide 57**


- seed(int | None, optional) – Seed which may be set to a fixed value for reproducibil-
    ity. Defaults toNone.
prepare_data()
Download the dataset if not available.
**Return type**
None

classanomalib.data.depth.mvtec_3d.MVTec3DDataset( _task_ , _root='./datasets/MVTec3D'_ , _category='bagel'_ ,
_transform=None_ , _split=None_ )
Bases:AnomalibDepthDataset
MVTec 3D dataset class.
**Parameters**

- task(TaskType) – Task type,classification,detectionorsegmentation
- root(Path | str) – Path to the root of the dataset Defaults to"./datasets/MVTec3D".
- category(str) – Sub-category of the dataset, e.g. ‘bagel’ Defaults to"bagel".
- transform(Transform, optional) – Transforms that should be applied to the input im-
    ages. Defaults toNone.
- split(str |Split | None) – Split of the dataset, usually Split.TRAIN or Split.TEST
    Defaults toNone.

anomalib.data.depth.mvtec_3d.make_mvtec_3d_dataset( _root_ , _split=None_ , _extensions=None_ )

```
Create MVTec 3D-AD samples by parsing the MVTec AD data file structure.
The files are expected to follow this structure: - path/to/dataset/split/category/image_filename.png -
path/to/dataset/ground_truth/category/mask_filename.png
This function creates a DataFrame to store the parsed information. The DataFrame follows this format:
```
```
path split label image_path mask_path la-
bel_index
0 datasets/name test de-
fect
```
```
file-
name.png
```
```
ground_truth/defect/filename_mask.png 1
```
```
Parameters
```
- root(Path) – Path to the dataset.
- split(str |Split| None, optional) – Dataset split (e.g., ‘train’ or ‘test’). Defaults
    toNone.
- extensions(Sequence[str] | None, optional) – List of file extensions to be in-
    cluded in the dataset. Defaults toNone.

**58 Chapter 3. Guides**


```
Examples
```
```
The following example shows how to get training samples from the MVTec 3D-AD ‘bagel’ category:
```
```
>>> from pathlib importPath
>>>root = Path('./MVTec3D')
>>>category ='bagel'
>>>path = root / category
>>>print(path)
PosixPath('MVTec3D/bagel')
```
```
>>>samples = create_mvtec_3d_ad_samples(path, split='train')
>>>print(samples.head())
path split label image_path mask_path ␣
˓→ label_index
MVTec3D/bagel train good MVTec3D/bagel/train/good/rgb/105.png MVTec3D/bagel/
˓→ground_truth/good/gt/105.png 0
MVTec3D/bagel train good MVTec3D/bagel/train/good/rgb/017.png MVTec3D/bagel/
˓→ground_truth/good/gt/017.png 0
```
```
Returns
An output DataFrame containing the samples of the dataset.
Return type
DataFrame
```
**Data Utils**

Image & Video Utils Learn more about anomalib API and CLI.

```
Data Transforms Learn how to use anomalib for your anomaly detection tasks.
Tiling Learn more about the internals of anomalib.
Synthetic Data Learn more about the internals of anomalib.
```
**Image and Video Utils**

**Path Utils**

Path Utils.

classanomalib.data.utils.path.DirType( _value_ , _names=None_ , _*_ , _module=None_ , _qualname=None_ ,
_type=None_ , _start=1_ , _boundary=None_ )
Bases:str,Enum
Dir type names.

anomalib.data.utils.path.contains_non_printable_characters( _path_ )

```
Check if the path contains non-printable characters.
Parameters
path(str | Path) – Path to check.
```
**3.3. Reference Guide 59**


```
Returns
True if the path contains non-printable characters, False otherwise.
Return type
bool
```
```
Examples
```
```
>>>contains_non_printable_characters("./datasets/MVTec/bottle/train/good/000.png")
False
```
```
>>>contains_non_printable_characters("./datasets/MVTec/bottle/train/good/000.png\0
˓→")
True
```
anomalib.data.utils.path.is_path_too_long( _path_ , _max_length=512_ )

```
Check if the path contains too long input.
Parameters
```
- path(str | Path) – Path to check.
- max_length(int) – Maximum length a path can be before it is considered too long. De-
    faults to 512.
**Returns**
True if the path contains too long input, False otherwise.
**Return type**
bool

```
Examples
```
```
>>>contains_too_long_input("./datasets/MVTec/bottle/train/good/000.png")
False
```
```
>>>contains_too_long_input("./datasets/MVTec/bottle/train/good/000.png" + "a" *␣
˓→4096)
True
```
anomalib.data.utils.path.resolve_path( _folder_ , _root=None_ )

```
Combine root and folder and returns the absolute path.
This allows users to pass either a root directory and relative paths, or absolute paths to each of the image sources.
This function makes sure that the samples dataframe always contains absolute paths.
Parameters
```
- folder(str | Path | None) – Folder location containing image or mask data.
- root(str | Path | None) – Root directory for the dataset.
**Return type**
Path

**60 Chapter 3. Guides**


anomalib.data.utils.path.validate_and_resolve_path( _folder_ , _root=None_ , _base_dir=None_ )

```
Validate and resolve the path.
Parameters
```
- folder(str | Path) – Folder location containing image or mask data.
- root(str | Path | None) – Root directory for the dataset.
- base_dir(str | Path | None) – Base directory to restrict file access.
**Returns**
Validated and resolved path.
**Return type**
Path

anomalib.data.utils.path.validate_path( _path_ , _base_dir=None_ , _should_exist=True_ )

```
Validate the path.
Parameters
```
- path(str | Path) – Path to validate.
- base_dir(str | Path) – Base directory to restrict file access.
- should_exist(bool) – If True, do not raise an exception if the path does not exist.
**Returns**
Validated path.
**Return type**
Path

```
Examples
```
```
>>>validate_path("./datasets/MVTec/bottle/train/good/000.png")
PosixPath('/abs/path/to/anomalib/datasets/MVTec/bottle/train/good/000.png')
```
```
>>>validate_path("./datasets/MVTec/bottle/train/good/000.png", base_dir="./
˓→datasets/MVTec")
PosixPath('/abs/path/to/anomalib/datasets/MVTec/bottle/train/good/000.png')
```
```
>>>validate_path("/path/to/unexisting/file")
Traceback (most recent call last):
File "<string>", line 1, in <module>
File "<string>", line 18, in validate_path
FileNotFoundError: Path does not exist: /path/to/unexisting/file
```
```
Accessing a file without read permission should raise PermissionError:
```
```
Note: Note that, we are using/usr/local/bindirectory as an example here. If this directory does not exist on
your system, this will raiseFileNotFoundErrorinstead ofPermissionError. You could change the directory
to any directory that you do not have read permission.
```
**3.3. Reference Guide 61**


```
>>>validate_path("/bin/bash", base_dir="/bin/")
Traceback (most recent call last):
File "<string>", line 1, in <module>
File "<string>", line 18, in validate_path
PermissionError: Read permission denied for the file: /usr/local/bin
```
**Download Utils**

Helper to show progress bars with _urlretrieve_ , check hash of file.

classanomalib.data.utils.download.DownloadInfo( _name_ , _url_ , _hashsum_ , _filename=None_ )

```
Bases:object
Info needed to download a dataset from a url.
```
classanomalib.data.utils.download.DownloadProgressBar( _iterable=None_ , _desc=None_ , _total=None_ ,
_leave=True_ , _file=None_ , _ncols=None_ ,
_mininterval=0.1_ , _maxinterval=10.0_ ,
_miniters=None_ , _use_ascii=None_ ,
_disable=False_ , _unit='it'_ , _unit_scale=False_ ,
_dynamic_ncols=False_ , _smoothing=0.3_ ,
_bar_format=None_ , _initial=0_ , _position=None_ ,
_postfix=None_ , _unit_divisor=1000_ ,
_write_bytes=None_ , _lock_args=None_ ,
_nrows=None_ , _colour=None_ , _delay=0_ ,
_gui=False_ , _**kwargs_ )
Bases:tqdm
Create progress bar for urlretrieve. Subclasses _tqdm_.
For information about the parameters in constructor, refer to _tqdm_ ’s documentation.
**Parameters**

- iterable(Iterable | None) – Iterable to decorate with a progressbar. Leave blank to
    manually manage the updates.
- desc(str | None) – Prefix for the progressbar.
- total(int | float | None) – The number of expected iterations. If unspecified,
    len(iterable) is used if possible. If float(“inf”) or as a last resort, only basic progress statistics
    are displayed (no ETA, no progressbar). If _gui_ is True and this parameter needs subsequent
    updating, specify an initial arbitrary large positive number, e.g. 9e9.
- leave(bool | None) – upon termination of iteration. If _None_ , will leave only if _position_
    is _0_.
- file(io.TextIOWrapper | io.StringIO | None) – Specifies where to output the
    progress messages (default: sys.stderr). Uses _file.write(str)_ and _file.flush()_ methods. For
    encoding, see _write_bytes_.
- ncols(int | None) – The width of the entire output message. If specified, dynamically
    resizes the progressbar to stay within this bound. If unspecified, attempts to use environment
    width. The fallback is a meter width of 10 and no limit for the counter and statistics. If 0,
    will not print any meter (only stats).
- mininterval(float | None) – Minimum progress display update interval [default: 0.1]
    seconds.

**62 Chapter 3. Guides**


- maxinterval(float | None) – Maximum progress display update interval [default: 10]
    seconds. Automatically adjusts _miniters_ to correspond to _mininterval_ after long display up-
    date lag. Only works if _dynamic_miniters_ or monitor thread is enabled.
- miniters(int | float | None) – Minimum progress display update interval, in itera-
    tions. If 0 and _dynamic_miniters_ , will automatically adjust to equal _mininterval_ (more CPU
    efficient, good for tight loops). If > 0, will skip display of specified number of iterations.
    Tweak this and _mininterval_ to get very efficient loops. If your progress is erratic with both
    fast and slow iterations (network, skipping items, etc) you should set miniters=1.
- use_ascii(str | bool | None) – If unspecified or False, use unicode (smooth blocks)
    to fill the meter. The fallback is to use ASCII characters “ 123456789#”.
- disable(bool | None) – Whether to disable the entire progressbar wrapper [default:
    False]. If set to None, disable on non-TTY.
- unit(str | None) – String that will be used to define the unit of each iteration [default:
    it].
- unit_scale(int | float | bool) – If 1 or True, the number of iterations will be re-
    duced/scaled automatically and a metric prefix following the International System of Units
    standard will be added (kilo, mega, etc.) [default: False]. If any other non-zero number, will
    scale _total_ and _n_.
- dynamic_ncols(bool | None) – If set, constantly alters _ncols_ and _nrows_ to the environ-
    ment (allowing for window resizes) [default: False].
- smoothing(float | None) – Exponential moving average smoothing factor for speed es-
    timates (ignored in GUI mode). Ranges from 0 (average speed) to 1 (current/instantaneous
    speed) [default: 0.3].
- bar_format(str | None) – Specify a custom bar string formatting. May impact per-
    formance. [default: ‘{l_bar}{bar}{r_bar}’], where l_bar=’{desc}: {percentage:3.0f}%|’
    and r_bar=’| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, ‘ ‘{rate_fmt}{postfix}]’ Possi-
    ble vars: l_bar, bar, r_bar, n, n_fmt, total, total_fmt, percentage, elapsed, elapsed_s, ncols,
    nrows, desc, unit, rate, rate_fmt, rate_noinv, rate_noinv_fmt, rate_inv, rate_inv_fmt, postfix,
    unit_divisor, remaining, remaining_s, eta. Note that a trailing “: “ is automatically removed
    after {desc} if the latter is empty.
- initial(int | float | None) – The initial counter value. Useful when restarting a
    progress bar [default: 0]. If using float, consider specifying _{n:.3f}_ or similar in _bar_format_ ,
    or specifying _unit_scale_.
- position(int | None) – Specify the line offset to print this bar (starting from 0) Auto-
    matic if unspecified. Useful to manage multiple bars at once (eg, from threads).
- postfix(dict | None) – Specify additional stats to display at the end of the bar. Calls
    _set_postfix(**postfix)_ if possible (dict).
- unit_divisor(float | None) – [default: 1000], ignored unless _unit_scale_ is True.
- write_bytes(bool | None) – If (default: None) and _file_ is unspecified, bytes will be
    written in Python 2. If _True_ will also write bytes. In all other cases will default to unicode.
- lock_args(tuple | None) – Passed to _refresh_ for intermediate output (initialisation, it-
    erating, and updating). nrows (int | None): The screen height. If specified, hides nested bars
    outside this bound. If unspecified, attempts to use environment height. The fallback is 20.
- colour(str | None) – Bar colour (e.g. ‘green’, ‘#00ff00’).
- delay(float | None) – Don’t display until [default: 0] seconds have elapsed.

**3.3. Reference Guide 63**


- gui(bool | None) – WARNING: internal parameter - do not use. Use tqdm.gui.tqdm(...)
    instead. If set, will attempt to use matplotlib animations for a graphical output [default:
    False].

```
Example
```
```
>>> withDownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=url.split(
˓→'/')[-1]) asp_bar:
>>> urllib.request.urlretrieve(url, filename=output_path, reporthook=p_bar.
˓→update_to)
```
```
update_to( chunk_number=1 , max_chunk_size=1 , total_size=None )
Progress bar hook for tqdm.
Based on https://stackoverflow.com/a/53877507 The implementor does not have to bother about passing
parameters to this as it gets them from urlretrieve. However the context needs a few parameters. Refer to
the example.
Parameters
```
- chunk_number(int, optional) – The current chunk being processed. Defaults to 1.
- max_chunk_size(int, optional) – Maximum size of each chunk. Defaults to 1.
- total_size(int, optional) – Total download size. Defaults to None.
**Return type**
None

anomalib.data.utils.download.check_hash( _file_path_ , _expected_hash_ , _algorithm='sha256'_ )

```
Raise value error if hash does not match the calculated hash of the file.
Parameters
```
- file_path(Path) – Path to file.
- expected_hash(str) – Expected hash of the file.
- algorithm(str) – Hashing algorithm to use (‘sha256’, ‘sha3_512’, etc.).
**Return type**
None

anomalib.data.utils.download.download_and_extract( _root_ , _info_ )

```
Download and extract a dataset.
Parameters
```
- root(Path) – Root directory where the dataset will be stored.
- info(DownloadInfo) – Info needed to download the dataset.
**Return type**
None

anomalib.data.utils.download.extract( _file_name_ , _root_ )

```
Extract a dataset.
Parameters
```
- file_name(Path) – Path of the file to be extracted.
- root(Path) – Root directory where the dataset will be stored.

**64 Chapter 3. Guides**


```
Return type
None
```
anomalib.data.utils.download.generate_hash( _file_path_ , _algorithm='sha256'_ )

```
Generate a hash of a file using the specified algorithm.
Parameters
```
- file_path(str | Path) – Path to the file to hash.
- algorithm(str) – The hashing algorithm to use (e.g., ‘sha256’, ‘sha3_512’).
**Returns**
The hexadecimal hash string of the file.
**Return type**
str
**Raises**
ValueError– If the specified hashing algorithm is not supported.

anomalib.data.utils.download.is_file_potentially_dangerous( _file_name_ )

```
Check if a file is potentially dangerous.
Parameters
file_name(str) – Filename.
Returns
True if the member is potentially dangerous, False otherwise.
Return type
bool
```
anomalib.data.utils.download.is_within_directory( _directory_ , _target_ )

```
Check if a target path is located within a given directory.
Parameters
```
- directory(Path) – path of the parent directory
- target(Path) – path of the target
**Returns**
True if the target is within the directory, False otherwise
**Return type**
(bool)

anomalib.data.utils.download.safe_extract( _tar_file_ , _root_ , _members_ )

```
Extract safe members from a tar archive.
Parameters
```
- tar_file(TarFile) – TarFile object.
- root(Path) – Root directory where the dataset will be stored.
- members(List[TarInfo]) – List of safe members to be extracted.
**Return type**
None

**3.3. Reference Guide 65**


**Image Utils**

Image Utils.

anomalib.data.utils.image.duplicate_filename( _path_ )

```
Check and duplicate filename.
This function checks the path and adds a suffix if it already exists on the file system.
Parameters
path(str | Path) – Input Path
```
```
Examples
```
```
>>>path = Path("datasets/MVTec/bottle/test/broken_large/000.png")
>>>path.exists()
True
```
```
If we pass this toduplicate_filenamefunction we would get the following: >>> duplicate_filename(path)
PosixPath(‘datasets/MVTec/bottle/test/broken_large/000_1.png’)
Returns
Duplicated output path.
Return type
Path
```
anomalib.data.utils.image.figure_to_array( _fig_ )

```
Convert a matplotlib figure to a numpy array.
Parameters
fig(Figure) – Matplotlib figure.
Returns
Numpy array containing the image.
Return type
np.ndarray
```
anomalib.data.utils.image.generate_output_image_filename( _input_path_ , _output_path_ )

```
Generate an output filename to save the inference image.
This function generates an output filaname by checking the input and output filenames. Input path is the input to
infer, and output path is the path to save the output predictions specified by the user.
The function expectsinput_pathto always be a file, not a directory. output_pathcould be a filename or
directory. If it is a filename, the function checks if the specified filename exists on the file system. If yes,
the function callsduplicate_filenameto duplicate the filename to avoid overwriting the existing file. If
output_pathis a directory, this function adds the parent and filenames ofinput_pathtooutput_path.
Parameters
```
- input_path(str | Path) – Path to the input image to infer.
- output_path(str | Path) – Path to output to save the predictions. Could be a filename
    or a directory.

**66 Chapter 3. Guides**


```
Examples
```
```
>>>input_path = Path("datasets/MVTec/bottle/test/broken_large/000.png")
>>>output_path = Path("datasets/MVTec/bottle/test/broken_large/000.png")
>>>generate_output_image_filename(input_path, output_path)
PosixPath('datasets/MVTec/bottle/test/broken_large/000_1.png')
```
```
>>>input_path = Path("datasets/MVTec/bottle/test/broken_large/000.png")
>>>output_path = Path("results/images")
>>>generate_output_image_filename(input_path, output_path)
PosixPath('results/images/broken_large/000.png')
```
```
Raises
ValueError– When theinput_pathis not a file.
Returns
The output filename to save the output predictions from the inferencer.
Return type
Path
```
anomalib.data.utils.image.get_image_filename( _filename_ )

```
Get image filename.
Parameters
filename(str | Path) – Filename to check.
Returns
Image filename.
Return type
Path
```
```
Examples
```
```
Assume that we have the following files in the directory:
```
```
$ ls
000.png 001.jpg 002.JPEG 003.tiff 004.png 005.txt
```
```
>>>get_image_filename("000.png")
PosixPath('000.png')
```
```
>>>get_image_filename("001.jpg")
PosixPath('001.jpg')
```
```
>>>get_image_filename("009.tiff")
Traceback (most recent call last):
File "<string>", line 1, in <module>
File "<string>", line 18, in get_image_filename
FileNotFoundError: File not found: 009.tiff
```
**3.3. Reference Guide 67**


```
>>>get_image_filename("005.txt")
Traceback (most recent call last):
File "<string>", line 1, in <module>
File "<string>", line 18, in get_image_filename
ValueError:``filename`` is not an image file. 005.txt
```
anomalib.data.utils.image.get_image_filenames( _path_ , _base_dir=None_ )

```
Get image filenames.
Parameters
```
- path(str | Path) – Path to image or image-folder.
- base_dir(Path) – Base directory to restrict file access.
**Returns**
List of image filenames.
**Return type**
list[Path]

```
Examples
```
```
Assume that we have the following files in the directory:
```
```
$ tree images
images
bad
003.png
004.jpg
good
000.png
001.tiff
```
```
We can get the image filenames with various ways:
```
```
>>>get_image_filenames("images/bad/003.png")
PosixPath('/home/sakcay/Projects/anomalib/images/bad/003.png')]
```
```
It is possible to recursively get the image filenames from a directory:
```
```
>>>get_image_filenames("images")
[PosixPath('/home/sakcay/Projects/anomalib/images/bad/003.png'),
PosixPath('/home/sakcay/Projects/anomalib/images/bad/004.jpg'),
PosixPath('/home/sakcay/Projects/anomalib/images/good/001.tiff'),
PosixPath('/home/sakcay/Projects/anomalib/images/good/000.png')]
```
```
If we want to restrict the file access to a specific directory, we can usebase_dirargument.
```
```
>>>get_image_filenames("images", base_dir="images/bad")
Traceback (most recent call last):
File "<string>", line 1, in <module>
File "<string>", line 18, in get_image_filenames
ValueError: Access denied: Path is outside the allowed directory.
```
**68 Chapter 3. Guides**


anomalib.data.utils.image.get_image_filenames_from_dir( _path_ )

```
Get image filenames from directory.
Parameters
path(str | Path) – Path to image directory.
Raises
ValueError– Whenpathis not a directory.
Returns
Image filenames.
Return type
list[Path]
```
```
Examples
```
```
Assume that we have the following files in the directory: $ ls 000.png 001.jpg 002.JPEG 003.tiff 004.png 005.png
```
```
>>>get_image_filenames_from_dir(".")
[PosixPath('000.png'), PosixPath('001.jpg'), PosixPath('002.JPEG'),
PosixPath('003.tiff'), PosixPath('004.png'), PosixPath('005.png')]
```
```
>>>get_image_filenames_from_dir("009.tiff")
Traceback (most recent call last):
File "<string>", line 1, in <module>
File "<string>", line 18, in get_image_filenames_from_dir
ValueError:``path`` is not a directory: 009.tiff
```
anomalib.data.utils.image.get_image_height_and_width( _image_size_ )

```
Get image height and width fromimage_sizevariable.
Parameters
image_size(int | Sequence[int] | None, optional) – Input image size.
Raises
ValueError– Image size not None, int or Sequence of values.
```
```
Examples
```
```
>>>get_image_height_and_width(image_size=256)
(256, 256)
```
```
>>>get_image_height_and_width(image_size=(256, 256))
(256, 256)
```
```
>>>get_image_height_and_width(image_size=(256, 256, 3))
(256, 256)
```
```
>>>get_image_height_and_width(image_size=256.)
Traceback (most recent call last):
File "<string>", line 1, in <module>
File "<string>", line 18, in get_image_height_and_width
ValueError:``image_size`` could be either int or tuple[int, int]
```
**3.3. Reference Guide 69**


```
Returns
A tuple containing image height and width values.
Return type
tuple[int | None, int | None]
```
anomalib.data.utils.image.is_image_file( _filename_ )

```
Check if the filename is an image file.
Parameters
filename(str | Path) – Filename to check.
Returns
True if the filename is an image file.
Return type
bool
```
```
Examples
```
```
>>>is_image_file("000.png")
True
```
```
>>>is_image_file("002.JPEG")
True
```
```
>>>is_image_file("009.tiff")
True
```
```
>>>is_image_file("002.avi")
False
```
anomalib.data.utils.image.pad_nextpow2( _batch_ )

```
Compute required padding from input size and return padded images.
Finds the largest dimension and computes a square image of dimensions that are of the power of 2. In case the
image dimension is odd, it returns the image with an extra padding on one side.
Parameters
batch(torch.Tensor) – Input images
Returns
Padded batch
Return type
batch
```
anomalib.data.utils.image.read_depth_image( _path_ )

```
Read tiff depth image from disk.
Parameters
path(str, Path) – path to the image file
```
**70 Chapter 3. Guides**


```
Example
```
```
>>>image = read_depth_image("test_image.tiff")
```
```
Return type
ndarray
Returns
image as numpy array
```
anomalib.data.utils.image.read_image( _path_ , _as_tensor=False_ )

```
Read image from disk in RGB format.
Parameters
```
- path(str, Path) – path to the image file
- as_tensor(bool, optional) – If True, returns the image as a tensor. Defaults to False.

```
Example
```
```
>>>image = read_image("test_image.jpg")
>>>type(image)
<class 'numpy.ndarray'>
>>>
>>>image = read_image("test_image.jpg", as_tensor=True)
>>>type(image)
<class 'torch.Tensor'>
```
```
Return type
Tensor|ndarray
Returns
image as numpy array
```
anomalib.data.utils.image.read_mask( _path_ , _as_tensor=False_ )

```
Read mask from disk.
Parameters
```
- path(str, Path) – path to the mask file
- as_tensor(bool, optional) – If True, returns the mask as a tensor. Defaults to False.
**Return type**
Tensor|ndarray

**3.3. Reference Guide 71**


```
Example
```
```
>>>mask = read_mask("test_mask.png")
>>>type(mask)
<class 'numpy.ndarray'>
>>>
>>>mask = read_mask("test_mask.png", as_tensor=True)
>>>type(mask)
<class 'torch.Tensor'>
```
anomalib.data.utils.image.save_image( _filename_ , _image_ , _root=None_ )

```
Save an image to the file system.
Parameters
```
- filename(Path | str) – Path or filename to which the image will be saved.
- image(np.ndarray | Figure) – Image that will be saved to the file system.
- root(Path, optional) – Root directory to save the image. If provided, the top level
    directory of an absolute filename will be overwritten. Defaults to None.
**Return type**
None

anomalib.data.utils.image.show_image( _image_ , _title='Image'_ )

```
Show an image on the screen.
Parameters
```
- image(np.ndarray | Figure) – Image that will be shown in the window.
- title(str, optional) – Title that will be given to that window. Defaults to “Image”.
**Return type**
None

**Video Utils**

Video utils.

classanomalib.data.utils.video.ClipsIndexer( _video_paths_ , _mask_paths_ , _clip_length_in_frames=2_ ,
_frames_between_clips=1_ )
Bases:VideoClips,ABC
Extension of torchvision’s VideoClips class that also returns the masks for each clip.
Subclasses should implement the get_mask method. By default, the class inherits the functionality of VideoClips,
which assumes that video_paths is a list of video files. If custom behaviour is required (e.g. video_paths is a list
of folders with single-frame images), the subclass should implement at least get_clip and _compute_frame_pts.
**Parameters**

- video_paths(list[str]) – List of video paths that make up the dataset.
- mask_paths(list[str]) – List of paths to the masks for each video in the dataset.
get_item( _idx_ )
Return a dictionary containing the clip, mask, video path and frame indices.

**72 Chapter 3. Guides**


```
Return type
dict[str,Any]
abstract get_mask( idx )
Return the masks for the given index.
Return type
Tensor|None
last_frame_idx( video_idx )
Return the index of the last frame for a given video.
Return type
int
```
anomalib.data.utils.video.convert_video( _input_path_ , _output_path_ , _codec='MP4V'_ )

```
Convert video file to a different codec.
Parameters
```
- input_path(Path) – Path to the input video.
- output_path(Path) – Path to the target output video.
- codec(str) – fourcc code of the codec that will be used for compression of the output file.
**Return type**
None

**Label Utils**

Label name enum class.

classanomalib.data.utils.label.LabelName( _value_ , _names=None_ , _*_ , _module=None_ , _qualname=None_ ,
_type=None_ , _start=1_ , _boundary=None_ )
Bases:int,Enum
Name of label.

**Bounding Box Utils**

Helper functions for processing bounding box detections and annotations.

anomalib.data.utils.boxes.boxes_to_anomaly_maps( _boxes_ , _scores_ , _image_size_ )

```
Convert bounding box coordinates to anomaly heatmaps.
Parameters
```
- boxes(list[torch.Tensor]) – A list of length B where each element is a tensor of shape
    (N, 4) containing the bounding box coordinates of the regions of interest in xyxy format.
- scores(list[torch.Tensor]) – A list of length B where each element is a 1D tensor of
    length N containing the anomaly scores for each region of interest.
- image_size(tuple[int, int]) – Image size of the output masks in (H, W) format.
**Returns**

**3.3. Reference Guide 73**


```
torch.Tensor of shape (B, H, W). The pixel locations within each bounding box are
collectively
assigned the anomaly score of the bounding box. In the case of overlapping bounding boxes,
the highest score is used.
Return type
Tensor
```
anomalib.data.utils.boxes.boxes_to_masks( _boxes_ , _image_size_ )

```
Convert bounding boxes to segmentations masks.
Parameters
```
- boxes(list[torch.Tensor]) – A list of length B where each element is a tensor of shape
    (N, 4) containing the bounding box coordinates of the regions of interest in xyxy format.
- image_size(tuple[int, int]) – Image size of the output masks in (H, W) format.
**Returns
torch.Tensor of shape (B, H, W) in which each slice is a binary mask showing the pixels
contained by a**
bounding box.
**Return type**
Tensor

anomalib.data.utils.boxes.masks_to_boxes( _masks_ , _anomaly_maps=None_ )

```
Convert a batch of segmentation masks to bounding box coordinates.
Parameters
```
- masks(torch.Tensor) – Input tensor of shape (B, 1, H, W), (B, H, W) or (H, W)
- anomaly_maps(Tensor | None, optional) – Anomaly maps of shape (B, 1, H, W), (B,
    H, W) or (H, W) which are used to determine an anomaly score for the converted bounding
    boxes.
**Returns
A list of length B where each element is a tensor of shape (N, 4)**
containing the bounding box coordinates of the objects in the masks in xyxy format.
**list[torch.Tensor]: A list of length B where each element is a tensor of length (N)**
containing an anomaly score for each of the converted boxes.
**Return type**
list[torch.Tensor]

anomalib.data.utils.boxes.scale_boxes( _boxes_ , _image_size_ , _new_size_ )

```
Scale bbox coordinates to a new image size.
Parameters
```
- boxes(torch.Tensor) – Boxes of shape (N, 4) - (x1, y1, x2, y2).
- image_size(Size) – Size of the original image in which the bbox coordinates were re-
    trieved.
- new_size(Size) – New image size to which the bbox coordinates will be scaled.
**Returns**
Updated boxes of shape (N, 4) - (x1, y1, x2, y2).

**74 Chapter 3. Guides**


```
Return type
Tensor
```
**Dataset Split Utils**

Dataset Split Utils.

This module contains function in regards to splitting normal images in training set, and creating validation sets from
test sets.

**These function are useful**

- when the test set does not contain any normal images.
- when the dataset doesn’t have a validation set.

classanomalib.data.utils.split.Split( _value_ , _names=None_ , _*_ , _module=None_ , _qualname=None_ ,
_type=None_ , _start=1_ , _boundary=None_ )
Bases:str,Enum
Split of a subset.

classanomalib.data.utils.split.TestSplitMode( _value_ , _names=None_ , _*_ , _module=None_ , _qualname=None_ ,
_type=None_ , _start=1_ , _boundary=None_ )
Bases:str,Enum
Splitting mode used to obtain subset.

classanomalib.data.utils.split.ValSplitMode( _value_ , _names=None_ , _*_ , _module=None_ , _qualname=None_ ,
_type=None_ , _start=1_ , _boundary=None_ )
Bases:str,Enum
Splitting mode used to obtain validation subset.

anomalib.data.utils.split.concatenate_datasets( _datasets_ )

```
Concatenate multiple datasets into a single dataset object.
Parameters
datasets(Sequence[AnomalibDataset]) – Sequence of at least two datasets.
Returns
Dataset that contains the combined samples of all input datasets.
Return type
AnomalibDataset
```
anomalib.data.utils.split.random_split( _dataset_ , _split_ratio_ , _label_aware=False_ , _seed=None_ )

```
Perform a random split of a dataset.
Parameters
```
- dataset(AnomalibDataset) – Source dataset
- split_ratio(Union[float, Sequence[float]]) – Fractions of the splits that will be
    produced. The values in the sequence must sum to 1. If a single value is passed, the ratio
    will be converted to [1-split_ratio, split_ratio].
- label_aware(bool) – When True, the relative occurrence of the different class labels of
    the source dataset will be maintained in each of the subsets.
- seed(int | None, optional) – Seed that can be passed if results need to be reproducible

**3.3. Reference Guide 75**


```
Return type
list[AnomalibDataset]
```
anomalib.data.utils.split.split_by_label( _dataset_ )

```
Split the dataset into the normal and anomalous subsets.
Return type
tuple[AnomalibDataset,AnomalibDataset]
```
**Data Transforms**

**Tiling**

Image Tiler.

classanomalib.data.utils.tiler.ImageUpscaleMode( _value_ , _names=None_ , _*_ , _module=None_ ,
_qualname=None_ , _type=None_ , _start=1_ ,
_boundary=None_ )
Bases:str,Enum
Type of mode when upscaling image.

exceptionanomalib.data.utils.tiler.StrideSizeError

```
Bases:Exception
StrideSizeError to raise exception when stride size is greater than the tile size.
```
classanomalib.data.utils.tiler.Tiler( _tile_size_ , _stride=None_ , _remove_border_count=0_ ,
_mode=ImageUpscaleMode.PADDING_ )
Bases:object
Tile Image into (non)overlapping Patches. Images are tiled in order to efficiently process large images.
**Parameters**

- tile_size(int|Sequence) – Tile dimension for each patch
- stride(int|Sequence|None) – Stride length between patches
- remove_border_count(int) – Number of border pixels to be removed from tile before
    untiling
- mode (ImageUpscaleMode) – Upscaling mode for image resize.Supported formats:
    padding, interpolation

```
Examples
```
```
>>> import torch
>>> from torchvision import transforms
>>> from skimage.data importcamera
>>>tiler = Tiler(tile_size=256,stride=128)
>>>image = transforms.ToTensor()(camera())
>>>tiles = tiler.tile(image)
>>>image.shape, tiles.shape
(torch.Size([3, 512, 512]), torch.Size([9, 3, 256, 256]))
```
**76 Chapter 3. Guides**


```
>>># Perform your operations on the tiles.
```
```
>>># Untile the patches to reconstruct the image
>>>reconstructed_image = tiler.untile(tiles)
>>>reconstructed_image.shape
torch.Size([1, 3, 512, 512])
```
```
tile( image , use_random_tiling=False )
Tiles an input image to either overlapping, non-overlapping or random patches.
Parameters
```
- image(Tensor) – Input image to tile.
- use_random_tiling(bool) – If True, randomly crops tiles from the image. If False, tiles
    the image in a regular grid.

```
Examples
```
```
>>> from anomalib.data.utils.tiler importTiler
>>>tiler = Tiler(tile_size=512,stride=256)
>>>image = torch.rand(size=(2, 3, 1024, 1024))
>>>image.shape
torch.Size([2, 3, 1024, 1024])
>>>tiles = tiler.tile(image)
>>>tiles.shape
torch.Size([18, 3, 512, 512])
```
```
Return type
Tensor
Returns
Tiles generated from the image.
```
```
untile( tiles )
Untiles patches to reconstruct the original input image.
If patches, are overlapping patches, the function averages the overlapping pixels, and return the recon-
structed image.
Parameters
tiles(Tensor) – Tiles from the input image, generated via tile()..
```
```
Examples
```
```
>>> from anomalib.data.utils.tiler importTiler
>>>tiler = Tiler(tile_size=512,stride=256)
>>>image = torch.rand(size=(2, 3, 1024, 1024))
>>>image.shape
torch.Size([2, 3, 1024, 1024])
>>>tiles = tiler.tile(image)
>>>tiles.shape
(continues on next page)
```
**3.3. Reference Guide 77**


```
(continued from previous page)
torch.Size([18, 3, 512, 512])
>>>reconstructed_image = tiler.untile(tiles)
>>>reconstructed_image.shape
torch.Size([2, 3, 1024, 1024])
>>>torch.equal(image, reconstructed_image)
True
```
```
Return type
Tensor
Returns
Output that is the reconstructed version of the input tensor.
```
anomalib.data.utils.tiler.compute_new_image_size( _image_size_ , _tile_size_ , _stride_ )

```
Check if image size is divisible by tile size and stride.
If not divisible, it resizes the image size to make it divisible.
Parameters
```
- image_size(tuple) – Original image size
- tile_size(tuple) – Tile size
- stride(tuple) – Stride

```
Examples
```
```
>>>compute_new_image_size(image_size=(512, 512), tile_size=(256, 256), stride=(128,
˓→128))
(512, 512)
```
```
>>>compute_new_image_size(image_size=(512, 512), tile_size=(222, 222), stride=(111,
˓→111))
(555, 555)
```
```
Returns
Updated image size that is divisible by tile size and stride.
Return type
tuple
```
anomalib.data.utils.tiler.downscale_image( _image_ , _size_ , _mode=ImageUpscaleMode.PADDING_ )

```
Opposite of upscaling. This image downscales image to a desired size.
Parameters
```
- image(torch.Tensor) – Input image
- size(tuple) – Size to which image is down scaled.
- mode(str, optional) – Downscaling mode. Defaults to “padding”.

**78 Chapter 3. Guides**


```
Examples
```
```
>>>x = torch.rand(1, 3, 512, 512)
>>>y = upscale_image(image, upscale_size=(555, 555), mode="padding")
>>>y = downscale_image(y, size=(512, 512), mode='padding')
>>>torch.allclose(x, y)
True
```
```
Returns
Downscaled image
Return type
Tensor
```
anomalib.data.utils.tiler.upscale_image( _image_ , _size_ , _mode=ImageUpscaleMode.PADDING_ )

```
Upscale image to the desired size via either padding or interpolation.
Parameters
```
- image(torch.Tensor) – Image
- size(tuple) – tuple to which image is upscaled.
- mode(str, optional) – Upscaling mode. Defaults to “padding”.

```
Examples
```
```
>>>image = torch.rand(1, 3, 512, 512)
>>>image = upscale_image(image, size=(555, 555), mode="padding")
>>>image.shape
torch.Size([1, 3, 555, 555])
```
```
>>>image = torch.rand(1, 3, 512, 512)
>>>image = upscale_image(image, size=(555, 555), mode="interpolation")
>>>image.shape
torch.Size([1, 3, 555, 555])
```
```
Returns
Upscaled image.
Return type
Tensor
```
**Synthetic Data Utils**

Utilities to generate synthetic data.

anomalib.data.utils.generators.random_2d_perlin( _shape_ , _res_ , _fade=<function <lambda>>_ )

```
Returns a random 2d perlin noise array.
Parameters
```
- shape(tuple) – Shape of the 2d map.

**3.3. Reference Guide 79**


- res(tuple[int | torch.Tensor, int | torch.Tensor]) – Tuple of scales for per-
    lin noise for height and width dimension.
- fade(_type_, optional) – Function used for fading the resulting 2d map. Defaults to
    equation 6*t**5-15*t**4+10*t**3.
**Returns**
Random 2d-array/tensor generated using perlin noise.
**Return type**
np.ndarray | torch.Tensor

Augmenter module to generates out-of-distribution samples for the DRAEM implementation.

classanomalib.data.utils.augmenter.Augmenter( _anomaly_source_path=None_ , _p_anomalous=0.5_ ,
_beta=(0.2, 1.0)_ )
Bases:object
Class that generates noisy augmentations of input images.
**Parameters**

- anomaly_source_path(str | None) – Path to a folder of images that will be used as
    source of the anomalous
- specified(noise. If not) –
- instead.(random noise will be used) –
- p_anomalous(float) – Probability that the anomalous perturbation will be applied to a
    given image.
- beta(float) – Parameter that determines the opacity of the noise mask.
augment_batch( _batch_ )
Generate anomalous augmentations for a batch of input images.
**Parameters**
batch(torch.Tensor) – Batch of input images
**Return type**
tuple[Tensor,Tensor]
**Returns**
- Augmented image to which anomalous perturbations have been added.
- Ground truth masks corresponding to the anomalous perturbations.
generate_perturbation( _height_ , _width_ , _anomaly_source_path=None_ )
Generate an image containing a random anomalous perturbation using a source image.
**Parameters**
- height(int) – height of the generated image.
- width(int) – (int): width of the generated image.
- anomaly_source_path(Path | str | None) – Path to an image file. If not provided,
random noise will be used
- instead.–
**Return type**
tuple[ndarray,ndarray]

**80 Chapter 3. Guides**


```
Returns
Image containing a random anomalous perturbation, and the corresponding ground truth
anomaly mask.
rand_augmenter()
Select 3 random transforms that will be applied to the anomaly source images.
Return type
Sequential
Returns
A selection of 3 transforms.
```
anomalib.data.utils.augmenter.nextpow2( _value_ )

```
Return the smallest power of 2 greater than or equal to the input value.
Return type
int
```
Dataset that generates synthetic anomalies.

This dataset can be used when there is a lack of real anomalous data.

classanomalib.data.utils.synthetic.SyntheticAnomalyDataset( _task_ , _transform_ , _source_samples_ )

```
Bases:AnomalibDataset
Dataset which reads synthetically generated anomalous images from a temporary folder.
Parameters
```
- task(str) – Task type, either “classification” or “segmentation”.
- transform(A.Compose) – Transform object describing the transforms that are applied to
    the inputs.
- source_samples(DataFrame) – Normal samples to which the anomalous augmentations
    will be applied.
classmethod from_dataset( _dataset_ )
Create a synthetic anomaly dataset from an existing dataset of normal images.
**Parameters**
dataset(AnomalibDataset) – Dataset consisting of only normal images that will be con-
verrted to a synthetic anomalous dataset with a 50/50 normal anomalous split.
**Return type**
SyntheticAnomalyDataset

anomalib.data.utils.synthetic.make_synthetic_dataset( _source_samples_ , _image_dir_ , _mask_dir_ ,
_anomalous_ratio=0.5_ )
Convert a set of normal samples into a mixed set of normal and synthetic anomalous samples.
The synthetic images will be saved to the file system in the specified root directory under <root>/images. For the
synthetic anomalous images, the masks will be saved under <root>/ground_truth.
**Parameters**

- source_samples(DataFrame) – Normal images that will be used as source for the syn-
    thetic anomalous images.
- image_dir(Path) – Directory to which the synthetic anomalous image files will be written.
- mask_dir(Path) – Directory to which the ground truth anomaly masks will be written.

**3.3. Reference Guide 81**


- anomalous_ratio(float) – Fraction of source samples that will be converted into anoma-
    lous samples.
**Return type**
DataFrame

**3.3.2 Models**

Model Components Learn more about components to design your own anomaly detection models.

```
Image Models Learn more about image anomaly detection models.
Video Models Learn more about video anomaly detection models.
```
**Model Components**

Feature Extractors Learn more about anomalib feature extractors to extract features from backbones.

```
Dimensionality Reduction Learn more about dimensionality reduction models.
Normalizing Flows Learn more aboutfreianormalizing flows model components.
Sampling Components Learn more about various sampling components.
Filters Learn more about filters for post-processing.
Classification Learn more about classification model components.
Cluster Learn more about cluster model components.
Statistical Components Learn more about classification model components.
```
**Feature Extractors**

Feature extractors.

classanomalib.models.components.feature_extractors.BackboneParams( _class_path_ ,
_init_args=<factory>_ )
Bases:object
Used for serializing the backbone.

classanomalib.models.components.feature_extractors.TimmFeatureExtractor( _backbone_ , _layers_ ,
_pre_trained=True_ ,
_re-
quires_grad=False_ )
Bases:Module
Extract features from a CNN.
**Parameters**

- backbone(nn.Module) – The backbone to which the feature extraction hooks are attached.
- layers(Iterable[str]) – List of layer names of the backbone to which the hooks are
    attached.

**82 Chapter 3. Guides**


- pre_trained(bool) – Whether to use a pre-trained backbone. Defaults to True.
- requires_grad(bool) – Whether to require gradients for the backbone. Defaults to False.
    Models likestfpmuse the feature extractor model as a trainable network. In such cases
    gradient computation is required.

```
Example
```
```
import torch
from anomalib.models.components.feature_extractors importTimmFeatureExtractor
```
```
model = TimmFeatureExtractor(model="resnet18", layers=['layer1', 'layer2','layer3
˓→'])
input = torch.rand((32, 3, 256, 256))
features = model(input)
```
```
print([layerforlayerin features.keys()])
# Output: ['layer1', 'layer2','layer3']
```
```
print([feature.shapeforfeatureinfeatures.values()]()
# Output: [torch.Size([32, 64, 64, 64]), torch.Size([32, 128, 32, 32]), torch.
˓→Size([32, 256, 16, 16])]
```
```
forward( inputs )
Forward-pass input tensor into the CNN.
Parameters
inputs(torch.Tensor) – Input tensor
Return type
dict[str,Tensor]
Returns
Feature map extracted from the CNN
```
```
Example
```
```
model = TimmFeatureExtractor(model="resnet50", layers=['layer3'])
input = torch.rand((32, 3, 256, 256))
features = model.forward(input)
```
classanomalib.models.components.feature_extractors.TorchFXFeatureExtractor( _backbone_ ,
_return_nodes_ ,
_weights=None_ ,
_re-
quires_grad=False_ ,
_tracer_kwargs=None_ )
Bases:Module
Extract features from a CNN.
**Parameters**

- backbone(str |BackboneParams| dict | nn.Module) – The backbone to which
    the feature extraction hooks are attached. If the name is provided, the model is loaded from

**3.3. Reference Guide 83**


```
torchvision. Otherwise, the model class can be provided and it will try to load the weights
from the provided weights file. Last, an instance of nn.Module can also be passed directly.
```
- return_nodes(Iterable[str]) – List of layer names of the backbone to which the hooks
    are attached. You can find the names of these nodes by usingget_graph_node_names
    function.
- weights(str | WeightsEnum | None) – Weights enum to use for the model. Torchvi-
    sion models requireWeightsEnum. These enums are defined intorchvision.models.
    <model>. You can pass the weights path for custom models.
- requires_grad(bool) – Models likestfpmuse the feature extractor for training. In such
    cases we should setrequires_gradtoTrue. Default isFalse.
- tracer_kwargs(dict | None) – a dictionary of keyword arguments for NodePathTracer
    (which passes them onto it’s parent class torch.fx.Tracer). Can be used to allow not trac-
    ing through a list of problematic modules, by passing a list of _leaf_modules_ as one of the
    _tracer_kwargs_.

```
Example
```
```
With torchvision models:
```
```
import torch
from anomalib.models.components.feature_extractors importTorchFXFeatureExtractor
from torchvision.models.efficientnet importEfficientNet_B5_Weights
```
```
feature_extractor = TorchFXFeatureExtractor(
backbone="efficientnet_b5",
return_nodes=["features.6.8"],
weights=EfficientNet_B5_Weights.DEFAULT
)
```
```
input = torch.rand((32, 3, 256, 256))
features = feature_extractor(input)
```
```
print([layerforlayerin features.keys()])
# Output: ["features.6.8"]
```
```
print([feature.shapeforfeatureinfeatures.values()])
# Output: [torch.Size([32, 304, 8, 8])]
```
```
With custom models:
```
```
import torch
from anomalib.models.components.feature_extractors importTorchFXFeatureExtractor
```
```
feature_extractor = TorchFXFeatureExtractor(
"path.to.CustomModel", ["linear_relu_stack.3"], weights="path/to/weights.pth"
)
```
```
input = torch.randn(1, 1, 28, 28)
features = feature_extractor(input)
```
```
(continues on next page)
```
**84 Chapter 3. Guides**


```
(continued from previous page)
print([layerforlayerin features.keys()])
# Output: ["linear_relu_stack.3"]
```
```
with model instances:
```
```
import torch
from anomalib.models.components.feature_extractors importTorchFXFeatureExtractor
from timm importcreate_model
```
```
model = create_model("resnet18", pretrained=True)
feature_extractor = TorchFXFeatureExtractor(model, ["layer1"])
```
```
input = torch.rand((32, 3, 256, 256))
features = feature_extractor(input)
```
```
print([layerforlayerin features.keys()])
# Output: ["layer1"]
```
```
print([feature.shapeforfeatureinfeatures.values()])
# Output: [torch.Size([32, 64, 64, 64])]
```
```
forward( inputs )
Extract features from the input.
Return type
dict[str,Tensor]
initialize_feature_extractor( backbone , return_nodes , weights=None , requires_grad=False ,
tracer_kwargs=None )
Extract features from a CNN.
Parameters
```
- backbone(BackboneParams| nn.Module) – The backbone to which the feature ex-
    traction hooks are attached. If the name is provided for BackboneParams, the model is
    loaded from torchvision. Otherwise, the model class can be provided and it will try to load
    the weights from the provided weights file. Last, an instance of the model can be provided
    as well, which will be used as-is.
- return_nodes(Iterable[str]) – List of layer names of the backbone to which
    the hooks are attached. You can find the names of these nodes by using
    get_graph_node_namesfunction.
- weights(str | WeightsEnum | None) – Weights enum to use for the model. Torchvi-
    sion models requireWeightsEnum. These enums are defined intorchvision.models.
    <model>. You can pass the weights path for custom models.
- requires_grad(bool) – Models likestfpmuse the feature extractor for training. In such
    cases we should setrequires_gradtoTrue. Default isFalse.
- tracer_kwargs(dict | None) – a dictionary of keyword arguments for NodePathTracer
    (which passes them onto it’s parent class torch.fx.Tracer). Can be used to allow not trac-
    ing through a list of problematic modules, by passing a list of _leaf_modules_ as one of the
    _tracer_kwargs_.
**Return type**
GraphModule

**3.3. Reference Guide 85**


```
Returns
Feature Extractor based on TorchFX.
```
anomalib.models.components.feature_extractors.dryrun_find_featuremap_dims( _feature_extractor_ ,
_input_size_ , _layers_ )
Dry run an empty image of _input_size_ size to get the featuremap tensors’ dimensions (num_features, resolution).
**Returns
maping of** **_layer -> dimensions dict_**
Each _dimension dict_ has two keys: _num_features_ (int) and **`** resolution`(tuple[int, int]).
**Return type**
tuple[int, int]

**Dimensionality Reduction**

Algorithms for decomposition and dimensionality reduction.

classanomalib.models.components.dimensionality_reduction.PCA( _n_components_ )

```
Bases:DynamicBufferMixin
Principle Component Analysis (PCA).
Parameters
n_components(float) – Number of components. Can be either integer number of components
or a ratio between 0-1.
```
```
Example
```
```
>>> import torch
>>> from anomalib.models.components importPCA
```
```
Create a PCA model with 2 components:
```
```
>>>pca = PCA(n_components=2)
```
```
Create a random embedding and fit a PCA model.
```
```
>>>embedding = torch.rand(1000, 5).cuda()
>>>pca = PCA(n_components=2)
>>>pca.fit(embedding)
```
```
Apply transformation:
```
```
>>>transformed = pca.transform(embedding)
>>>transformed.shape
torch.Size([1000, 2])
```
```
fit( dataset )
Fits the PCA model to the dataset.
Parameters
dataset(torch.Tensor) – Input dataset to fit the model.
Return type
None
```
**86 Chapter 3. Guides**


```
Example
```
```
>>>pca.fit(embedding)
>>>pca.singular_vectors
tensor([9.6053, 9.2763], device='cuda:0')
```
```
>>>pca.mean
tensor([0.4859, 0.4959, 0.4906, 0.5010, 0.5042], device='cuda:0')
```
```
fit_transform( dataset )
Fit and transform PCA to dataset.
Parameters
dataset(torch.Tensor) – Dataset to which the PCA if fit and transformed
Return type
Tensor
Returns
Transformed dataset
```
```
Example
```
```
>>>pca.fit_transform(embedding)
>>>transformed_embedding = pca.fit_transform(embedding)
>>>transformed_embedding.shape
torch.Size([1000, 2])
```
```
forward( features )
Transform the features.
Parameters
features(torch.Tensor) – Input features
Return type
Tensor
Returns
Transformed features
```
```
Example
```
```
>>>pca(embedding).shape
torch.Size([1000, 2])
```
```
inverse_transform( features )
Inverses the transformed features.
Parameters
features(torch.Tensor) – Transformed features
Return type
Tensor
Returns
Inverse features
```
**3.3. Reference Guide 87**


```
Example
```
```
>>>inverse_embedding = pca.inverse_transform(transformed_embedding)
>>>inverse_embedding.shape
torch.Size([1000, 5])
```
```
transform( features )
Transform the features based on singular vectors calculated earlier.
Parameters
features(torch.Tensor) – Input features
Return type
Tensor
Returns
Transformed features
```
```
Example
```
```
>>>pca.transform(embedding)
>>>transformed_embedding = pca.transform(embedding)
```
```
>>>embedding.shape
torch.Size([1000, 5])
#
>>>transformed_embedding.shape
torch.Size([1000, 2])
```
classanomalib.models.components.dimensionality_reduction.SparseRandomProjection( _eps=0.1_ ,
_ran-
dom_state=None_ )
Bases:object
Sparse Random Projection using PyTorch operations.
**Parameters**

- eps(float, optional) – Minimum distortion rate parameter for calculating Johnson-
    Lindenstrauss minimum dimensions. Defaults to0.1.
- random_state(int | None, optional) – Uses the seed to set the random state for sam-
    ple_without_replacement function. Defaults toNone.

```
Example
```
```
To fit and transform the embedding tensor, use the following code:
```
```
import torch
from anomalib.models.components importSparseRandomProjection
```
```
sparse_embedding = torch.rand(1000, 5).cuda()
model = SparseRandomProjection(eps=0.1)
```
```
Fit the model and transform the embedding tensor:
```
**88 Chapter 3. Guides**


```
model.fit(sparse_embedding)
projected_embedding = model.transform(sparse_embedding)
```
```
print(projected_embedding.shape)
# Output: torch.Size([1000, 5920])
```
```
fit( embedding )
Generate sparse matrix from the embedding tensor.
Parameters
embedding(torch.Tensor) – embedding tensor for generating embedding
Returns
Return self to be used as
```
```
>>>model = SparseRandomProjection()
>>>model = model.fit()
```
```
Return type
( SparseRandomProjection )
transform( embedding )
Project the data by using matrix product with the random matrix.
Parameters
embedding(torch.Tensor) – Embedding of shape (n_samples, n_features) The input data
to project into a smaller dimensional space
Returns
Sparse matrix of shape
(n_samples, n_components) Projected array.
Return type
projected_embedding (torch.Tensor)
```
```
Example
```
```
>>>projected_embedding = model.transform(embedding)
>>>projected_embedding.shape
torch.Size([1000, 5920])
```
**Normalizing Flows**

All In One Block Layer.

classanomalib.models.components.flow.AllInOneBlock( _dims_in_ , _dims_c=None_ ,
_subnet_constructor=None_ , _affine_clamping=2.0_ ,
_gin_block=False_ , _global_affine_init=1.0_ ,
_global_affine_type='SOFTPLUS'_ ,
_permute_soft=False_ ,
_learned_householder_permutation=0_ ,
_reverse_permutation=False_ )

**3.3. Reference Guide 89**


```
Bases:InvertibleModule
Module combining the most common operations in a normalizing flow or similar model.
It combines affine coupling, permutation, and global affine transformation (‘ActNorm’). It can also be used as
GIN coupling block, perform learned householder permutations, and use an inverted pre-permutation. The affine
transformation includes a soft clamping mechanism, first used in Real-NVP. The block as a whole performs the
following computation:
```
```
𝑦=𝑉 𝑅Ψ(𝑠global)⊙Coupling
```
### (︁

### 𝑅−^1 𝑉−^1 𝑥

### )︁

```
+𝑡global
```
- The inverse pre-permutation of x (i.e.𝑅−^1 𝑉−^1 ) is optional (seereverse_permutationbelow).
- The learned householder reflection matrix 𝑉 is also optional all together (see
    learned_householder_permutationbelow).
- For the coupling, the input is split into𝑥 1 ,𝑥 2 along the channel dimension. Then the output of the coupling
    operation is the two halves𝑢= concat(𝑢 1 ,𝑢 2 ).

```
𝑢 1 =𝑥 1 ⊙exp
```
### (︁

```
𝛼tanh
```
### (︀

### 𝑠(𝑥 2 )

### )︀)︁

### +𝑡(𝑥 2 )

### 𝑢 2 =𝑥 2

```
Becausetanh(𝑠)∈[− 1 ,1], this clamping mechanism prevents exploding values in the exponential. The
hyperparameter𝛼can be adjusted.
```
```
Parameters
```
- subnet_constructor(Callable|None) – class or callablef, called asf(channels_in,
    channels_out)and should return a torch.nn.Module. Predicts coupling coefficients𝑠,𝑡.
- affine_clamping(float) – clamp the output of the multiplicative coefficients before ex-
    ponentiation to +/-affine_clamping(see𝛼above).
- gin_block(bool) – Turn the block into a GIN block from Sorrenson et al, 2019. Makes it
    so that the coupling operations as a whole is volume preserving.
- global_affine_init(float) – Initial value for the global affine scaling𝑠global.
- global_affine_init–'SIGMOID','SOFTPLUS', or'EXP'. Defines the activation to be
    used on the beta for the global affine scaling (Ψabove).
- permute_soft(bool) – bool, whether to sample the permutation matrix𝑅from𝑆𝑂(𝑁),
    or to use hard permutations instead. Note,permute_soft=Trueis very slow when working
    with >512 dimensions.
- learned_householder_permutation(int) – Int, if >0, turn on the matrix𝑉above, that
    represents multiple learned householder reflections. Slow if large number. Dubious whether
    it actually helps network performance.
- reverse_permutation(bool) – Reverse the permutation before the block, as introduced
    by Putzky et al, 2019. Turns on the𝑅−^1 𝑉−^1 pre-multiplication above.

```
forward( x , c=None , rev=False , jac=True )
See base class docstring.
Return type
tuple[tuple[Tensor],Tensor]
```
**90 Chapter 3. Guides**


```
output_dims( input_dims )
Output dimensions of the layer.
Parameters
input_dims(list[tuple[int]]) – Input dimensions.
Returns
Output dimensions.
Return type
list[tuple[int]]
```
**Sampling Components**

Sampling methods.

classanomalib.models.components.sampling.KCenterGreedy( _embedding_ , _sampling_ratio_ )

```
Bases:object
Implements k-center-greedy method.
Parameters
```
- embedding(torch.Tensor) – Embedding vector extracted from a CNN
- sampling_ratio(float) – Ratio to choose coreset size from the embedding size.

```
Example
```
```
>>>embedding.shape
torch.Size([219520, 1536])
>>>sampler = KCenterGreedy(embedding=embedding)
>>>sampled_idxs = sampler.select_coreset_idxs()
>>>coreset = embedding[sampled_idxs]
>>>coreset.shape
torch.Size([219, 1536])
```
```
get_new_idx()
Get index value of a sample.
Based on minimum distance of the cluster
Returns
```
## Index

```
Return type
int
reset_distances()
Reset minimum distances.
Return type
None
sample_coreset( selected_idxs=None )
Select coreset from the embedding.
```
**3.3. Reference Guide 91**


```
Parameters
selected_idxs(list[int] |None) – index of samples already selected. Defaults to an
empty set.
Returns
Output coreset
Return type
Tensor
```
```
Example
```
```
>>>embedding.shape
torch.Size([219520, 1536])
>>>sampler = KCenterGreedy(...)
>>>coreset = sampler.sample_coreset()
>>>coreset.shape
torch.Size([219, 1536])
```
```
select_coreset_idxs( selected_idxs=None )
Greedily form a coreset to minimize the maximum distance of a cluster.
Parameters
selected_idxs(list[int] |None) – index of samples already selected. Defaults to an
empty set.
Return type
list[int]
Returns
indices of samples selected to minimize distance to cluster centers
update_distances( cluster_centers )
Update min distances given cluster centers.
Parameters
cluster_centers(list[int]) – indices of cluster centers
Return type
None
```
**Filters**

Implements filters used by models.

classanomalib.models.components.filters.GaussianBlur2d( _sigma_ , _channels=1_ , _kernel_size=None_ ,
_normalize=True_ , _border_type='reflect'_ ,
_padding='same'_ )
Bases:Module
Compute GaussianBlur in 2d.
Makes use of kornia functions, but most notably the kernel is not computed during the forward pass, and does
not depend on the input size. As a caveat, the number of channels that are expected have to be provided during
initialization.

**92 Chapter 3. Guides**


```
forward( input_tensor )
Blur the input with the computed Gaussian.
Parameters
input_tensor(torch.Tensor) – Input tensor to be blurred.
Returns
Blurred output tensor.
Return type
Tensor
```
**Classification**

Classification modules.

classanomalib.models.components.classification.FeatureScalingMethod( _value_ , _names=None_ , _*_ ,
_module=None_ ,
_qualname=None_ ,
_type=None_ , _start=1_ ,
_boundary=None_ )
Bases:str,Enum
Determines how the feature embeddings are scaled.

classanomalib.models.components.classification.KDEClassifier( _n_pca_components=16_ , _fea-
ture_scaling_method=FeatureScalingMethod.SCALE_ ,
_max_training_points=40000_ )
Bases:Module
Classification module for KDE-based anomaly detection.
**Parameters**

- n_pca_components(int, optional) – Number of PCA components. Defaults to 16.
- feature_scaling_method(FeatureScalingMethod, optional) – Scaling method
    applied to features before passing to KDE. Options are _norm_ (normalize to unit vector length)
    and _scale_ (scale to max length observed in training).
- max_training_points(int, optional) – Maximum number of training points to fit the
    KDE model. Defaults to 40000.
compute_kde_scores( _features_ , _as_log_likelihood=False_ )
Compute the KDE scores.
**The scores calculated from the KDE model are converted to densities. If** **_as_log_likelihood_** **is set to
true then**
the log of the scores are calculated.

```
Parameters
```
- features(torch.Tensor) – Features to which the PCA model is fit.
- as_log_likelihood(bool | None, optional) – If true, gets log likelihood scores.
    Defaults to False.
**Returns**
Score

**3.3. Reference Guide 93**


```
Return type
(torch.Tensor)
```
```
static compute_probabilities( scores )
Convert density scores to anomaly probabilities (see https://www.desmos.com/calculator/ifju7eesg7).
Parameters
scores(torch.Tensor) – density of an image.
Return type
Tensor
Returns
probability that image with {density} is anomalous
fit( embeddings )
Fit a kde model to embeddings.
Parameters
embeddings(torch.Tensor) – Input embeddings to fit the model.
Return type
bool
Returns
Boolean confirming whether the training is successful.
forward( features )
Make predictions on extracted features.
Return type
Tensor
pre_process( feature_stack , max_length=None )
Pre-process the CNN features.
Parameters
```
- feature_stack(torch.Tensor) – Features extracted from CNN
- max_length(Tensor | None) – Used to unit normalize the feature_stack vector. If
    max_lenis not provided, the length is calculated from thefeature_stack. Defaults
    to None.
**Returns**
Stacked features and length
**Return type**
(Tuple)
predict( _features_ )
Predicts the probability that the features belong to the anomalous class.
**Parameters**
features(torch.Tensor) – Feature from which the output probabilities are detected.
**Return type**
Tensor
**Returns**
Detection probabilities

**94 Chapter 3. Guides**


**Cluster**

Clustering algorithm implementations using PyTorch.

classanomalib.models.components.cluster.GaussianMixture( _n_components_ , _n_iter=100_ , _tol=0.001_ )

```
Bases:DynamicBufferMixin
Gaussian Mixture Model.
Parameters
```
- n_components(int) – Number of components.
- n_iter(int) – Maximum number of iterations to perform. Defaults to 100.
- tol(float) – Convergence threshold. Defaults to1e-3.

```
Example
```
```
The following examples shows how to fit a Gaussian Mixture Model to some data and get the cluster means and
predicted labels and log-likelihood scores of the data.
```
```
>>> import torch
>>> from anomalib.models.components.cluster importGaussianMixture
>>>model = GaussianMixture(n_components=2)
>>>data = torch.tensor(
... [
... [2, 1], [2, 2], [2, 3],
... [7, 5], [8, 5], [9, 5],
... ]
...).float()
>>>model.fit(data)
>>>model.means # get the means of the gaussians
tensor([[8., 5.],
[2., 2.]])
>>>model.predict(data) # get the predicted cluster label of each sample
tensor([1, 1, 1, 0, 0, 0])
>>>model.score_samples(data) # get the log-likelihood score of each sample
tensor([3.8295, 4.5795, 3.8295, 3.8295, 4.5795, 3.8295])
```
```
fit( data )
Fit the model to the data.
Parameters
data(Tensor) – Data to fit the model to. Tensor of shape (n_samples, n_features).
Return type
None
predict( data )
Predict the cluster labels of the data.
Parameters
data(Tensor) – Samples to assign to clusters. Tensor of shape (n_samples, n_features).
Returns
Tensor of shape (n_samples,) containing the predicted cluster label of each sample.
```
**3.3. Reference Guide 95**


```
Return type
Tensor
score_samples( data )
Assign a likelihood score to each sample in the data.
Parameters
data(Tensor) – Samples to assign scores to. Tensor of shape (n_samples, n_features).
Returns
Tensor of shape (n_samples,) containing the log-likelihood score of each sample.
Return type
Tensor
```
classanomalib.models.components.cluster.KMeans( _n_clusters_ , _max_iter=10_ )

```
Bases:object
Initialize the KMeans object.
Parameters
```
- n_clusters(int) – The number of clusters to create.
- max_iter(int, optional)) – The maximum number of iterations to run the algorithm.
    Defaults to 10.
fit( _inputs_ )
Fit the K-means algorithm to the input data.
**Parameters**
inputs(torch.Tensor) – Input data of shape (batch_size, n_features).
**Returns**
A tuple containing the labels of the input data with respect to the identified clusters and the
cluster centers themselves. The labels have a shape of (batch_size,) and the cluster centers
have a shape of (n_clusters, n_features).
**Return type**
tuple
**Raises**
ValueError– If the number of clusters is less than or equal to 0.
predict( _inputs_ )
Predict the labels of input data based on the fitted model.
**Parameters**
inputs(torch.Tensor) – Input data of shape (batch_size, n_features).
**Returns**
The predicted labels of the input data with respect to the identified clusters.
**Return type**
torch.Tensor
**Raises**
AttributeError– If the KMeans object has not been fitted to input data.

**96 Chapter 3. Guides**


**Stats Components**

Statistical functions.

classanomalib.models.components.stats.GaussianKDE( _dataset=None_ )

```
Bases:DynamicBufferMixin
Gaussian Kernel Density Estimation.
Parameters
dataset(Tensor | None, optional) – Dataset on which to fit the KDE model. Defaults to
None.
static cov( tensor )
Calculate the unbiased covariance matrix.
Parameters
tensor(torch.Tensor) – Input tensor from which covariance matrix is computed.
Return type
Tensor
Returns
Output covariance matrix.
fit( dataset )
Fit a KDE model to the input dataset.
Parameters
dataset(torch.Tensor) – Input dataset.
Return type
None
Returns
None
forward( features )
Get the KDE estimates from the feature map.
Parameters
features(torch.Tensor) – Feature map extracted from the CNN
Return type
Tensor
Returns: KDE Estimates
```
classanomalib.models.components.stats.MultiVariateGaussian

```
Bases:DynamicBufferMixin,Module
Multi Variate Gaussian Distribution.
fit( embedding )
Fit multi-variate gaussian distribution to the input embedding.
Parameters
embedding(torch.Tensor) – Embedding vector extracted from CNN.
Return type
list[Tensor]
```
**3.3. Reference Guide 97**


```
Returns
Mean and the covariance of the embedding.
forward( embedding )
Calculate multivariate Gaussian distribution.
Parameters
embedding(torch.Tensor) – CNN features whose dimensionality is reduced via either
random sampling or PCA.
Return type
list[Tensor]
Returns
mean and inverse covariance of the multi-variate gaussian distribution that fits the features.
```
**Image Models**

CFA Coupled-hypersphere-based Feature Adaptation for Target-Oriented Anomaly Localization

```
C-Flow Real-Time Unsupervised Anomaly Detection via Conditional Normalizing Flows
CS-Flow Fully Convolutional Cross-Scale-Flows for Image-based Defect Detection
DFKDE Deep Feature Kernel Density Estimation
DFM Probabilistic Modeling of Deep Features for Out-of-Distribution and Adversarial Detection
DRAEM DRÆM – A discriminatively trained reconstruction embedding for surface anomaly detection
DSR DSR – A Dual Subspace Re-Projection Network for Surface Anomaly Detection
Efficient AD EfficientAD: Accurate Visual Anomaly Detection at Millisecond-Level Latencies
FastFlow FastFlow: Unsupervised Anomaly Detection and Localization via 2D Normalizing Flows
GANomaly GANomaly: Semi-Supervised Anomaly Detection via Adversarial Training
PaDiM PaDiM: A Patch Distribution Modeling Framework for Anomaly Detection and Localization
Patchcore Towards Total Recall in Industrial Anomaly Detection
Reverse Distillation Anomaly Detection via Reverse Distillation from One-Class Embedding.
R-KDE Region-Based Kernel Density Estimation (RKDE)
STFPM Student-Teacher Feature Pyramid Matching for Unsupervised Anomaly Detection
U-Flow U-Flow: A U-shaped Normalizing Flow for Anomaly Detection with Unsupervised Threshold
```
WinCLIP WinCLIP: Zero-/Few-Shot Anomaly Classification and Segmentation

**98 Chapter 3. Guides**


### CFA

Lightning Implementatation of the CFA Model.

CFA: Coupled-hypersphere-based Feature Adaptation for Target-Oriented Anomaly Localization

Paper https://arxiv.org/abs/2206.04325

classanomalib.models.image.cfa.lightning_model.Cfa( _backbone='wide_resnet50_2'_ , _gamma_c=1_ ,
_gamma_d=1_ , _num_nearest_neighbors=3_ ,
_num_hard_negative_features=3_ , _radius=1e-05_ )
Bases:AnomalyModule
CFA: Coupled-hypersphere-based Feature Adaptation for Target-Oriented Anomaly Localization.
**Parameters**

- backbone(str) – Backbone CNN network Defaults to"wide_resnet50_2".
- gamma_c(int, optional) – gamma_c value from the paper. Defaults to 1.
- gamma_d(int, optional) – gamma_d value from the paper. Defaults to 1.
- num_nearest_neighbors(int) – Number of nearest neighbors. Defaults to 3.
- num_hard_negative_features(int) – Number of hard negative features. Defaults to 3.
- radius(float) – Radius of the hypersphere to search the soft boundary. Defaults to1e-5.
backward( _loss_ , _*args_ , _**kwargs_ )
Perform backward-pass for the CFA model.
**Parameters**
- loss(torch.Tensor) – Loss value.
- *args– Arguments.
- **kwargs– Keyword arguments.
**Return type**
None
configure_optimizers()
Configure optimizers for the CFA Model.
**Returns**
Adam optimizer for each decoder
**Return type**
Optimizer
property learning_type: LearningType
Return the learning type of the model.
**Returns**
Learning type of the model.
**Return type**
LearningType
on_train_start()
Initialize the centroid for the memory bank computation.

**3.3. Reference Guide 99**


```
Return type
None
property trainer_arguments: dict[str, Any]
CFA specific trainer arguments.
training_step( batch , *args , **kwargs )
Perform the training step for the CFA model.
Parameters
```
- batch(dict[str, str | torch.Tensor]) – Batch input.
- *args– Arguments.
- **kwargs– Keyword arguments.
**Returns**
Loss value.
**Return type**
STEP_OUTPUT
validation_step( _batch_ , _*args_ , _**kwargs_ )
Perform the validation step for the CFA model.
**Parameters**
- batch(dict[str, str | torch.Tensor]) – Input batch.
- *args– Arguments.
- **kwargs– Keyword arguments.
**Returns**
Anomaly map computed by the model.
**Return type**
dict

Torch Implementatation of the CFA Model.

CFA: Coupled-hypersphere-based Feature Adaptation for Target-Oriented Anomaly Localization

Paper https://arxiv.org/abs/2206.04325

classanomalib.models.image.cfa.torch_model.CfaModel( _backbone_ , _gamma_c_ , _gamma_d_ ,
_num_nearest_neighbors_ ,
_num_hard_negative_features_ , _radius_ )
Bases:DynamicBufferMixin
Torch implementation of the CFA Model.
**Parameters**

- backbone(str) – Backbone CNN network.
- gamma_c(int) – gamma_c parameter from the paper.
- gamma_d(int) – gamma_d parameter from the paper.
- num_nearest_neighbors(int) – Number of nearest neighbors.
- num_hard_negative_features(int) – Number of hard negative features.
- radius(float) – Radius of the hypersphere to search the soft boundary.

**100 Chapter 3. Guides**


```
compute_distance( target_oriented_features )
Compute distance using target oriented features.
Parameters
target_oriented_features(torch.Tensor) – Target oriented features computed using
the descriptor.
Returns
Distance tensor.
Return type
Tensor
forward( input_tensor )
Forward pass.
Parameters
input_tensor(torch.Tensor) – Input tensor.
Raises
ValueError– When the memory bank is not initialized.
Returns
Loss or anomaly map depending on the train/eval mode.
Return type
Tensor
get_scale( input_size )
Get the scale of the feature map.
Parameters
input_size(tuple[int, int]) – Input size of the image tensor.
Return type
Size
initialize_centroid( data_loader )
Initialize the Centroid of the Memory Bank.
Parameters
data_loader(DataLoader) – Train Dataloader.
Returns
Memory Bank.
Return type
Tensor
```
Loss function for the Cfa Model Implementation.

classanomalib.models.image.cfa.loss.CfaLoss( _num_nearest_neighbors_ , _num_hard_negative_features_ ,
_radius_ )
Bases:Module
Cfa Loss.
**Parameters**

- num_nearest_neighbors(int) – Number of nearest neighbors.
- num_hard_negative_features(int) – Number of hard negative features.
- radius(float) – Radius of the hypersphere to search the soft boundary.

**3.3. Reference Guide 101**


```
forward( distance )
Compute the CFA loss.
Parameters
distance(torch.Tensor) – Distance computed using target oriented features.
Returns
CFA loss.
Return type
Tensor
```
Anomaly Map Generator for the CFA model implementation.

classanomalib.models.image.cfa.anomaly_map.AnomalyMapGenerator( _num_nearest_neighbors_ ,
_sigma=4_ )
Bases:Module
Generate Anomaly Heatmap.
compute_anomaly_map( _score_ , _image_size=None_ )
Compute anomaly map based on the score.
**Parameters**

- score(torch.Tensor) – Score tensor.
- image_size(tuple[int, int] | torch.Size | None, optional) – Size of the
    input image.
**Returns**
Anomaly map.
**Return type**
Tensor
compute_score( _distance_ , _scale_ )
Compute score based on the distance.
**Parameters**
- distance(torch.Tensor) – Distance tensor computed using target oriented features.
- scale(tuple[int, int]) – Height and width of the largest feature map.
**Returns**
Score value.
**Return type**
Tensor
forward( _**kwargs_ )
Return anomaly map.
**Raises**
distance` and scale keys are not found–
**Returns**
Anomaly heatmap.
**Return type**
Tensor

**102 Chapter 3. Guides**


**C-Flow**

Cflow.

Real-Time Unsupervised Anomaly Detection via Conditional Normalizing Flows.

For more details, see the paper: Real-Time Unsupervised Anomaly Detection via Conditional Normalizing Flows.

classanomalib.models.image.cflow.lightning_model.Cflow( _backbone='wide_resnet50_2'_ ,
_layers=('layer2','layer3','layer4')_ ,
_pre_trained=True_ , _fiber_batch_size=64_ ,
_decoder='freia-cflow'_ ,
_condition_vector=128_ , _coupling_blocks=8_ ,
_clamp_alpha=1.9_ , _permute_soft=False_ ,
_lr=0.0001_ )
Bases:AnomalyModule
PL Lightning Module for the CFLOW algorithm.
**Parameters**

- backbone (str, optional) – Backbone CNN architecture. Defaults to
    "wide_resnet50_2".
- layers(Sequence[str], optional) – Layers to extract features from. Defaults to(
    "layer2", "layer3", "layer4").
- pre_trained(bool, optional) – Whether to use pre-trained weights. Defaults toTrue.
- fiber_batch_size(int, optional) – Fiber batch size. Defaults to 64.
- decoder(str, optional) – Decoder architecture. Defaults to"freia-cflow".
- condition_vector(int, optional) – Condition vector size. Defaults to 128.
- coupling_blocks(int, optional) – Number of coupling blocks. Defaults to 8.
- clamp_alpha(float, optional) – Clamping value for the alpha parameter. Defaults to
    1.9.
- permute_soft(bool, optional) – Whether to use soft permutation. Defaults toFalse.
- lr(float, optional) – Learning rate. Defaults to0.0001.
configure_optimizers()
Configure optimizers for each decoder.
**Returns**
Adam optimizer for each decoder
**Return type**
Optimizer
property learning_type: LearningType
Return the learning type of the model.
**Returns**
Learning type of the model.
**Return type**
LearningType

**3.3. Reference Guide 103**


```
property trainer_arguments: dict[str, Any]
C-FLOW specific trainer arguments.
training_step( batch , *args , **kwargs )
Perform the training step of CFLOW.
For each batch, decoder layers are trained with a dynamic fiber batch size. Training step is performed
manually as multiple training steps are involved
per batch of input images
```
```
Parameters
```
- batch(dict[str, str | torch.Tensor]) – Input batch
- *args– Arguments.
- **kwargs– Keyword arguments.
**Return type**
Union[Tensor,Mapping[str,Any],None]
**Returns**
Loss value for the batch

```
validation_step( batch , *args , **kwargs )
Perform the validation step of CFLOW.
Similar to the training step, encoder features are extracted from the CNN for each batch, and
anomaly map is computed.
```
```
Parameters
```
- batch(dict[str, str | torch.Tensor]) – Input batch
- *args– Arguments.
- **kwargs– Keyword arguments.
**Return type**
Union[Tensor,Mapping[str,Any],None]
**Returns**
Dictionary containing images, anomaly maps, true labels and masks. These are required in
_validation_epoch_end_ for feature concatenation.

PyTorch model for CFlow model implementation.

classanomalib.models.image.cflow.torch_model.CflowModel( _backbone_ , _layers_ , _pre_trained=True_ ,
_fiber_batch_size=64_ ,
_decoder='freia-cflow'_ ,
_condition_vector=128_ ,
_coupling_blocks=8_ , _clamp_alpha=1.9_ ,
_permute_soft=False_ )
Bases:Module
CFLOW: Conditional Normalizing Flows.
**Parameters**

- backbone(str) – Backbone CNN architecture.

**104 Chapter 3. Guides**


- layers(Sequence[str]) – Layers to extract features from.
- pre_trained(bool) – Whether to use pre-trained weights. Defaults toTrue.
- fiber_batch_size(int) – Fiber batch size. Defaults to 64.
- decoder(str) – Decoder architecture. Defaults to"freia-cflow".
- condition_vector(int) – Condition vector size. Defaults to 128.
- coupling_blocks(int) – Number of coupling blocks. Defaults to 8.
- clamp_alpha(float) – Clamping value for the alpha parameter. Defaults to1.9.
- permute_soft(bool) – Whether to use soft permutation. Defaults toFalse.
forward( _images_ )
Forward-pass images into the network to extract encoder features and compute probability.
**Parameters**
images(Tensor) – Batch of images.
**Return type**
Tensor
**Returns**
Predicted anomaly maps.

Anomaly Map Generator for CFlow model implementation.

classanomalib.models.image.cflow.anomaly_map.AnomalyMapGenerator( _pool_layers_ )

```
Bases:Module
Generate Anomaly Heatmap.
compute_anomaly_map( distribution , height , width , image_size )
Compute the layer map based on likelihood estimation.
Parameters
```
- distribution(list[torch.Tensor]) – List of likelihoods for each layer.
- height(list[int]) – List of heights of the feature maps.
- width(list[int]) – List of widths of the feature maps.
- image_size(tuple[int, int] | torch.Size | None) – Size of the input image.
**Return type**
Tensor
**Returns**
Final Anomaly Map
forward( _**kwargs_ )
Return anomaly_map.
Expects _distribution_ , _height_ and ‘width’ keywords to be passed explicitly

**3.3. Reference Guide 105**


```
Example
```
```
>>>anomaly_map_generator = AnomalyMapGenerator(image_size=tuple(hparams.model.
˓→input_size),
>>> pool_layers=pool_layers)
>>>output = self.anomaly_map_generator(distribution=dist, height=height,␣
˓→width=width)
```
```
Raises
ValueError– distribution , height and ‘width’ keys are not found
Returns
anomaly map
Return type
torch.Tensor
```
**CS-Flow**

Fully Convolutional Cross-Scale-Flows for Image-based Defect Detection.

https://arxiv.org/pdf/2110.02855.pdf

classanomalib.models.image.csflow.lightning_model.Csflow( _cross_conv_hidden_channels=1024_ ,
_n_coupling_blocks=4_ , _clamp=3_ ,
_num_channels=3_ )
Bases:AnomalyModule
Fully Convolutional Cross-Scale-Flows for Image-based Defect Detection.
**Parameters**

- n_coupling_blocks(int) – Number of coupling blocks in the model. Defaults to 4.
- cross_conv_hidden_channels(int) – Number of hidden channels in the cross convolu-
    tion. Defaults to 1024.
- clamp(int) – Clamp value for glow layer. Defaults to 3.
- num_channels(int) – Number of channels in the model. Defaults to 3.
configure_optimizers()
Configure optimizers.
**Returns**
Adam optimizer
**Return type**
Optimizer
property learning_type: LearningType
Return the learning type of the model.
**Returns**
Learning type of the model.
**Return type**
LearningType

**106 Chapter 3. Guides**


```
property trainer_arguments: dict[str, Any]
CS-Flow-specific trainer arguments.
training_step( batch , *args , **kwargs )
Perform the training step of CS-Flow.
Parameters
```
- batch(dict[str, str | torch.Tensor]) – Input batch
- args– Arguments.
- kwargs– Keyword arguments.
**Return type**
Union[Tensor,Mapping[str,Any],None]
**Returns**
Loss value
validation_step( _batch_ , _*args_ , _**kwargs_ )
Perform the validation step for CS Flow.
**Parameters**
- batch(torch.Tensor) – Input batch
- args– Arguments.
- kwargs– Keyword arguments.
**Returns**
Dictionary containing the anomaly map, scores, etc.
**Return type**
dict[str, torch.Tensor]

PyTorch model for CS-Flow implementation.

classanomalib.models.image.csflow.torch_model.CsFlowModel( _input_size_ ,
_cross_conv_hidden_channels_ ,
_n_coupling_blocks=4_ , _clamp=3_ ,
_num_channels=3_ )
Bases:Module
CS Flow Module.
**Parameters**

- input_size(tuple[int, int]) – Input image size.
- cross_conv_hidden_channels(int) – Number of hidden channels in the cross convolu-
    tion.
- n_coupling_blocks(int) – Number of coupling blocks. Defaults to 4.
- clamp(float) – Clamp value for the coupling blocks. Defaults to 3.
- num_channels(int) – Number of channels in the input image. Defaults to 3.
forward( _images_ )
Forward method of the model.
**Parameters**
images(torch.Tensor) – Input images.

**3.3. Reference Guide 107**


```
Returns
During training: tuple containing the z_distribution for three scales
and the sum of log determinant of the Jacobian. During evaluation: tuple containing
anomaly maps and anomaly scores
Return type
tuple[torch.Tensor, torch.Tensor]
```
Loss function for the CS-Flow Model Implementation.

classanomalib.models.image.csflow.loss.CsFlowLoss( _*args_ , _**kwargs_ )

```
Bases:Module
Loss function for the CS-Flow Model Implementation.
forward( z_dist , jacobians )
Compute the loss CS-Flow.
Parameters
```
- z_dist(torch.Tensor) – Latent space image mappings from NF.
- jacobians(torch.Tensor) – Jacobians of the distribution
**Return type**
Tensor
**Returns**
Loss value

Anomaly Map Generator for CS-Flow model.

classanomalib.models.image.csflow.anomaly_map.AnomalyMapGenerator( _input_dims_ ,
_mode=AnomalyMapMode.ALL_ )
Bases:Module
Anomaly Map Generator for CS-Flow model.
**Parameters**

- input_dims(tuple[int, int, int]) – Input dimensions.
- mode(AnomalyMapMode) – Anomaly map mode. Defaults toAnomalyMapMode.ALL.
forward( _inputs_ )
Get anomaly maps by taking mean of the z-distributions across channels.
By default it computes anomaly maps for all the scales as it gave better performance on initial tests. Use
AnomalyMapMode.MAXfor the largest scale as mentioned in the paper.
**Parameters**
- inputs(torch.Tensor) – z-distributions for the three scales.
- mode(AnomalyMapMode) – Anomaly map mode.
**Returns**
Anomaly maps.
**Return type**
Tensor

**108 Chapter 3. Guides**


classanomalib.models.image.csflow.anomaly_map.AnomalyMapMode( _value_ , _names=None_ , _*_ ,
_module=None_ , _qualname=None_ ,
_type=None_ , _start=1_ ,
_boundary=None_ )
Bases:str,Enum
Generate anomaly map from all the scales or the max.

### DFKDE

DFKDE: Deep Feature Kernel Density Estimation.

classanomalib.models.image.dfkde.lightning_model.Dfkde( _backbone='resnet18'_ , _layers=('layer4',)_ ,
_pre_trained=True_ , _n_pca_components=16_ ,
_fea-
ture_scaling_method=FeatureScalingMethod.SCALE_ ,
_max_training_points=40000_ )
Bases:MemoryBankMixin,AnomalyModule
DFKDE: Deep Feature Kernel Density Estimation.
**Parameters**

- backbone(str) – Pre-trained model backbone. Defaults to"resnet18".
- layers(Sequence[str], optional) – Layers to extract features from. Defaults to(
    "layer4",).
- pre_trained(bool, optional) – Boolean to check whether to use a pre_trained back-
    bone. Defaults toTrue.
- n_pca_components(int, optional) – Number of PCA components. Defaults to 16.
- feature_scaling_method (FeatureScalingMethod, optional) – Feature scaling
    method. Defaults toFeatureScalingMethod.SCALE.
- max_training_points(int, optional) – Number of training points to fit the KDE
    model. Defaults to 40000.
static configure_optimizers()
DFKDE doesn’t require optimization, therefore returns no optimizers.
**Return type**
None
fit()
Fit a KDE Model to the embedding collected from the training set.
**Return type**
None
property learning_type: LearningType
Return the learning type of the model.
**Returns**
Learning type of the model.
**Return type**
LearningType

**3.3. Reference Guide 109**


```
property trainer_arguments: dict[str, Any]
Return DFKDE-specific trainer arguments.
training_step( batch , *args , **kwargs )
Perform the training step of DFKDE. For each batch, features are extracted from the CNN.
Parameters
```
- (batch(batch) – dict[str, str | torch.Tensor]): Batch containing image filename, image,
    label and mask
- args– Arguments.
- kwargs– Keyword arguments.
**Return type**
None
**Returns**
Deep CNN features.
validation_step( _batch_ , _*args_ , _**kwargs_ )
Perform the validation step of DFKDE.
Similar to the training step, features are extracted from the CNN for each batch.
**Parameters**
- batch(dict[str, str | torch.Tensor]) – Input batch
- args– Arguments.
- kwargs– Keyword arguments.
**Return type**
Union[Tensor,Mapping[str,Any],None]
**Returns**
Dictionary containing probability, prediction and ground truth values.

Normality model of DFKDE.

classanomalib.models.image.dfkde.torch_model.DfkdeModel( _backbone_ , _layers_ , _pre_trained=True_ ,
_n_pca_components=16_ , _fea-
ture_scaling_method=FeatureScalingMethod.SCALE_ ,
_max_training_points=40000_ )
Bases:Module
Normality Model for the DFKDE algorithm.
**Parameters**

- backbone(str) – Pre-trained model backbone.
- layers(Sequence[str]) – Layers to extract features from.
- pre_trained(bool, optional) – Boolean to check whether to use a pre_trained back-
    bone. Defaults toTrue.
- n_pca_components(int, optional) – Number of PCA components. Defaults to 16.
- feature_scaling_method (FeatureScalingMethod, optional) – Feature scaling
    method. Defaults toFeatureScalingMethod.SCALE.

**110 Chapter 3. Guides**


- max_training_points(int, optional) – Number of training points to fit the KDE
    model. Defaults to 40000.
forward( _batch_ )
Prediction by normality model.
**Parameters**
batch(torch.Tensor) – Input images.
**Returns**
Predictions
**Return type**
Tensor
get_features( _batch_ )
Extract features from the pretrained network.
**Parameters**
batch(torch.Tensor) – Image batch.
**Returns**
torch.Tensor containing extracted features.
**Return type**
Tensor

### DFM

DFM: Deep Feature Modeling.

https://arxiv.org/abs/1909.11786

classanomalib.models.image.dfm.lightning_model.Dfm( _backbone='resnet50'_ , _layer='layer3'_ ,
_pre_trained=True_ , _pooling_kernel_size=4_ ,
_pca_level=0.97_ , _score_type='fre'_ )
Bases:MemoryBankMixin,AnomalyModule
DFM: Deep Featured Kernel Density Estimation.
**Parameters**

- backbone(str) – Backbone CNN network Defaults to"resnet50".
- layer(str) – Layer to extract features from the backbone CNN Defaults to"layer3".
- pre_trained(bool, optional) – Boolean to check whether to use a pre_trained back-
    bone. Defaults toTrue.
- pooling_kernel_size(int, optional) – Kernel size to pool features extracted from
    the CNN. Defaults to 4.
- pca_level(float, optional) – Ratio from which number of components for PCA are
    calculated. Defaults to0.97.
- score_type(str, optional) – Scoring type. Options are _fre_ and _nll_. Defaults tofre.
static configure_optimizers()
DFM doesn’t require optimization, therefore returns no optimizers.
**Return type**
None

**3.3. Reference Guide 111**


```
fit()
Fit a PCA transformation and a Gaussian model to dataset.
Return type
None
property learning_type: LearningType
Return the learning type of the model.
Returns
Learning type of the model.
Return type
LearningType
property trainer_arguments: dict[str, Any]
Return DFM-specific trainer arguments.
training_step( batch , *args , **kwargs )
Perform the training step of DFM.
For each batch, features are extracted from the CNN.
Parameters
```
- batch(dict[str, str | torch.Tensor]) – Input batch
- args– Arguments.
- kwargs– Keyword arguments.
**Return type**
None
**Returns**
Deep CNN features.
validation_step( _batch_ , _*args_ , _**kwargs_ )
Perform the validation step of DFM.
Similar to the training step, features are extracted from the CNN for each batch.
**Parameters**
- batch(dict[str, str | torch.Tensor]) – Input batch
- args– Arguments.
- kwargs– Keyword arguments.
**Return type**
Union[Tensor,Mapping[str,Any],None]
**Returns**
Dictionary containing FRE anomaly scores and anomaly maps.

PyTorch model for DFM model implementation.

classanomalib.models.image.dfm.torch_model.DFMModel( _backbone_ , _layer_ , _pre_trained=True_ ,
_pooling_kernel_size=4_ , _n_comps=0.97_ ,
_score_type='fre'_ )
Bases:Module
Model for the DFM algorithm.

**112 Chapter 3. Guides**


```
Parameters
```
- backbone(str) – Pre-trained model backbone.
- layer(str) – Layer from which to extract features.
- pre_trained(bool, optional) – Boolean to check whether to use a pre_trained back-
    bone. Defaults toTrue.
- pooling_kernel_size(int, optional) – Kernel size to pool features extracted from
    the CNN. Defaults to 4.
- n_comps(float, optional) – Ratio from which number of components for PCA are cal-
    culated. Defaults to0.97.
- score_type(str, optional) – Scoring type. Options are _fre_ and _nll_. Anomaly Defaults
    tofre. Segmentation is supported with _fre_ only. If using _nll_ , set _task_ in config.yaml to
    classification Defaults toclassification.
fit( _dataset_ )
Fit a pca transformation and a Gaussian model to dataset.
**Parameters**
dataset(torch.Tensor) – Input dataset to fit the model.
**Return type**
None
forward( _batch_ )
Compute score from input images.
**Parameters**
batch(torch.Tensor) – Input images
**Returns**
Scores
**Return type**
Tensor
get_features( _batch_ )
Extract features from the pretrained network.
**Parameters**
batch(torch.Tensor) – Image batch.
**Returns**
torch.Tensor containing extracted features.
**Return type**
Tensor
score( _features_ , _feature_shapes_ )
Compute scores.
Scores are either PCA-based feature reconstruction error (FRE) scores or the Gaussian density-based NLL
scores
**Parameters**
- features(torch.Tensor) – semantic features on which PCA and density modeling is
performed.

**3.3. Reference Guide 113**


- feature_shapes(tuple) – shape of _features_ tensor. Used to generate anomaly map of
    correct shape.
**Returns**
numpy array of scores
**Return type**
score (torch.Tensor)

classanomalib.models.image.dfm.torch_model.SingleClassGaussian

```
Bases:DynamicBufferMixin
Model Gaussian distribution over a set of points.
fit( dataset )
Fit a Gaussian model to dataset X.
Covariance matrix is not calculated directly using:C = X.X^TInstead, it is represented in terms of the
Singular Value Decomposition of X:X = U.S.V^THence,C = U.S^2.U^TThis simplifies the calculation
of the log-likelihood without requiring full matrix inversion.
Parameters
dataset(torch.Tensor) – Input dataset to fit the model.
Return type
None
forward( dataset )
Provide the same functionality as fit.
Transforms the input dataset based on singular values calculated earlier.
Parameters
dataset(torch.Tensor) – Input dataset
Return type
None
score_samples( features )
Compute the NLL (negative log likelihood) scores.
Parameters
features(torch.Tensor) – semantic features on which density modeling is performed.
Returns
Torch tensor of scores
Return type
nll (torch.Tensor)
```
### DRAEM

DRÆM - A discriminatively trained reconstruction embedding for surface anomaly detection.

Paper https://arxiv.org/abs/2108.07610

classanomalib.models.image.draem.lightning_model.Draem( _enable_sspcab=False_ ,
_sspcab_lambda=0.1_ ,
_anomaly_source_path=None_ , _beta=(0.1,
1.0)_ )

**114 Chapter 3. Guides**


```
Bases:AnomalyModule
DRÆM: A discriminatively trained reconstruction embedding for surface anomaly detection.
Parameters
```
- enable_sspcab(bool) – Enable SSPCAB training. Defaults toFalse.
- sspcab_lambda(float) – SSPCAB loss weight. Defaults to0.1.
- anomaly_source_path(str | None) – Path to folder that contains the anomaly source
    images. Random noise will be used if left empty. Defaults toNone.
configure_optimizers()
Configure the Adam optimizer.
**Return type**
Optimizer
property learning_type: LearningType
Return the learning type of the model.
**Returns**
Learning type of the model.
**Return type**
LearningType
setup_sspcab()
Prepare the model for the SSPCAB training step by adding forward hooks for the SSPCAB layer activations.
**Return type**
None
property trainer_arguments: dict[str, Any]
Return DRÆM-specific trainer arguments.
training_step( _batch_ , _*args_ , _**kwargs_ )
Perform the training step of DRAEM.
Feeds the original image and the simulated anomaly image through the network and computes the training
loss.
**Parameters**
- batch(dict[str, str | torch.Tensor]) – Batch containing image filename, image,
label and mask
- args– Arguments.
- kwargs– Keyword arguments.
**Return type**
Union[Tensor,Mapping[str,Any],None]
**Returns**
Loss dictionary
validation_step( _batch_ , _*args_ , _**kwargs_ )
Perform the validation step of DRAEM. The Softmax predictions of the anomalous class are used as
anomaly map.
**Parameters**

**3.3. Reference Guide 115**


- batch(dict[str, str | torch.Tensor]) – Batch of input images
- args– Arguments.
- kwargs– Keyword arguments.
**Return type**
Union[Tensor,Mapping[str,Any],None]
**Returns**
Dictionary to which predicted anomaly maps have been added.

PyTorch model for the DRAEM model implementation.

classanomalib.models.image.draem.torch_model.DraemModel( _sspcab=False_ )

```
Bases:Module
DRAEM PyTorch model consisting of the reconstructive and discriminative sub networks.
Parameters
sspcab(bool) – Enable SSPCAB training. Defaults toFalse.
forward( batch )
Compute the reconstruction and anomaly mask from an input image.
Parameters
batch(torch.Tensor) – batch of input images
Return type
Tensor|tuple[Tensor,Tensor]
Returns
Predicted confidence values of the anomaly mask. During training the reconstructed input
images are returned as well.
```
Loss function for the DRAEM model implementation.

classanomalib.models.image.draem.loss.DraemLoss

```
Bases:Module
Overall loss function of the DRAEM model.
The total loss consists of the sum of the L2 loss and Focal loss between the reconstructed image and the input
image, and the Structural Similarity loss between the predicted and GT anomaly masks.
forward( input_image , reconstruction , anomaly_mask , prediction )
Compute the loss over a batch for the DRAEM model.
Return type
Tensor
```
### DSR

This is the implementation of the DSR paper.

Model Type: Segmentation

**116 Chapter 3. Guides**


**Description**

DSR is a quantized-feature based algorithm that consists of an autoencoder with one encoder and two decoders, coupled
with an anomaly detection module. DSR learns a codebook of quantized representations on ImageNet, which are then
used to encode input images. These quantized representations also serve to sample near-in-distribution anomalies,
since they do not rely on external datasets. Training takes place in three phases. The encoder and “general object
decoder”, as well as the codebook, are pretrained on ImageNet. Defects are then generated at the feature level using
the codebook on the quantized representations, and are used to train the object-specific decoder as well as the anomaly
detection module. In the final phase of training, the upsampling module is trained on simulated image-level smudges
in order to output more robust anomaly maps.

**Architecture**

```
PyTorch model for the DSR model implementation.
```
classanomalib.models.image.dsr.torch_model.AnomalyDetectionModule( _in_channels_ , _out_channels_ ,
_base_width_ )
Bases:Module
Anomaly detection module.
Module that detects the preseßnce of an anomaly by comparing two images reconstructed by the object specific
decoder and the general object decoder.
**Parameters**

- in_channels(int) – Number of input channels.
- out_channels(int) – Number of output channels.
- base_width(int) – Base dimensionality of the layers of the autoencoder.
forward( _batch_real_ , _batch_anomaly_ )
Computes the anomaly map over corresponding real and anomalous images.
**Parameters**
- batch_real(torch.Tensor) – Batch of real, non defective images.
- batch_anomaly(torch.Tensor) – Batch of potentially anomalous images.
**Return type**
Tensor

**3.3. Reference Guide 117**


```
Returns
The anomaly segmentation map.
```
classanomalib.models.image.dsr.torch_model.DecoderBot( _in_channels_ , _num_hiddens_ ,
_num_residual_layers_ ,
_num_residual_hiddens_ )
Bases:Module
General appearance decoder module to reconstruct images while keeping possible anomalies.
**Parameters**

- in_channels(int) – Number of input channels.
- num_hiddens(int) – Number of hidden channels.
- num_residual_layers(int) – Number of residual layers in residual stack.
- num_residual_hiddens(int) – Number of channels in residual layers.
forward( _inputs_ )
Decode quantized feature maps into an image.
**Parameters**
inputs(torch.Tensor) – Quantized feature maps.
**Return type**
Tensor
**Returns**
Decoded image.

classanomalib.models.image.dsr.torch_model.DiscreteLatentModel( _num_hiddens_ ,
_num_residual_layers_ ,
_num_residual_hiddens_ ,
_num_embeddings_ ,
_embedding_dim_ )
Bases:Module
Discrete Latent Model.
Autoencoder quantized model that encodes the input images into quantized feature maps and generates a recon-
structed image using the general appearance decoder.
**Parameters**

- num_hiddens(int) – Number of hidden channels.
- num_residual_layers(int) – Number of residual layers in residual stacks.
- num_residual_hiddens(int) – Number of channels in residual layers.
- num_embeddings(int) – Size of embedding dictionary.
- embedding_dim(int) – Dimension of embeddings.
forward( _batch_ , _anomaly_mask=None_ , _anom_str_lo=None_ , _anom_str_hi=None_ )
Generate quantized feature maps.
Generates quantized feature maps of batch of input images as well as their reconstruction based on the
general appearance decoder.
**Parameters**
- batch(Tensor) – Batch of input images.

**118 Chapter 3. Guides**


- anomaly_mask(Tensor | None) – Anomaly mask to be used to generate anomalies on
    the quantized feature maps.
- anom_str_lo(torch.Tensor | None) – Strength of generated anomaly lo.
- anom_str_hi(torch.Tensor | None) – Strength of generated anomaly hi.
**Returns
If generating an anomaly mask:**
- General object decoder-decoded anomalous image
- Reshaped ground truth anomaly map
- Non defective quantized lo feature
- Non defective quantized hi feature
- Non quantized subspace encoded defective lo feature
- Non quantized subspace encoded defective hi feature
**Else:**
- General object decoder-decoded image
- Quantized lo feature
- Quantized hi feature
**Return type**
dict[str, torch.Tensor]
generate_fake_anomalies_joined( _features_ , _embeddings_ , _memory_torch_original_ , _mask_ , _strength_ )
Generate quantized anomalies.
**Parameters**
- features(torch.Tensor) – Features on which the anomalies will be generated.
- embeddings(torch.Tensor) – Embeddings to use to generate the anomalies.
- memory_torch_original(torch.Tensor) – Weight of embeddings.
- mask(torch.Tensor) – Original anomaly mask.
- strength(float) – Strength of generated anomaly.
**Returns**
Anomalous embedding.
**Return type**
torch.Tensor
property vq_vae_bot:VectorQuantizer
Returnself._vq_vae_bot.
property vq_vae_top:VectorQuantizer
Returnself._vq_vae_top.

classanomalib.models.image.dsr.torch_model.DsrModel( _latent_anomaly_strength=0.2_ ,
_embedding_dim=128_ , _num_embeddings=4096_ ,
_num_hiddens=128_ , _num_residual_layers=2_ ,
_num_residual_hiddens=64_ )

**3.3. Reference Guide 119**


```
Bases:Module
DSR PyTorch model.
Consists of the discrete latent model, image reconstruction network, subspace restriction modules, anomaly
detection module and upsampling module.
Parameters
```
- embedding_dim(int) – Dimension of codebook embeddings.
- num_embeddings(int) – Number of embeddings.
- latent_anomaly_strength(float) – Strength of the generated anomalies in the latent
    space.
- num_hiddens(int) – Number of output channels in residual layers.
- num_residual_layers(int) – Number of residual layers.
- num_residual_hiddens(int) – Number of intermediate channels.
forward( _batch_ , _anomaly_map_to_generate=None_ )
Compute the anomaly mask from an input image.
**Parameters**
- batch(torch.Tensor) – Batch of input images.
- anomaly_map_to_generate(torch.Tensor | None) – anomaly map to use to gener-
ate quantized defects.
- 2 (If not training phase) –
- None.(should be) –
**Returns
If testing:**
- ”anomaly_map”: Upsampled anomaly map
- ”pred_score”: Image score
**If training phase 2:**
- ”recon_feat_hi”: Reconstructed non-quantized hi features of defect (F~_hi)
- ”recon_feat_lo”: Reconstructed non-quantized lo features of defect (F~_lo)
- ”embedding_bot”: Quantized features of non defective img (Q_hi)
- ”embedding_top”: Quantized features of non defective img (Q_lo)
- ”obj_spec_image”: Object-specific-decoded image (I_spc)
- ”anomaly_map”: Predicted segmentation mask (M)
- ”true_mask”: Resized ground-truth anomaly map (M_gt)
**If training phase 3:**
- ”anomaly_map”: Reconstructed anomaly map
**Return type**
dict[str, torch.Tensor]

**120 Chapter 3. Guides**


```
load_pretrained_discrete_model_weights( ckpt )
Load pre-trained model weights.
Return type
None
```
classanomalib.models.image.dsr.torch_model.EncoderBot( _in_channels_ , _num_hiddens_ ,
_num_residual_layers_ ,
_num_residual_hiddens_ )
Bases:Module
Encoder module for bottom quantized feature maps.
**Parameters**

- in_channels(int) – Number of input channels.
- num_hiddens(int) – Number of hidden channels.
- num_residual_layers(int) – Number of residual layers in residual stacks.
- num_residual_hiddens(int) – Number of channels in residual layers.
forward( _batch_ )
Encode inputs to be quantized into the bottom feature map.
**Parameters**
batch(torch.Tensor) – Batch of input images.
**Return type**
Tensor
**Returns**
Encoded feature maps.

classanomalib.models.image.dsr.torch_model.EncoderTop( _in_channels_ , _num_hiddens_ ,
_num_residual_layers_ ,
_num_residual_hiddens_ )
Bases:Module
Encoder module for top quantized feature maps.
**Parameters**

- in_channels(int) – Number of input channels.
- num_hiddens(int) – Number of hidden channels.
- num_residual_layers(int) – Number of residual layers in residual stacks.
- num_residual_hiddens(int) – Number of channels in residual layers.
forward( _batch_ )
Encode inputs to be quantized into the top feature map.
**Parameters**
batch(torch.Tensor) – Batch of input images.
**Return type**
Tensor
**Returns**
Encoded feature maps.

**3.3. Reference Guide 121**


classanomalib.models.image.dsr.torch_model.FeatureDecoder( _base_width_ , _out_channels=1_ )

```
Bases:Module
Feature decoder for the subspace restriction network.
Parameters
```
- base_width(int) – Base dimensionality of the layers of the autoencoder.
- out_channels(int) – Number of output channels.
forward( ___ , ____ , _b3_ )
Decode a batch of latent features to a non-quantized representation.
**Parameters**
- _(torch.Tensor) – Top latent feature layer.
- __(torch.Tensor) – Middle latent feature layer.
- b3(torch.Tensor) – Bottom latent feature layer.
**Return type**
Tensor
**Returns**
Decoded non-quantized representation.

classanomalib.models.image.dsr.torch_model.FeatureEncoder( _in_channels_ , _base_width_ )

```
Bases:Module
Feature encoder for the subspace restriction network.
Parameters
```
- in_channels(int) – Number of input channels.
- base_width(int) – Base dimensionality of the layers of the autoencoder.
forward( _batch_ )
Encode a batch of input features to the latent space.
**Parameters**
batch(torch.Tensor) – Batch of input images.
**Return type**
tuple[Tensor,Tensor,Tensor]
Returns: Encoded feature maps.

classanomalib.models.image.dsr.torch_model.ImageReconstructionNetwork( _in_channels_ ,
_num_hiddens_ ,
_num_residual_layers_ ,
_num_residual_hiddens_ )
Bases:Module
Image Reconstruction Network.
Image reconstruction network that reconstructs the image from a quantized representation.
**Parameters**

- in_channels(int) – Number of input channels.
- num_hiddens(int) – Number of output channels in residual layers.

**122 Chapter 3. Guides**


- num_residual_layers(int) – Number of residual layers.
- num_residual_hiddens(int) – Number of intermediate channels.
forward( _inputs_ )
Reconstructs an image from a quantized representation.
**Parameters**
inputs(torch.Tensor) – Quantized features.
**Return type**
Tensor
**Returns**
Reconstructed image.

classanomalib.models.image.dsr.torch_model.Residual( _in_channels_ , _out_channels_ ,
_num_residual_hiddens_ )
Bases:Module
Residual layer.
**Parameters**

- in_channels(int) – Number of input channels.
- out_channels(int) – Number of output channels.
- num_residual_hiddens(int) – Number of intermediate channels.
forward( _batch_ )
Compute residual layer.
**Parameters**
batch(torch.Tensor) – Batch of input images.
**Return type**
Tensor
**Returns**
Computed feature maps.

classanomalib.models.image.dsr.torch_model.ResidualStack( _in_channels_ , _num_hiddens_ ,
_num_residual_layers_ ,
_num_residual_hiddens_ )
Bases:Module
Stack of residual layers.
**Parameters**

- in_channels(int) – Number of input channels.
- num_hiddens(int) – Number of output channels in residual layers.
- num_residual_layers(int) – Number of residual layers.
- num_residual_hiddens(int) – Number of intermediate channels.
forward( _batch_ )
Compute residual stack.
**Parameters**
batch(torch.Tensor) – Batch of input images.

**3.3. Reference Guide 123**


```
Return type
Tensor
Returns
Computed feature maps.
```
classanomalib.models.image.dsr.torch_model.SubspaceRestrictionModule( _base_width_ )

```
Bases:Module
Subspace Restriction Module.
Subspace restriction module that restricts the appearance subspace into configurations that agree with normal
appearances and applies quantization.
Parameters
base_width(int) – Base dimensionality of the layers of the autoencoder.
forward( batch , quantization )
Generate the quantized anomaly-free representation of an anomalous image.
Parameters
```
- batch(torch.Tensor) – Batch of input images.
- quantization(function | object) – Quantization function.
**Return type**
tuple[Tensor,Tensor]
**Returns**
Reconstructed batch of non-quantized features and corresponding quantized features.

classanomalib.models.image.dsr.torch_model.SubspaceRestrictionNetwork( _in_channels=64_ ,
_out_channels=64_ ,
_base_width=64_ )
Bases:Module
Subspace Restriction Network.
Subspace restriction network that reconstructs the input image into a non-quantized configuration that agrees
with normal appearances.
**Parameters**

- in_channels(int) – Number of input channels.
- out_channels(int) – Number of output channels.
- base_width(int) – Base dimensionality of the layers of the autoencoder.
forward( _batch_ )
Reconstruct non-quantized representation from batch.
Generate non-quantized feature maps from potentially anomalous images, to be quantized into non-
anomalous quantized representations.
**Parameters**
batch(torch.Tensor) – Batch of input images.
**Return type**
Tensor
**Returns**
Reconstructed non-quantized representation.

**124 Chapter 3. Guides**


classanomalib.models.image.dsr.torch_model.UnetDecoder( _base_width_ , _out_channels=1_ )

```
Bases:Module
Decoder of the Unet network.
Parameters
```
- base_width(int) – Base dimensionality of the layers of the autoencoder.
- out_channels(int) – Number of output channels.
forward( _b1_ , _b2_ , _b3_ , _b4_ )
Decodes latent represnetations into an image.
**Parameters**
- b1(torch.Tensor) – First (top level) quantized feature map.
- b2(torch.Tensor) – Second quantized feature map.
- b3(torch.Tensor) – Third quantized feature map.
- b4(torch.Tensor) – Fourth (bottom level) quantized feature map.
**Return type**
Tensor
**Returns**
Reconstructed image.

classanomalib.models.image.dsr.torch_model.UnetEncoder( _in_channels_ , _base_width_ )

```
Bases:Module
Encoder of the Unet network.
Parameters
```
- in_channels(int) – Number of input channels.
- base_width(int) – Base dimensionality of the layers of the autoencoder.
forward( _batch_ )
Encodes batch of images into a latent representation.
**Parameters**
batch(torch.Tensor) – Quantized features.
**Return type**
tuple[Tensor,Tensor,Tensor,Tensor]
**Returns**
Latent representations of the input batch.

classanomalib.models.image.dsr.torch_model.UnetModel( _in_channels=64_ , _out_channels=64_ ,
_base_width=64_ )
Bases:Module
Autoencoder model that reconstructs the input image.
**Parameters**

- in_channels(int) – Number of input channels.
- out_channels(int) – Number of output channels.
- base_width(int) – Base dimensionality of the layers of the autoencoder.

**3.3. Reference Guide 125**


```
forward( batch )
Reconstructs an input batch of images.
Parameters
batch(torch.Tensor) – Batch of input images.
Return type
Tensor
Returns
Reconstructed images.
```
classanomalib.models.image.dsr.torch_model.UpsamplingModule( _in_channels=8_ , _out_channels=2_ ,
_base_width=64_ )
Bases:Module
Module that upsamples the generated anomaly mask to full resolution.
**Parameters**

- in_channels(int) – Number of input channels.
- out_channels(int) – Number of output channels.
- base_width(int) – Base dimensionality of the layers of the autoencoder.
forward( _batch_real_ , _batch_anomaly_ , _batch_segmentation_map_ )
Computes upsampled segmentation maps.
**Parameters**
- batch_real(torch.Tensor) – Batch of real, non defective images.
- batch_anomaly(torch.Tensor) – Batch of potentially anomalous images.
- batch_segmentation_map(torch.Tensor) – Batch of anomaly segmentation maps.
**Return type**
Tensor
**Returns**
Upsampled anomaly segmentation maps.

classanomalib.models.image.dsr.torch_model.VectorQuantizer( _num_embeddings_ , _embedding_dim_ )

```
Bases:Module
Module that quantizes a given feature map using learned quantization codebooks.
Parameters
```
- num_embeddings(int) – Size of embedding codebook.
- embedding_dim(int) – Dimension of embeddings.
property embedding: Tensor
Return embedding.
forward( _inputs_ )
Calculates quantized feature map.
**Parameters**
inputs(torch.Tensor) – Non-quantized feature maps.
**Return type**
Tensor

**126 Chapter 3. Guides**


```
Returns
Quantized feature maps.
```
DSR - A Dual Subspace Re-Projection Network for Surface Anomaly Detection.

Paper https://link.springer.com/chapter/10.1007/978-3-031-19821-2_31

classanomalib.models.image.dsr.lightning_model.Dsr( _latent_anomaly_strength=0.2_ ,
_upsampling_train_ratio=0.7_ )
Bases:AnomalyModule
DSR: A Dual Subspace Re-Projection Network for Surface Anomaly Detection.
**Parameters**

- latent_anomaly_strength(float) – Strength of the generated anomalies in the latent
    space. Defaults to 0.2
- upsampling_train_ratio(float) – Ratio of training steps for the upsampling module.
    Defaults to 0.7
configure_optimizers()
Configure the Adam optimizer for training phases 2 and 3.
Does not train the discrete model (phase 1)
**Returns**
Dictionary of optimizers
**Return type**
dict[str, torch.optim.Optimizer | torch.optim.lr_scheduler.LRScheduler]
property learning_type: LearningType
Return the learning type of the model.
**Returns**
Learning type of the model.
**Return type**
LearningType
on_train_epoch_start()
Display a message when starting to train the upsampling module.
**Return type**
None
on_train_start()
Load pretrained weights of the discrete model when starting training.
**Return type**
None
prepare_pretrained_model()
Download pre-trained models if they don’t exist.
**Return type**
Path
property trainer_arguments: dict[str, Any]
Required trainer arguments.

**3.3. Reference Guide 127**


```
training_step( batch )
Training Step of DSR.
Feeds the original image and the simulated anomaly mask during first phase. During second phase, feeds
a generated anomalous image to train the upsampling module.
Parameters
batch(dict[str, str | Tensor]) – Batch containing image filename, image, label and
mask
Returns
Loss dictionary
Return type
STEP_OUTPUT
validation_step( batch , *args , **kwargs )
Validation step of DSR.
The Softmax predictions of the anomalous class are used as anomaly map.
Parameters
```
- batch(dict[str, str | Tensor]) – Batch of input images
- *args– unused
- **kwargs– unused
**Returns**
Dictionary to which predicted anomaly maps have been added.
**Return type**
STEP_OUTPUT

Anomaly generator for the DSR model implementation.

classanomalib.models.image.dsr.anomaly_generator.DsrAnomalyGenerator( _p_anomalous=0.5_ )

```
Bases:Module
Anomaly generator of the DSR model.
The anomaly is generated using a Perlin noise generator on the two quantized representations of an image. This
generator is only used during the second phase of training! The third phase requires generating smudges over
the input images.
Parameters
p_anomalous(float, optional) – Probability to generate an anomalous image.
augment_batch( batch )
Generate anomalous augmentations for a batch of input images.
Parameters
batch(Tensor) – Batch of input images
Returns
Ground truth masks corresponding to the anomalous perturbations.
Return type
Tensor
generate_anomaly( height , width )
Generate an anomalous mask.
```
**128 Chapter 3. Guides**


```
Parameters
```
- height(int) – Height of generated mask.
- width(int) – Width of generated mask.
**Returns**
Generated mask.
**Return type**
Tensor

Loss function for the DSR model implementation.

classanomalib.models.image.dsr.loss.DsrSecondStageLoss

```
Bases:Module
Overall loss function of the second training phase of the DSR model.
The total loss consists of:
```
- MSE loss between non-anomalous quantized input image and anomalous subspace-reconstructed non-
    quantized input (hi and lo)
- MSE loss between input image and reconstructed image through object-specific decoder,
- Focal loss between computed segmentation mask and ground truth mask.
forward( _recon_nq_hi_ , _recon_nq_lo_ , _qu_hi_ , _qu_lo_ , _input_image_ , _gen_img_ , _seg_ , _anomaly_mask_ )
Compute the loss over a batch for the DSR model.
**Parameters**
- recon_nq_hi(Tensor) – Reconstructed non-quantized hi feature
- recon_nq_lo(Tensor) – Reconstructed non-quantized lo feature
- qu_hi(Tensor) – Non-defective quantized hi feature
- qu_lo(Tensor) – Non-defective quantized lo feature
- input_image(Tensor) – Original image
- gen_img(Tensor) – Object-specific decoded image
- seg(Tensor) – Computed anomaly map
- anomaly_mask(Tensor) – Ground truth anomaly map
**Returns**
Total loss
**Return type**
Tensor

classanomalib.models.image.dsr.loss.DsrThirdStageLoss

```
Bases:Module
Overall loss function of the third training phase of the DSR model.
The loss consists of a focal loss between the computed segmentation mask and the ground truth mask.
forward( pred_mask , true_mask )
Compute the loss over a batch for the DSR model.
Parameters
```
**3.3. Reference Guide 129**


- pred_mask(Tensor) – Computed anomaly map
- true_mask(Tensor) – Ground truth anomaly map
**Returns**
Total loss
**Return type**
Tensor

**Efficient AD**

EfficientAd: Accurate Visual Anomaly Detection at Millisecond-Level Latencies.

https://arxiv.org/pdf/2303.14535.pdf.

classanomalib.models.image.efficient_ad.lightning_model.EfficientAd( _imagenet_dir='./datasets/imagenette'_ ,
_teacher_out_channels=384_ ,
_model_size=EfficientAdModelSize.S_ ,
_lr=0.0001_ ,
_weight_decay=1e-05_ ,
_padding=False_ ,
_pad_maps=True_ ,
_batch_size=1_ )
Bases:AnomalyModule
PL Lightning Module for the EfficientAd algorithm.
**Parameters**

- imagenet_dir(Path|str) – directory path for the Imagenet dataset Defaults to./
    datasets/imagenette.
- teacher_out_channels(int) – number of convolution output channels Defaults to 384.
- model_size (str) – size of student and teacher model Defaults to
    EfficientAdModelSize.S.
- lr(float) – learning rate Defaults to0.0001.
- weight_decay(float) – optimizer weight decay Defaults to0.00001.
- padding(bool) – use padding in convoluional layers Defaults toFalse.
- pad_maps(bool) – relevant if padding is set to False. In this case, pad_maps = True pads the
    output anomaly maps so that their size matches the size in the padding = True case. Defaults
    toTrue.
- batch_size(int) – batch size for imagenet dataloader Defaults to 1.
configure_optimizers()
Configure optimizers.
**Return type**
Optimizer
configure_transforms( _image_size=None_ )
Default transform for Padim.
**Return type**
Transform

**130 Chapter 3. Guides**


```
property learning_type: LearningType
Return the learning type of the model.
Returns
Learning type of the model.
Return type
LearningType
map_norm_quantiles( dataloader )
Calculate 90% and 99.5% quantiles of the student(st) and autoencoder(ae).
Parameters
dataloader(DataLoader) – Dataloader of the respective dataset.
Returns
Dictionary of both the 90% and 99.5% quantiles of both the student and autoencoder feature
maps.
Return type
dict[str, torch.Tensor]
on_train_start()
Called before the first training epoch.
First sets up the pretrained teacher model, then prepares the imagenette data, and finally calculates or loads
the channel-wise mean and std of the training dataset and push to the model.
Return type
None
on_validation_start()
Calculate the feature map quantiles of the validation dataset and push to the model.
Return type
None
prepare_imagenette_data( image_size )
Prepare ImageNette dataset transformations.
Parameters
image_size(tuple[int, int] | torch.Size) – Image size.
Return type
None
prepare_pretrained_model()
Prepare the pretrained teacher model.
Return type
None
teacher_channel_mean_std( dataloader )
Calculate the mean and std of the teacher models activations.
Adapted from https://math.stackexchange.com/a/2148949
Parameters
dataloader(DataLoader) – Dataloader of the respective dataset.
Returns
Dictionary of channel-wise mean and std
```
**3.3. Reference Guide 131**


```
Return type
dict[str, torch.Tensor]
property trainer_arguments: dict[str, Any]
Return EfficientAD trainer arguments.
training_step( batch , *args , **kwargs )
Perform the training step for EfficientAd returns the student, autoencoder and combined loss.
Parameters
```
- (batch(batch) – dict[str, str | torch.Tensor]): Batch containing image filename, image,
    label and mask
- args– Additional arguments.
- kwargs– Additional keyword arguments.
**Return type**
dict[str,Tensor]
**Returns**
Loss.
validation_step( _batch_ , _*args_ , _**kwargs_ )
Perform the validation step of EfficientAd returns anomaly maps for the input image batch.
**Parameters**
- batch(dict[str, str | torch.Tensor]) – Input batch
- args– Additional arguments.
- kwargs– Additional keyword arguments.
**Return type**
Union[Tensor,Mapping[str,Any],None]
**Returns**
Dictionary containing anomaly maps.

Torch model for student, teacher and autoencoder model in EfficientAd.

classanomalib.models.image.efficient_ad.torch_model.EfficientAdModel( _teacher_out_channels_ ,
_model_size=EfficientAdModelSize.S_ ,
_padding=False_ ,
_pad_maps=True_ )
Bases:Module
EfficientAd model.
**Parameters**

- teacher_out_channels(int) – number of convolution output channels of the pre-trained
    teacher model
- model_size(str) – size of student and teacher model
- padding(bool) – use padding in convoluional layers Defaults toFalse.
- pad_maps(bool) – relevant if padding is set to False. In this case, pad_maps = True pads the
    output anomaly maps so that their size matches the size in the padding = True case. Defaults
    toTrue.

**132 Chapter 3. Guides**


```
choose_random_aug_image( image )
Choose a random augmentation function and apply it to the input image.
Parameters
image(torch.Tensor) – Input image.
Returns
Augmented image.
Return type
Tensor
forward( batch , batch_imagenet=None , normalize=True )
Perform the forward-pass of the EfficientAd models.
Parameters
```
- batch(torch.Tensor) – Input images.
- batch_imagenet(torch.Tensor) – ImageNet batch. Defaults to None.
- normalize(bool) – Normalize anomaly maps or not
**Returns**
Predictions
**Return type**
Tensor
is_set( _p_dic_ )
Check if any of the parameters in the parameter dictionary is set.
**Parameters**
p_dic(nn.ParameterDict) – Parameter dictionary.
**Returns**
Boolean indicating whether any of the parameters in the parameter dictionary is set.
**Return type**
bool

**FastFlow**

FastFlow Lightning Model Implementation.

https://arxiv.org/abs/2111.07677

classanomalib.models.image.fastflow.lightning_model.Fastflow( _backbone='resnet18'_ ,
_pre_trained=True_ , _flow_steps=8_ ,
_conv3x3_only=False_ ,
_hidden_ratio=1.0_ )
Bases:AnomalyModule
PL Lightning Module for the FastFlow algorithm.
**Parameters**

- backbone(str) – Backbone CNN network Defaults toresnet18.
- pre_trained(bool, optional) – Boolean to check whether to use a pre_trained back-
    bone. Defaults toTrue.

**3.3. Reference Guide 133**


- flow_steps(int, optional) – Flow steps. Defaults to 8.
- conv3x3_only(bool, optinoal) – Use only conv3x3 in fast_flow model. Defaults to
    False.
- hidden_ratio(float, optional) – Ratio to calculate hidden var channels. Defaults to
    **``** 1.0`.
configure_optimizers()
Configure optimizers for each decoder.
**Returns**
Adam optimizer for each decoder
**Return type**
Optimizer
property learning_type: LearningType
Return the learning type of the model.
**Returns**
Learning type of the model.
**Return type**
LearningType
property trainer_arguments: dict[str, Any]
Return FastFlow trainer arguments.
training_step( _batch_ , _*args_ , _**kwargs_ )
Perform the training step input and return the loss.
**Parameters**
- (batch(batch) – dict[str, str | torch.Tensor]): Input batch
- args– Additional arguments.
- kwargs– Additional keyword arguments.
**Returns**
Dictionary containing the loss value.
**Return type**
STEP_OUTPUT
validation_step( _batch_ , _*args_ , _**kwargs_ )
Perform the validation step and return the anomaly map.
**Parameters**
- batch(dict[str, str | torch.Tensor]) – Input batch
- args– Additional arguments.
- kwargs– Additional keyword arguments.
**Returns**
batch dictionary containing anomaly-maps.
**Return type**
STEP_OUTPUT | None

FastFlow Torch Model Implementation.

**134 Chapter 3. Guides**


classanomalib.models.image.fastflow.torch_model.FastflowModel( _input_size_ , _backbone_ ,
_pre_trained=True_ , _flow_steps=8_ ,
_conv3x3_only=False_ ,
_hidden_ratio=1.0_ )
Bases:Module
FastFlow.
Unsupervised Anomaly Detection and Localization via 2D Normalizing Flows.
**Parameters**

- input_size(tuple[int, int]) – Model input size.
- backbone(str) – Backbone CNN network
- pre_trained(bool, optional) – Boolean to check whether to use a pre_trained back-
    bone. Defaults toTrue.
- flow_steps(int, optional) – Flow steps. Defaults to 8.
- conv3x3_only(bool, optinoal) – Use only conv3x3 in fast_flow model. Defaults to
    False.
- hidden_ratio(float, optional) – Ratio to calculate hidden var channels. Defaults to
    1.0.
**Raises**
ValueError– When the backbone is not supported.
forward( _input_tensor_ )
Forward-Pass the input to the FastFlow Model.
**Parameters**
input_tensor(torch.Tensor) – Input tensor.
**Returns
During training, return**
(hidden_variables, log-of-the-jacobian-determinants). During the validation/test, return
the anomaly map.
**Return type**
Tensor | list[torch.Tensor] | tuple[list[torch.Tensor]]

Loss function for the FastFlow Model Implementation.

classanomalib.models.image.fastflow.loss.FastflowLoss( _*args_ , _**kwargs_ )

```
Bases:Module
FastFlow Loss.
forward( hidden_variables , jacobians )
Calculate the Fastflow loss.
Parameters
```
- hidden_variables (list[torch.Tensor]) – Hidden variables from the fastflow
    model. f: X -> Z
- jacobians(list[torch.Tensor]) – Log of the jacobian determinants from the fastflow
    model.

**3.3. Reference Guide 135**


```
Returns
Fastflow loss computed based on the hidden variables and the log of the Jacobians.
Return type
Tensor
```
FastFlow Anomaly Map Generator Implementation.

classanomalib.models.image.fastflow.anomaly_map.AnomalyMapGenerator( _input_size_ )

```
Bases:Module
Generate Anomaly Heatmap.
Parameters
input_size(ListConfig | tuple) – Input size.
forward( hidden_variables )
Generate Anomaly Heatmap.
This implementation generates the heatmap based on the flow maps computed from the normalizing flow
(NF) FastFlow blocks. Each block yields a flow map, which overall is stacked and averaged to an anomaly
map.
Parameters
hidden_variables(list[torch.Tensor]) – List of hidden variables from each NF Fast-
Flow block.
Returns
Anomaly Map.
Return type
Tensor
```
**GANomaly**

GANomaly: Semi-Supervised Anomaly Detection via Adversarial Training.

https://arxiv.org/abs/1805.06725

classanomalib.models.image.ganomaly.lightning_model.Ganomaly( _batch_size=32_ , _n_features=64_ ,
_latent_vec_size=100_ ,
_extra_layers=0_ ,
_add_final_conv_layer=True_ ,
_wadv=1_ , _wcon=50_ , _wenc=1_ ,
_lr=0.0002_ , _beta1=0.5_ ,
_beta2=0.999_ )
Bases:AnomalyModule
PL Lightning Module for the GANomaly Algorithm.
**Parameters**

- batch_size(int) – Batch size. Defaults to 32.
- n_features(int) – Number of features layers in the CNNs. Defaults to 64.
- latent_vec_size(int) – Size of autoencoder latent vector. Defaults to 100.
- extra_layers(int, optional) – Number of extra layers for encoder/decoder. Defaults
    to 0.

**136 Chapter 3. Guides**


- add_final_conv_layer(bool, optional) – Add convolution layer at the end. Defaults
    toTrue.
- wadv(int, optional) – Weight for adversarial loss. Defaults to 1.
- wcon(int, optional) – Image regeneration weight. Defaults to 50.
- wenc(int, optional) – Latent vector encoder weight. Defaults to 1.
- lr(float, optional) – Learning rate. Defaults to0.0002.
- beta1(float, optional) – Adam beta1. Defaults to0.5.
- beta2(float, optional) – Adam beta2. Defaults to0.999.
configure_optimizers()
Configure optimizers for each decoder.
**Returns**
Adam optimizer for each decoder
**Return type**
Optimizer
property learning_type: LearningType
Return the learning type of the model.
**Returns**
Learning type of the model.
**Return type**
LearningType
on_test_batch_end( _outputs_ , _batch_ , _batch_idx_ , _dataloader_idx=0_ )
Normalize outputs based on min/max values.
**Return type**
None
on_test_start()
Reset min max values before test batch starts.
**Return type**
None
on_validation_batch_end( _outputs_ , _batch_ , _batch_idx_ , _dataloader_idx=0_ )
Normalize outputs based on min/max values.
**Return type**
None
on_validation_start()
Reset min and max values for current validation epoch.
**Return type**
None
test_step( _batch_ , _batch_idx_ , _*args_ , _**kwargs_ )
Update min and max scores from the current step.
**Return type**
Union[Tensor,Mapping[str,Any],None]

**3.3. Reference Guide 137**


```
property trainer_arguments: dict[str, Any]
Return GANomaly trainer arguments.
training_step( batch , batch_idx )
Perform the training step.
Parameters
```
- batch(dict[str, str | torch.Tensor]) – Input batch containing images.
- batch_idx(int) – Batch index.
- optimizer_idx(int) – Optimizer which is being called for current training step.
**Returns**
Loss
**Return type**
STEP_OUTPUT
validation_step( _batch_ , _*args_ , _**kwargs_ )
Update min and max scores from the current step.
**Parameters**
- batch (dict[str, str | torch.Tensor]) – Predicted difference between z and
z_hat.
- args– Additional arguments.
- kwargs– Additional keyword arguments.
**Returns**
Output predictions.
**Return type**
(STEP_OUTPUT)

Torch models defining encoder, decoder, Generator and Discriminator.

Code adapted from https://github.com/samet-akcay/ganomaly.

classanomalib.models.image.ganomaly.torch_model.GanomalyModel( _input_size_ , _num_input_channels_ ,
_n_features_ , _latent_vec_size_ ,
_extra_layers=0_ ,
_add_final_conv_layer=True_ )
Bases:Module
Ganomaly Model.
**Parameters**

- input_size(tuple[int, int]) – Input dimension.
- num_input_channels(int) – Number of input channels.
- n_features(int) – Number of features layers in the CNNs.
- latent_vec_size(int) – Size of autoencoder latent vector.
- extra_layers(int, optional) – Number of extra layers for encoder/decoder. Defaults
    to 0.
- add_final_conv_layer(bool, optional) – Add convolution layer at the end. Defaults
    toTrue.

**138 Chapter 3. Guides**


```
forward( batch )
Get scores for batch.
Parameters
batch(torch.Tensor) – Images
Returns
Regeneration scores.
Return type
Tensor
static weights_init( module )
Initialize DCGAN weights.
Parameters
module(nn.Module) – [description]
Return type
None
```
Loss function for the GANomaly Model Implementation.

classanomalib.models.image.ganomaly.loss.DiscriminatorLoss

```
Bases:Module
Discriminator loss for the GANomaly model.
forward( pred_real , pred_fake )
Compute the loss for a predicted batch.
Parameters
```
- pred_real(torch.Tensor) – Discriminator predictions for the real image.
- pred_fake(torch.Tensor) – Discriminator predictions for the fake image.
**Returns**
The computed discriminator loss.
**Return type**
Tensor

classanomalib.models.image.ganomaly.loss.GeneratorLoss( _wadv=1_ , _wcon=50_ , _wenc=1_ )

```
Bases:Module
Generator loss for the GANomaly model.
Parameters
```
- wadv(int, optional) – Weight for adversarial loss. Defaults to 1.
- wcon(int, optional) – Image regeneration weight. Defaults to 50.
- wenc(int, optional) – Latent vector encoder weight. Defaults to 1.
forward( _latent_i_ , _latent_o_ , _images_ , _fake_ , _pred_real_ , _pred_fake_ )
Compute the loss for a batch.
**Parameters**
- latent_i(torch.Tensor) – Latent features of the first encoder.
- latent_o(torch.Tensor) – Latent features of the second encoder.
- images(torch.Tensor) – Real image that served as input of the generator.

**3.3. Reference Guide 139**


- fake(torch.Tensor) – Generated image.
- pred_real(torch.Tensor) – Discriminator predictions for the real image.
- pred_fake(torch.Tensor) – Discriminator predictions for the fake image.
**Returns**
The computed generator loss.
**Return type**
Tensor

**Padim**

PaDiM: a Patch Distribution Modeling Framework for Anomaly Detection and Localization.

Paper https://arxiv.org/abs/2011.08785

classanomalib.models.image.padim.lightning_model.Padim( _backbone='resnet18'_ , _layers=['layer1',
'layer2','layer3']_ , _pre_trained=True_ ,
_n_features=None_ )
Bases:MemoryBankMixin,AnomalyModule
PaDiM: a Patch Distribution Modeling Framework for Anomaly Detection and Localization.
**Parameters**

- backbone(str) – Backbone CNN network Defaults toresnet18.
- layers(list[str]) – Layers to extract features from the backbone CNN Defaults to[
    "layer1", "layer2", "layer3"].
- pre_trained(bool, optional) – Boolean to check whether to use a pre_trained back-
    bone. Defaults toTrue.
- n_features(int, optional) – Number of features to retain in the dimension reduction
    step. Default values from the paper are available for: resnet18 (100), wide_resnet50_2 (550).
    Defaults toNone.
static configure_optimizers()
PADIM doesn’t require optimization, therefore returns no optimizers.
**Return type**
None
configure_transforms( _image_size=None_ )
Default transform for Padim.
**Return type**
Transform
fit()
Fit a Gaussian to the embedding collected from the training set.
**Return type**
None
property learning_type: LearningType
Return the learning type of the model.
**Returns**
Learning type of the model.

**140 Chapter 3. Guides**


```
Return type
LearningType
property trainer_arguments: dict[str, int | float]
Return PADIM trainer arguments.
Since the model does not require training, we limit the max_epochs to 1. Since we need to run training
epoch before validation, we also set the sanity steps to 0
training_step( batch , *args , **kwargs )
Perform the training step of PADIM. For each batch, hierarchical features are extracted from the CNN.
Parameters
```
- batch(dict[str, str | torch.Tensor]) – Batch containing image filename, image,
    label and mask
- args– Additional arguments.
- kwargs– Additional keyword arguments.
**Return type**
None
**Returns**
Hierarchical feature map
validation_step( _batch_ , _*args_ , _**kwargs_ )
Perform a validation step of PADIM.
Similar to the training step, hierarchical features are extracted from the CNN for each batch.
**Parameters**
- batch(dict[str, str | torch.Tensor]) – Input batch
- args– Additional arguments.
- kwargs– Additional keyword arguments.
**Return type**
Union[Tensor,Mapping[str,Any],None]
**Returns**
Dictionary containing images, features, true labels and masks. These are required in _valida-
tion_epoch_end_ for feature concatenation.

PyTorch model for the PaDiM model implementation.

classanomalib.models.image.padim.torch_model.PadimModel( _layers_ , _backbone='resnet18'_ ,
_pre_trained=True_ , _n_features=None_ )
Bases:Module
Padim Module.
**Parameters**

- layers(list[str]) – Layers used for feature extraction
- backbone(str, optional) – Pre-trained model backbone. Defaults to “resnet18”. De-
    faults toresnet18.
- pre_trained(bool, optional) – Boolean to check whether to use a pre_trained back-
    bone. Defaults toTrue.

**3.3. Reference Guide 141**


- n_features(int, optional) – Number of features to retain in the dimension reduction
    step. Default values from the paper are available for: resnet18 (100), wide_resnet50_2 (550).
    Defaults toNone.
forward( _input_tensor_ )
Forward-pass image-batch (N, C, H, W) into model to extract features.
**Parameters**
- input_tensor(Tensor) – Image-batch (N, C, H, W)
- input_tensor– torch.Tensor:
**Return type**
Tensor
**Returns**
Features from single/multiple layers.

```
Example
```
```
>>>x = torch.randn(32, 3, 224, 224)
>>>features = self.extract_features(input_tensor)
>>>features.keys()
dict_keys(['layer1', 'layer2','layer3'])
```
```
>>>[v.shapeforvin features.values()]
[torch.Size([32, 64, 56, 56]),
torch.Size([32, 128, 28, 28]),
torch.Size([32, 256, 14, 14])]
```
```
generate_embedding( features )
Generate embedding from hierarchical feature map.
Parameters
features (dict[str, torch.Tensor]) – Hierarchical feature map from a CNN
(ResNet18 or WideResnet)
Return type
Tensor
Returns
Embedding vector
```
**PatchCore**

Towards Total Recall in Industrial Anomaly Detection.

Paper https://arxiv.org/abs/2106.08265.

classanomalib.models.image.patchcore.lightning_model.Patchcore( _backbone='wide_resnet50_2'_ ,
_layers=('layer2','layer3')_ ,
_pre_trained=True_ ,
_coreset_sampling_ratio=0.1_ ,
_num_neighbors=9_ )

**142 Chapter 3. Guides**


```
Bases:MemoryBankMixin,AnomalyModule
PatchcoreLightning Module to train PatchCore algorithm.
Parameters
```
- backbone(str) – Backbone CNN network Defaults towide_resnet50_2.
- layers(list[str]) – Layers to extract features from the backbone CNN Defaults to[
    "layer2", "layer3"].
- pre_trained(bool, optional) – Boolean to check whether to use a pre_trained back-
    bone. Defaults toTrue.
- coreset_sampling_ratio(float, optional) – Coreset sampling ratio to subsample
    embedding. Defaults to0.1.
- num_neighbors(int, optional) – Number of nearest neighbors. Defaults to 9.
configure_optimizers()
Configure optimizers.
**Returns**
Do not set optimizers by returning None.
**Return type**
None
configure_transforms( _image_size=None_ )
Default transform for Padim.
**Return type**
Transform
fit()
Apply subsampling to the embedding collected from the training set.
**Return type**
None
property learning_type: LearningType
Return the learning type of the model.
**Returns**
Learning type of the model.
**Return type**
LearningType
property trainer_arguments: dict[str, Any]
Return Patchcore trainer arguments.
training_step( _batch_ , _*args_ , _**kwargs_ )
Generate feature embedding of the batch.
**Parameters**
- batch(dict[str, str | torch.Tensor]) – Batch containing image filename, image,
label and mask
- args– Additional arguments.
- kwargs– Additional keyword arguments.

**3.3. Reference Guide 143**


```
Returns
Embedding Vector
Return type
dict[str, np.ndarray]
validation_step( batch , *args , **kwargs )
Get batch of anomaly maps from input image batch.
Parameters
```
- batch(dict[str, str | torch.Tensor]) – Batch containing image filename, image,
    label and mask
- args– Additional arguments.
- kwargs– Additional keyword arguments.
**Returns**
Image filenames, test images, GT and predicted label/masks
**Return type**
dict[str, Any]

PyTorch model for the PatchCore model implementation.

classanomalib.models.image.patchcore.torch_model.PatchcoreModel( _layers_ ,
_backbone='wide_resnet50_2'_ ,
_pre_trained=True_ ,
_num_neighbors=9_ )
Bases:DynamicBufferMixin,Module
Patchcore Module.
**Parameters**

- layers(list[str]) – Layers used for feature extraction
- backbone(str, optional) – Pre-trained model backbone. Defaults toresnet18.
- pre_trained(bool, optional) – Boolean to check whether to use a pre_trained back-
    bone. Defaults toTrue.
- num_neighbors(int, optional) – Number of nearest neighbors. Defaults to 9.
compute_anomaly_score( _patch_scores_ , _locations_ , _embedding_ )
Compute Image-Level Anomaly Score.
**Parameters**
- patch_scores(torch.Tensor) – Patch-level anomaly scores
- locations(Tensor) – Memory bank locations of the nearest neighbor for each patch
location
- embedding(Tensor) – The feature embeddings that generated the patch scores
**Returns**
Image-level anomaly scores
**Return type**
Tensor

**144 Chapter 3. Guides**


```
static euclidean_dist( x , y )
Calculate pair-wise distance between row vectors in x and those in y.
Replaces torch cdist with p=2, as cdist is not properly exported to onnx and openvino format. Resulting
matrix is indexed by x vectors in rows and y vectors in columns.
Parameters
```
- x(Tensor) – input tensor 1
- y(Tensor) – input tensor 2
**Return type**
Tensor
**Returns**
Matrix of distances between row vectors in x and y.
forward( _input_tensor_ )
Return Embedding during training, or a tuple of anomaly map and anomaly score during testing.
Steps performed: 1. Get features from a CNN. 2. Generate embedding based on the features. 3. Compute
anomaly map in test mode.
**Parameters**
input_tensor(torch.Tensor) – Input tensor
**Returns**
Embedding for training, anomaly map and anomaly score for testing.
**Return type**
Tensor | dict[str, torch.Tensor]
generate_embedding( _features_ )
Generate embedding from hierarchical feature map.
**Parameters**
- features(dict[str,Tensor]) – Hierarchical feature map from a CNN (ResNet18 or
WideResnet)
- features– dict[str:Tensor]:
**Return type**
Tensor
**Returns**
Embedding vector
nearest_neighbors( _embedding_ , _n_neighbors_ )
Nearest Neighbours using brute force method and euclidean norm.
**Parameters**
- embedding(torch.Tensor) – Features to compare the distance with the memory bank.
- n_neighbors(int) – Number of neighbors to look at
**Returns**
Patch scores. Tensor: Locations of the nearest neighbor(s).
**Return type**
Tensor

**3.3. Reference Guide 145**


```
static reshape_embedding( embedding )
Reshape Embedding.
Reshapes Embedding to the following format:
```
- [Batch, Embedding, Patch, Patch] to [Batch*Patch*Patch, Embedding]

```
Parameters
embedding(torch.Tensor) – Embedding tensor extracted from CNN features.
Returns
Reshaped embedding tensor.
Return type
Tensor
```
```
subsample_embedding( embedding , sampling_ratio )
Subsample embedding based on coreset sampling and store to memory.
Parameters
```
- embedding(np.ndarray) – Embedding tensor from the CNN
- sampling_ratio(float) – Coreset sampling ratio
**Return type**
None

**Reverse Distillation**

Anomaly Detection via Reverse Distillation from One-Class Embedding.

https://arxiv.org/abs/2201.10703v2

classanomalib.models.image.reverse_distillation.lightning_model.ReverseDistillation( _backbone='wide_resnet50_2'_ ,
_lay-
ers=('layer1',
'layer2',
'layer3')_ ,
_anomaly_map_mode=AnomalyMapGenerationMode.ADD_ ,
_pre_trained=True_ )
Bases:AnomalyModule
PL Lightning Module for Reverse Distillation Algorithm.
**Parameters**

- backbone(str) – Backbone of CNN network Defaults towide_resnet50_2.
- layers(list[str]) – Layers to extract features from the backbone CNN Defaults to[
    "layer1", "layer2", "layer3"].
- anomaly_map_mode(AnomalyMapGenerationMode, optional) – Mode to generate
    anomaly map. Defaults toAnomalyMapGenerationMode.ADD.
- pre_trained(bool, optional) – Boolean to check whether to use a pre_trained back-
    bone. Defaults toTrue.

**146 Chapter 3. Guides**


```
configure_optimizers()
Configure optimizers for decoder and bottleneck.
Returns
Adam optimizer for each decoder
Return type
Optimizer
property learning_type: LearningType
Return the learning type of the model.
Returns
Learning type of the model.
Return type
LearningType
property trainer_arguments: dict[str, Any]
Return Reverse Distillation trainer arguments.
training_step( batch , *args , **kwargs )
Perform a training step of Reverse Distillation Model.
Features are extracted from three layers of the Encoder model. These are passed to the bottleneck layer that
are passed to the decoder network. The loss is then calculated based on the cosine similarity between the
encoder and decoder features.
Parameters
```
- (batch(batch) – dict[str, str | torch.Tensor]): Input batch
- args– Additional arguments.
- kwargs– Additional keyword arguments.
**Return type**
Union[Tensor,Mapping[str,Any],None]
**Returns**
Feature Map
validation_step( _batch_ , _*args_ , _**kwargs_ )
Perform a validation step of Reverse Distillation Model.
Similar to the training step, encoder/decoder features are extracted from the CNN for each batch, and
anomaly map is computed.
**Parameters**
- batch(dict[str, str | torch.Tensor]) – Input batch
- args– Additional arguments.
- kwargs– Additional keyword arguments.
**Return type**
Union[Tensor,Mapping[str,Any],None]
**Returns**
Dictionary containing images, anomaly maps, true labels and masks. These are required in
_validation_epoch_end_ for feature concatenation.

PyTorch model for Reverse Distillation.

**3.3. Reference Guide 147**


classanomalib.models.image.reverse_distillation.torch_model.ReverseDistillationModel( _backbone_ ,
_in-
put_size_ ,
_lay-
ers_ ,
_anomaly_map_mode_ ,
_pre_trained=True_ )
Bases:Module
Reverse Distillation Model.
**To reproduce results in the paper, use torchvision model for the encoder:**
self.encoder = torchvision.models.wide_resnet50_2(pretrained=True)

```
Parameters
```
- backbone(str) – Name of the backbone used for encoder and decoder.
- input_size(tuple[int, int]) – Size of input image.
- layers(list[str]) – Name of layers from which the features are extracted.
- anomaly_map_mode(str) – Mode used to generate anomaly map. Options are between
    multiplyandadd.
- pre_trained(bool, optional) – Boolean to check whether to use a pre_trained back-
    bone. Defaults toTrue.

```
forward( images )
Forward-pass images to the network.
During the training mode the model extracts features from encoder and decoder networks. During evalua-
tion mode, it returns the predicted anomaly map.
Parameters
images(torch.Tensor) – Batch of images
Returns
Encoder and decoder features
in training mode, else anomaly maps.
Return type
torch.Tensor | list[torch.Tensor] | tuple[list[torch.Tensor]]
```
Loss function for Reverse Distillation.

classanomalib.models.image.reverse_distillation.loss.ReverseDistillationLoss( _*args_ ,
_**kwargs_ )
Bases:Module
Loss function for Reverse Distillation.
forward( _encoder_features_ , _decoder_features_ )
Compute cosine similarity loss based on features from encoder and decoder.
Based on the official code: https://github.com/hq-deng/RD4AD/blob/
6554076872c65f8784f6ece8cfb39ce77e1aee12/main.py#L33C25-L33C25 Calculates loss from flat-
tened arrays of features, see https://github.com/hq-deng/RD4AD/issues/22
**Parameters**

**148 Chapter 3. Guides**


- encoder_features(list[torch.Tensor]) – List of features extracted from encoder
- decoder_features(list[torch.Tensor]) – List of features extracted from decoder
**Returns**
Cosine similarity loss
**Return type**
Tensor

Compute Anomaly map.

classanomalib.models.image.reverse_distillation.anomaly_map.AnomalyMapGenerationMode( _value_ ,
_names=None_ ,
_*_ ,
_mod-
ule=None_ ,
_qual-
name=None_ ,
_type=None_ ,
_start=1_ ,
_bound-
ary=None_ )
Bases:str,Enum
Type of mode when generating anomaly imape.

classanomalib.models.image.reverse_distillation.anomaly_map.AnomalyMapGenerator( _image_size_ ,
_sigma=4_ ,
_mode=AnomalyMapGenerationMode.MULTIPLY_ )
Bases:Module
Generate Anomaly Heatmap.
**Parameters**

- image_size (ListConfig, tuple) – Size of original image used for upscaling the
    anomaly map.
- sigma(int) – Standard deviation of the gaussian kernel used to smooth anomaly map. De-
    faults to 4.
- mode (AnomalyMapGenerationMode, optional) – Operation used to gen-
    erate anomaly map. Options are AnomalyMapGenerationMode.ADD and
    AnomalyMapGenerationMode.MULTIPLY. Defaults toAnomalyMapGenerationMode.
    MULTIPLY.
**Raises**
ValueError– In case modes other than multiply and add are passed.
forward( _student_features_ , _teacher_features_ )
Compute anomaly map given encoder and decoder features.
**Parameters**
- student_features(list[torch.Tensor]) – List of encoder features
- teacher_features(list[torch.Tensor]) – List of decoder features
**Returns**
Anomaly maps of length batch.

**3.3. Reference Guide 149**


```
Return type
Tensor
```
### R-KDE

Region Based Anomaly Detection With Real-Time Training and Analysis.

https://ieeexplore.ieee.org/abstract/document/8999287

classanomalib.models.image.rkde.lightning_model.Rkde( _roi_stage=RoiStage.RCNN_ ,
_roi_score_threshold=0.001_ ,
_min_box_size=25_ , _iou_threshold=0.3_ ,
_max_detections_per_image=100_ ,
_n_pca_components=16_ , _fea-
ture_scaling_method=FeatureScalingMethod.SCALE_ ,
_max_training_points=40000_ )
Bases:MemoryBankMixin,AnomalyModule
Region Based Anomaly Detection With Real-Time Training and Analysis.
**Parameters**

- roi_stage(RoiStage, optional) – Processing stage from which rois are extracted. De-
    faults toRoiStage.RCNN.
- roi_score_threshold(float, optional) – Mimumum confidence score for the region
    proposals. Defaults to0.001.
- min_size(int, optional) – Minimum size in pixels for the region proposals. Defaults
    to 25.
- iou_threshold(float, optional) – Intersection-Over-Union threshold used during
    NMS. Defaults to0.3.
- max_detections_per_image(int, optional) – Maximum number of region proposals
    per image. Defaults to 100.
- n_pca_components(int, optional) – Number of PCA components. Defaults to 16.
- feature_scaling_method(FeatureScalingMethod, optional) – Scaling method
    applied to features before passing to KDE. Options are _norm_ (normalize to unit vector length)
    and _scale_ (scale to max length observed in training). Defaults toFeatureScalingMethod.
    SCALE.
- max_training_points(int, optional) – Maximum number of training points to fit the
    KDE model. Defaults to 40000.
static configure_optimizers()
RKDE doesn’t require optimization, therefore returns no optimizers.
**Return type**
None
fit()
Fit a KDE Model to the embedding collected from the training set.
**Return type**
None

**150 Chapter 3. Guides**


```
property learning_type: LearningType
Return the learning type of the model.
Returns
Learning type of the model.
Return type
LearningType
property trainer_arguments: dict[str, Any]
Return R-KDE trainer arguments.
Returns
Arguments for the trainer.
Return type
dict[str, Any]
training_step( batch , *args , **kwargs )
Perform a training Step of RKDE. For each batch, features are extracted from the CNN.
Parameters
```
- batch(dict[str, str | torch.Tensor]) – Batch containing image filename, image,
    label and mask
- args– Additional arguments.
- kwargs– Additional keyword arguments.
**Return type**
None
**Returns**
Deep CNN features.
validation_step( _batch_ , _*args_ , _**kwargs_ )
Perform a validation Step of RKde.
Similar to the training step, features are extracted from the CNN for each batch.
**Parameters**
- batch(dict[str, str | torch.Tensor]) – Batch containing image filename, image,
label and mask
- args– Additional arguments.
- kwargs– Additional keyword arguments.
**Return type**
Union[Tensor,Mapping[str,Any],None]
**Returns**
Dictionary containing probability, prediction and ground truth values.

Torch model for region-based anomaly detection.

classanomalib.models.image.rkde.torch_model.RkdeModel( _roi_stage=RoiStage.RCNN_ ,
_roi_score_threshold=0.001_ ,
_min_box_size=25_ , _iou_threshold=0.3_ ,
_max_detections_per_image=100_ ,
_n_pca_components=16_ , _fea-
ture_scaling_method=FeatureScalingMethod.SCALE_ ,
_max_training_points=40000_ )

**3.3. Reference Guide 151**


```
Bases:Module
Torch Model for the Region-based Anomaly Detection Model.
Parameters
```
- roi_stage(RoiStage, optional) – Processing stage from which rois are extracted. De-
    faults toRoiStage.RCNN.
- roi_score_threshold(float, optional) – Mimumum confidence score for the region
    proposals. Defaults to0.001.
- min_size(int, optional) – Minimum size in pixels for the region proposals. Defaults
    to 25.
- iou_threshold(float, optional) – Intersection-Over-Union threshold used during
    NMS. Defaults to0.3.
- max_detections_per_image(int, optional) – Maximum number of region proposals
    per image. Defaults to 100.
- n_pca_components(int, optional) – Number of PCA components. Defaults to 16.
- feature_scaling_method(FeatureScalingMethod, optional) – Scaling method
    applied to features before passing to KDE. Options are _norm_ (normalize to unit vector length)
    and _scale_ (scale to max length observed in training). Defaults toFeatureScalingMethod.
    SCALE.
- max_training_points(int, optional) – Maximum number of training points to fit the
    KDE model. Defaults to 40000.
fit( _embeddings_ )
Fit the model using a set of collected embeddings.
**Parameters**
embeddings(torch.Tensor) – Input embeddings to fit the model.
**Return type**
bool
**Returns**
Boolean confirming whether the training is successful.
forward( _batch_ )
Prediction by normality model.
**Parameters**
batch(torch.Tensor) – Input images.
**Returns
The extracted features (when in training mode),**
or the predicted rois and corresponding anomaly scores.
**Return type**
Tensor | tuple[torch.Tensor, torch.Tensor]

Region-based Anomaly Detection with Real Time Training and Analysis.

Feature Extractor.

**152 Chapter 3. Guides**


classanomalib.models.image.rkde.feature_extractor.FeatureExtractor

```
Bases:Module
Feature Extractor module for Region-based anomaly detection.
forward( batch , rois )
Perform a forward pass of the feature extractor.
Parameters
```
- batch(torch.Tensor) – Batch of input images of shape [B, C, H, W].
- rois(torch.Tensor) – torch.Tensor of shape [N, 5] describing the regions-of-interest in
    the batch.
**Returns**
torch.Tensor containing a 4096-dimensional feature vector for every RoI location.
**Return type**
Tensor

Region-based Anomaly Detection with Real Time Training and Analysis.

Region Extractor.

classanomalib.models.image.rkde.region_extractor.RegionExtractor( _stage=RoiStage.RCNN_ ,
_score_threshold=0.001_ ,
_min_size=25_ ,
_iou_threshold=0.3_ ,
_max_detections_per_image=100_ )
Bases:Module
Extracts regions from the image.
**Parameters**

- stage(RoiStage, optional) – Processing stage from which rois are extracted. Defaults
    toRoiStage.RCNN.
- score_threshold(float, optional) – Mimumum confidence score for the region pro-
    posals. Defaults to0.001.
- min_size(int, optional) – Minimum size in pixels for the region proposals. Defaults
    to 25.
- iou_threshold(float, optional) – Intersection-Over-Union threshold used during
    NMS. Defaults to0.3.
- max_detections_per_image(int, optional) – Maximum number of region proposals
    per image. Defaults to 100.
forward( _batch_ )
Forward pass of the model.
**Parameters**
batch(torch.Tensor) – Batch of input images of shape [B, C, H, W].
**Raises**
ValueError– Whenstageis not one ofrcnnorrpn.
**Returns**

**3.3. Reference Guide 153**


```
Predicted regions, tensor of shape [N, 5] where N is the number of predicted regions in
the batch,
and where each row describes the index of the image in the batch and the 4 bounding box
coordinates.
Return type
Tensor
post_process_box_predictions( pred_boxes , pred_scores )
Post-processes the box predictions.
The post-processing consists of removing small boxes, applying nms, and keeping only the k boxes with
the highest confidence score.
Parameters
```
- pred_boxes(torch.Tensor) – Box predictions of shape (N, 4).
- pred_scores(torch.Tensor) – torch.Tensor of shape () with a confidence score for each
    box prediction.
**Returns**
Post-processed box predictions of shape (N, 4).
**Return type**
list[torch.Tensor]

classanomalib.models.image.rkde.region_extractor.RoiStage( _value_ , _names=None_ , _*_ , _module=None_ ,
_qualname=None_ , _type=None_ , _start=1_ ,
_boundary=None_ )
Bases:str,Enum
Processing stage from which rois are extracted.

### STFPM

STFPM: Student-Teacher Feature Pyramid Matching for Unsupervised Anomaly Detection.

https://arxiv.org/abs/2103.04257

classanomalib.models.image.stfpm.lightning_model.Stfpm( _backbone='resnet18'_ , _layers=('layer1',
'layer2','layer3')_ )
Bases:AnomalyModule
PL Lightning Module for the STFPM algorithm.
**Parameters**

- backbone(str) – Backbone CNN network Defaults toresnet18.
- layers(list[str]) – Layers to extract features from the backbone CNN Defaults to[
    "layer1", "layer2", "layer3"].
configure_optimizers()
Configure optimizers.
**Returns**
SGD optimizer
**Return type**
Optimizer

**154 Chapter 3. Guides**


```
property learning_type: LearningType
Return the learning type of the model.
Returns
Learning type of the model.
Return type
LearningType
property trainer_arguments: dict[str, Any]
Required trainer arguments.
training_step( batch , *args , **kwargs )
Perform a training step of STFPM.
For each batch, teacher and student and teacher features are extracted from the CNN.
Parameters
```
- batch(dict[str, str | torch.Tensor]) – Input batch.
- args– Additional arguments.
- kwargs– Additional keyword arguments.
**Return type**
Union[Tensor,Mapping[str,Any],None]
**Returns**
Loss value
validation_step( _batch_ , _*args_ , _**kwargs_ )
Perform a validation Step of STFPM.
Similar to the training step, student/teacher features are extracted from the CNN for each batch, and anomaly
map is computed.
**Parameters**
- batch(dict[str, str | torch.Tensor]) – Input batch
- args– Additional arguments
- kwargs– Additional keyword arguments
**Return type**
Union[Tensor,Mapping[str,Any],None]
**Returns**
Dictionary containing images, anomaly maps, true labels and masks. These are required in
_validation_epoch_end_ for feature concatenation.

PyTorch model for the STFPM model implementation.

classanomalib.models.image.stfpm.torch_model.STFPMModel( _layers_ , _backbone='resnet18'_ )

```
Bases:Module
STFPM: Student-Teacher Feature Pyramid Matching for Unsupervised Anomaly Detection.
Parameters
```
- layers(list[str]) – Layers used for feature extraction.
- backbone(str, optional) – Pre-trained model backbone. Defaults toresnet18.

**3.3. Reference Guide 155**


```
forward( images )
Forward-pass images into the network.
During the training mode the model extracts the features from the teacher and student networks. During
the evaluation mode, it returns the predicted anomaly map.
Parameters
images(torch.Tensor) – Batch of images.
Return type
Tensor|dict[str,Tensor] |tuple[dict[str,Tensor]]
Returns
Teacher and student features when in training mode, otherwise the predicted anomaly maps.
```
Loss function for the STFPM Model Implementation.

classanomalib.models.image.stfpm.loss.STFPMLoss

```
Bases:Module
Feature Pyramid Loss This class implmenents the feature pyramid loss function proposed in STFPM paper.
```
```
Example
```
```
>>> from anomalib.models.components.feature_extractors importTimmFeatureExtractor
>>> from anomalib.models.stfpm.loss importSTFPMLoss
>>> from torchvision.models importresnet18
```
```
>>>layers = ['layer1','layer2', 'layer3']
>>>teacher_model = TimmFeatureExtractor(model=resnet18(pretrained=True),␣
˓→layers=layers)
>>>student_model = TimmFeatureExtractor(model=resnet18(pretrained=False),␣
˓→layers=layers)
>>>loss = Loss()
```
```
>>>inp = torch.rand((4, 3, 256, 256))
>>>teacher_features = teacher_model(inp)
>>>student_features = student_model(inp)
>>>loss(student_features, teacher_features)
tensor(51.2015, grad_fn=<SumBackward0>)
```
```
compute_layer_loss( teacher_feats , student_feats )
Compute layer loss based on Equation (1) in Section 3.2 of the paper.
Parameters
```
- teacher_feats(torch.Tensor) – Teacher features
- student_feats(torch.Tensor) – Student features
**Return type**
Tensor
**Returns**
L2 distance between teacher and student features.

**156 Chapter 3. Guides**


```
forward( teacher_features , student_features )
Compute the overall loss via the weighted average of the layer losses computed by the cosine similarity.
Parameters
```
- teacher_features(dict[str, torch.Tensor]) – Teacher features
- student_features(dict[str, torch.Tensor]) – Student features
**Return type**
Tensor
**Returns**
Total loss, which is the weighted average of the layer losses.

Anomaly Map Generator for the STFPM model implementation.

classanomalib.models.image.stfpm.anomaly_map.AnomalyMapGenerator

```
Bases:Module
Generate Anomaly Heatmap.
compute_anomaly_map( teacher_features , student_features , image_size )
Compute the overall anomaly map via element-wise production the interpolated anomaly maps.
Parameters
```
- teacher_features(dict[str, torch.Tensor]) – Teacher features
- student_features(dict[str, torch.Tensor]) – Student features
- image_size(tuple[int, int]) – Image size to which the anomaly map should be re-
    sized.
**Return type**
Tensor
**Returns**
Final anomaly map
compute_layer_map( _teacher_features_ , _student_features_ , _image_size_ )
Compute the layer map based on cosine similarity.
**Parameters**
- teacher_features(torch.Tensor) – Teacher features
- student_features(torch.Tensor) – Student features
- image_size(tuple[int, int]) – Image size to which the anomaly map should be re-
sized.
**Return type**
Tensor
**Returns**
Anomaly score based on cosine similarity.
forward( _**kwargs_ )
Return anomaly map.
Expects _teach_features_ and _student_features_ keywords to be passed explicitly.
**Parameters**
kwargs(dict[str, torch.Tensor]) – Keyword arguments

**3.3. Reference Guide 157**


```
Example
```
```
>>>anomaly_map_generator = AnomalyMapGenerator(image_size=tuple(hparams.model.
˓→input_size))
>>>output = self.anomaly_map_generator(
teacher_features=teacher_features,
student_features=student_features
)
```
```
Raises
ValueError– teach_features and student_features keys are not found
Returns
anomaly map
Return type
torch.Tensor
```
**U-Flow**

This is the implementation of the U-Flow paper.

Model Type: Segmentation

**Description**

U-Flow is a U-Shaped normalizing flow-based probability distribution estimator. The method consists of three phases.
(1) Multi-scale feature extraction: a rich multi-scale representation is obtained with MSCaiT, by combining pre-trained
image Transformers acting at different image scales. It can also be used any other feature extractor, such as ResNet. (2)
U-shaped Normalizing Flow: by adapting the widely used U-like architecture to NFs, a fully invertible architecture is
designed. This architecture is capable of merging the information from different scales while ensuring independence
both intra- and inter-scales. To make it fully invertible, split and invertible up-sampling operations are used. (3)
Anomaly score and segmentation computation: besides generating the anomaly map based on the likelihood of test
data, we also propose to adapt the a contrario framework to obtain an automatic threshold by controlling the allowed
number of false alarms.

**Architecture**

```
U-Flow torch model.
```
**158 Chapter 3. Guides**


classanomalib.models.image.uflow.torch_model.AffineCouplingSubnet( _kernel_size_ ,
_subnet_channels_ratio_ )
Bases:object
Class for building the Affine Coupling subnet.
It is passed as an argument to the _AllInOneBlock_ module.
**Parameters**

- kernel_size(int) – Kernel size.
- subnet_channels_ratio(float) – Subnet channels ratio.

classanomalib.models.image.uflow.torch_model.UflowModel( _input_size=(448, 448)_ , _flow_steps=4_ ,
_backbone='mcait'_ , _affine_clamp=2.0_ ,
_affine_subnet_channels_ratio=1.0_ ,
_permute_soft=False_ )
Bases:Module
U-Flow model.
**Parameters**

- input_size(tuple[int, int]) – Input image size.
- flow_steps(int) – Number of flow steps.
- backbone(str) – Backbone name.
- affine_clamp(float) – Affine clamp.
- affine_subnet_channels_ratio(float) – Affine subnet channels ratio.
- permute_soft(bool) – Whether to use soft permutation.
build_flow( _flow_steps_ )
Build the flow model.
First we start with the input nodes, which have to match the feature extractor output. Then, we build the
U-Shaped flow. Starting from the bottom (the coarsest scale), the flow is built as follows:
1. Pass the input through a Flow Stage ( _build_flow_stage_ ).
2. Split the output of the flow stage into two parts, one that goes directly to the output,
3. and the other is up-sampled, and will be concatenated with the output of the next flow stage (next scale)
4. Repeat steps 1-3 for the next scale.
Finally, we build the Flow graph using the input nodes, the flow stages, and the output nodes.
**Parameters**
flow_steps(int) – Number of flow steps.
**Returns**
Flow model.
**Return type**
ff.GraphINN
build_flow_stage( _in_node_ , _flow_steps_ , _condition_node=None_ )
Build a flow stage, which is a sequence of flow steps.
Each flow stage is essentially a sequence of _flow_steps_ Glow blocks ( _AllInOneBlock_ ).

**3.3. Reference Guide 159**


```
Parameters
```
- in_node(ff.Node) – Input node.
- flow_steps(int) – Number of flow steps.
- condition_node(ff.Node) – Condition node.
**Returns**
List of flow steps.
**Return type**
List[ff.Node]
encode( _features_ )
Return
**Return type**
tuple[Tensor,Tensor]
forward( _image_ )
Return anomaly map.
**Return type**
Tensor

U-Flow: A U-shaped Normalizing Flow for Anomaly Detection with Unsupervised Threshold.

https://arxiv.org/pdf/2211.12353.pdf

classanomalib.models.image.uflow.lightning_model.Uflow( _backbone='mcait'_ , _flow_steps=4_ ,
_affine_clamp=2.0_ ,
_affine_subnet_channels_ratio=1.0_ ,
_permute_soft=False_ )
Bases:AnomalyModule
PL Lightning Module for the UFLOW algorithm.
configure_optimizers()
Return optimizer and scheduler.
**Return type**
tuple[list[LightningOptimizer],list[LRScheduler]]
configure_transforms( _image_size=None_ )
Default transform for Padim.
**Return type**
Transform
property learning_type: LearningType
Return the learning type of the model.
**Returns**
Learning type of the model.
**Return type**
LearningType
property trainer_arguments: dict[str, Any]
Return EfficientAD trainer arguments.

**160 Chapter 3. Guides**


```
training_step( batch , *args , **kwargs )
Training step.
Return type
Union[Tensor,Mapping[str,Any],None]
validation_step( batch , *args , **kwargs )
Validation step.
Return type
Union[Tensor,Mapping[str,Any],None]
```
UFlow Anomaly Map Generator Implementation.

classanomalib.models.image.uflow.anomaly_map.AnomalyMapGenerator( _input_size_ )

```
Bases:Module
Generate Anomaly Heatmap and segmentation.
static binomial_test( z , window_size , probability_thr , high_precision=False )
The binomial test applied to validate or reject the null hypothesis that the pixel is normal.
The null hypothesis is that the pixel is normal, and the alternative hypothesis is that the pixel is anomalous.
The binomial test is applied to a window around the pixel, and the number of pixels in the window that ares
anomalous is compared to the number of pixels that are expected to be anomalous under the null hypothesis.
Parameters
```
- z(Tensor) – Latent variable from the UFlow model. Tensor of shape (N, Cl, Hl, Wl),
    where N is the batch size, Cl is
- channels(the number of) –
- variables(and Hl and Wl are the height and width of the latent) –
- respectively.–
- window_size(int) – Window size for the binomial test.
- probability_thr(float) – Probability threshold for the binomial test.
- high_precision(bool) – Whether to use high precision for the binomial test.
**Return type**
Tensor
**Returns**
Log of the probability of the null hypothesis.
compute_anomaly_map( _latent_variables_ )
Generate a likelihood-based anomaly map, from latent variables.
**Parameters**
- latent_variables(list[Tensor]) – List of latent variables from the UFlow model.
Each element is a tensor of shape
- (N–
- Cl–
- Hl–
- Wl)–
- size(where N is the batch) –

**3.3. Reference Guide 161**


- channels(Cl is the number of) –
- and(and Hl and Wl are the height) –
- variables(width of the latent) –
- respectively–
- l.(for each scale) –
**Return type**
Tensor
**Returns**
Final Anomaly Map. Tensor of shape (N, 1, H, W), where N is the batch size, and H and W
are the height and width of the input image, respectively.
compute_anomaly_mask( _z_ , _window_size=7_ , _binomial_probability_thr=0.5_ , _high_precision=False_ )
This method is not used in the basic functionality of training and testing.
It is a bit slow, so we decided to leave it as an option for the user. It is included as it is part of the U-Flow
paper, and can be called separately if an unsupervised anomaly segmentation is needed.
Generate an anomaly mask, from latent variables. It is based on the NFA (Number of False Alarms) method,
which is a statistical method to detect anomalies. The NFA is computed as the log of the probability of
the null hypothesis, which is that all pixels are normal. First, we compute a list of candidate pixels, with
suspiciously high values of z^2, by applying a binomial test to each pixel, looking at a window around it.
Then, to compute the NFA values (actually the log-NFA), we evaluate how probable is that a pixel belongs
to the normal distribution. The null-hypothesis is that under normality assumptions, all candidate pixels
are uniformly distributed. Then, the detection is based on the concentration of candidate pixels.
**Parameters**
- z(list[torch.Tensor]) – List of latent variables from the UFlow model. Each element
is a tensor of shape (N, Cl, Hl, Wl), where N is the batch size, Cl is the number of channels,
and Hl and Wl are the height and width of the latent variables, respectively, for each scale
l.
- window_size(int) – Window size for the binomial test. Defaults to 7.
- binomial_probability_thr(float) – Probability threshold for the binomial test. De-
faults to 0.5
- high_precision(bool) – Whether to use high precision for the binomial test. Defaults
to False.
**Return type**
Tensor
**Returns**
Anomaly mask. Tensor of shape (N, 1, H, W), where N is the batch size, and H and W are
the height and width of the input image, respectively.
forward( _latent_variables_ )
Return anomaly map.
**Return type**
Tensor

**162 Chapter 3. Guides**


**WinCLIP**

WinCLIP: Zero-/Few-Shot Anomaly Classification and Segmentation.

Paper https://arxiv.org/abs/2303.14814

classanomalib.models.image.winclip.lightning_model.WinClip( _class_name=None_ , _k_shot=0_ ,
_scales=(2, 3)_ ,
_few_shot_source=None_ )
Bases:AnomalyModule
WinCLIP Lightning model.
**Parameters**

- class_name(str, optional) – The name of the object class used in the prompt ensemble.
    Defaults toNone.
- k_shot(int) – The number of reference images for few-shot inference. Defaults to 0.
- scales(tuple[int], optional) – The scales of the sliding windows used for multiscale
    anomaly detection. Defaults to(2, 3).
- few_shot_source(str | Path, optional) – Path to a folder of reference images used
    for few-shot inference. Defaults toNone.
collect_reference_images( _dataloader_ )
Collect reference images for few-shot inference.
The reference images are collected by iterating the training dataset until the required number of images are
collected.
**Returns**
A tensor containing the reference images.
**Return type**
ref_images (Tensor)
static configure_optimizers()
WinCLIP doesn’t require optimization, therefore returns no optimizers.
**Return type**
None
configure_transforms( _image_size=None_ )
Configure the default transforms used by the model.
**Return type**
Transform
property learning_type: LearningType
The learning type of the model.
WinCLIP is a zero-/few-shot model, depending on the user configuration. Therefore, the learning type
is set toLearningType.FEW_SHOTwhenk_shotis greater than zero andLearningType.ZERO_SHOT
otherwise.
load_state_dict( _state_dict_ , _strict=True_ )
Load the state dict of the model.
Before loading the state dict, we restore the parameters of the frozen backbone to ensure that the model is
loaded correctly. We also restore the auxiliary objects like threshold classes and normalization metrics.

**3.3. Reference Guide 163**


```
Return type
Any
state_dict()
Return the state dict of the model.
Before returning the state dict, we remove the parameters of the frozen backbone to reduce the size of the
checkpoint.
Return type
OrderedDict[str,Any]
property trainer_arguments: dict[str, int | float]
Set model-specific trainer arguments.
validation_step( batch , *args , **kwargs )
Validation Step of WinCLIP.
Return type
dict
```
PyTorch model for the WinCLIP implementation.

classanomalib.models.image.winclip.torch_model.WinClipModel( _class_name=None_ ,
_reference_images=None_ , _scales=(2,
3)_ , _apply_transform=False_ )
Bases:DynamicBufferMixin,BufferListMixin,Module
PyTorch module that implements the WinClip model for image anomaly detection.
**Parameters**

- class_name(str, optional) – The name of the object class used in the prompt ensemble.
    Defaults toNone.
- reference_images(torch.Tensor, optional) – Tensor of shape(K, C, H, W)con-
    taining the reference images. Defaults toNone.
- scales(tuple[int], optional) – The scales of the sliding windows used for multi-scale
    anomaly detection. Defaults to(2, 3).
- apply_transform(bool, optional) – Whether to apply the default CLIP transform to
    the input images. Defaults toFalse.
clip
The CLIP model used for image and text encoding.
**Type**
CLIP
grid_size
The size of the feature map grid.
**Type**
tuple[int]
k_shot
The number of reference images used for few-shot anomaly detection.
**Type**
int

**164 Chapter 3. Guides**


```
scales
The scales of the sliding windows used for multi-scale anomaly detection.
Type
tuple[int]
masks
The masks representing the sliding window locations.
Type
list[torch.Tensor] | None
_text_embeddings
The text embeddings for the compositional prompt ensemble.
Type
torch.Tensor | None
_visual_embeddings
The multi-scale embeddings for the reference images.
Type
list[torch.Tensor] | None
_patch_embeddings
The patch embeddings for the reference images.
Type
torch.Tensor | None
encode_image( batch )
Encode the batch of images to obtain image embeddings, window embeddings, and patch embeddings.
The image embeddings and patch embeddings are obtained by passing the batch of images through the
model. The window embeddings are obtained by masking the feature map and passing it through the
transformer. A forward hook is used to retrieve the intermediate feature map and share computation between
the image and window embeddings.
Parameters
batch(torch.Tensor) – Batch of input images of shape(N, C, H, W).
Returns
A tuple containing the image embeddings, window embeddings, and patch embeddings re-
spectively.
Return type
Tuple[torch.Tensor, List[torch.Tensor], torch.Tensor]
```
```
Examples
```
```
>>>model = WinClipModel()
>>>model.prepare_masks()
>>>batch = torch.rand((1, 3, 240, 240))
>>>image_embeddings, window_embeddings, patch_embeddings = model.encode_
˓→image(batch)
>>>image_embeddings.shape
torch.Size([1, 640])
>>>[embedding.shapeforembeddingin window_embeddings]
(continues on next page)
```
**3.3. Reference Guide 165**


```
(continued from previous page)
[torch.Size([1, 196, 640]), torch.Size([1, 169, 640])]
>>>patch_embeddings.shape
torch.Size([1, 225, 896])
```
```
forward( batch )
Forward-pass through the model to obtain image and pixel scores.
Parameters
batch(torch.Tensor) – Batch of input images of shape(batch_size, C, H, W).
Returns
Tuple containing the image scores and pixel scores.
Return type
Tuple[torch.Tensor, torch.Tensor]
property patch_embeddings: Tensor
The patch embeddings used by the model.
setup( class_name=None , reference_images=None )
Setup WinCLIP.
WinCLIP’s setup stage consists of collecting the text and visual embeddings used during inference. The
following steps are performed, depending on the arguments passed to the model: - Collect text embeddings
for zero-shot inference. - Collect reference images for few-shot inference. The k_shot attribute is updated
based on the number of reference images.
The setup method is called internally by the constructor. However, it can also be called manually to update
the text and visual embeddings after the model has been initialized.
Parameters
```
- class_name(str) – The name of the object class used in the prompt ensemble.
- reference_images(torch.Tensor) – Tensor of shape(batch_size, C, H, W)con-
    taining the reference images.
**Return type**
None

```
Examples
```
```
>>>model = WinClipModel()
>>>model.setup("transistor")
>>>model.text_embeddings.shape
torch.Size([2, 640])
```
```
>>>ref_images = torch.rand(2, 3, 240, 240)
>>>model = WinClipModel()
>>>model.setup("transistor", ref_images)
>>>model.k_shot
2
>>>model.visual_embeddings[0].shape
torch.Size([2, 196, 640])
```
**166 Chapter 3. Guides**


```
>>>model = WinClipModel("transistor")
>>>model.k_shot
0
>>>model.setup(reference_images=ref_images)
>>>model.k_shot
2
```
```
>>>model = WinClipModel(class_name="transistor", reference_images=ref_images)
>>>model.text_embeddings.shape
torch.Size([2, 640])
>>>model.visual_embeddings[0].shape
torch.Size([2, 196, 640])
```
```
property text_embeddings: Tensor
The text embeddings used by the model.
property transform: Compose
The transform used by the model.
To obtain the transforms, we retrieve the transforms from the clip backbone. Since the original transforms
are intended for PIL images, we prepend a ToPILImage transform to the list of transforms.
property visual_embeddings: list[Tensor]
The visual embeddings used by the model.
```
**Video Models**

### AI VAD

### AI VAD

Attribute-based Representations for Accurate and Interpretable Video Anomaly Detection.

Paper https://arxiv.org/pdf/2212.00789.pdf

classanomalib.models.video.ai_vad.lightning_model.AiVad( _box_score_thresh=0.7_ ,
_persons_only=False_ ,
_min_bbox_area=100_ ,
_max_bbox_overlap=0.65_ ,
_enable_foreground_detections=True_ ,
_foreground_kernel_size=3_ ,
_foreground_binary_threshold=18_ ,
_n_velocity_bins=1_ ,
_use_velocity_features=True_ ,
_use_pose_features=True_ ,
_use_deep_features=True_ ,
_n_components_velocity=2_ ,
_n_neighbors_pose=1_ ,
_n_neighbors_deep=1_ )
Bases:MemoryBankMixin,AnomalyModule
AI-VAD: Attribute-based Representations for Accurate and Interpretable Video Anomaly Detection.
**Parameters**

**3.3. Reference Guide 167**


- box_score_thresh(float) – Confidence threshold for bounding box predictions. De-
    faults to0.7.
- persons_only(bool) – When enabled, only regions labeled as person are included. De-
    faults toFalse.
- min_bbox_area(int) – Minimum bounding box area. Regions with a surface area lower
    than this value are excluded. Defaults to 100.
- max_bbox_overlap(float) – Maximum allowed overlap between bounding boxes. De-
    faults to0.65.
- enable_foreground_detections(bool) – Add additional foreground detections based
    on pixel difference between consecutive frames. Defaults toTrue.
- foreground_kernel_size(int) – Gaussian kernel size used in foreground detection. De-
    faults to 3.
- foreground_binary_threshold(int) – Value between 0 and 255 which acts as binary
    threshold in foreground detection. Defaults to 18.
- n_velocity_bins(int) – Number of discrete bins used for velocity histogram features.
    Defaults to 1.
- use_velocity_features(bool) – Flag indicating if velocity features should be used. De-
    faults toTrue.
- use_pose_features(bool) – Flag indicating if pose features should be used. Defaults to
    True.
- use_deep_features(bool) – Flag indicating if deep features should be used. Defaults to
    True.
- n_components_velocity(int) – Number of components used by GMM density estima-
    tion for velocity features. Defaults to 2.
- n_neighbors_pose(int) – Number of neighbors used in KNN density estimation for pose
    features. Defaults to 1.
- n_neighbors_deep(int) – Number of neighbors used in KNN density estimation for deep
    features. Defaults to 1.
static configure_optimizers()
AI-VAD training does not involve fine-tuning of NN weights, no optimizers needed.
**Return type**
None
configure_transforms( _image_size=None_ )
AI-VAD does not need a transform, as the region- and feature-extractors apply their own transforms.
**Return type**
Transform|None
fit()
Fit the density estimators to the extracted features from the training set.
**Return type**
None
property learning_type: LearningType
Return the learning type of the model.

**168 Chapter 3. Guides**


```
Returns
Learning type of the model.
Return type
LearningType
property trainer_arguments: dict[str, Any]
AI-VAD specific trainer arguments.
training_step( batch )
Training Step of AI-VAD.
Extract features from the batch of clips and update the density estimators.
Parameters
batch(dict[str, str | torch.Tensor]) – Batch containing image filename, image,
label and mask
Return type
None
validation_step( batch , *args , **kwargs )
Perform the validation step of AI-VAD.
Extract boxes and box scores..
Parameters
```
- batch(dict[str, str | torch.Tensor]) – Input batch
- *args– Arguments.
- **kwargs– Keyword arguments.
**Return type**
Union[Tensor,Mapping[str,Any],None]
**Returns**
Batch dictionary with added boxes and box scores.

PyTorch model for AI-VAD model implementation.

Paper https://arxiv.org/pdf/2212.00789.pdf

classanomalib.models.video.ai_vad.torch_model.AiVadModel( _box_score_thresh=0.8_ ,
_persons_only=False_ ,
_min_bbox_area=100_ ,
_max_bbox_overlap=0.65_ ,
_enable_foreground_detections=True_ ,
_foreground_kernel_size=3_ ,
_foreground_binary_threshold=18_ ,
_n_velocity_bins=8_ ,
_use_velocity_features=True_ ,
_use_pose_features=True_ ,
_use_deep_features=True_ ,
_n_components_velocity=5_ ,
_n_neighbors_pose=1_ ,
_n_neighbors_deep=1_ )
Bases:Module
AI-VAD model.

**3.3. Reference Guide 169**


```
Parameters
```
- box_score_thresh(float) – Confidence threshold for region extraction stage. Defaults
    to0.8.
- persons_only(bool) – When enabled, only regions labeled as person are included. De-
    faults toFalse.
- min_bbox_area(int) – Minimum bounding box area. Regions with a surface area lower
    than this value are excluded. Defaults to 100.
- max_bbox_overlap(float) – Maximum allowed overlap between bounding boxes. De-
    faults to0.65.
- enable_foreground_detections(bool) – Add additional foreground detections based
    on pixel difference between consecutive frames. Defaults toTrue.
- foreground_kernel_size(int) – Gaussian kernel size used in foreground detection. De-
    faults to 3.
- foreground_binary_threshold(int) – Value between 0 and 255 which acts as binary
    threshold in foreground detection. Defaults to 18.
- n_velocity_bins(int) – Number of discrete bins used for velocity histogram features.
    Defaults to 8.
- use_velocity_features(bool) – Flag indicating if velocity features should be used. De-
    faults toTrue.
- use_pose_features(bool) – Flag indicating if pose features should be used. Defaults to
    True.
- use_deep_features(bool) – Flag indicating if deep features should be used. Defaults to
    True.
- n_components_velocity(int) – Number of components used by GMM density estima-
    tion for velocity features. Defaults to 5.
- n_neighbors_pose(int) – Number of neighbors used in KNN density estimation for pose
    features. Defaults to 1.
- n_neighbors_deep(int) – Number of neighbors used in KNN density estimation for deep
    features. Defaults to 1.
forward( _batch_ )
Forward pass through AI-VAD model.
**Parameters**
batch(torch.Tensor) – Input image of shape (N, L, C, H, W)
**Returns**
List of bbox locations for each image. list[torch.Tensor]: List of per-bbox anomaly scores for
each image. list[torch.Tensor]: List of per-image anomaly scores.
**Return type**
list[torch.Tensor]

Feature extraction module for AI-VAD model implementation.

classanomalib.models.video.ai_vad.features.DeepExtractor

```
Bases:Module
Deep feature extractor.
Extracts the deep (appearance) features from the input regions.
```
**170 Chapter 3. Guides**


```
forward( batch , boxes , batch_size )
Extract deep features using CLIP encoder.
Parameters
```
- batch(torch.Tensor) – Batch of RGB input images of shape (N, 3, H, W)
- boxes(torch.Tensor) – Bounding box coordinates of shaspe (M, 5). First column indi-
    cates batch index of the bbox.
- batch_size(int) – Number of images in the batch.
**Returns**
Deep feature tensor of shape (M, 512)
**Return type**
Tensor

classanomalib.models.video.ai_vad.features.FeatureExtractor( _n_velocity_bins=8_ ,
_use_velocity_features=True_ ,
_use_pose_features=True_ ,
_use_deep_features=True_ )
Bases:Module
Feature extractor for AI-VAD.
**Parameters**

- n_velocity_bins(int) – Number of discrete bins used for velocity histogram features.
    Defaults to 8.
- use_velocity_features(bool) – Flag indicating if velocity features should be used. De-
    faults toTrue.
- use_pose_features(bool) – Flag indicating if pose features should be used. Defaults to
    True.
- use_deep_features(bool) – Flag indicating if deep features should be used. Defaults to
    True.
forward( _rgb_batch_ , _flow_batch_ , _regions_ )
Forward pass through the feature extractor.
Extract any combination of velocity, pose and deep features depending on configuration.
**Parameters**
- rgb_batch(torch.Tensor) – Batch of RGB images of shape (N, 3, H, W)
- flow_batch(torch.Tensor) – Batch of optical flow images of shape (N, 2, H, W)
- regions(list[dict]) – Region information per image in batch.
**Returns**
Feature dictionary per image in batch.
**Return type**
list[dict]

classanomalib.models.video.ai_vad.features.FeatureType( _value_ , _names=None_ , _*_ , _module=None_ ,
_qualname=None_ , _type=None_ , _start=1_ ,
_boundary=None_ )
Bases:str,Enum

**3.3. Reference Guide 171**


```
Names of the different feature streams used in AI-VAD.
```
classanomalib.models.video.ai_vad.features.PoseExtractor( _*args_ , _**kwargs_ )

```
Bases:Module
Pose feature extractor.
Extracts pose features based on estimated body landmark keypoints.
forward( batch , boxes )
Extract pose features using a human keypoint estimation model.
Parameters
```
- batch(torch.Tensor) – Batch of RGB input images of shape (N, 3, H, W)
- boxes(torch.Tensor) – Bounding box coordinates of shaspe (M, 5). First column indi-
    cates batch index of the bbox.
**Returns**
list of pose feature tensors for each image.
**Return type**
list[torch.Tensor]

classanomalib.models.video.ai_vad.features.VelocityExtractor( _n_bins=8_ )

```
Bases:Module
Velocity feature extractor.
Extracts histograms of optical flow magnitude and direction.
Parameters
n_bins(int) – Number of direction bins used for the feature histograms.
forward( flows , boxes )
Extract velocioty features by filling a histogram.
Parameters
```
- flows(torch.Tensor) – Batch of optical flow images of shape (N, 2, H, W)
- boxes(torch.Tensor) – Bounding box coordinates of shaspe (M, 5). First column indi-
    cates batch index of the bbox.
**Returns**
Velocity feature tensor of shape (M, n_bins)
**Return type**
Tensor

Regions extraction module of AI-VAD model implementation.

classanomalib.models.video.ai_vad.regions.RegionExtractor( _box_score_thresh=0.8_ ,
_persons_only=False_ ,
_min_bbox_area=100_ ,
_max_bbox_overlap=0.65_ ,
_enable_foreground_detections=True_ ,
_foreground_kernel_size=3_ ,
_foreground_binary_threshold=18_ )
Bases:Module
Region extractor for AI-VAD.

**172 Chapter 3. Guides**


```
Parameters
```
- box_score_thresh(float) – Confidence threshold for bounding box predictions. De-
    faults to0.8.
- persons_only(bool) – When enabled, only regions labeled as person are included. De-
    faults toFalse.
- min_bbox_area(int) – Minimum bounding box area. Regions with a surface area lower
    than this value are excluded. Defaults to 100.
- max_bbox_overlap(float) – Maximum allowed overlap between bounding boxes. De-
    faults to0.65.
- enable_foreground_detections(bool) – Add additional foreground detections based
    on pixel difference between consecutive frames. Defaults toTrue.
- foreground_kernel_size(int) – Gaussian kernel size used in foreground detection. De-
    faults to 3.
- foreground_binary_threshold(int) – Value between 0 and 255 which acts as binary
    threshold in foreground detection. Defaults to 18.
add_foreground_boxes( _regions_ , _first_frame_ , _last_frame_ , _kernel_size_ , _binary_threshold_ )
Add any foreground regions that were not detected by the region extractor.
This method adds regions that likely belong to the foreground of the video scene, but were not detected by
the region extractor module. The foreground pixels are determined by taking the pixel difference between
two consecutive video frames and applying a binary threshold. The final detections consist of all connected
components in the foreground that do not fall in one of the bounding boxes predicted by the region extractor.
**Parameters**
- regions(list[dict[str, torch.Tensor]]) – Region detections for a batch of im-
ages, generated by the region extraction module.
- first_frame(torch.Tensor) – video frame at time t-1
- last_frame(torch.Tensor) – Video frame time t
- kernel_size(int) – Kernel size for Gaussian smoothing applied to input frames
- binary_threshold(int) – Binary threshold used in foreground detection, should be in
range [0, 255]
**Returns**
region detections with foreground regions appended
**Return type**
list[dict[str, torch.Tensor]]
forward( _first_frame_ , _last_frame_ )
Perform forward-pass through region extractor.
**Parameters**
- first_frame(torch.Tensor) – Batch of input images of shape (N, C, H, W) forming
the first frames in the clip.
- last_frame(torch.Tensor) – Batch of input images of shape (N, C, H, W) forming the
last frame in the clip.
**Returns**
List of Mask RCNN predictions for each image in the batch.

**3.3. Reference Guide 173**


```
Return type
list[dict]
post_process_bbox_detections( regions )
Post-process the region detections.
The region detections are filtered based on class label, bbox area and overlap with other regions.
Parameters
regions(list[dict[str, torch.Tensor]]) – Region detections for a batch of images,
generated by the region extraction module.
Returns
Filtered regions
Return type
list[dict[str, torch.Tensor]]
static subsample_regions( regions , indices )
Subsample the items in a region dictionary based on a Tensor of indices.
Parameters
```
- regions(dict[str, torch.Tensor]) – Region detections for a single image in the
    batch.
- indices(torch.Tensor) – Indices of region detections that should be kept.
**Returns**
Subsampled region detections.
**Return type**
dict[str, torch.Tensor]

Optical Flow extraction module for AI-VAD implementation.

classanomalib.models.video.ai_vad.flow.FlowExtractor( _*args_ , _**kwargs_ )

```
Bases:Module
Optical Flow extractor.
Computes the pixel displacement between 2 consecutive frames from a video clip.
forward( first_frame , last_frame )
Forward pass through the flow extractor.
Parameters
```
- first_frame(torch.Tensor) – Batch of starting frames of shape (N, 3, H, W).
- last_frame(torch.Tensor) – Batch of last frames of shape (N, 3, H, W).
**Returns**
Estimated optical flow map of shape (N, 2, H, W).
**Return type**
Tensor
pre_process( _first_frame_ , _last_frame_ )
Resize inputs to dimensions required by backbone.
**Parameters**
- first_frame(torch.Tensor) – Starting frame of optical flow computation.

**174 Chapter 3. Guides**


- last_frame(torch.Tensor) – Last frame of optical flow computation.
**Returns**
Preprocessed first and last frame.
**Return type**
tuple[torch.Tensor, torch.Tensor]

Density estimation module for AI-VAD model implementation.

classanomalib.models.video.ai_vad.density.BaseDensityEstimator( _*args_ , _**kwargs_ )

```
Bases:Module,ABC
Base density estimator.
abstract fit()
Compose model using collected features.
Return type
None
forward( features )
Update or predict depending on training status.
Return type
Tensor|tuple[Tensor,Tensor] |None
abstract predict( features )
Predict the density of a set of features.
Return type
Tensor|tuple[Tensor,Tensor]
abstract update( features , group=None )
Update the density model with a new set of features.
Return type
None
```
classanomalib.models.video.ai_vad.density.CombinedDensityEstimator( _use_pose_features=True_ ,
_use_deep_features=True_ ,
_use_velocity_features=False_ ,
_n_neighbors_pose=1_ ,
_n_neighbors_deep=1_ ,
_n_components_velocity=5_ )
Bases:BaseDensityEstimator
Density estimator for AI-VAD.
Combines density estimators for the different feature types included in the model.
**Parameters**

- use_pose_features(bool) – Flag indicating if pose features should be used. Defaults to
    True.
- use_deep_features(bool) – Flag indicating if deep features should be used. Defaults to
    True.
- use_velocity_features(bool) – Flag indicating if velocity features should be used. De-
    faults toFalse.

**3.3. Reference Guide 175**


- n_neighbors_pose(int) – Number of neighbors used in KNN density estimation for pose
    features. Defaults to 1.
- n_neighbors_deep(int) – Number of neighbors used in KNN density estimation for deep
    features. Defaults to 1.
- n_components_velocity(int) – Number of components used by GMM density estima-
    tion for velocity features. Defaults to 5.
fit()
Fit the density estimation models on the collected features.
**Return type**
None
predict( _features_ )
Predict the region- and image-level anomaly scores for an image based on a set of features.
**Parameters**
features(dict[Tensor]) – Dictionary containing extracted features for a single frame.
**Returns**
Region-level anomaly scores for all regions withing the frame. Tensor: Frame-level anomaly
score for the frame.
**Return type**
Tensor
update( _features_ , _group=None_ )
Update the density estimators for the different feature types.
**Parameters**
- features(dict[FeatureType, torch.Tensor]) – Dictionary containing extracted
features for a single frame.
- group(str) – Identifier of the video from which the frame was sampled. Used for grouped
density estimation.
**Return type**
None

classanomalib.models.video.ai_vad.density.GMMEstimator( _n_components=2_ )

```
Bases:BaseDensityEstimator
Density estimation based on Gaussian Mixture Model.
Parameters
n_components(int) – Number of components used in the GMM. Defaults to 2.
fit()
Fit the GMM and compute normalization statistics.
Return type
None
predict( features , normalize=True )
Predict the density of a set of feature vectors.
Parameters
```
- features(torch.Tensor) – Input feature vectors.

**176 Chapter 3. Guides**


- normalize(bool) – Flag indicating if the density should be normalized to min-max stats
    of the feature bank. Defaults toTrue.
**Returns**
Density scores of the input feature vectors.
**Return type**
Tensor
update( _features_ , _group=None_ )
Update the feature bank.
**Return type**
None

classanomalib.models.video.ai_vad.density.GroupedKNNEstimator( _n_neighbors_ )

```
Bases:DynamicBufferMixin,BaseDensityEstimator
Grouped KNN density estimator.
Keeps track of the group (e.g. video id) from which the features were sampled for normalization purposes.
Parameters
n_neighbors(int) – Number of neighbors used in KNN search.
fit()
Fit the KNN model by stacking the feature vectors and computing the normalization statistics.
Return type
None
predict( features , group=None , n_neighbors=1 , normalize=True )
Predict the (normalized) density for a set of features.
Parameters
```
- features(torch.Tensor) – Input features that will be compared to the density model.
- group(str, optional) – Group (video id) from which the features originate. If passed,
    all features of the same group in the memory bank will be excluded from the density esti-
    mation. Defaults toNone.
- n_neighbors(int) – Number of neighbors used in the KNN search. Defaults to 1.
- normalize(bool) – Flag indicating if the density should be normalized to min-max stats
    of the feature bank. Defatuls toTrue.
**Returns**
Mean (normalized) distances of input feature vectors to k nearest neighbors in feature bank.
**Return type**
Tensor
update( _features_ , _group=None_ )
Update the internal feature bank while keeping track of the group.
**Parameters**
- features(torch.Tensor) – Feature vectors extracted from a video frame.
- group(str) – Identifier of the group (video) from which the frame was sampled.
**Return type**
None

**3.3. Reference Guide 177**


**3.3.3 Engine**

Anomalib engine.

classanomalib.engine.Engine( _callbacks=None_ , _normalization=NormalizationMethod.MIN_MAX_ ,
_threshold='F1AdaptiveThreshold'_ , _task=TaskType.SEGMENTATION_ ,
_image_metrics=None_ , _pixel_metrics=None_ , _logger=None_ ,
_default_root_dir='results'_ , _**kwargs_ )
Bases:object
Anomalib Engine.

```
Note: Refer to PyTorch Lightning’s Trainer for a list of parameters for details on other Trainer parameters.
```
```
Parameters
```
- callbacks(list[Callback]) – Add a callback or list of callbacks.
- normalization(NORMALIZATION, optional) – Normalization method. Defaults to Nor-
    malizationMethod.MIN_MAX.
- threshold(THRESHOLD) – Thresholding method. Defaults to “F1AdaptiveThreshold”.
- task(TaskType, optional) – Task type. Defaults to TaskType.SEGMENTATION.
- image_metrics (list[str] | str | dict[str, dict[str, Any]] | None,
    optional) – Image metrics to be used for evaluation. Defaults to None.
- pixel_metrics (list[str] | str | dict[str, dict[str, Any]] | None,
    optional) – Pixel metrics to be used for evaluation. Defaults to None.
- default_root_dir(str, optional) – Default root directory for the trainer. The results
    will be saved in this directory. Defaults toresults.
- **kwargs– PyTorch Lightning Trainer arguments.

```
export( model , export_type , export_root=None , transform=None , ov_args=None , ckpt_path=None )
Export the model in PyTorch, ONNX or OpenVINO format.
Parameters
```
- model(AnomalyModule) – Trained model.
- export_type(ExportType) – Export type.
- export_root(str | Path | None, optional) – Path to the output directory. If it is
    not set, the model is exported to trainer.default_root_dir. Defaults to None.
- transform(Transform | None, optional) – Input transform to include in the ex-
    ported model. If not provided, the engine will try to use the transform from the datamodule
    or dataset. Defaults to None.
- ov_args(dict[str, Any] | None, optional) – This is optional and used only for
    OpenVINO’s model optimizer. Defaults to None.
- ckpt_path(str | Path | None) – Checkpoint path. If provided, the model will be
    loaded from this path.
**Returns**
Path to the exported model.

**178 Chapter 3. Guides**


```
Return type
Path
Raises
```
- ValueError– If Dataset, Datamodule, and transform are not provided.
- TypeError– If path to the transform file is not a string or Path.

```
CLI Usage:
```
1. **To export as a torch** .pt **file you can run the following command.**
    `python anomalib export --model Padim --export_mode TORCH --data MVTec
    `
2. **To export as an ONNX** .onnx **file you can run the following command.**
    `python anomalib export --model Padim --export_mode ONNX --data Visa
    --input_size "[256,256]"`
3. **To export as an OpenVINO** .xml **and** .bin **file you can run the following command.**
    `python anomalib export --model Padim --export_mode OPENVINO --data
    Visa --input_size "[256,256]"`
4. **You can also overrride OpenVINO model optimizer by adding the** --mo_args.<key>
    **arguments.**
       `python anomalib export --model Padim --export_mode OPENVINO --data
       Visa --input_size "[256,256]" --mo_args.compress_to_fp16 False `

```
fit( model , train_dataloaders=None , val_dataloaders=None , datamodule=None , ckpt_path=None )
Fit the model using the trainer.
Parameters
```
- model(AnomalyModule) – Model to be trained.
- train_dataloaders(TRAIN_DATALOADERS | None, optional) – Train dataloaders.
    Defaults to None.
- val_dataloaders(EVAL_DATALOADERS | None, optional) – Validation dataload-
    ers. Defaults to None.
- datamodule(AnomalibDataModule| None, optional) – Lightning datamodule. If
    provided, dataloaders will be instantiated from this. Defaults to None.
- ckpt_path(str | None, optional) – Checkpoint path. If provided, the model will
    be loaded from this path. Defaults to None.
**Return type**
None

```
CLI Usage:
```
1. **you can pick a model, and you can run through the MVTec dataset.**
    `python anomalib fit --model anomalib.models.Padim`
2. **Of course, you can override the various values with commands.**
    `python anomalib fit --model anomalib.models.Padim --data <CONFIG |
    CLASS_PATH_OR_NAME> --trainer.max_epochs 3`
4. **If you have a ready configuration file, run it like this.**
    `python anomalib fit --config <config_file_path>`

**3.3. Reference Guide 179**


```
property model: AnomalyModule
Property to get the model.
Raises
UnassignedError– When the model is not assigned yet.
Returns
Anomaly model.
Return type
AnomalyModule
property normalization_callback: NormalizationCallback | None
TheNormalizationCallbackcallback in the trainer.callbacks list, orNoneif it doesn’t exist.
Returns
Normalization callback, if available.
Return type
NormalizationCallback | None
Raises
ValueError– If there are multiple normalization callbacks.
predict( model=None , dataloaders=None , datamodule=None , dataset=None , return_predictions=None ,
ckpt_path=None )
Predict using the model using the trainer.
Sets up the trainer and the dataset task if not already set up. Then validates the model if needed and a
validation dataloader is available. Finally, predicts using the model.
Parameters
```
- model(AnomalyModule | None, optional) – Model to be used for prediction. De-
    faults to None.
- dataloaders(EVAL_DATALOADERS | None, optional) – An iterable or collection of
    iterables specifying predict samples. Defaults to None.
- datamodule(AnomalibDataModule| None, optional) – AAnomalibDataModule
    that defines thepredict_dataloaderhook. The datamodule can also be a dataset that
    will be wrapped in a torch Dataloader. Defaults to None.
- dataset (Dataset | PredictDataset | None, optional) – A Dataset or
    PredictDatasetthat will be used to create a dataloader. Defaults to None.
- return_predictions(bool | None, optional) – Whether to return predictions.
    Trueby default except when an accelerator that spawns processes is used (not supported).
    Defaults to None.
- ckpt_path(str | None, optional) – Either"best","last","hpc"or path to the
    checkpoint you wish to predict. IfNoneand the model instance was passed, use the current
    weights. Otherwise, the best model checkpoint from the previoustrainer.fitcall will
    be loaded if a checkpoint callback is configured. Defaults to None.
**Returns**
Predictions.
**Return type**
_PREDICT_OUTPUT | None

```
CLI Usage:
```
**180 Chapter 3. Guides**


1. **you can pick a model.**
    `python anomalib predict --model anomalib.models.Padim anomalib predict
    --model Padim --data datasets/MVTec/bottle/test/broken_large`
2. **Of course, you can override the various values with commands.**
    `python anomalib predict --model anomalib.models.Padim --data <CONFIG |
    CLASS_PATH_OR_NAME> `
4. **If you have a ready configuration file, run it like this.**
    `python anomalib predict --config <config_file_path>
    --return_predictions`
5. **You can also point to a folder with image or a single image instead of passing a dataset.**
    `python anomalib predict --model Padim --data <PATH_TO_IMAGE_OR_FOLDER>
    --ckpt_path <PATH_TO_CHECKPOINT> `

```
test( model=None , dataloaders=None , ckpt_path=None , verbose=True , datamodule=None )
Test the model using the trainer.
Sets up the trainer and the dataset task if not already set up. Then validates the model if needed and finally
tests the model.
Parameters
```
- model(AnomalyModule | None, optional) – The model to be tested. Defaults to
    None.
- dataloaders(EVAL_DATALOADERS | None, optional) – An iterable or collection of
    iterables specifying test samples. Defaults to None.
- ckpt_path(str | None, optional) – Either"best","last","hpc"or path to the
    checkpoint you wish to test. IfNoneand the model instance was passed, use the current
    weights. Otherwise, the best model checkpoint from the previoustrainer.fitcall will
    be loaded if a checkpoint callback is configured. Defaults to None.
- verbose(bool, optional) – If True, prints the test results. Defaults to True.
- datamodule(AnomalibDataModule| None, optional) – AAnomalibDataModule
    that defines thetest_dataloaderhook. Defaults to None.
**Returns**
A List of dictionaries containing the test results. 1 dict per dataloader.
**Return type**
_EVALUATE_OUTPUT

```
Examples
```
```
# fit and test a one-class model >>> from anomalib.data import MVTec >>> from anomalib.models import
Padim >>> from anomalib.engine import Engine
```
```
>>>datamodule = MVTec()
>>>model = Padim()
>>>model.learning_type
<LearningType.ONE_CLASS:'one_class'>
```
**3.3. Reference Guide 181**


```
>>>engine = Engine()
>>>engine.fit(model, datamodule=datamodule)
>>>engine.test(model, datamodule=datamodule)
```
```
# Test a zero-shot model >>> from anomalib.data import MVTec >>> from anomalib.models import Padim
>>> from anomalib.engine import Engine
```
```
>>>datamodule = MVTec(image_size=240, normalization="clip")
>>>model = Padim()
>>>model.learning_type
<LearningType.ZERO_SHOT:'zero_shot'>
```
```
>>>engine = Engine()
>>>engine.test(model, datamodule=datamodule)
```
```
CLI Usage:
```
1. **you can pick a model.**
    `python anomalib test --model anomalib.models.Padim`
2. **Of course, you can override the various values with commands.**
    `python anomalib test --model anomalib.models.Padim --data <CONFIG |
    CLASS_PATH_OR_NAME> `
4. **If you have a ready configuration file, run it like this.**
    `python anomalib test --config <config_file_path>`

```
property threshold_callback: _ThresholdCallback | None
TheThresholdCallbackcallback in the trainer.callbacks list, orNoneif it doesn’t exist.
Returns
Threshold callback, if available.
Return type
_ThresholdCallback | None
Raises
ValueError– If there are multiple threshold callbacks.
train( model , train_dataloaders=None , val_dataloaders=None , test_dataloaders=None , datamodule=None ,
ckpt_path=None )
Fits the model and then calls test on it.
Parameters
```
- model(AnomalyModule) – Model to be trained.
- train_dataloaders(TRAIN_DATALOADERS | None, optional) – Train dataloaders.
    Defaults to None.
- val_dataloaders(EVAL_DATALOADERS | None, optional) – Validation dataload-
    ers. Defaults to None.
- test_dataloaders (EVAL_DATALOADERS | None, optional) – Test dataloaders.
    Defaults to None.
- datamodule(AnomalibDataModule| None, optional) – Lightning datamodule. If
    provided, dataloaders will be instantiated from this. Defaults to None.

**182 Chapter 3. Guides**


- ckpt_path(str | None, optional) – Checkpoint path. If provided, the model will
    be loaded from this path. Defaults to None.
**Return type**
List[Mapping[str,float]]

```
CLI Usage:
```
1. **you can pick a model, and you can run through the MVTec dataset.**
    `python anomalib train --model anomalib.models.Padim --data MVTec`
2. **Of course, you can override the various values with commands.**
    `python anomalib train --model anomalib.models.Padim --data <CONFIG |
    CLASS_PATH_OR_NAME> --trainer.max_epochs 3`
4. **If you have a ready configuration file, run it like this.**
    `python anomalib train --config <config_file_path>`

```
property trainer: Trainer
Property to get the trainer.
Raises
UnassignedError– When the trainer is not assigned yet.
Returns
Lightning Trainer.
Return type
Trainer
validate( model=None , dataloaders=None , ckpt_path=None , verbose=True , datamodule=None )
Validate the model using the trainer.
Parameters
```
- model(AnomalyModule | None, optional) – Model to be validated. Defaults to
    None.
- dataloaders(EVAL_DATALOADERS | None, optional) – Dataloaders to be used for
    validation. Defaults to None.
- ckpt_path(str | None, optional) – Checkpoint path. If provided, the model will
    be loaded from this path. Defaults to None.
- verbose(bool, optional) – Boolean to print the validation results. Defaults to True.
- datamodule (AnomalibDataModule | None, optional) – A datamodule
    AnomalibDataModulethat defines theval_dataloaderhook. Defaults to None.
**Returns**
Validation results.
**Return type**
_EVALUATE_OUTPUT | None

```
CLI Usage:
```
1. **you can pick a model.**
    `python anomalib validate --model anomalib.models.Padim`

**3.3. Reference Guide 183**


2. **Of course, you can override the various values with commands.**
    `python anomalib validate --model anomalib.models.Padim --data <CONFIG
    | CLASS_PATH_OR_NAME>`
4. **If you have a ready configuration file, run it like this.**
    `python anomalib validate --config <config_file_path>`

**3.3.4 Metrics**

Custom anomaly evaluation metrics.

classanomalib.metrics.AUPR( _**kwargs_ )

```
Bases:PrecisionRecallCurve
Area under the PR curve.
This metric computes the area under the precision-recall curve.
Parameters
kwargs(Any) – Additional arguments to the TorchMetrics base class.
```
```
Examples
```
```
To compute the metric for a set of predictions and ground truth targets:
```
```
>>>true = torch.tensor([0, 1, 1, 1, 0, 0, 0, 0, 1, 1])
>>>pred = torch.tensor([0.59, 0.35, 0.72, 0.33, 0.73, 0.81, 0.30, 0.05, 0.04, 0.
˓→48])
```
```
>>>metric = AUPR()
>>>metric(pred, true)
tensor(0.4899)
```
```
It is also possible to update the metric state incrementally within batches:
```
```
>>> forbatchin dataloader:
... # Compute prediction and target tensors
... metric.update(pred, true)
>>>metric.compute()
```
```
Once the metric has been computed, we can plot the PR curve:
```
```
>>>figure, title = metric.generate_figure()
```
```
compute()
First compute PR curve, then compute area under the curve.
Return type
Tensor
Returns
Value of the AUPR metric
```
**184 Chapter 3. Guides**


```
generate_figure()
Generate a figure containing the PR curve as well as the random baseline and the AUC.
Returns
Tuple containing both the PR curve and the figure title to be used for logging
Return type
tuple[Figure, str]
update( preds , target )
Update state with new values.
Need to flatten new values as PrecicionRecallCurve expects them in this format for binary classification.
Parameters
```
- preds(torch.Tensor) – predictions of the model
- target(torch.Tensor) – ground truth targets
**Return type**
None

classanomalib.metrics.AUPRO( _dist_sync_on_step=False_ , _process_group=None_ , _dist_sync_fn=None_ ,
_fpr_limit=0.3_ , _num_thresholds=None_ )
Bases:Metric
Area under per region overlap (AUPRO) Metric.
**Parameters**

- dist_sync_on_step (bool) – Synchronize metric state across processes at each
    forward()before returning the value at the step. Default:False
- process_group(Optional[Any]) – Specify the process group on which synchronization
    is called. Default:None(which selects the entire world)
- dist_sync_fn(Optional[Callable]) – Callback that performs the allgather operation
    on the metric state. WhenNone, DDP will be used to perform the allgather. Default:None
- fpr_limit(float) – Limit for the false positive rate. Defaults to0.3.
- num_thresholds(int) – Number of thresholds to use for computing the roc curve. Defaults
    toNone. IfNone, the roc curve is computed with the thresholds returned bytorchmetrics.
    functional.classification.thresholds.

```
Examples
```
```
>>> import torch
>>> from anomalib.metrics importAUPRO
...
>>>labels = torch.randint(low=0, high=2, size=(1, 10, 5), dtype=torch.float32)
>>>preds = torch.rand_like(labels)
...
>>>aupro = AUPRO(fpr_limit=0.3)
>>>aupro(preds, labels)
tensor(0.4321)
```
```
Increasing the fpr_limit will increase the AUPRO value:
```
**3.3. Reference Guide 185**


```
>>>aupro = AUPRO(fpr_limit=0.7)
>>>aupro(preds, labels)
tensor(0.5271)
```
```
compute()
Fist compute PRO curve, then compute and scale area under the curve.
Returns
Value of the AUPRO metric
Return type
Tensor
compute_pro( cca , target , preds )
Compute the pro/fpr value-pairs until the fpr specified by self.fpr_limit.
It leverages the fact that the overlap corresponds to the tpr, and thus computes the overall PRO curve by
aggregating per-region tpr/fpr values produced by ROC-construction.
Returns
tuple containing final fpr and tpr values.
Return type
tuple[torch.Tensor, torch.Tensor]
generate_figure()
Generate a figure containing the PRO curve and the AUPRO.
Returns
Tuple containing both the figure and the figure title to be used for logging
Return type
tuple[Figure, str]
static interp1d( old_x , old_y , new_x )
Interpolate a 1D signal linearly to new sampling points.
Parameters
```
- old_x(torch.Tensor) – original 1-D x values (same size as y)
- old_y(torch.Tensor) – original 1-D y values (same size as x)
- new_x(torch.Tensor) – x-values where y should be interpolated at
**Returns**
y-values at corresponding new_x values.
**Return type**
Tensor
perform_cca()
Perform the Connected Component Analysis on the self.target tensor.
**Raises**
ValueError– ValueError is raised if self.target doesn’t conform with requirements imposed
by kornia for connected component analysis.
**Returns**
Components labeled from 0 to N.

**186 Chapter 3. Guides**


```
Return type
Tensor
update( preds , target )
Update state with new values.
Parameters
```
- preds(torch.Tensor) – predictions of the model
- target(torch.Tensor) – ground truth targets
**Return type**
None

classanomalib.metrics.AUROC( _thresholds=None_ , _ignore_index=None_ , _validate_args=True_ , _**kwargs_ )

```
Bases:BinaryROC
Area under the ROC curve.
```
```
Examples
```
```
>>> import torch
>>> from anomalib.metrics importAUROC
...
>>>preds = torch.tensor([0.13, 0.26, 0.08, 0.92, 0.03])
>>>target = torch.tensor([0, 0, 1, 1, 0])
...
>>>auroc = AUROC()
>>>auroc(preds, target)
tensor(0.6667)
```
```
It is possible to update the metric state incrementally:
```
```
>>>auroc.update(preds[:2], target[:2])
>>>auroc.update(preds[2:], target[2:])
>>>auroc.compute()
tensor(0.6667)
```
```
To plot the ROC curve, use thegenerate_figuremethod:
```
```
>>>fig, title = auroc.generate_figure()
```
```
compute()
First compute ROC curve, then compute area under the curve.
Returns
Value of the AUROC metric
Return type
Tensor
generate_figure()
Generate a figure containing the ROC curve, the baseline and the AUROC.
Returns
Tuple containing both the figure and the figure title to be used for logging
```
**3.3. Reference Guide 187**


```
Return type
tuple[Figure, str]
update( preds , target )
Update state with new values.
Need to flatten new values as ROC expects them in this format for binary classification.
Parameters
```
- preds(torch.Tensor) – predictions of the model
- target(torch.Tensor) – ground truth targets
**Return type**
None

classanomalib.metrics.AnomalyScoreDistribution( _**kwargs_ )

```
Bases:Metric
Mean and standard deviation of the anomaly scores of normal training data.
compute()
Compute stats.
Return type
tuple[Tensor,Tensor,Tensor,Tensor]
update( *args , anomaly_scores=None , anomaly_maps=None , **kwargs )
Update the precision-recall curve metric.
Return type
None
```
classanomalib.metrics.BinaryPrecisionRecallCurve( _thresholds=None_ , _ignore_index=None_ ,
_validate_args=True_ , _**kwargs_ )
Bases:BinaryPrecisionRecallCurve
Binary precision-recall curve with without threshold prediction normalization.
update( _preds_ , _target_ )
Update metric state with new predictions and targets.
Unlike the base class, this accepts raw predictions and targets.
**Parameters**

- preds(Tensor) – Predicted probabilities
- target(Tensor) – Ground truth labels
**Return type**
None

classanomalib.metrics.F1AdaptiveThreshold( _default_value=0.5_ , _**kwargs_ )

```
Bases:BinaryPrecisionRecallCurve,BaseThreshold
Anomaly Score Threshold.
This class computes/stores the threshold that determines the anomalous label given anomaly scores. It initially
computes the adaptive threshold to find the optimal f1_score and stores the computed adaptive threshold value.
Parameters
default_value(float) – Default value of the threshold. Defaults to0.5.
```
**188 Chapter 3. Guides**


```
Examples
```
```
To find the best threshold that maximizes the F1 score, we could run the following:
```
```
>>> from anomalib.metrics importF1AdaptiveThreshold
>>> import torch
...
>>>labels = torch.tensor([0, 0, 0, 1, 1])
>>>preds = torch.tensor([2.3, 1.6, 2.6, 7.9, 3.3])
...
>>>adaptive_threshold = F1AdaptiveThreshold(default_value=0.5)
>>>threshold = adaptive_threshold(preds, labels)
>>>threshold
tensor(3.3000)
```
```
compute()
Compute the threshold that yields the optimal F1 score.
Compute the F1 scores while varying the threshold. Store the optimal threshold as attribute and return the
maximum value of the F1 score.
Return type
Tensor
Returns
Value of the F1 score at the optimal threshold.
```
classanomalib.metrics.F1Score( _threshold=0.5_ , _multidim_average='global'_ , _ignore_index=None_ ,
_validate_args=True_ , _**kwargs_ )
Bases:BinaryF1Score
This is a wrapper around torchmetrics’ BinaryF1Score.
The idea behind this is to retain the current configuration otherwise the one from torchmetrics requirestaskas
a parameter.

classanomalib.metrics.ManualThreshold( _default_value=0.5_ , _**kwargs_ )

```
Bases:BaseThreshold
Initialize Manual Threshold.
Parameters
```
- default_value(float, optional) – Default threshold value. Defaults to0.5.
- kwargs– Any keyword arguments.

```
Examples
```
```
>>> from anomalib.metrics importManualThreshold
>>> import torch
...
>>>manual_threshold = ManualThreshold(default_value=0.5)
...
>>>labels = torch.randint(low=0, high=2, size=(5,))
>>>preds = torch.rand(5)
...
(continues on next page)
```
**3.3. Reference Guide 189**


```
(continued from previous page)
>>>threshold = manual_threshold(preds, labels)
>>>threshold
tensor(0.5000, dtype=torch.float64)
```
```
As the threshold is manually set, the threshold value is the same as thedefault_value.
```
```
>>>labels = torch.randint(low=0, high=2, size=(5,))
>>>preds = torch.rand(5)
>>>threshold = manual_threshold(preds2, labels2)
>>>threshold
tensor(0.5000, dtype=torch.float64)
```
```
The threshold value remains the same even if the inputs change.
compute()
Compute the threshold.
In case of manual thresholding, the threshold is already set and does not need to be computed.
Returns
Value of the optimal threshold.
Return type
torch.Tensor
update( *args , **kwargs )
Do nothing.
Parameters
```
- *args– Any positional arguments.
- **kwargs– Any keyword arguments.
**Return type**
None

classanomalib.metrics.MinMax( _**kwargs_ )

```
Bases:Metric
Track the min and max values of the observations in each batch.
Parameters
```
- full_state_update(bool, optional) – Whether to update the state with the new val-
    ues. Defaults toTrue.
- kwargs– Any keyword arguments.

**190 Chapter 3. Guides**


```
Examples
```
```
>>> from anomalib.metrics importMinMax
>>> import torch
...
>>>predictions = torch.tensor([0.0807, 0.6329, 0.0559, 0.9860, 0.3595])
>>>minmax = MinMax()
>>>minmax(predictions)
(tensor(0.0559), tensor(0.9860))
```
```
It is possible to update the minmax values with a new tensor of predictions.
```
```
>>>new_predictions = torch.tensor([0.3251, 0.3169, 0.3072, 0.6247, 0.9999])
>>>minmax.update(new_predictions)
>>>minmax.compute()
(tensor(0.0559), tensor(0.9999))
```
```
compute()
Return min and max values.
Return type
tuple[Tensor,Tensor]
update( predictions , *args , **kwargs )
Update the min and max values.
Return type
None
```
classanomalib.metrics.PRO( _threshold=0.5_ , _**kwargs_ )

```
Bases:Metric
Per-Region Overlap (PRO) Score.
This metric computes the macro average of the per-region overlap between the predicted anomaly masks and the
ground truth masks.
Parameters
```
- threshold(float) – Threshold used to binarize the predictions. Defaults to0.5.
- kwargs– Additional arguments to the TorchMetrics base class.

```
Example
```
```
Import the metric from the package:
```
```
>>> import torch
>>> from anomalib.metrics importPRO
```
```
Create randompredsandlabelstensors:
```
```
>>>labels = torch.randint(low=0, high=2, size=(1, 10, 5), dtype=torch.float32)
>>>preds = torch.rand_like(labels)
```
```
Compute the PRO score for labels and preds:
```
**3.3. Reference Guide 191**


```
>>>pro = PRO(threshold=0.5)
>>>pro.update(preds, labels)
>>>pro.compute()
tensor(0.5433)
```
```
Note: Note that the example above shows random predictions and labels. Therefore, the PRO score above may
not be reproducible.
```
```
compute()
Compute the macro average of the PRO score across all regions in all batches.
Return type
Tensor
```
```
Example
```
```
To compute the metric based on the state accumulated from multiple batches, use thecomputemethod:
```
```
>>>pro.compute()
tensor(0.5433)
```
```
update( predictions , targets )
Compute the PRO score for the current batch.
Parameters
```
- predictions(torch.Tensor) – Predicted anomaly masks (Bx1xHxW)
- targets(torch.Tensor) – Ground truth anomaly masks (Bx1xHxW)
**Return type**
None

```
Example
```
```
To update the metric state for the current batch, use theupdatemethod:
```
```
>>>pro.update(preds, labels)
```
**3.3.5 Loggers**

Load PyTorch Lightning Loggers.

classanomalib.loggers.AnomalibCometLogger( _api_key=None_ , _save_dir=None_ , _project_name=None_ ,
_rest_api_key=None_ , _experiment_name=None_ ,
_experiment_key=None_ , _offline=False_ , _prefix=''_ , _**kwargs_ )
Bases:ImageLoggerBase,CometLogger
Logger for comet.
Adds interface foradd_imagein the logger rather than calling the experiment object.

**192 Chapter 3. Guides**


```
Note: Same as the CometLogger provided by PyTorch Lightning and the doc string is reproduced below.
```
```
Track your parameters, metrics, source code and more using Comet.
Install it with pip:
```
```
pip install comet-ml
```
```
Comet requires either an API Key (online mode) or a local directory path (offline mode).
Parameters
```
- api_key(str|None) – Required in online mode. API key, found on Comet.ml. If not given,
    this will be loaded from the environment variable COMET_API_KEY or ~/.comet.config if
    either exists. Defaults toNone.
- save_dir(str|None) – Required in offline mode. The path for the directory to save local
    comet logs. If given, this also sets the directory for saving checkpoints. Defaults toNone.
- project_name(str|None) – Optional. Send your experiment to a specific project. Other-
    wise will be sent to Uncategorized Experiments. If the project name does not already exist,
    Comet.ml will create a new project. Defaults toNone.
- rest_api_key(str|None) – Optional. Rest API key found in Comet.ml settings. This is
    used to determine version number Defaults toNone.
- experiment_name(str|None) – Optional. String representing the name for this particular
    experiment on Comet.ml. Defaults toNone.
- experiment_key(str|None) – Optional. If set, restores from existing experiment. De-
    faults toNone.
- offline(bool) – If api_key and save_dir are both given, this determines whether the ex-
    periment will be in online or offline mode. This is useful if you use save_dir to control the
    checkpoints directory and have a ~/.comet.config file but still want to run offline experiments.
    Defaults toNone.
- prefix(str) – A string to put at the beginning of metric keys. Defaults to"".
- kwargs– Additional arguments like _workspace_ , _log_code_ , etc. used byCometExperiment
    can be passed as keyword arguments in this logger.
**Raises**
- ModuleNotFoundError– If required Comet package is not installed on the device.
- MisconfigurationException– If neitherapi_keynorsave_dirare passed as argu-
ments.

```
Example
```
```
>>> from anomalib.loggers importAnomalibCometLogger
>>> from anomalib.engine importEngine
...
>>>comet_logger = AnomalibCometLogger()
>>>engine = Engine(logger=comet_logger)
```
```
See also:
```
**3.3. Reference Guide 193**


- Comet Documentation

```
add_image( image , name=None , **kwargs )
Interface to add image to comet logger.
Parameters
```
- image(np.ndarray | Figure) – Image to log.
- name(str | None) – The tag of the image Defaults toNone.
- kwargs– Accepts only _global_step_ (int). The step at which to log the image.
**Return type**
None

classanomalib.loggers.AnomalibTensorBoardLogger( _save_dir_ , _name='default'_ , _version=None_ ,
_log_graph=False_ , _default_hp_metric=True_ ,
_prefix=''_ , _**kwargs_ )
Bases:ImageLoggerBase,TensorBoardLogger
Logger for tensorboard.
Adds interface for _add_image_ in the logger rather than calling the experiment object.

```
Note: Same as the Tensorboard Logger provided by PyTorch Lightning and the doc string is reproduced below.
```
```
Logs are saved toos.path.join(save_dir, name, version). This is the default logger in Lightning, it
comes preinstalled.
```
```
Example
```
```
>>> from anomalib.engine importEngine
>>> from anomalib.loggers importAnomalibTensorBoardLogger
...
>>>logger = AnomalibTensorBoardLogger("tb_logs", name="my_model")
>>>engine = Engine(logger=logger)
```
```
Parameters
```
- save_dir(str) – Save directory
- name(str | None) – Experiment name. Defaults to'default'. If it is the empty string
    then no per-experiment subdirectory is used. Default:'default'.
- version(int | str | None) – Experiment version. If version is not specified the logger
    inspects the save directory for existing versions, then automatically assigns the next available
    version. If it is a string then it is used as the run-specific subdirectory name, otherwise
    'version_${version}'is used. Defaults toNone
- log_graph(bool) – Adds the computational graph to tensorboard. This requires that the
    user has defined the _self.example_input_array_ attribute in their model. Defaults toFalse.
- default_hp_metric(bool) – Enables a placeholder metric with keyhp_metricwhen
    log_hyperparamsis called without a metric (otherwise calls to log_hyperparams without
    a metric are ignored). Defaults toTrue.
- prefix(str) – A string to put at the beginning of metric keys. Defaults to''.

**194 Chapter 3. Guides**


- **kwargs – Additional arguments like _comment_ , _filename_suffix_ , etc. used by
    SummaryWritercan be passed as keyword arguments in this logger.

```
add_image( image , name=None , **kwargs )
Interface to add image to tensorboard logger.
Parameters
```
- image(np.ndarray | Figure) – Image to log
- name(str | None) – The tag of the image Defaults toNone.
- kwargs– Accepts only _global_step_ (int). The step at which to log the image.
**Return type**
None

classanomalib.loggers.AnomalibWandbLogger( _name=None_ , _save_dir=None_ , _offline=False_ , _id=None_ ,
_anonymous=None_ , _version=None_ , _project=None_ ,
_log_model=False_ , _experiment=None_ , _prefix=''_ , _**kwargs_ )
Bases:ImageLoggerBase,WandbLogger
Logger for wandb.
Adds interface for _add_image_ in the logger rather than calling the experiment object.

```
Note: Same as the wandb Logger provided by PyTorch Lightning and the doc string is reproduced below.
```
```
Log using Weights and Biases.
Install it with pip:
```
```
$ pip install wandb
```
```
Parameters
```
- name(str|None) – Display name for the run. Defaults toNone.
- save_dir(str|None) – Path where data is saved (wandb dir by default). Defaults toNone.
- offline(bool|None) – Run offline (data can be streamed later to wandb servers). Defaults
    toFalse.
- id(str|None) – Sets the version, mainly used to resume a previous run. Defaults toNone.
- anonymous(bool|None) – Enables or explicitly disables anonymous logging. Defaults to
    None.
- version(str|None) – Same as id. Defaults toNone.
- project(str|None) – The name of the project to which this run will belong. Defaults to
    None.
- log_model(str|bool) – Save checkpoints in wandb dir to upload on W&B servers. De-
    faults toFalse.
- experiment(type[Run] |type[RunDisabled] |None) – WandB experiment object. Au-
    tomatically set when creating a run. Defaults toNone.
- prefix(str|None) – A string to put at the beginning of metric keys. Defaults to''.
- **kwargs– Arguments passed towandb.init()like _entity_ , _group_ , _tags_ , etc.

**3.3. Reference Guide 195**


```
Raises
```
- ImportError– If required WandB package is not installed on the device.
- MisconfigurationException – If both log_model and offline``is set to
    ``True.

```
Example
```
```
>>> from anomalib.loggers importAnomalibWandbLogger
>>> from anomalib.engine importEngine
...
>>>wandb_logger = AnomalibWandbLogger()
>>>engine = Engine(logger=wandb_logger)
```
```
Note: When logging manually through wandb.log or trainer.logger.experiment.log , make sure to use com-
mit=False so the logging step does not increase.
```
```
See also:
```
- Tutorial on how to use W&B with PyTorch Lightning
- W&B Documentation

```
add_image( image , name=None , **kwargs )
Interface to add image to wandb logger.
Parameters
```
- image(np.ndarray | Figure) – Image to log
- name(str | None) – The tag of the image Defaults toNone.
- kwargs– Additional arguments to _wandb.Image_
**Return type**
None
save()
Upload images to wandb server. :rtype:None

```
Note: There is a limit on the number of images that can be logged together to the wandb server.
```
**3.3.6 Callbacks**

Callbacks for Anomalib models.

classanomalib.callbacks.GraphLogger

```
Bases:Callback
Log model graph to respective logger.
```
**196 Chapter 3. Guides**


```
Examples
```
```
Log model graph to Tensorboard
```
```
>>> from anomalib.callbacks importGraphLogger
>>> from anomalib.loggers importAnomalibTensorBoardLogger
>>> from anomalib.engine importEngine
...
>>>logger = AnomalibTensorBoardLogger()
>>>callbacks = [GraphLogger()]
>>>engine = Engine(logger=logger, callbacks=callbacks)
```
```
Log model graph to Comet
```
```
>>> from anomalib.loggers importAnomalibCometLogger
>>> from anomalib.engine importEngine
...
>>>logger = AnomalibCometLogger()
>>>callbacks = [GraphLogger()]
>>>engine = Engine(logger=logger, callbacks=callbacks)
```
```
on_train_end( trainer , pl_module )
Unwatch model if configured for wandb and log it model graph in Tensorboard if specified.
Parameters
```
- trainer(Trainer) – Trainer object which contans reference to loggers.
- pl_module(LightningModule) – LightningModule object which is logged.
**Return type**
None
on_train_start( _trainer_ , _pl_module_ )
Log model graph to respective logger.
**Parameters**
- trainer(Trainer) – Trainer object which contans reference to loggers.
- pl_module(LightningModule) – LightningModule object which is logged.
**Return type**
None

classanomalib.callbacks.LoadModelCallback( _weights_path_ )

```
Bases:Callback
Callback that loads the model weights from the state dict.
```
**3.3. Reference Guide 197**


```
Examples
```
```
>>> from anomalib.callbacks importLoadModelCallback
>>> from anomalib.engine importEngine
...
>>>callbacks = [LoadModelCallback(weights_path="path/to/weights.pt")]
>>>engine = Engine(callbacks=callbacks)
```
```
setup( trainer , pl_module , stage=None )
Call when inference begins.
Loads the model weights fromweights_pathinto the PyTorch module.
Return type
None
```
classanomalib.callbacks.ModelCheckpoint( _dirpath=None_ , _filename=None_ , _monitor=None_ , _verbose=False_ ,
_save_last=None_ , _save_top_k=1_ , _save_weights_only=False_ ,
_mode='min'_ , _auto_insert_metric_name=True_ ,
_every_n_train_steps=None_ , _train_time_interval=None_ ,
_every_n_epochs=None_ , _save_on_train_epoch_end=None_ ,
_enable_version_counter=True_ )
Bases:ModelCheckpoint
Anomalib Model Checkpoint Callback.
This class overrides the Lightning ModelCheckpoint callback to enable saving checkpoints without running any
training steps. This is useful for zero-/few-shot models, where the fit sequence only consists of validation.
To enable saving checkpoints without running any training steps, we need to override two checks which are being
called in theon_validation_endmethod of the parent class: -_should_save_on_train_epoch_end: This
method checks whether the checkpoint should be saved at the end of a
training epoch, or at the end of the validation sequence. We modify this method to default to
saving at the end of the validation sequence when the model is of zero- or few-shot type, unless
save_on_train_epoch_endis specifically set by the user.

- _should_skip_saving_checkpoint **: This method checks whether the checkpoint should be saved**
    **at all. We modify**
       this method to allow saving during both theFITTINGandVALIDATINGstates. In addition, we allow
       saving if the global step has not changed since the last checkpoint, but only for zero- and few-shot
       models. This is needed because both the last global step and the last checkpoint remain unchanged
       during zero-/few-shot training, which would otherwise prevent saving checkpoints during validation.

classanomalib.callbacks.TimerCallback

```
Bases:Callback
Callback that measures the training and testing time of a PyTorch Lightning module.
```
**198 Chapter 3. Guides**


```
Examples
```
```
>>> from anomalib.callbacks importTimerCallback
>>> from anomalib.engine importEngine
...
>>>callbacks = [TimerCallback()]
>>>engine = Engine(callbacks=callbacks)
```
```
on_fit_end( trainer , pl_module )
Call when fit ends.
Prints the time taken for training.
Parameters
```
- trainer(Trainer) – PyTorch Lightning trainer.
- pl_module(LightningModule) – Current training module.
**Return type**
None
**Returns**
None
on_fit_start( _trainer_ , _pl_module_ )
Call when fit begins.
Sets the start time to the time training started.
**Parameters**
- trainer(Trainer) – PyTorch Lightning trainer.
- pl_module(LightningModule) – Current training module.
**Return type**
None
**Returns**
None
on_test_end( _trainer_ , _pl_module_ )
Call when the test ends.
Prints the time taken for testing and the throughput in frames per second.
**Parameters**
- trainer(Trainer) – PyTorch Lightning trainer.
- pl_module(LightningModule) – Current training module.
**Return type**
None
**Returns**
None
on_test_start( _trainer_ , _pl_module_ )
Call when the test begins.
Sets the start time to the time testing started. Goes over all the test dataloaders and adds the number of
images in each.

**3.3. Reference Guide 199**


```
Parameters
```
- trainer(Trainer) – PyTorch Lightning trainer.
- pl_module(LightningModule) – Current training module.
**Return type**
None
**Returns**
None

**3.3.7 CLI**

Anomalib CLI.

classanomalib.cli.AnomalibCLI( _args=None_ )

```
Bases:object
Implementation of a fully configurable CLI tool for anomalib.
The advantage of this tool is its flexibility to configure the pipeline from both the CLI and a configuration file
(.yaml or .json). It is even possible to use both the CLI and a configuration file simultaneously. For more details,
the reader could refer to PyTorch Lightning CLI documentation.
save_config_kwargsis set tooverwrite=Trueso that theSaveConfigCallbackoverwrites the config if it
already exists.
add_arguments_to_parser( parser )
Extend trainer’s arguments to add engine arguments. :rtype:None
```
```
Note: SinceEngineparameters are manually added, any change to theEngineclass should be reflected
manually.
```
```
add_export_arguments( parser )
Add export arguments to the parser.
Return type
None
add_predict_arguments( parser )
Add predict arguments to the parser.
Return type
None
add_subcommands( **kwargs )
Initialize base subcommands and add anomalib specific on top of it.
Return type
None
add_train_arguments( parser )
Add train arguments to the parser.
Return type
None
```
**200 Chapter 3. Guides**


```
add_trainer_arguments( parser , subcommand )
Add train arguments to the parser.
Return type
None
static anomalib_subcommands()
Return a dictionary of subcommands and their description.
Return type
dict[str,dict[str,str]]
before_instantiate_classes()
Modify the configuration to properly instantiate classes and sets up tiler.
Return type
None
property export: Callable
Export the model using engine’s export method.
property fit: Callable
Fit the model using engine’s fit method.
init_parser( **kwargs )
Method that instantiates the argument parser.
Return type
ArgumentParser
instantiate_classes()
Instantiate classes depending on the subcommand.
For trainer related commands it instantiates all the model, datamodule and trainer classes. But for sub-
commands we do not want to instantiate any trainer specific classes such as datamodule, model, etc This is
because the subcommand is responsible for instantiating and executing code based on the passed config
Return type
None
instantiate_engine()
Instantiate the engine. :rtype:None
```
```
Note: Most of the code in this method is taken fromLightningCLI’sinstantiate_trainermethod.
Refer to that method for more details.
```
```
property predict: Callable
Predict using engine’s predict method.
static subcommands()
Skip predict subcommand as it is added later.
Return type
dict[str,set[str]]
property test: Callable
Test the model using engine’s test method.
```
**3.3. Reference Guide 201**


```
property train: Callable
Train the model using engine’s train method.
property validate: Callable
Validate the model using engine’s validate method.
```
**3.3.8 Deployment**

Functions for Inference and model deployment.

classanomalib.deploy.ExportType( _value_ , _names=None_ , _*_ , _module=None_ , _qualname=None_ , _type=None_ ,
_start=1_ , _boundary=None_ )
Bases:str,Enum
Model export type.

```
Examples
```
```
>>> from anomalib.deploy importExportType
>>>ExportType.ONNX
'onnx'
>>>ExportType.OPENVINO
'openvino'
>>>ExportType.TORCH
'torch'
```
classanomalib.deploy.OpenVINOInferencer( _path_ , _metadata=None_ , _device='AUTO'_ , _task=None_ ,
_config=None_ )
Bases:Inferencer
OpenVINO implementation for the inference.
**Parameters**

- path(str | Path) – Path to the openvino onnx, xml or bin file.
- metadata(str | Path | dict, optional) – Path to metadata file or a dict object
    defining the metadata. Defaults toNone.
- device(str | None, optional) – Device to run the inference on (AUTO, CPU, GPU,
    NPU). Defaults toAUTO.
- task(TaskType | None, optional) – Task type. Defaults toNone.
- config(dict | None, optional) – Configuration parameters for the inference Defaults
    toNone.

**202 Chapter 3. Guides**


```
Examples
```
```
Assume that we have an OpenVINO IR model and metadata files in the following structure:
```
```
$ tree weights
./weights
model.bin
model.xml
metadata.json
```
```
We could then createOpenVINOInferenceras follows:
```
```
>>> from anomalib.deploy.inferencers importOpenVINOInferencer
>>>inferencer = OpenVINOInferencer(
... path="weights/model.xml",
... metadata="weights/metadata.json",
... device="CPU",
...)
```
```
This will ensure that the model is loaded on theCPUdevice and the metadata is loaded from themetadata.json
file. To make a prediction, we can simply call thepredictmethod:
```
```
>>>prediction = inferencer.predict(image="path/to/image.jpg")
```
```
Alternatively we can also pass the image as a PIL image or numpy array:
```
```
>>> from PIL importImage
>>>image = Image.open("path/to/image.jpg")
>>>prediction = inferencer.predict(image=image)
```
```
>>> import numpy as np
>>>image = np.random.rand(224, 224, 3)
>>>prediction = inferencer.predict(image=image)
```
```
predictionwill be anImageResultobject containing the prediction results. For example, to visualize the
heatmap, we can do the following:
```
```
>>> from matplotlib importpyplot as plt
>>>plt.imshow(result.heatmap)
```
```
It is also possible to visualize the true and predicted masks if the task isTaskType.SEGMENTATION:
```
```
>>>plt.imshow(result.gt_mask)
>>>plt.imshow(result.pred_mask)
```
```
forward( image )
Forward-Pass input tensor to the model.
Parameters
image(np.ndarray) – Input tensor.
Returns
Output predictions.
Return type
np.ndarray
```
**3.3. Reference Guide 203**


```
load_model( path )
Load the OpenVINO model.
Parameters
path(str | Path | tuple[bytes, bytes]) – Path to the onnx or xml and bin files or
tuple of .xml and .bin data as bytes.
Returns
Input and Output blob names
together with the Executable network.
Return type
[tuple[str, str, ExecutableNetwork]]
post_process( predictions , metadata=None )
Post process the output predictions.
Parameters
```
- predictions(np.ndarray) – Raw output predicted by the model.
- metadata(Dict, optional) – Metadata. Post-processing step sometimes requires ad-
    ditional metadata such as image shape. This variable comprises such info. Defaults to
    None.
**Returns**
Post processed prediction results.
**Return type**
dict[str, Any]
pre_process( _image_ )
Pre-process the input image by applying transformations.
**Parameters**
image(np.ndarray) – Input image.
**Returns**
pre-processed image.
**Return type**
np.ndarray
predict( _image_ , _metadata=None_ )
Perform a prediction for a given input image.
The main workflow is (i) pre-processing, (ii) forward-pass, (iii) post-process.
**Parameters**
- image(Union[str, np.ndarray]) – Input image whose output is to be predicted. It
could be either a path to image or numpy array itself.
- metadata(dict[str,Any] |None) – Metadata information such as shape, threshold.
**Returns**
Prediction results to be visualized.
**Return type**
ImageResult

**204 Chapter 3. Guides**


classanomalib.deploy.TorchInferencer( _path_ , _device='auto'_ )

```
Bases:Inferencer
PyTorch implementation for the inference.
Parameters
```
- path(str | Path) – Path to Torch model weights.
- device(str) – Device to use for inference. Options areauto,cpu,cuda. Defaults toauto.

```
Examples
```
```
Assume that we have a Torchptmodel and metadata files in the following structure:
```
```
>>> from anomalib.deploy.inferencers importTorchInferencer
>>>inferencer = TorchInferencer(path="path/to/torch/model.pt", device="cpu")
```
```
This will ensure that the model is loaded on theCPUdevice. To make a prediction, we can simply call the
predictmethod:
```
```
>>> from anomalib.data.utils importread_image
>>>image = read_image("path/to/image.jpg")
>>>result = inferencer.predict(image)
```
```
resultwill be anImageResultobject containing the prediction results. For example, to visualize the heatmap,
we can do the following:
```
```
>>> from matplotlib importpyplot as plt
>>>plt.imshow(result.heatmap)
```
```
It is also possible to visualize the true and predicted masks if the task isTaskType.SEGMENTATION:
```
```
>>>plt.imshow(result.gt_mask)
>>>plt.imshow(result.pred_mask)
```
```
forward( image )
Forward-Pass input tensor to the model.
Parameters
image(torch.Tensor) – Input tensor.
Returns
Output predictions.
Return type
Tensor
load_model( path )
Load the PyTorch model.
Parameters
path(str | Path) – Path to the Torch model.
Returns
Torch model.
Return type
(nn.Module)
```
**3.3. Reference Guide 205**


```
post_process( predictions , metadata=None )
Post process the output predictions.
Parameters
```
- predictions(Tensor | list[torch.Tensor] | dict[str, torch.Tensor]) –
    Raw output predicted by the model.
- metadata(dict, optional) – Meta data. Post-processing step sometimes requires ad-
    ditional meta data such as image shape. This variable comprises such info. Defaults to
    None.
**Returns**
Post processed prediction results.
**Return type**
dict[str, str | float | np.ndarray]
pre_process( _image_ )
Pre process the input image.
**Parameters**
image(np.ndarray) – Input image
**Returns**
pre-processed image.
**Return type**
Tensor
predict( _image_ , _metadata=None_ )
Perform a prediction for a given input image.
The main workflow is (i) pre-processing, (ii) forward-pass, (iii) post-process.
**Parameters**
- image(Union[str, np.ndarray]) – Input image whose output is to be predicted. It
could be either a path to image or numpy array itself.
- metadata(dict[str,Any] |None) – Metadata information such as shape, threshold.
**Returns**
Prediction results to be visualized.
**Return type**
ImageResult

anomalib.deploy.export_to_onnx( _model_ , _export_root_ , _transform=None_ , _task=None_ ,
_export_type=ExportType.ONNX_ )
Export model to onnx.
**Parameters**

- model(AnomalyModule) – Model to export.
- export_root(Path) – Path to the root folder of the exported model.
- transform(Transform, optional) – Input transforms used for the model. If not pro-
    vided, the transform is taken from the model. Defaults toNone.
- task(TaskType | None) – Task type. Defaults toNone.

**206 Chapter 3. Guides**


- export_type(ExportType) – Mode to export the model. Since this method is used by
    OpenVINO export as well, we need to pass the export type so that the right export path is
    created. Defaults toExportType.ONNX.
**Returns**
Path to the exported onnx model.
**Return type**
Path

```
Examples
```
```
Export the Lightning Model to ONNX:
```
```
>>> from anomalib.models importPatchcore
>>> from anomalib.data importVisa
>>> from anomalib.deploy importexport_to_onnx
...
>>>datamodule = Visa()
>>>model = Patchcore()
...
>>>export_to_onnx(
... model=model,
... export_root="path/to/export",
... transform=datamodule.test_data.transform,
... task=datamodule.test_data.task
...)
```
```
Using Custom Transforms: This example shows how to use a customComposeobject for thetransformargu-
ment.
```
```
>>>export_to_onnx(
... model=model,
... export_root="path/to/export",
... task="segmentation",
...)
```
anomalib.deploy.export_to_openvino( _export_root_ , _model_ , _transform=None_ , _ov_args=None_ , _task=None_ )

```
Convert onnx model to OpenVINO IR.
Parameters
```
- export_root(Path) – Path to the export folder.
- model(AnomalyModule) – AnomalyModule to export.
- transform(Transform, optional) – Input transforms used for the model. If not pro-
    vided, the transform is taken from the model. Defaults toNone.
- ov_args(dict[str,Any] |None) – Model optimizer arguments for OpenVINO model con-
    version. Defaults toNone.
- task(TaskType | None) – Task type. Defaults toNone.
**Returns**
Path to the exported onnx model.
**Return type**
Path

**3.3. Reference Guide 207**


```
Raises
ModuleNotFoundError– If OpenVINO is not installed.
Returns
Path to the exported OpenVINO IR.
Return type
Path
```
```
Examples
```
```
Export the Lightning Model to OpenVINO IR: This example demonstrates how to export the Lightning Model
to OpenVINO IR.
```
```
>>> from anomalib.models importPatchcore
>>> from anomalib.data importVisa
>>> from anomalib.deploy importexport_to_openvino
...
>>>datamodule = Visa()
>>>model = Patchcore()
...
>>>export_to_openvino(
... export_root="path/to/export",
... model=model,
... input_size=(224, 224),
... transform=datamodule.test_data.transform,
... task=datamodule.test_data.task
...)
```
```
Using Custom Transforms: This example shows how to use a customTransformobject for thetransform
argument.
```
```
>>> from torchvision.transforms.v2 importResize
>>>transform = Resize(224, 224)
...
>>>export_to_openvino(
... export_root="path/to/export",
... model=model,
... transform=transform,
... task="segmentation",
...)
```
anomalib.deploy.export_to_torch( _model_ , _export_root_ , _transform=None_ , _task=None_ )

```
Export AnomalibModel to torch.
Parameters
```
- model(AnomalyModule) – Model to export.
- export_root(Path) – Path to the output folder.
- transform(Transform, optional) – Input transforms used for the model. If not pro-
    vided, the transform is taken from the model. Defaults toNone.
- task(TaskType | None) – Task type. Defaults toNone.
**Returns**
Path to the exported pytorch model.

**208 Chapter 3. Guides**


```
Return type
Path
```
```
Examples
```
```
Assume that we have a model to train and we want to export it to torch format.
```
```
>>> from anomalib.data importVisa
>>> from anomalib.models importPatchcore
>>> from anomalib.engine importEngine
...
>>>datamodule = Visa()
>>>model = Patchcore()
>>>engine = Engine()
...
>>>engine.fit(model, datamodule)
```
```
Now that we have a model trained, we can export it to torch format.
```
```
>>> from anomalib.deploy importexport_to_torch
...
>>>export_to_torch(
... model=model,
... export_root="path/to/export",
... transform=datamodule.test_data.transform,
... task=datamodule.test_data.task,
...)
```
**3.4 How to Guide**

This section contains how-to guides for anomalib.

```
Warning: This section is under construction
```
**3.4.1 Tutorials**

Data Learn more about anomalib datamodules.

```
Models Learn more about image and video models.
Engine Learn more about anomalib Engine.
```
Metrics Learn more about anomalib metrics

Loggers Learn more about anomalib loggers

Callbacks Learn more about anomalib callbacks

CLI Learn more about anomalib CLI

Deployment Learn more about anomalib’s deployment capabilities

Pipelines Learn more about anomalib hpo, sweep and benchmarking pipelines

**3.4. How to Guide 209**


**Data Tutorials**

This section contains tutorials on how to fully utilize the data components of anomalib.

Train on Custom Data. Learn more about how to useFolderdataset to train anomalib models on your custom data.

**Custom Data**

This tutorial will show you how to train anomalib models on your custom data. More specifically, we will show you
how to use the _Folder_ dataset to train anomalib models on your custom data.

```
Warning: This tutorial assumes that you have already installed anomalib. If not, please refer to the installation
section.
```
**Note:** We will use our hazelnut_toy dataset to show the capabilities of the _Folder_ dataset, but you can use any dataset
you want.

We will split the section to two tasks: Classification and Segmentation.

**Classification Dataset**

In certain use-cases, ground-truth masks for the abnormal images may not be available. In such cases, we could use
the classification task to train a model that will be able to detect the abnormal images in the test set.

We will split this section into two tasks:

- Classification with normal and abnormal images, and
- Classification with only normal images.

**With Normal and Abnormal Images**

We could use _Folder_ datamodule to train a model on this dataset. We could run the following python code to create
the custom datamodule:

**Code Syntax**

**API**

# Import the datamodule
from anomalib.data importFolder

# Create the datamodule
datamodule = Folder(
name="hazelnut_toy",
root="datasets/hazelnut_toy",
normal_dir="good",
(continues on next page)

**210 Chapter 3. Guides**


(continued from previous page)
abnormal_dir="crack",
task="classification",
)

# Setup the datamodule
datamodule.setup()

**Note:** As can be seen above, we only need to specify thetaskargument toclassification. We could have also
useTaskType.CLASSIFICATIONinstead ofclassification.

The _Folder_ datamodule will create training, validation, test and prediction datasets and dataloaders for us. We can
access the datasets and dataloaders by following the same approach as in the segmentation task.

When we check the samples from the dataloaders, we will see that themaskkey is not present in the samples. This is
because we do not need the masks for the classification task.

i, train_data = next(enumerate(datamodule.train_dataloader()))
print(train_data.keys())
# dict_keys(['image_path','label', 'image'])

i, val_data = next(enumerate(datamodule.val_dataloader()))
print(val_data.keys())
# dict_keys(['image_path','label', 'image'])

i, test_data = next(enumerate(datamodule.test_dataloader()))
print(test_data.keys())
# dict_keys(['image_path','label', 'image'])

Training the model is as simple as running the following command:

# Import the model and engine
from anomalib.models import Patchcore
from anomalib.engine import Engine

# Create the model and engine
model = Patchcore()
engine = Engine(task="classification")

# Train a Patchcore model on the given datamodule
engine.train(datamodule=datamodule, model=model)

where we train a Patchcore model on this custom dataset with default model parameters.

**3.4. How to Guide 211**


### CLI

Here is the dataset config to create the same custom datamodule:

class_path: anomalib.data.Folder
init_args:
name: "MVTec"
root: "datasets/MVTec/transistor"
normal_dir: "train/good"
abnormal_dir: "test/bent_lead"
normal_test_dir: "test/good"
normal_split_ratio: 0
extensions: [".png"]
image_size: [256, 256]
train_batch_size: 32
eval_batch_size: 32
num_workers: 8
task: classification
train_transform: null
eval_transform: null
test_split_mode: from_dir
test_split_ratio: 0.2
val_split_mode: same_as_test
val_split_ratio: 0.5
seed: null

Assume that we have saved the above config file asclassification.yaml. We could run the following CLI command
to train a Patchcore model on above dataset:

anomalib train --data classification.yaml --model anomalib.models.Patchcore --task␣
˓→CLASSIFICATION

**Note:** As can be seen above, we also need to specify thetaskargument toCLASSIFICATIONto explicitly tell anomalib
that we want to train a classification model. This is because the defaulttaskisSEGMENTATIONwithinEngine.

**With Only Normal Images**

There are certain cases where we only have normal images in our dataset but would like to train a classification model.

This could be done in two ways:

- Train the model and skip the validation and test steps, as we do not have abnormal images to validate and test the
    model on, or
- Use the synthetic anomaly generation feature to create abnormal images from normal images, and perform the
    validation and test steps.

For now we will focus on the second approach.

**212 Chapter 3. Guides**


**With Validation and Testing via Synthetic Anomalies**

If we want to check the performance of the model, we will need to have abnormal images to validate and test the model
on. During the validation stage, these anomalous images are used to normalize the anomaly scores and find the best
threshold that separates normal and abnormal images.

Anomalib provides synthetic anomaly generation capabilities to create abnormal images from normal images so we
could check the performance. We could use the _Folder_ datamodule to train a model on this dataset.

**Code Syntax**

**API**

# Import the datamodule
from anomalib.data importFolder
from anomalib.data.utils import TestSplitMode

# Create the datamodule
datamodule = Folder(
name="hazelnut_toy",
root="datasets/hazelnut_toy",
normal_dir="good",
test_split_mode=TestSplitMode.SYNTHETIC,
task="classification",
)

# Setup the datamodule
datamodule.setup()

Once the datamodule is setup, the rest of the process is the same as in the previous classification example.

# Import the model and engine
from anomalib.models import Patchcore
from anomalib.engine import Engine

# Create the model and engine
model = Patchcore()
engine = Engine(task="classification")

# Train a Patchcore model on the given datamodule
engine.train(datamodule=datamodule, model=model)

where we train a Patchcore model on this custom dataset with default model parameters.

**3.4. How to Guide 213**


### CLI

Here is the CLI command to create the same custom datamodule with only normal images. We only need to change
thetest_split_modeargument toSYNTHETICto generate synthetic anomalies.

class_path: anomalib.data.Folder
init_args:
root: "datasets/hazelnut_toy"
normal_dir: "good"
normalization: imagenet
test_split_mode: synthetic
task: classification

Assume that we have saved the above config file asnormal.yaml. We could run the following CLI command to train
a Patchcore model on above dataset:

anomalib train --data normal.yaml --model anomalib.models.Patchcore --task CLASSIFICATION

**Note:** As shown in the previous classification example, we, again, need to specify thetaskargument to
CLASSIFICATIONto explicitly tell anomalib that we want to train a classification model. This is because the default
taskisSEGMENTATIONwithinEngine.

**Segmentation Dataset**

Assume that we have a dataset in which the training set contains only normal images, and the test set contains both
normal and abnormal images. We also have masks for the abnormal images in the test set. We want to train an anomaly
segmentation model that will be able to detect the abnormal regions in the test set.

**With Normal and Abnormal Images**

We could use _Folder_ datamodule to load the hazelnut dataset in a format that is readable by Anomalib’s models.

**Code Syntax**

### API

We could run the following python code to create the custom datamodule:

# Import the datamodule
from anomalib.data importFolder

# Create the datamodule
datamodule = Folder(
name="hazelnut_toy",
root="datasets/hazelnut_toy",
normal_dir="good",
abnormal_dir="crack",
mask_dir="mask/crack",
normal_split_ratio=0.2,
(continues on next page)

**214 Chapter 3. Guides**


```
(continued from previous page)
```
)

# Setup the datamodule
datamodule.setup()

The _Folder_ datamodule will create training, validation, test and prediction datasets and dataloaders for us. We can
access the datasets and dataloaders using the following attributes:

# Access the datasets
train_dataset = datamodule.train_data
val_dataset = datamodule.val_data
test_dataset = datamodule.test_data

# Access the dataloaders
train_dataloader = datamodule.train_dataloader()
val_dataloader = datamodule.val_dataloader()
test_dataloader = datamodule.test_dataloader()

To check what individual samples from dataloaders look like, we can run the following command:

i, train_data = next(enumerate(datamodule.train_dataloader()))
print(train_data.keys())
# dict_keys(['image_path','label', 'image', 'mask_path', 'mask'])

i, val_data = next(enumerate(datamodule.val_dataloader()))
print(val_data.keys())
# dict_keys(['image_path','label', 'image', 'mask_path', 'mask'])

i, test_data = next(enumerate(datamodule.test_dataloader()))
print(test_data.keys())
# dict_keys(['image_path','label', 'image', 'mask_path', 'mask'])

We could check the shape of the images and masks using the following commands:

print(train_data["image"].shape)
# torch.Size([32, 3, 256, 256])

print(train_data["mask"].shape)
# torch.Size([32, 256, 256])

Training the model is as simple as running the following command:

# Import the model and engine
from anomalib.models import Patchcore
from anomalib.engine import Engine

# Create the model and engine
model = Patchcore()
engine = Engine()

# Train a Patchcore model on the given datamodule
engine.train(datamodule=datamodule, model=model)

where we train a Patchcore model on this custom dataset with default model parameters.

**3.4. How to Guide 215**


### CLI

Here is the CLI command to create the same custom datamodule:

class_path: anomalib.data.Folder
init_args:
root: "datasets/hazelnut_toy"
normal_dir: "good"
abnormal_dir: "crack"
mask_dir: "mask/crack"
normal_split_ratio: 0.2
normalization: imagenet

Assume that we have saved the above config file assegmentation.yaml. We could run the following CLI command
to train a Patchcore model on above dataset:

anomalib train --data segmentation.yaml --model anomalib.models.Patchcore

This example demonstrates how to create a segmentation dataset with normal and abnormal images. We could expand
this example to create a segmentation dataset with only normal images.

**With Only Normal Images**

There are certain cases where we only have normal images in our dataset but would like to train a segmentation model.
This could be done in two ways:

- Train the model and skip the validation and test steps, as we do not have abnormal images to validate and test the
    model on, or
- Use the synthetic anomaly generation feature to create abnormal images from normal images, and perform the
    validation and test steps.

For now we will focus on the second approach.

**With Validation and Testing via Synthetic Anomalies**

We could use the synthetic anomaly generation feature again to create abnormal images from normal images. We
could then use the _Folder_ datamodule to train a model on this dataset. Here is the python code to create the custom
datamodule:

**Code Syntax**

### API

We could run the following python code to create the custom datamodule:

# Import the datamodule
from anomalib.data importFolder
from anomalib.data.utils import TestSplitMode

# Create the datamodule
datamodule = Folder(
(continues on next page)

**216 Chapter 3. Guides**


(continued from previous page)
name="hazelnut_toy",
root="datasets/hazelnut_toy",
normal_dir="good",
test_split_mode=TestSplitMode.SYNTHETIC,
)

# Setup the datamodule
datamodule.setup()

As can be seen from the code above, we only need to specify thetest_split_modeargument toSYNTHETIC. The
_Folder_ datamodule will create training, validation, test and prediction datasets and dataloaders for us.

To check what individual samples from dataloaders look like, we can run the following command:

i, train_data = next(enumerate(datamodule.train_dataloader()))
print(train_data.keys())
# dict_keys(['image_path','label', 'image', 'mask_path', 'mask'])

i, val_data = next(enumerate(datamodule.val_dataloader()))
print(val_data.keys())
# dict_keys(['image_path','label', 'image', 'mask_path', 'mask'])

i, test_data = next(enumerate(datamodule.test_dataloader()))
print(test_data.keys())
# dict_keys(['image_path','label', 'image', 'mask_path', 'mask'])

We could check the shape of the images and masks using the following commands:

print(train_data["image"].shape)
# torch.Size([32, 3, 256, 256])

print(train_data["mask"].shape)
# torch.Size([32, 256, 256])

### CLI

Here is the CLI command to create the same custom datamodule with only normal images. We only need to change
thetest_split_modeargument toSYNTHETICto generate synthetic anomalies.

class_path: anomalib.data.Folder
init_args:
root: "datasets/hazelnut_toy"
normal_dir: "good"
normalization: imagenet
train_batch_size: 32
eval_batch_size: 32
num_workers: 8
task: segmentation
train_transform: null
eval_transform: null
test_split_mode: synthetic
test_split_ratio: 0.2
(continues on next page)

**3.4. How to Guide 217**


```
(continued from previous page)
val_split_mode: same_as_test
val_split_ratio: 0.5
seed: null
```
Assume that we have saved the above config file assynthetic.yaml. We could run the following CLI command to
train a Patchcore model on above dataset:

anomalib train --data synthetic.yaml --model anomalib.models.Patchcore

**Model Tutorials**

This section contains tutorials on how to use the different model components of anomalib.

Feature Extractors Learn more about how to use the different backbones as feature extractors.

**Feature extractors**

This guide demonstrates how different backbones can be used as feature extractors for anomaly detection models. Most
of these models use Timm Feature Extractor except **CSFLOW** which uses TorchFx Feature Extractor. Here we show
how to use API and CLI to use different backbones as feature extractors.

**See also:**

For specifics of implementation refer to implementation classesTimm Feature ExtractorandTorchFx Feature
Extractor

**Available backbones and layers**

**Timm**

Available Timm models are listed on Timm GitHub page.

In most cases, we want to use a pretrained backbone, so can get a list of all such models using the following code:

import timm
# list all pretrained models in timm
formodel_namein timm.list_models(pretrained=True):
print(model_name)

Once we have a model selected we can obtain available layer names using the following code:

import timm
model = timm.create_model("resnet18", features_only=True)
# Print module names
print(model.feature_info.module_name())
>>>['act1', 'layer1','layer2','layer3', 'layer4']

model = timm.create_model("mobilenetv3_large_100", features_only=True)
print(model.feature_info.module_name())
>>>['blocks.0.0','blocks.1.1','blocks.2.2', 'blocks.4.1', 'blocks.6.0']

**218 Chapter 3. Guides**


```
We can then use selected model name and layer names with either API or using config file.
```
```
TorchFX
```
```
When using TorchFX for feature extraction, you can use either model name, custom model, or instance of model. In
this guide, we will cover pretrained models from Torchvision passed by name. For use of the custom model or instance
of a model refer toTorchFxFeatureExtractor class examples.
Available torchvision models are listed on Torchvision models page.
We can get layer names for selected model using the following code:
```
```
# Import model and function to list names
from torchvision.models importresnet18
from torchvision.models.feature_extraction importget_graph_node_names
```
```
# Make an instance of model with default (latest) weights
model = resnet18(weights="DEFAULT")
```
```
# Get and print node (layer) names
train_nodes, eval_nodes = get_graph_node_names(model)
print(eval_nodes)
>>>['x','conv1','bn1', 'relu', 'maxpool', ...,'layer4.1.relu_1', 'avgpool','flatten',
˓→'fc']
```
```
As a result, we get a list of all model nodes, which is quite long.
Now for example, if we want only output from the last node in the block namedlayer4, we specifylayer4.1.relu_1.
If we want to avoid writinglayer4.1.relu_1to get the last output oflayer4block, we can shorten it tolayer4.
We can then use selected model name and layer names with either API or using config file.
See also:
Additional info about TorchFX feature extraction can be found on PyTorch FX page and feature_extraction documen-
tation page.
```
```
Warning: Some models might not support every backbone.
```
```
Backbone and layer selection
```
### API

```
When using API, we need to specifybackboneandlayerswhen instantiating the model with a non-default backbone.
```
1 # Import the required modules
2 from anomalib.data importMVTec
3 from anomalib.models import Padim
4 from anomalib.engine import Engine
5
6 # Initialize the datamodule, model, and engine
7 datamodule = MVTec(num_workers=0)
8 # Specify backbone and layers
(continues on next page)

```
3.4. How to Guide 219
```

(continued from previous page)
9 model = Padim(backbone="resnet18", layers=["layer1", "layer2"])
10 engine = Engine(image_metrics=["AUROC"], pixel_metrics=["AUROC"])
11
12 # Train the model
13 engine.fit(datamodule=datamodule, model=model)

### CLI

```
In the following example config, we can see that we need to specify two parameters: thebackboneandlayerslist.
```
1 model:
2 class_path: anomalib.models.Padim
3 init_args:
4 layers:
5 - blocks.1.1
6 - blocks.2.2
7 input_size: null
8 backbone: mobilenetv3_large_100
9 pre_trained: true
10 n_features: 50

```
Then we can train using:
```
```
anomalib train --config <path/to/config>
```
**3.5 Topic Guide**

```
This section contains design documents and other internals of anomalib.
```
```
Warning: This section is under construction
```
**3.6 Developer Guide**

```
This section contains information for developers who want to contribute to Anomalib.
Anomalib Software Design Document. Learn more about the architecture, components, and design decisions of
Anomalib.
Contribution Guidelies. Learn how to contribute to Anomalib.
Code Review Checklist. Learn the criteria for reviewing code.
```
```
220 Chapter 3. Guides
```

**3.6.1 Software Design Document**

The Software Design Document (SDD) provides a comprehensive overview of the design and architecture of Anomalib.
It is intended for developers, architects, and anyone interested in the technical underpinnings of the software.

**1. Introduction**

**Purpose**

The purpose of this document is to give a detailed overview of the architecture, components, and design decisions of
Anomalib. It serves as a guide for new developers and a reference for all stakeholders involved in development.

**Scope**

This section outlines the scope of the software, including its intended functionality, user base, and integration with
other systems or technologies.

**2. Overall Description**

**Product Perspective**

Anomalib is a deep learning library that aims to collect state-of-the-art anomaly detection algorithms for benchmarking
on both public and private datasets. Anomalib provides several ready-to-use implementations of anomaly detection
algorithms described in the recent literature, as well as a set of tools that facilitate the development and implementation
of custom models. The library has a strong focus on visual anomaly detection, where the goal of the algorithm is to
detect and/or localize anomalies within images or videos in a dataset.

**Product Features**

- Modular Design
    Offers a plug-and-play architecture that allows users to easily modify the existing anomaly detection algorithms
    or add new ones based on their specific requirements.
- Wide Algorithm Support
    Includes implementations of the state-of-the-art (sota) anomaly detection algorithms, covering a variety of tech-
    niques such as autoencoders, GANs, flow-based models, and more.
- Real-time Detection
    With a support of multiple inferencers such as Torch, Lightning, ONNX, OpenVINO and Gradio, Anomalib is
    capable of performing real-time anomaly detection on streaming data.
- Visualization Tools
    Comes with a suite of visualization tools for analyzing anomalies, understanding model decisions, and interpret-
    ing results.
- Customization and Extensibility
    Provides APIs for customization and extension, enabling researchers and practitioners to experiment with sota
    algorithms and techniques.

**3.6. Developer Guide 221**


- Comprehensive Documentation and Community Support
    Features detailed documentation and a supportive community forum to assist users in implementation and trou-
    bleshooting.

**User Classes and Characteristics**

- Research Scientists
    Academics and researchers conducting studies in anomaly detection, requiring a flexible toolset for experiment-
    ing with different models and techniques.
- Data Scientists and Analysts
    Professionals looking to incorporate anomaly detection into their data analysis workflows without diving deeper
    into algorithmic complexities.
- Machine Learning Engineers
    Individuals focused on developing and deploying anomaly detection models, seeking a library that offers sota
    algorithms.
- Industry Practitioners
    Specialists in sectors like manufacturing, healthcare and security, who need to detect and respond to anomalies
    as part of risk management and quality control processes.

**Operating Environment**

Anomalib is designed to be environment-agnostic, capable of running on a variety of platforms including standalone
desktop environments, on-premises servers, and cloud-based infrastructures. The library supports both CPU and GPU-
based processing, with the latter providing significant performance improvements for large-scale anomaly detection
tasks. The library also supports edge devices for real-time anomaly detection in IoT and embedded systems.

**Assumptions and Dependencies**

Anomalib’s effectiveness and performance are dependent on the quality and granularity of the input data, as well as the
appropriateness of the chosen anomaly detection algorithms for the specific use case. It assumes that users have a basic
understanding of anomaly detection principles and are familiar with Python programming for custom implementations.

**3. System Architecture**

**Architectural Overview**

Anomalib is designed with a focus on modular, scalable, and flexible architecture to meet the diverse needs of anomaly
detection across various domains.

**222 Chapter 3. Guides**


**High-Level Workflow**

As shown in Figure _1_ , Anomalib’s high-level workflow consists of the following components:

- **Training [Optional]** : Involves training the anomaly detection model to learn patterns and characteristics of
    normal behavior. This step is optional for unsupervised algorithms. For zero-shot learning, the model is not
    trained on any specific dataset.
- **Validation/Test** : Validates the model by finding the threshold for anomaly detection and testing the model’s
    performance on a validation or test dataset.
- **Optimization [Optional]** : Involves optimizing the model by converting it to a more efficient form, such as a
    compressed or quantized model. This step utilizes tools such as ONNX, OpenVINO for optimization.
- **Deployment** : Deploys the model to a production environment for real-time or batch anomaly detection. This
    step uses multiple tools such as Torch, Lightning, ONNX, OpenVINO and Gradio for deployment.

**High-Level Architecture**

**3.6. Developer Guide 223**


As shown in Figure _2_ , Anomalib’s high-level architecture consists of the following components: API/CLI, Components
and Entry Points.

- **API/CLI** : Provides an interface for users to interact with the library, including APIs for data ingestion, config-
    uration, and result export, as well as a command-line interface for running anomaly detection tasks. Both API
    and CLI provides the same functionality.
- **Components** : Consists of the core components of Anomalib, including, Datamodules, Models, Callbacks, Met-
    rics, Visualizers and Engine.
- **Entry Points** : Includes the main entry points for running anomaly detection tasks, such astrain,validate,
    test,export, andpredict.
**4. Detailed System Design**

Here we will provide a detailed overview of the design and architecture of Anomalib, including the CLI, core compo-
nents and entyrpoints.

### CLI

As shown in Figure _3_ , Anomalib’s command-line interface (CLI) provides a user-friendly way to interact with the
library, allowing users to run anomaly detection tasks, configure algorithms, and visualize results. The CLI is imple-
mented using LightningCLI, a command-line interface framework for PyTorch Lightning that uses jsonargparse for
argument parsing. This allows users to sync their CLI arguments with their Python code, making it easier to maintain
and update.

**224 Chapter 3. Guides**


**Datamodules**

Datamodules in anomalib are responsible for preparing the data for training, validation, and testing. They are imple-
mented using PyTorch Lightning’sLightningDataModuleclass, which provides a set of APIs for loading, prepro-
cessing, and splitting the data. The datamodules are designed to be modular and extensible, allowing users to easily
modify the data loading and preprocessing steps based on their specific requirements. Anomalib has a base datamod-
ule, which is sub-classes by image, video and depth datamodules to provide the necessary functionality for loading and
preprocessing image, video and depth data.

**Models**

Anomalib provides a collection of anomaly models within the image and video domains. The models are implemented
sub-classing PyTorch Lightning’sLightningModuleclass, which is calledAnomalyModule, which provides a set
of APIs for defining the model architecture, loss function, and optimization algorithm. The models are designed to
be modular and extensible, allowing users to easily modify the model architecture and training workflow based on
their specific requirements. Anomalib has a base model, which is sub-classes by algorithm implementations such as
autoencoders, GANs, flow-based models, and more. The figure below demonstrates the high-level overview of model
architecture, and the implementation of Patchcore model.

**Callbacks**

We use LightningCallbacks to add functionality to the training loop, such as logging, early stopping, and model check-
pointing. For more details, refer to the Lightning Callbacks documentation. In addition to the built-in callbacks, we
have implemented custom callbacks for anomaly detection tasks, which could be accessed through thecallbacks
module.

**Metrics**

Similar to callbacks, we make use of Lightning Torchmetrics package, which provides a collection of metrics for
evaluating model performance. For more details, refer to the Torchmetrics documentation. We have also implemented
custom metrics such as PRO, sPRO and PIMO for anomaly detection tasks, which could be accessed through the
metricsmodule.

**Visualizers**

Anomalib provides a suite of visualization tools for analyzing anomalies, understanding model decisions, and inter-
preting results. Currently we have implemented visualizers for image and metrics, which could be accessed through
ImageVisualizerandMetricsVisualizerclasses, respectively. The figure below demonstrates the high-level
overview of visualizers.

**3.6. Developer Guide 225**


**226 Chapter 3. Guides**


**3.6. Developer Guide 227**


**Engine**

The figure below demonstrates the high-level architecture of the engine, which is responsible for orchestrating the
anomaly detection workflow, including data preparation, model training, validation, testing, export and predict.

Under the hood,Engineuses Lightning’sTrainerto manage the training workflow. One of the main distinguishing
features of theEngineis its ability of caching arguments to create the Trainer class after the datamodule and model are
created or modified. This allows the user to modify the datamodule and model without having to re-create the Trainer
class.

Another important feature of theEngineis its ability to export the trained model to OpenVINO format for deployment
on edge devices. This is achieved by using theopenvinomodule, which provides a set of tools for converting PyTorch
models to OpenVINO format.

Overall, Anomalib is designed with a focus on modular, scalable, and flexible architecture to meet the diverse needs
of anomaly detection across various domains. The library provides a comprehensive set of tools for data preparation,
model training, validation, testing, export, and predict, as well as a suite of visualization tools for analyzing anomalies,
understanding model decisions, and interpreting results.

**3.6.2 Contribution Guidelines**

**Getting Started**

To get started with Anomalib development, follow these steps:

**Fork and Clone the Repository**

First, fork the Anomalib repository by following the GitHub documentation on forking a repo. Then, clone your forked
repository to your local machine and create a new branch frommain.

**Set Up Your Development Environment**

Set up your development environment to start contributing. This involves installing the required dependencies and
setting up pre-commit hooks for code quality checks. Note that this guide assumes you are using Conda for package
management. However, the steps are similar for other package managers.

**228 Chapter 3. Guides**


**Development Environment Setup Instructions**

1. Create and activate a new Conda environment:

```
conda create -n anomalib_dev python=3.10
conda activate anomalib_dev
```
```
Install the base and development requirements:
```
```
pip install -r requirements/installer.txt -r requirements/dev.txt
anomalib install -v
```
```
Optionally, for a full installation with all dependencies:
```
```
pip install -e .[all]
```
2. Install and configure pre-commit hooks:

```
pre-commit install
```
```
Pre-commit hooks help ensure code quality and consistency. After each commit,pre-commitwill automatically
run the configured checks for the changed file. If you would like to manually run the checks for all files, use:
```
```
pre-commit run --all-files
```
```
To bypass pre-commit hooks temporarily (e.g., for a work-in-progress commit), use:
```
```
git commit -m'WIP commit'--no-verify
```
```
However, make sure to address any pre-commit issues before finalizing your pull request.
```
**Making Changes**

1. **Write Code:** Follow the project’s coding standards and write your code with clear intent. Ensure your code is
    well-documented and includes examples where appropriate. For code quality we use ruff, whose configuration
    is inpyproject.tomlfile.
2. **Add Tests:** If your code includes new functionality, add corresponding tests using pytest to maintain coverage
    and reliability.
3. **Update Documentation:** If you’ve changed APIs or added new features, update the documentation accordingly.
    Ensure your docstrings are clear and follow Google’s docstring guide.
4. **Pass Tests and Quality Checks:** Ensure the test suite passes and that your code meets quality standards by
    running:

```
pre-commit run --all-files
pytest tests/
```
5. **Update the Changelog:** For significant changes, add a summary to the CHANGELOG.
6. **Check Licensing:** Ensure you own the code or have rights to use it, adhering to appropriate licensing.
7. **Sign Your Commits:** Use signed commits to certify that you have the right to submit the code under the project’s
    license:

**3.6. Developer Guide 229**


```
git commit -S -m "Your detailed commit message"
```
```
For more on signing commits, see GitHub’s guide on signing commits.
```
**Submitting Pull Requests**

Once you’ve followed the above steps and are satisfied with your changes:

1. Push your changes to your forked repository.
2. Go to the original Anomalib repository you forked and click “New pull request”.
3. Choose your fork and the branch with your changes to open a pull request.
4. Fill in the pull request template with the necessary details about your changes.

We look forward to your contributions!

**3.6.3 Code Review Checklist**

This checklist is a guide for reviewing code changes. It can be used as a reference for both authors and reviewers to
ensure that the code meets the project’s standards and requirements.

**Code Quality**

- Is the code clear and understandable?
- Does the code follow the project’s coding conventions and style guide (naming conventions, spacing, indentation,
    etc.)?
- Are there any redundant or unnecessary parts of the code?
- Is there duplicated code that could be refactored into a reusable function/method?
- Are there any magic numbers or strings that should be constants or configurations?

**Architecture and Design**

- Is the code change consistent with the overall architecture of the system?
- Are the classes, modules, and functions well-organized and appropriately sized?
- Are design patterns used appropriately and consistently?
- Does the change introduce any potential scalability issues?
- Is there a clear separation of concerns (e.g., UI, business logic, data access)?

**230 Chapter 3. Guides**


**Functionality**

- Does the code do what it’s supposed to do?
- Are all edge cases considered and handled?
- Is there any dead or commented-out code that should be removed?
- Are there any debugging or logging statements that need to be removed or adjusted?

**Security**

- Are all data inputs validated and sanitized to prevent SQL injection, XSS, etc.?
- Are passwords and sensitive data properly encrypted or secured?
- Are there any potential security vulnerabilities introduced or exposed by the code change?
- Is authentication and authorization handled properly?

**Performance**

- Are there any obvious performance issues or bottlenecks?
- Is the code optimized for time and space complexity where necessary?
- Are large data sets or files handled efficiently?
- Is caching implemented appropriately?

**Testing**

- Are there unit tests covering the new functionality or changes?
- Do the existing tests need to be updated or extended?
- Is there appropriate error handling and logging in the tests?
- Do all tests pass?
- Is there enough coverage for critical paths in the code?

**Documentation and Comments**

- Is the new code adequately commented for clarity?
- Is the documentation (README, API docs, inline comments) updated to reflect the changes?
- Are complex algorithms or decisions well-explained?
- Are there any assumptions or limitations that need to be documented?

**3.6. Developer Guide 231**


**Compatibility**

- Is the code compatible with all targeted environments (operating systems, browsers, devices)?
- Does the change maintain backward compatibility or is a migration path provided?
- Are there any dependencies added or updated? If so, are they necessary and properly vetted?

**Reviewer’s General Feedback**

- Provide any general feedback or suggestions for improvements.
- Highlight any areas of excellence or particularly clever solutions.

**3.7 Awards and Recognition**

Anomalib is committed to excellence and innovation in open-source software development. Over the years, our efforts
have been recognized by various prestigious organizations and communities. This page highlights the awards and
recognitions we have received, reflecting our dedication to our goals and mission.

**3.7.1 2024**

```
Passing Badge - OpenSSF Best Practices
```
- **Date:** January 25, 2024
- **Details:** Anomalib received the passing grade for the Open Source Initiative Best Practices Badge. This badge
    recognizes our commitment to high-quality coding standards and community engagement.

Learn more about this award

**3.7.2 2023**

```
Anomalib as a Pytorch EcoSystem Project
```
- **Date:** December 15, 2023
- **Details:** Anomalib has been accepted as a Pytorch Ecosystem Project. This recognition highlights our commit-
    ment to the Pytorch community and our efforts to make anomaly detection accessible to everyone.

**232 Chapter 3. Guides**


**3.7.3 Acknowledgements**

We extend our heartfelt thanks to our contributors, users, and the community who have supported us throughout our
journey. These recognitions belong not only to the core team but to everyone who has been a part of this project.

_Last Updated: 29-Jan-24_

**3.8 Community Engagement**

Anomalib has recently been gaining traction in the developer community. It has recently been the subject of several
hackathons, events, and challenges, showcasing its adaptability and impact in real-world scenarios.

**3.8.1 Events**

**Keynote - Visual Anomaly Detection Workshop, CVPR24**

- **Date:** June 2024
- **Location:** Seattle, Washington, USA
- **Details:** A keynote showcasing anomalib at the Anomaly Detection workshop at CVPR. Link to the event.

**Keynote - Anomaly Detection Workshop, NVPHBV**

- **Date:** November 2023
- **Location:** Amsterdam, Netherlands
- **Details:** A keynote showcasing anomalib at the Anomaly Detection workshop at NVPHBV. The workshop was
    organized by the Dutch Society for Pattern Recognition (NVPHBV). Link to the event.

**Talk - AI Grunn Tech Event**

- **Date:** November 2023
- **Location:** Groningen, Netherlands
- **Details:** A talk showcasing anomalib at the tech event. Link to the event.

**3.8.2 Challenges**

**Chips Challenge on Hackster.io**

- **Date:** December 2023
- **Overview:** A coding marathon where participants created innovative solutions using Anomalib and OpenVINO.
- **Outcome:** Help Chip find all the bad produce at his farm-to-table business using anomaly detection by sharing
    your knowledge, running a Jupyter notebook or creating an app with OpenVINO.
- **Details:** Chips Challenge

**3.8. Community Engagement 233**


**234 Chapter 3. Guides**


**PYTHON MODULE INDEX**

a

anomalib.callbacks, 196
anomalib.cli, 200
anomalib.data.base.datamodule, 19
anomalib.data.base.dataset, 18
anomalib.data.base.depth, 21
anomalib.data.base.video, 21
anomalib.data.depth.folder_3d, 53
anomalib.data.depth.mvtec_3d, 56
anomalib.data.image.btech, 22
anomalib.data.image.folder, 27
anomalib.data.image.kolektor, 31
anomalib.data.image.mvtec, 34
anomalib.data.image.visa, 38
anomalib.data.utils.augmenter, 80
anomalib.data.utils.boxes, 73
anomalib.data.utils.download, 62
anomalib.data.utils.generators, 79
anomalib.data.utils.image, 66
anomalib.data.utils.label, 73
anomalib.data.utils.path, 59
anomalib.data.utils.split, 75
anomalib.data.utils.synthetic, 81
anomalib.data.utils.tiler, 76
anomalib.data.utils.video, 72
anomalib.data.video.avenue, 42
anomalib.data.video.shanghaitech, 47
anomalib.data.video.ucsd_ped, 50
anomalib.deploy, 202
anomalib.engine, 178
anomalib.loggers, 192
anomalib.metrics, 184
anomalib.models.components.classification, 93
anomalib.models.components.cluster, 95
anomalib.models.components.dimensionality_reduction,
86
anomalib.models.components.feature_extractors,
82
anomalib.models.components.filters, 92
anomalib.models.components.flow, 89
anomalib.models.components.sampling, 91
anomalib.models.components.stats, 97

```
anomalib.models.image.cfa.anomaly_map, 102
anomalib.models.image.cfa.lightning_model, 99
anomalib.models.image.cfa.loss, 101
anomalib.models.image.cfa.torch_model, 100
anomalib.models.image.cflow.anomaly_map, 105
anomalib.models.image.cflow.lightning_model,
103
anomalib.models.image.cflow.torch_model, 104
anomalib.models.image.csflow.anomaly_map, 108
anomalib.models.image.csflow.lightning_model,
106
anomalib.models.image.csflow.loss, 108
anomalib.models.image.csflow.torch_model, 107
anomalib.models.image.dfkde.lightning_model,
109
anomalib.models.image.dfkde.torch_model, 110
anomalib.models.image.dfm.lightning_model,
111
anomalib.models.image.dfm.torch_model, 112
anomalib.models.image.draem.lightning_model,
114
anomalib.models.image.draem.loss, 116
anomalib.models.image.draem.torch_model, 116
anomalib.models.image.dsr.anomaly_generator,
128
anomalib.models.image.dsr.lightning_model,
127
anomalib.models.image.dsr.loss, 129
anomalib.models.image.dsr.torch_model, 117
anomalib.models.image.efficient_ad.lightning_model,
130
anomalib.models.image.efficient_ad.torch_model,
132
anomalib.models.image.fastflow.anomaly_map,
136
anomalib.models.image.fastflow.lightning_model,
133
anomalib.models.image.fastflow.loss, 135
anomalib.models.image.fastflow.torch_model,
134
anomalib.models.image.ganomaly.lightning_model,
136
```
### 235


anomalib.models.image.ganomaly.loss, 139
anomalib.models.image.ganomaly.torch_model,
138
anomalib.models.image.padim.lightning_model,
140
anomalib.models.image.padim.torch_model, 141
anomalib.models.image.patchcore.lightning_model,
142
anomalib.models.image.patchcore.torch_model,
144
anomalib.models.image.reverse_distillation.anomaly_map,
149
anomalib.models.image.reverse_distillation.lightning_model,
146
anomalib.models.image.reverse_distillation.loss,
148
anomalib.models.image.reverse_distillation.torch_model,
147
anomalib.models.image.rkde.feature_extractor,
152
anomalib.models.image.rkde.lightning_model,
150
anomalib.models.image.rkde.region_extractor,
153
anomalib.models.image.rkde.torch_model, 151
anomalib.models.image.stfpm.anomaly_map, 157
anomalib.models.image.stfpm.lightning_model,
154
anomalib.models.image.stfpm.loss, 156
anomalib.models.image.stfpm.torch_model, 155
anomalib.models.image.uflow.anomaly_map, 161
anomalib.models.image.uflow.lightning_model,
160
anomalib.models.image.uflow.torch_model, 158
anomalib.models.image.winclip.lightning_model,
163
anomalib.models.image.winclip.torch_model,
164
anomalib.models.video.ai_vad.density, 175
anomalib.models.video.ai_vad.features, 170
anomalib.models.video.ai_vad.flow, 174
anomalib.models.video.ai_vad.lightning_model,
167
anomalib.models.video.ai_vad.regions, 172
anomalib.models.video.ai_vad.torch_model, 169

## Python Module Index


**INDEX**

Symbols

_patch_embeddings ( _anoma-
lib.models.image.winclip.torch_model.WinClipModel
attribute_ ), 165
_text_embeddings ( _anoma-
lib.models.image.winclip.torch_model.WinClipModel
attribute_ ), 165
_visual_embeddings ( _anoma-
lib.models.image.winclip.torch_model.WinClipModel
attribute_ ), 165

A

add_arguments_to_parser() ( _anoma-
lib.cli.AnomalibCLI method_ ), 200
add_export_arguments()( _anomalib.cli.AnomalibCLI
method_ ), 200
add_foreground_boxes() ( _anoma-
lib.models.video.ai_vad.regions.RegionExtractor
method_ ), 173
add_image()( _anomalib.loggers.AnomalibCometLogger
method_ ), 194
add_image() ( _anoma-
lib.loggers.AnomalibTensorBoardLogger
method_ ), 195
add_image() ( _anoma-
lib.loggers.AnomalibWandbLogger method_ ),
196
add_predict_arguments() ( _anoma-
lib.cli.AnomalibCLI method_ ), 200
add_subcommands() ( _anomalib.cli.AnomalibCLI
method_ ), 200
add_train_arguments() ( _anomalib.cli.AnomalibCLI
method_ ), 200
add_trainer_arguments() ( _anoma-
lib.cli.AnomalibCLI method_ ), 200
AffineCouplingSubnet ( _class in anoma-
lib.models.image.uflow.torch_model_ ), 158
AiVad ( _class in anoma-
lib.models.video.ai_vad.lightning_model_ ),
167
AiVadModel ( _class in anoma-
lib.models.video.ai_vad.torch_model_ ), 169

```
AllInOneBlock ( class in anoma-
lib.models.components.flow ), 89
anomalib.callbacks
module, 196
anomalib.cli
module, 200
anomalib.data.base.datamodule
module, 19
anomalib.data.base.dataset
module, 18
anomalib.data.base.depth
module, 21
anomalib.data.base.video
module, 21
anomalib.data.depth.folder_3d
module, 53
anomalib.data.depth.mvtec_3d
module, 56
anomalib.data.image.btech
module, 22
anomalib.data.image.folder
module, 27
anomalib.data.image.kolektor
module, 31
anomalib.data.image.mvtec
module, 34
anomalib.data.image.visa
module, 38
anomalib.data.utils.augmenter
module, 80
anomalib.data.utils.boxes
module, 73
anomalib.data.utils.download
module, 62
anomalib.data.utils.generators
module, 79
anomalib.data.utils.image
module, 66
anomalib.data.utils.label
module, 73
anomalib.data.utils.path
module, 59
```
### 237


anomalib.data.utils.split
module, 75
anomalib.data.utils.synthetic
module, 81
anomalib.data.utils.tiler
module, 76
anomalib.data.utils.video
module, 72
anomalib.data.video.avenue
module, 42
anomalib.data.video.shanghaitech
module, 47
anomalib.data.video.ucsd_ped
module, 50
anomalib.deploy
module, 202
anomalib.engine
module, 178
anomalib.loggers
module, 192
anomalib.metrics
module, 184
anomalib.models.components.classification
module, 93
anomalib.models.components.cluster
module, 95
anomalib.models.components.dimensionality_reduction
module, 86
anomalib.models.components.feature_extractors
module, 82
anomalib.models.components.filters
module, 92
anomalib.models.components.flow
module, 89
anomalib.models.components.sampling
module, 91
anomalib.models.components.stats
module, 97
anomalib.models.image.cfa.anomaly_map
module, 102
anomalib.models.image.cfa.lightning_model
module, 99
anomalib.models.image.cfa.loss
module, 101
anomalib.models.image.cfa.torch_model
module, 100
anomalib.models.image.cflow.anomaly_map
module, 105
anomalib.models.image.cflow.lightning_model
module, 103
anomalib.models.image.cflow.torch_model
module, 104
anomalib.models.image.csflow.anomaly_map
module, 108

```
anomalib.models.image.csflow.lightning_model
module, 106
anomalib.models.image.csflow.loss
module, 108
anomalib.models.image.csflow.torch_model
module, 107
anomalib.models.image.dfkde.lightning_model
module, 109
anomalib.models.image.dfkde.torch_model
module, 110
anomalib.models.image.dfm.lightning_model
module, 111
anomalib.models.image.dfm.torch_model
module, 112
anomalib.models.image.draem.lightning_model
module, 114
anomalib.models.image.draem.loss
module, 116
anomalib.models.image.draem.torch_model
module, 116
anomalib.models.image.dsr.anomaly_generator
module, 128
anomalib.models.image.dsr.lightning_model
module, 127
anomalib.models.image.dsr.loss
module, 129
anomalib.models.image.dsr.torch_model
module, 117
anomalib.models.image.efficient_ad.lightning_model
module, 130
anomalib.models.image.efficient_ad.torch_model
module, 132
anomalib.models.image.fastflow.anomaly_map
module, 136
anomalib.models.image.fastflow.lightning_model
module, 133
anomalib.models.image.fastflow.loss
module, 135
anomalib.models.image.fastflow.torch_model
module, 134
anomalib.models.image.ganomaly.lightning_model
module, 136
anomalib.models.image.ganomaly.loss
module, 139
anomalib.models.image.ganomaly.torch_model
module, 138
anomalib.models.image.padim.lightning_model
module, 140
anomalib.models.image.padim.torch_model
module, 141
anomalib.models.image.patchcore.lightning_model
module, 142
anomalib.models.image.patchcore.torch_model
module, 144
```
**238 Index**


anomalib.models.image.reverse_distillation.anomaly_map
module, 149
anomalib.models.image.reverse_distillation.lightning_model
module, 146
anomalib.models.image.reverse_distillation.loss
module, 148
anomalib.models.image.reverse_distillation.torch_model
module, 147
anomalib.models.image.rkde.feature_extractor
module, 152
anomalib.models.image.rkde.lightning_model
module, 150
anomalib.models.image.rkde.region_extractor
module, 153
anomalib.models.image.rkde.torch_model
module, 151
anomalib.models.image.stfpm.anomaly_map
module, 157
anomalib.models.image.stfpm.lightning_model
module, 154
anomalib.models.image.stfpm.loss
module, 156
anomalib.models.image.stfpm.torch_model
module, 155
anomalib.models.image.uflow.anomaly_map
module, 161
anomalib.models.image.uflow.lightning_model
module, 160
anomalib.models.image.uflow.torch_model
module, 158
anomalib.models.image.winclip.lightning_model
module, 163
anomalib.models.image.winclip.torch_model
module, 164
anomalib.models.video.ai_vad.density
module, 175
anomalib.models.video.ai_vad.features
module, 170
anomalib.models.video.ai_vad.flow
module, 174
anomalib.models.video.ai_vad.lightning_model
module, 167
anomalib.models.video.ai_vad.regions
module, 172
anomalib.models.video.ai_vad.torch_model
module, 169
anomalib_subcommands()( _anomalib.cli.AnomalibCLI
static method_ ), 201
AnomalibCLI( _class in anomalib.cli_ ), 200
AnomalibCometLogger( _class in anomalib.loggers_ ), 192
AnomalibDataModule ( _class in anoma-
lib.data.base.datamodule_ ), 19
AnomalibDataset ( _class in anoma-
lib.data.base.dataset_ ), 18

```
AnomalibDepthDataset ( class in anoma-
lib.data.base.depth ), 21
AnomalibTensorBoardLogger ( class in anoma-
lib.loggers ), 194
AnomalibVideoDataModule ( class in anoma-
lib.data.base.video ), 21
AnomalibVideoDataset ( class in anoma-
lib.data.base.video ), 21
AnomalibWandbLogger( class in anomalib.loggers ), 195
AnomalyDetectionModule ( class in anoma-
lib.models.image.dsr.torch_model ), 117
AnomalyMapGenerationMode ( class in anoma-
lib.models.image.reverse_distillation.anomaly_map ),
149
AnomalyMapGenerator ( class in anoma-
lib.models.image.cfa.anomaly_map ), 102
AnomalyMapGenerator ( class in anoma-
lib.models.image.cflow.anomaly_map ), 105
AnomalyMapGenerator ( class in anoma-
lib.models.image.csflow.anomaly_map ), 108
AnomalyMapGenerator ( class in anoma-
lib.models.image.fastflow.anomaly_map ),
136
AnomalyMapGenerator ( class in anoma-
lib.models.image.reverse_distillation.anomaly_map ),
149
AnomalyMapGenerator ( class in anoma-
lib.models.image.stfpm.anomaly_map ), 157
AnomalyMapGenerator ( class in anoma-
lib.models.image.uflow.anomaly_map ), 161
AnomalyMapMode ( class in anoma-
lib.models.image.csflow.anomaly_map ), 108
AnomalyScoreDistribution ( class in anoma-
lib.metrics ), 188
apply_cls1_split() ( anomalib.data.image.visa.Visa
method ), 39
augment_batch() ( anoma-
lib.data.utils.augmenter.Augmenter method ),
80
augment_batch() ( anoma-
lib.models.image.dsr.anomaly_generator.DsrAnomalyGenerator
method ), 128
Augmenter( class in anomalib.data.utils.augmenter ), 80
AUPR( class in anomalib.metrics ), 184
AUPRO( class in anomalib.metrics ), 185
AUROC( class in anomalib.metrics ), 187
Avenue( class in anomalib.data.video.avenue ), 42
AvenueDataset( class in anomalib.data.video.avenue ),
45
```
```
B
BackboneParams ( class in anoma-
lib.models.components.feature_extractors ),
82
```
**Index 239**


backward()( _anomalib.models.image.cfa.lightning_model.Cfa
method_ ), 99
BaseDensityEstimator ( _class in anoma-
lib.models.video.ai_vad.density_ ), 175
before_instantiate_classes() ( _anoma-
lib.cli.AnomalibCLI method_ ), 201
BinaryPrecisionRecallCurve ( _class in anoma-
lib.metrics_ ), 188
binomial_test() ( _anoma-
lib.models.image.uflow.anomaly_map.AnomalyMapGenerator
static method_ ), 161
boxes_to_anomaly_maps() ( _in module anoma-
lib.data.utils.boxes_ ), 73
boxes_to_masks() ( _in module anoma-
lib.data.utils.boxes_ ), 74
BTech( _class in anomalib.data.image.btech_ ), 22
BTechDataset( _class in anomalib.data.image.btech_ ), 25
build_flow() ( _anoma-
lib.models.image.uflow.torch_model.UflowModel
method_ ), 159
build_flow_stage() ( _anoma-
lib.models.image.uflow.torch_model.UflowModel
method_ ), 159

C

category( _anomalib.data.base.datamodule.AnomalibDataModule
property_ ), 20
category( _anomalib.data.base.dataset.AnomalibDataset
property_ ), 19
Cfa ( _class in anoma-
lib.models.image.cfa.lightning_model_ ), 99
CfaLoss( _class in anomalib.models.image.cfa.loss_ ), 101
CfaModel ( _class in anoma-
lib.models.image.cfa.torch_model_ ), 100
Cflow ( _class in anoma-
lib.models.image.cflow.lightning_model_ ),
103
CflowModel ( _class in anoma-
lib.models.image.cflow.torch_model_ ), 104
check_hash() ( _in module anoma-
lib.data.utils.download_ ), 64
choose_random_aug_image() ( _anoma-
lib.models.image.efficient_ad.torch_model.EfficientAdModel
method_ ), 132
clip( _anomalib.models.image.winclip.torch_model.WinClipModel
attribute_ ), 164
ClipsIndexer( _class in anomalib.data.utils.video_ ), 72
collate_fn() ( _in module anoma-
lib.data.base.datamodule_ ), 21
collect_reference_images() ( _anoma-
lib.models.image.winclip.lightning_model.WinClip
method_ ), 163
CombinedDensityEstimator ( _class in anoma-
lib.models.video.ai_vad.density_ ), 175

```
compute()( anomalib.metrics.AnomalyScoreDistribution
method ), 188
compute()( anomalib.metrics.AUPR method ), 184
compute()( anomalib.metrics.AUPRO method ), 186
compute()( anomalib.metrics.AUROC method ), 187
compute() ( anomalib.metrics.F1AdaptiveThreshold
method ), 189
compute() ( anomalib.metrics.ManualThreshold
method ), 190
compute()( anomalib.metrics.MinMax method ), 191
compute()( anomalib.metrics.PRO method ), 192
compute_anomaly_map() ( anoma-
lib.models.image.cfa.anomaly_map.AnomalyMapGenerator
method ), 102
compute_anomaly_map() ( anoma-
lib.models.image.cflow.anomaly_map.AnomalyMapGenerator
method ), 105
compute_anomaly_map() ( anoma-
lib.models.image.stfpm.anomaly_map.AnomalyMapGenerator
method ), 157
compute_anomaly_map() ( anoma-
lib.models.image.uflow.anomaly_map.AnomalyMapGenerator
method ), 161
compute_anomaly_mask() ( anoma-
lib.models.image.uflow.anomaly_map.AnomalyMapGenerator
method ), 162
compute_anomaly_score() ( anoma-
lib.models.image.patchcore.torch_model.PatchcoreModel
method ), 144
compute_distance() ( anoma-
lib.models.image.cfa.torch_model.CfaModel
method ), 100
compute_kde_scores() ( anoma-
lib.models.components.classification.KDEClassifier
method ), 93
compute_layer_loss() ( anoma-
lib.models.image.stfpm.loss.STFPMLoss
method ), 156
compute_layer_map() ( anoma-
lib.models.image.stfpm.anomaly_map.AnomalyMapGenerator
method ), 157
compute_new_image_size() ( in module anoma-
lib.data.utils.tiler ), 78
compute_pro() ( anomalib.metrics.AUPRO method ),
186
compute_probabilities() ( anoma-
lib.models.components.classification.KDEClassifier
static method ), 94
compute_score() ( anoma-
lib.models.image.cfa.anomaly_map.AnomalyMapGenerator
method ), 102
concatenate_datasets() ( in module anoma-
lib.data.utils.split ), 75
configure_optimizers() ( anoma-
```
**240 Index**


_lib.models.image.cfa.lightning_model.Cfa
method_ ), 99
configure_optimizers() ( _anoma-
lib.models.image.cflow.lightning_model.Cflow
method_ ), 103
configure_optimizers() ( _anoma-
lib.models.image.csflow.lightning_model.Csflow
method_ ), 106
configure_optimizers() ( _anoma-
lib.models.image.dfkde.lightning_model.Dfkde
static method_ ), 109
configure_optimizers() ( _anoma-
lib.models.image.dfm.lightning_model.Dfm
static method_ ), 111
configure_optimizers() ( _anoma-
lib.models.image.draem.lightning_model.Draem
method_ ), 115
configure_optimizers() ( _anoma-
lib.models.image.dsr.lightning_model.Dsr
method_ ), 127
configure_optimizers() ( _anoma-
lib.models.image.efficient_ad.lightning_model.EfficientAd
method_ ), 130
configure_optimizers() ( _anoma-
lib.models.image.fastflow.lightning_model.Fastflow
method_ ), 134
configure_optimizers() ( _anoma-
lib.models.image.ganomaly.lightning_model.Ganomaly
method_ ), 137
configure_optimizers() ( _anoma-
lib.models.image.padim.lightning_model.Padim
static method_ ), 140
configure_optimizers() ( _anoma-
lib.models.image.patchcore.lightning_model.Patchcore
method_ ), 143
configure_optimizers() ( _anoma-
lib.models.image.reverse_distillation.lightning_model.ReverseDistillation
method_ ), 146
configure_optimizers() ( _anoma-
lib.models.image.rkde.lightning_model.Rkde
static method_ ), 150
configure_optimizers() ( _anoma-
lib.models.image.stfpm.lightning_model.Stfpm
method_ ), 154
configure_optimizers() ( _anoma-
lib.models.image.uflow.lightning_model.Uflow
method_ ), 160
configure_optimizers() ( _anoma-
lib.models.image.winclip.lightning_model.WinClip
static method_ ), 163
configure_optimizers() ( _anoma-
lib.models.video.ai_vad.lightning_model.AiVad
static method_ ), 168
configure_transforms() ( _anoma-_

```
lib.models.image.efficient_ad.lightning_model.EfficientAd
method ), 130
configure_transforms() ( anoma-
lib.models.image.padim.lightning_model.Padim
method ), 140
configure_transforms() ( anoma-
lib.models.image.patchcore.lightning_model.Patchcore
method ), 143
configure_transforms() ( anoma-
lib.models.image.uflow.lightning_model.Uflow
method ), 160
configure_transforms() ( anoma-
lib.models.image.winclip.lightning_model.WinClip
method ), 163
configure_transforms() ( anoma-
lib.models.video.ai_vad.lightning_model.AiVad
method ), 168
contains_non_printable_characters()( in module
anomalib.data.utils.path ), 59
convert_video() ( in module anoma-
lib.data.utils.video ), 73
cov()( anomalib.models.components.stats.GaussianKDE
static method ), 97
Csflow ( class in anoma-
lib.models.image.csflow.lightning_model ),
106
CsFlowLoss ( class in anoma-
lib.models.image.csflow.loss ), 108
CsFlowModel ( class in anoma-
lib.models.image.csflow.torch_model ), 107
```
```
D
DecoderBot ( class in anoma-
lib.models.image.dsr.torch_model ), 118
DeepExtractor ( class in anoma-
lib.models.video.ai_vad.features ), 170
Dfkde ( class in anoma-
lib.models.image.dfkde.lightning_model ),
109
DfkdeModel ( class in anoma-
lib.models.image.dfkde.torch_model ), 110
Dfm ( class in anoma-
lib.models.image.dfm.lightning_model ), 111
DFMModel ( class in anoma-
lib.models.image.dfm.torch_model ), 112
DirType( class in anomalib.data.utils.path ), 59
DiscreteLatentModel ( class in anoma-
lib.models.image.dsr.torch_model ), 118
DiscriminatorLoss ( class in anoma-
lib.models.image.ganomaly.loss ), 139
download_and_extract() ( in module anoma-
lib.data.utils.download ), 64
DownloadInfo( class in anomalib.data.utils.download ),
62
```
**Index 241**


DownloadProgressBar ( _class in anoma-
lib.data.utils.download_ ), 62
downscale_image() ( _in module anoma-
lib.data.utils.tiler_ ), 78
Draem ( _class in anoma-
lib.models.image.draem.lightning_model_ ),
114
DraemLoss ( _class in anoma-
lib.models.image.draem.loss_ ), 116
DraemModel ( _class in anoma-
lib.models.image.draem.torch_model_ ), 116
dryrun_find_featuremap_dims()( _in module anoma-
lib.models.components.feature_extractors_ ), 86
Dsr ( _class in anoma-
lib.models.image.dsr.lightning_model_ ), 127
DsrAnomalyGenerator ( _class in anoma-
lib.models.image.dsr.anomaly_generator_ ),
128
DsrModel ( _class in anoma-
lib.models.image.dsr.torch_model_ ), 119
DsrSecondStageLoss ( _class in anoma-
lib.models.image.dsr.loss_ ), 129
DsrThirdStageLoss ( _class in anoma-
lib.models.image.dsr.loss_ ), 129
duplicate_filename() ( _in module anoma-
lib.data.utils.image_ ), 66

E

EfficientAd ( _class in anoma-
lib.models.image.efficient_ad.lightning_model_ ),
130
EfficientAdModel ( _class in anoma-
lib.models.image.efficient_ad.torch_model_ ),
132
embedding( _anomalib.models.image.dsr.torch_model.VectorQuantizer
property_ ), 126
encode()( _anomalib.models.image.uflow.torch_model.UflowModel
method_ ), 160
encode_image() ( _anoma-
lib.models.image.winclip.torch_model.WinClipModel
method_ ), 165
EncoderBot ( _class in anoma-
lib.models.image.dsr.torch_model_ ), 121
EncoderTop ( _class in anoma-
lib.models.image.dsr.torch_model_ ), 121
Engine( _class in anomalib.engine_ ), 178
euclidean_dist() ( _anoma-
lib.models.image.patchcore.torch_model.PatchcoreModel
static method_ ), 144
eval_transform ( _anoma-
lib.data.base.datamodule.AnomalibDataModule
property_ ), 20
export( _anomalib.cli.AnomalibCLI property_ ), 201
export()( _anomalib.engine.Engine method_ ), 178

```
export_to_onnx()( in module anomalib.deploy ), 206
export_to_openvino()( in module anomalib.deploy ),
207
export_to_torch()( in module anomalib.deploy ), 208
ExportType( class in anomalib.deploy ), 202
extract()( in module anomalib.data.utils.download ),
64
```
```
F
F1AdaptiveThreshold( class in anomalib.metrics ), 188
F1Score( class in anomalib.metrics ), 189
Fastflow ( class in anoma-
lib.models.image.fastflow.lightning_model ),
133
FastflowLoss ( class in anoma-
lib.models.image.fastflow.loss ), 135
FastflowModel ( class in anoma-
lib.models.image.fastflow.torch_model ), 134
FeatureDecoder ( class in anoma-
lib.models.image.dsr.torch_model ), 121
FeatureEncoder ( class in anoma-
lib.models.image.dsr.torch_model ), 122
FeatureExtractor ( class in anoma-
lib.models.image.rkde.feature_extractor ),
152
FeatureExtractor ( class in anoma-
lib.models.video.ai_vad.features ), 171
FeatureScalingMethod ( class in anoma-
lib.models.components.classification ), 93
FeatureType ( class in anoma-
lib.models.video.ai_vad.features ), 171
figure_to_array() ( in module anoma-
lib.data.utils.image ), 66
fit( anomalib.cli.AnomalibCLI property ), 201
fit()( anomalib.engine.Engine method ), 179
fit()( anomalib.models.components.classification.KDEClassifier
method ), 94
fit()( anomalib.models.components.cluster.GaussianMixture
method ), 95
fit() ( anomalib.models.components.cluster.KMeans
method ), 96
fit()( anomalib.models.components.dimensionality_reduction.PCA
method ), 86
fit()( anomalib.models.components.dimensionality_reduction.SparseRandomProjection
method ), 89
fit()( anomalib.models.components.stats.GaussianKDE
method ), 97
fit()( anomalib.models.components.stats.MultiVariateGaussian
method ), 97
fit()( anomalib.models.image.dfkde.lightning_model.Dfkde
method ), 109
fit()( anomalib.models.image.dfm.lightning_model.Dfm
method ), 111
```
**242 Index**


fit()( _anomalib.models.image.dfm.torch_model.DFMModel
method_ ), 113
fit()( _anomalib.models.image.dfm.torch_model.SingleClassGaussian
method_ ), 114
fit()( _anomalib.models.image.padim.lightning_model.Padim
method_ ), 140
fit()( _anomalib.models.image.patchcore.lightning_model.Patchcore
method_ ), 143
fit()( _anomalib.models.image.rkde.lightning_model.Rkde
method_ ), 150
fit()( _anomalib.models.image.rkde.torch_model.RkdeModel
method_ ), 152
fit()( _anomalib.models.video.ai_vad.density.BaseDensityEstimator
method_ ), 175
fit()( _anomalib.models.video.ai_vad.density.CombinedDensityEstimator
method_ ), 176
fit()( _anomalib.models.video.ai_vad.density.GMMEstimator
method_ ), 176
fit()( _anomalib.models.video.ai_vad.density.GroupedKNNEstimator
method_ ), 177
fit()( _anomalib.models.video.ai_vad.lightning_model.AiVad
method_ ), 168
fit_transform() ( _anoma-
lib.models.components.dimensionality_reduction.PCA
method_ ), 87
FlowExtractor ( _class in anoma-
lib.models.video.ai_vad.flow_ ), 174
Folder( _class in anomalib.data.image.folder_ ), 27
Folder3D( _class in anomalib.data.depth.folder_3d_ ), 53
Folder3DDataset ( _class in anoma-
lib.data.depth.folder_3d_ ), 55
FolderDataset( _class in anomalib.data.image.folder_ ),
29
forward() ( _anomalib.deploy.OpenVINOInferencer
method_ ), 203
forward() ( _anomalib.deploy.TorchInferencer method_ ),
205
forward()( _anomalib.models.components.classification.KDEClassifier
method_ ), 94
forward()( _anomalib.models.components.dimensionality_reduction.PCA
method_ ), 87
forward()( _anomalib.models.components.feature_extractors.TimmFeatureExtractor
method_ ), 83
forward()( _anomalib.models.components.feature_extractors.TorchFXFeatureExtractor
method_ ), 85
forward()( _anomalib.models.components.filters.GaussianBlur2d
method_ ), 92
forward()( _anomalib.models.components.flow.AllInOneBlock
method_ ), 90
forward()( _anomalib.models.components.stats.GaussianKDE
method_ ), 97
forward()( _anomalib.models.components.stats.MultiVariateGaussian
method_ ), 98
forward()( _anomalib.models.image.cfa.anomaly_map.AnomalyMapGenerator_

```
method ), 102
forward() ( anomalib.models.image.cfa.loss.CfaLoss
method ), 102
forward()( anomalib.models.image.cfa.torch_model.CfaModel
method ), 101
forward()( anomalib.models.image.cflow.anomaly_map.AnomalyMapGenerator
method ), 105
forward()( anomalib.models.image.cflow.torch_model.CflowModel
method ), 105
forward()( anomalib.models.image.csflow.anomaly_map.AnomalyMapGenerator
method ), 108
forward()( anomalib.models.image.csflow.loss.CsFlowLoss
method ), 108
forward()( anomalib.models.image.csflow.torch_model.CsFlowModel
method ), 107
forward()( anomalib.models.image.dfkde.torch_model.DfkdeModel
method ), 111
forward()( anomalib.models.image.dfm.torch_model.DFMModel
method ), 113
forward()( anomalib.models.image.dfm.torch_model.SingleClassGaussian
method ), 114
forward()( anomalib.models.image.draem.loss.DraemLoss
method ), 116
forward()( anomalib.models.image.draem.torch_model.DraemModel
method ), 116
forward()( anomalib.models.image.dsr.loss.DsrSecondStageLoss
method ), 129
forward()( anomalib.models.image.dsr.loss.DsrThirdStageLoss
method ), 129
forward()( anomalib.models.image.dsr.torch_model.AnomalyDetectionModule
method ), 117
forward()( anomalib.models.image.dsr.torch_model.DecoderBot
method ), 118
forward()( anomalib.models.image.dsr.torch_model.DiscreteLatentModel
method ), 118
forward()( anomalib.models.image.dsr.torch_model.DsrModel
method ), 120
forward()( anomalib.models.image.dsr.torch_model.EncoderBot
method ), 121
forward()( anomalib.models.image.dsr.torch_model.EncoderTop
method ), 121
forward()( anomalib.models.image.dsr.torch_model.FeatureDecoder
method ), 122
forward()( anomalib.models.image.dsr.torch_model.FeatureEncoder
method ), 122
forward()( anomalib.models.image.dsr.torch_model.ImageReconstructionNetwork
method ), 123
forward()( anomalib.models.image.dsr.torch_model.Residual
method ), 123
forward()( anomalib.models.image.dsr.torch_model.ResidualStack
method ), 123
forward()( anomalib.models.image.dsr.torch_model.SubspaceRestrictionModule
method ), 124
forward()( anomalib.models.image.dsr.torch_model.SubspaceRestrictionNetwork
```
**Index 243**


_method_ ), 124
forward()( _anomalib.models.image.dsr.torch_model.UnetDecoder
method_ ), 125
forward()( _anomalib.models.image.dsr.torch_model.UnetEncoder
method_ ), 125
forward()( _anomalib.models.image.dsr.torch_model.UnetModel
method_ ), 125
forward()( _anomalib.models.image.dsr.torch_model.UpsamplingModule
method_ ), 126
forward()( _anomalib.models.image.dsr.torch_model.VectorQuantizer
method_ ), 126
forward()( _anomalib.models.image.efficient_ad.torch_model.EfficientAdModel
method_ ), 133
forward()( _anomalib.models.image.fastflow.anomaly_map.AnomalyMapGenerator
method_ ), 136
forward()( _anomalib.models.image.fastflow.loss.FastflowLoss
method_ ), 135
forward()( _anomalib.models.image.fastflow.torch_model.FastflowModel
method_ ), 135
forward()( _anomalib.models.image.ganomaly.loss.DiscriminatorLoss
method_ ), 139
forward()( _anomalib.models.image.ganomaly.loss.GeneratorLoss
method_ ), 139
forward()( _anomalib.models.image.ganomaly.torch_model.GanomalyModel
method_ ), 138
forward()( _anomalib.models.image.padim.torch_model.PadimModel
method_ ), 142
forward()( _anomalib.models.image.patchcore.torch_model.PatchcoreModel
method_ ), 145
forward()( _anomalib.models.image.reverse_distillation.anomaly_map.AnomalyMapGenerator
method_ ), 149
forward()( _anomalib.models.image.reverse_distillation.loss.ReverseDistillationLoss
method_ ), 148
forward()( _anomalib.models.image.reverse_distillation.torch_model.ReverseDistillationModel
method_ ), 148
forward()( _anomalib.models.image.rkde.feature_extractor.FeatureExtractor
method_ ), 153
forward()( _anomalib.models.image.rkde.region_extractor.RegionExtractor
method_ ), 153
forward()( _anomalib.models.image.rkde.torch_model.RkdeModel
method_ ), 152
forward()( _anomalib.models.image.stfpm.anomaly_map.AnomalyMapGenerator
method_ ), 157
forward()( _anomalib.models.image.stfpm.loss.STFPMLoss
method_ ), 156
forward()( _anomalib.models.image.stfpm.torch_model.STFPMModel
method_ ), 155
forward()( _anomalib.models.image.uflow.anomaly_map.AnomalyMapGenerator
method_ ), 162
forward()( _anomalib.models.image.uflow.torch_model.UflowModel
method_ ), 160
forward()( _anomalib.models.image.winclip.torch_model.WinClipModel
method_ ), 166
forward()( _anomalib.models.video.ai_vad.density.BaseDensityEstimator_

```
method ), 175
forward()( anomalib.models.video.ai_vad.features.DeepExtractor
method ), 170
forward()( anomalib.models.video.ai_vad.features.FeatureExtractor
method ), 171
forward()( anomalib.models.video.ai_vad.features.PoseExtractor
method ), 172
forward()( anomalib.models.video.ai_vad.features.VelocityExtractor
method ), 172
forward()( anomalib.models.video.ai_vad.flow.FlowExtractor
method ), 174
forward()( anomalib.models.video.ai_vad.regions.RegionExtractor
method ), 173
forward()( anomalib.models.video.ai_vad.torch_model.AiVadModel
method ), 170
from_dataset() ( anoma-
lib.data.utils.synthetic.SyntheticAnomalyDataset
class method ), 81
```
```
G
Ganomaly ( class in anoma-
lib.models.image.ganomaly.lightning_model ),
136
GanomalyModel ( class in anoma-
lib.models.image.ganomaly.torch_model ),
138
GaussianBlur2d ( class in anoma-
lib.models.components.filters ), 92
GaussianKDE ( class in anoma-
lib.models.components.stats ), 97
GaussianMixture ( class in anoma-
lib.models.components.cluster ), 95
generate_anomaly() ( anoma-
lib.models.image.dsr.anomaly_generator.DsrAnomalyGenerator
method ), 128
generate_embedding() ( anoma-
lib.models.image.padim.torch_model.PadimModel
method ), 142
generate_embedding() ( anoma-
lib.models.image.patchcore.torch_model.PatchcoreModel
method ), 145
generate_fake_anomalies_joined() ( anoma-
lib.models.image.dsr.torch_model.DiscreteLatentModel
method ), 119
generate_figure()( anomalib.metrics.AUPR method ),
184
generate_figure() ( anomalib.metrics.AUPRO
method ), 186
generate_figure() ( anomalib.metrics.AUROC
method ), 187
generate_hash() ( in module anoma-
lib.data.utils.download ), 65
generate_output_image_filename() ( in module
anomalib.data.utils.image ), 66
```
**244 Index**


generate_perturbation() ( _anoma-
lib.data.utils.augmenter.Augmenter method_ ),
80
GeneratorLoss ( _class in anoma-
lib.models.image.ganomaly.loss_ ), 139
get_clip()( _anomalib.data.video.shanghaitech.ShanghaiTechTestClipsIndexer
method_ ), 49
get_clip()( _anomalib.data.video.ucsd_ped.UCSDpedClipsIndexer
method_ ), 51
get_features() ( _anoma-
lib.models.image.dfkde.torch_model.DfkdeModel
method_ ), 111
get_features() ( _anoma-
lib.models.image.dfm.torch_model.DFMModel
method_ ), 113
get_image_filename() ( _in module anoma-
lib.data.utils.image_ ), 67
get_image_filenames() ( _in module anoma-
lib.data.utils.image_ ), 68
get_image_filenames_from_dir() ( _in module
anomalib.data.utils.image_ ), 68
get_image_height_and_width()( _in module anoma-
lib.data.utils.image_ ), 69
get_item() ( _anomalib.data.utils.video.ClipsIndexer
method_ ), 72
get_mask() ( _anomalib.data.utils.video.ClipsIndexer
method_ ), 73
get_mask()( _anomalib.data.video.shanghaitech.ShanghaiTechTestClipsIndexer
method_ ), 49
get_mask()( _anomalib.data.video.shanghaitech.ShanghaiTechTrainClipsIndexer
method_ ), 49
get_mask()( _anomalib.data.video.ucsd_ped.UCSDpedClipsIndexer
method_ ), 51
get_new_idx() ( _anoma-
lib.models.components.sampling.KCenterGreedy
method_ ), 91
get_scale() ( _anoma-
lib.models.image.cfa.torch_model.CfaModel
method_ ), 101
GMMEstimator ( _class in anoma-
lib.models.video.ai_vad.density_ ), 176
GraphLogger( _class in anomalib.callbacks_ ), 196
grid_size( _anomalib.models.image.winclip.torch_model.WinClipModel
attribute_ ), 164
GroupedKNNEstimator ( _class in anoma-
lib.models.video.ai_vad.density_ ), 177

H

has_anomalous ( _anoma-
lib.data.base.dataset.AnomalibDataset prop-
erty_ ), 19
has_normal( _anomalib.data.base.dataset.AnomalibDataset
property_ ), 19

```
I
ImageReconstructionNetwork ( class in anoma-
lib.models.image.dsr.torch_model ), 122
ImageUpscaleMode( class in anomalib.data.utils.tiler ),
76
init_parser() ( anomalib.cli.AnomalibCLI method ),
201
initialize_centroid() ( anoma-
lib.models.image.cfa.torch_model.CfaModel
method ), 101
initialize_feature_extractor() ( anoma-
lib.models.components.feature_extractors.TorchFXFeatureExtractor
method ), 85
instantiate_classes() ( anomalib.cli.AnomalibCLI
method ), 201
instantiate_engine() ( anomalib.cli.AnomalibCLI
method ), 201
interp1d()( anomalib.metrics.AUPRO static method ),
186
inverse_transform() ( anoma-
lib.models.components.dimensionality_reduction.PCA
method ), 87
is_file_potentially_dangerous() ( in module
anomalib.data.utils.download ), 65
is_image_file() ( in module anoma-
lib.data.utils.image ), 70
is_path_too_long() ( in module anoma-
lib.data.utils.path ), 60
is_set()( anomalib.models.image.efficient_ad.torch_model.EfficientAdModel
method ), 133
is_within_directory() ( in module anoma-
lib.data.utils.download ), 65
```
```
K
k_shot( anomalib.models.image.winclip.torch_model.WinClipModel
attribute ), 164
KCenterGreedy ( class in anoma-
lib.models.components.sampling ), 91
KDEClassifier ( class in anoma-
lib.models.components.classification ), 93
KMeans( class in anomalib.models.components.cluster ),
96
Kolektor( class in anomalib.data.image.kolektor ), 31
KolektorDataset ( class in anoma-
lib.data.image.kolektor ), 33
```
```
L
LabelName( class in anomalib.data.utils.label ), 73
last_frame_idx() ( anoma-
lib.data.utils.video.ClipsIndexer method ),
73
learning_type ( anoma-
lib.models.image.cfa.lightning_model.Cfa
property ), 99
```
**Index 245**


learning_type ( _anoma-
lib.models.image.cflow.lightning_model.Cflow
property_ ), 103
learning_type ( _anoma-
lib.models.image.csflow.lightning_model.Csflow
property_ ), 106
learning_type ( _anoma-
lib.models.image.dfkde.lightning_model.Dfkde
property_ ), 109
learning_type ( _anoma-
lib.models.image.dfm.lightning_model.Dfm
property_ ), 112
learning_type ( _anoma-
lib.models.image.draem.lightning_model.Draem
property_ ), 115
learning_type ( _anoma-
lib.models.image.dsr.lightning_model.Dsr
property_ ), 127
learning_type ( _anoma-
lib.models.image.efficient_ad.lightning_model.EfficientAd
property_ ), 130
learning_type ( _anoma-
lib.models.image.fastflow.lightning_model.Fastflow
property_ ), 134
learning_type ( _anoma-
lib.models.image.ganomaly.lightning_model.Ganomaly
property_ ), 137
learning_type ( _anoma-
lib.models.image.padim.lightning_model.Padim
property_ ), 140
learning_type ( _anoma-
lib.models.image.patchcore.lightning_model.Patchcore
property_ ), 143
learning_type ( _anoma-
lib.models.image.reverse_distillation.lightning_model.ReverseDistillation
property_ ), 147
learning_type ( _anoma-
lib.models.image.rkde.lightning_model.Rkde
property_ ), 150
learning_type ( _anoma-
lib.models.image.stfpm.lightning_model.Stfpm
property_ ), 154
learning_type ( _anoma-
lib.models.image.uflow.lightning_model.Uflow
property_ ), 160
learning_type ( _anoma-
lib.models.image.winclip.lightning_model.WinClip
property_ ), 163
learning_type ( _anoma-
lib.models.video.ai_vad.lightning_model.AiVad
property_ ), 168
load_model() ( _anomalib.deploy.OpenVINOInferencer
method_ ), 203
load_model() ( _anomalib.deploy.TorchInferencer_

```
method ), 205
load_pretrained_discrete_model_weights()
( anomalib.models.image.dsr.torch_model.DsrModel
method ), 120
load_state_dict() ( anoma-
lib.models.image.winclip.lightning_model.WinClip
method ), 163
LoadModelCallback( class in anomalib.callbacks ), 197
```
```
M
make_avenue_dataset() ( in module anoma-
lib.data.video.avenue ), 46
make_btech_dataset() ( in module anoma-
lib.data.image.btech ), 26
make_folder3d_dataset() ( in module anoma-
lib.data.depth.folder_3d ), 56
make_folder_dataset() ( in module anoma-
lib.data.image.folder ), 30
make_kolektor_dataset() ( in module anoma-
lib.data.image.kolektor ), 33
make_mvtec_3d_dataset() ( in module anoma-
lib.data.depth.mvtec_3d ), 58
make_mvtec_dataset() ( in module anoma-
lib.data.image.mvtec ), 37
make_shanghaitech_dataset() ( in module anoma-
lib.data.video.shanghaitech ), 49
make_synthetic_dataset() ( in module anoma-
lib.data.utils.synthetic ), 81
make_ucsd_dataset() ( in module anoma-
lib.data.video.ucsd_ped ), 52
ManualThreshold( class in anomalib.metrics ), 189
map_norm_quantiles() ( anoma-
lib.models.image.efficient_ad.lightning_model.EfficientAd
method ), 131
masks( anomalib.models.image.winclip.torch_model.WinClipModel
attribute ), 165
masks_to_boxes() ( in module anoma-
lib.data.utils.boxes ), 74
MinMax( class in anomalib.metrics ), 190
model( anomalib.engine.Engine property ), 179
ModelCheckpoint( class in anomalib.callbacks ), 198
module
anomalib.callbacks, 196
anomalib.cli, 200
anomalib.data.base.datamodule, 19
anomalib.data.base.dataset, 18
anomalib.data.base.depth, 21
anomalib.data.base.video, 21
anomalib.data.depth.folder_3d, 53
anomalib.data.depth.mvtec_3d, 56
anomalib.data.image.btech, 22
anomalib.data.image.folder, 27
anomalib.data.image.kolektor, 31
anomalib.data.image.mvtec, 34
```
**246 Index**


```
anomalib.data.image.visa, 38
anomalib.data.utils.augmenter, 80
anomalib.data.utils.boxes, 73
anomalib.data.utils.download, 62
anomalib.data.utils.generators, 79
anomalib.data.utils.image, 66
anomalib.data.utils.label, 73
anomalib.data.utils.path, 59
anomalib.data.utils.split, 75
anomalib.data.utils.synthetic, 81
anomalib.data.utils.tiler, 76
anomalib.data.utils.video, 72
anomalib.data.video.avenue, 42
anomalib.data.video.shanghaitech, 47
anomalib.data.video.ucsd_ped, 50
anomalib.deploy, 202
anomalib.engine, 178
anomalib.loggers, 192
anomalib.metrics, 184
anomalib.models.components.classification,
93
anomalib.models.components.cluster, 95
anomalib.models.components.dimensionality_reduction,
86
anomalib.models.components.feature_extractors,
82
anomalib.models.components.filters, 92
anomalib.models.components.flow, 89
anomalib.models.components.sampling, 91
anomalib.models.components.stats, 97
anomalib.models.image.cfa.anomaly_map,
102
anomalib.models.image.cfa.lightning_model,
99
anomalib.models.image.cfa.loss, 101
anomalib.models.image.cfa.torch_model,
100
anomalib.models.image.cflow.anomaly_map,
105
anomalib.models.image.cflow.lightning_model,
103
anomalib.models.image.cflow.torch_model,
104
anomalib.models.image.csflow.anomaly_map,
108
anomalib.models.image.csflow.lightning_model,
106
anomalib.models.image.csflow.loss, 108
anomalib.models.image.csflow.torch_model,
107
anomalib.models.image.dfkde.lightning_model,
109
anomalib.models.image.dfkde.torch_model,
110
```
```
anomalib.models.image.dfm.lightning_model,
111
anomalib.models.image.dfm.torch_model,
112
anomalib.models.image.draem.lightning_model,
114
anomalib.models.image.draem.loss, 116
anomalib.models.image.draem.torch_model,
116
anomalib.models.image.dsr.anomaly_generator,
128
anomalib.models.image.dsr.lightning_model,
127
anomalib.models.image.dsr.loss, 129
anomalib.models.image.dsr.torch_model,
117
anomalib.models.image.efficient_ad.lightning_model,
130
anomalib.models.image.efficient_ad.torch_model,
132
anomalib.models.image.fastflow.anomaly_map,
136
anomalib.models.image.fastflow.lightning_model,
133
anomalib.models.image.fastflow.loss, 135
anomalib.models.image.fastflow.torch_model,
134
anomalib.models.image.ganomaly.lightning_model,
136
anomalib.models.image.ganomaly.loss, 139
anomalib.models.image.ganomaly.torch_model,
138
anomalib.models.image.padim.lightning_model,
140
anomalib.models.image.padim.torch_model,
141
anomalib.models.image.patchcore.lightning_model,
142
anomalib.models.image.patchcore.torch_model,
144
anomalib.models.image.reverse_distillation.anomaly_map,
149
anomalib.models.image.reverse_distillation.lightning_model,
146
anomalib.models.image.reverse_distillation.loss,
148
anomalib.models.image.reverse_distillation.torch_model,
147
anomalib.models.image.rkde.feature_extractor,
152
anomalib.models.image.rkde.lightning_model,
150
anomalib.models.image.rkde.region_extractor,
153
```
**Index 247**


anomalib.models.image.rkde.torch_model,
151
anomalib.models.image.stfpm.anomaly_map,
157
anomalib.models.image.stfpm.lightning_model,
154
anomalib.models.image.stfpm.loss, 156
anomalib.models.image.stfpm.torch_model,
155
anomalib.models.image.uflow.anomaly_map,
161
anomalib.models.image.uflow.lightning_model,
160
anomalib.models.image.uflow.torch_model,
158
anomalib.models.image.winclip.lightning_model,
163
anomalib.models.image.winclip.torch_model,
164
anomalib.models.video.ai_vad.density, 175
anomalib.models.video.ai_vad.features,
170
anomalib.models.video.ai_vad.flow, 174
anomalib.models.video.ai_vad.lightning_model,
167
anomalib.models.video.ai_vad.regions, 172
anomalib.models.video.ai_vad.torch_model,
169
MultiVariateGaussian ( _class in anoma-
lib.models.components.stats_ ), 97
MVTec( _class in anomalib.data.image.mvtec_ ), 34
MVTec3D( _class in anomalib.data.depth.mvtec_3d_ ), 57
MVTec3DDataset ( _class in anoma-
lib.data.depth.mvtec_3d_ ), 58
MVTecDataset( _class in anomalib.data.image.mvtec_ ), 36

N

name( _anomalib.data.base.datamodule.AnomalibDataModule
property_ ), 20
name ( _anomalib.data.base.dataset.AnomalibDataset
property_ ), 19
name ( _anomalib.data.depth.folder_3d.Folder3D prop-
erty_ ), 54
name ( _anomalib.data.depth.folder_3d.Folder3DDataset
property_ ), 55
name( _anomalib.data.image.folder.Folder property_ ), 29
name ( _anomalib.data.image.folder.FolderDataset prop-
erty_ ), 30
nearest_neighbors() ( _anoma-
lib.models.image.patchcore.torch_model.PatchcoreModel
method_ ), 145
nextpow2()( _in module anomalib.data.utils.augmenter_ ),
81

```
normalization_callback ( anomalib.engine.Engine
property ), 180
```
```
O
on_fit_end() ( anomalib.callbacks.TimerCallback
method ), 199
on_fit_start() ( anomalib.callbacks.TimerCallback
method ), 199
on_test_batch_end() ( anoma-
lib.models.image.ganomaly.lightning_model.Ganomaly
method ), 137
on_test_end() ( anomalib.callbacks.TimerCallback
method ), 199
on_test_start() ( anomalib.callbacks.TimerCallback
method ), 199
on_test_start() ( anoma-
lib.models.image.ganomaly.lightning_model.Ganomaly
method ), 137
on_train_end() ( anomalib.callbacks.GraphLogger
method ), 197
on_train_epoch_start() ( anoma-
lib.models.image.dsr.lightning_model.Dsr
method ), 127
on_train_start() ( anomalib.callbacks.GraphLogger
method ), 197
on_train_start() ( anoma-
lib.models.image.cfa.lightning_model.Cfa
method ), 99
on_train_start() ( anoma-
lib.models.image.dsr.lightning_model.Dsr
method ), 127
on_train_start() ( anoma-
lib.models.image.efficient_ad.lightning_model.EfficientAd
method ), 131
on_validation_batch_end() ( anoma-
lib.models.image.ganomaly.lightning_model.Ganomaly
method ), 137
on_validation_start() ( anoma-
lib.models.image.efficient_ad.lightning_model.EfficientAd
method ), 131
on_validation_start() ( anoma-
lib.models.image.ganomaly.lightning_model.Ganomaly
method ), 137
OpenVINOInferencer( class in anomalib.deploy ), 202
output_dims() ( anoma-
lib.models.components.flow.AllInOneBlock
method ), 90
```
```
P
pad_nextpow2()( in module anomalib.data.utils.image ),
70
Padim ( class in anoma-
lib.models.image.padim.lightning_model ),
140
```
**248 Index**


PadimModel ( _class in anoma-
lib.models.image.padim.torch_model_ ), 141
patch_embeddings ( _anoma-
lib.models.image.winclip.torch_model.WinClipModel
property_ ), 166
Patchcore ( _class in anoma-
lib.models.image.patchcore.lightning_model_ ),
142
PatchcoreModel ( _class in anoma-
lib.models.image.patchcore.torch_model_ ),
144
PCA ( _class in anoma-
lib.models.components.dimensionality_reduction_ ),
86
perform_cca() ( _anomalib.metrics.AUPRO method_ ),
186
PoseExtractor ( _class in anoma-
lib.models.video.ai_vad.features_ ), 172
post_process() ( _anoma-
lib.deploy.OpenVINOInferencer method_ ),
204
post_process() ( _anomalib.deploy.TorchInferencer
method_ ), 205
post_process_bbox_detections() ( _anoma-
lib.models.video.ai_vad.regions.RegionExtractor
method_ ), 174
post_process_box_predictions() ( _anoma-
lib.models.image.rkde.region_extractor.RegionExtractor
method_ ), 154
pre_process()( _anomalib.deploy.OpenVINOInferencer
method_ ), 204
pre_process() ( _anomalib.deploy.TorchInferencer
method_ ), 206
pre_process() ( _anoma-
lib.models.components.classification.KDEClassifier
method_ ), 94
pre_process() ( _anoma-
lib.models.video.ai_vad.flow.FlowExtractor
method_ ), 174
predict( _anomalib.cli.AnomalibCLI property_ ), 201
predict() ( _anomalib.deploy.OpenVINOInferencer
method_ ), 204
predict() ( _anomalib.deploy.TorchInferencer method_ ),
206
predict()( _anomalib.engine.Engine method_ ), 180
predict()( _anomalib.models.components.classification.KDEClassifier
method_ ), 94
predict()( _anomalib.models.components.cluster.GaussianMixture
method_ ), 95
predict()( _anomalib.models.components.cluster.KMeans
method_ ), 96
predict()( _anomalib.models.video.ai_vad.density.BaseDensityEstimator
method_ ), 175
predict()( _anomalib.models.video.ai_vad.density.CombinedDensityEstimator_

```
method ), 176
predict()( anomalib.models.video.ai_vad.density.GMMEstimator
method ), 176
predict()( anomalib.models.video.ai_vad.density.GroupedKNNEstimator
method ), 177
predict_dataloader() ( anoma-
lib.data.base.datamodule.AnomalibDataModule
method ), 20
prepare_data() ( anoma-
lib.data.depth.mvtec_3d.MVTec3D method ),
58
prepare_data() ( anomalib.data.image.btech.BTech
method ), 24
prepare_data() ( anoma-
lib.data.image.kolektor.Kolektor method ),
32
prepare_data() ( anomalib.data.image.mvtec.MVTec
method ), 36
prepare_data() ( anomalib.data.image.visa.Visa
method ), 40
prepare_data() ( anomalib.data.video.avenue.Avenue
method ), 44
prepare_data() ( anoma-
lib.data.video.shanghaitech.ShanghaiTech
method ), 48
prepare_data() ( anoma-
lib.data.video.ucsd_ped.UCSDped method ),
51
prepare_imagenette_data() ( anoma-
lib.models.image.efficient_ad.lightning_model.EfficientAd
method ), 131
prepare_pretrained_model() ( anoma-
lib.models.image.dsr.lightning_model.Dsr
method ), 127
prepare_pretrained_model() ( anoma-
lib.models.image.efficient_ad.lightning_model.EfficientAd
method ), 131
PRO( class in anomalib.metrics ), 191
```
```
R
rand_augmenter() ( anoma-
lib.data.utils.augmenter.Augmenter method ),
81
random_2d_perlin() ( in module anoma-
lib.data.utils.generators ), 79
random_split()( in module anomalib.data.utils.split ),
75
read_depth_image() ( in module anoma-
lib.data.utils.image ), 70
read_image()( in module anomalib.data.utils.image ),
71
read_mask()( in module anomalib.data.utils.image ), 71
RegionExtractor ( class in anoma-
lib.models.image.rkde.region_extractor ),
```
**Index 249**


### 153

RegionExtractor ( _class in anoma-
lib.models.video.ai_vad.regions_ ), 172
reset_distances() ( _anoma-
lib.models.components.sampling.KCenterGreedy
method_ ), 91
reshape_embedding() ( _anoma-
lib.models.image.patchcore.torch_model.PatchcoreModel
static method_ ), 145
Residual ( _class in anoma-
lib.models.image.dsr.torch_model_ ), 123
ResidualStack ( _class in anoma-
lib.models.image.dsr.torch_model_ ), 123
resolve_path()( _in module anomalib.data.utils.path_ ),
60
ReverseDistillation ( _class in anoma-
lib.models.image.reverse_distillation.lightning_model_ ),
146
ReverseDistillationLoss ( _class in anoma-
lib.models.image.reverse_distillation.loss_ ),
148
ReverseDistillationModel ( _class in anoma-
lib.models.image.reverse_distillation.torch_model_ ),
147
Rkde ( _class in anoma-
lib.models.image.rkde.lightning_model_ ),
150
RkdeModel ( _class in anoma-
lib.models.image.rkde.torch_model_ ), 151
RoiStage ( _class in anoma-
lib.models.image.rkde.region_extractor_ ),
154

S

safe_extract() ( _in module anoma-
lib.data.utils.download_ ), 65
sample_coreset() ( _anoma-
lib.models.components.sampling.KCenterGreedy
method_ ), 91
samples ( _anomalib.data.base.dataset.AnomalibDataset
property_ ), 19
samples( _anomalib.data.base.video.AnomalibVideoDataset
property_ ), 22
save() ( _anomalib.loggers.AnomalibWandbLogger
method_ ), 196
save_image()( _in module anomalib.data.utils.image_ ),
72
scale_boxes()( _in module anomalib.data.utils.boxes_ ),
74
scales( _anomalib.models.image.winclip.torch_model.WinClipModel
attribute_ ), 164
score()( _anomalib.models.image.dfm.torch_model.DFMModel
method_ ), 113

```
score_samples() ( anoma-
lib.models.components.cluster.GaussianMixture
method ), 96
score_samples() ( anoma-
lib.models.image.dfm.torch_model.SingleClassGaussian
method ), 114
select_coreset_idxs() ( anoma-
lib.models.components.sampling.KCenterGreedy
method ), 92
setup() ( anomalib.callbacks.LoadModelCallback
method ), 198
setup()( anomalib.data.base.datamodule.AnomalibDataModule
method ), 20
setup()( anomalib.models.image.winclip.torch_model.WinClipModel
method ), 166
setup_sspcab() ( anoma-
lib.models.image.draem.lightning_model.Draem
method ), 115
ShanghaiTech ( class in anoma-
lib.data.video.shanghaitech ), 47
ShanghaiTechDataset ( class in anoma-
lib.data.video.shanghaitech ), 48
ShanghaiTechTestClipsIndexer ( class in anoma-
lib.data.video.shanghaitech ), 49
ShanghaiTechTrainClipsIndexer( class in anoma-
lib.data.video.shanghaitech ), 49
show_image()( in module anomalib.data.utils.image ),
72
SingleClassGaussian ( class in anoma-
lib.models.image.dfm.torch_model ), 114
SparseRandomProjection ( class in anoma-
lib.models.components.dimensionality_reduction ),
88
Split( class in anomalib.data.utils.split ), 75
split_by_label() ( in module anoma-
lib.data.utils.split ), 76
state_dict() ( anoma-
lib.models.image.winclip.lightning_model.WinClip
method ), 164
Stfpm ( class in anoma-
lib.models.image.stfpm.lightning_model ),
154
STFPMLoss( class in anomalib.models.image.stfpm.loss ),
156
STFPMModel ( class in anoma-
lib.models.image.stfpm.torch_model ), 155
StrideSizeError, 76
subcommands() ( anomalib.cli.AnomalibCLI static
method ), 201
subsample() ( anoma-
lib.data.base.dataset.AnomalibDataset
method ), 19
subsample_embedding() ( anoma-
lib.models.image.patchcore.torch_model.PatchcoreModel
```
**250 Index**


_method_ ), 146
subsample_regions() ( _anoma-
lib.models.video.ai_vad.regions.RegionExtractor
static method_ ), 174
SubspaceRestrictionModule ( _class in anoma-
lib.models.image.dsr.torch_model_ ), 124
SubspaceRestrictionNetwork ( _class in anoma-
lib.models.image.dsr.torch_model_ ), 124
SyntheticAnomalyDataset ( _class in anoma-
lib.data.utils.synthetic_ ), 81

T

teacher_channel_mean_std() ( _anoma-
lib.models.image.efficient_ad.lightning_model.EfficientAd
method_ ), 131
test( _anomalib.cli.AnomalibCLI property_ ), 201
test()( _anomalib.engine.Engine method_ ), 181
test_dataloader() ( _anoma-
lib.data.base.datamodule.AnomalibDataModule
method_ ), 20
test_step() ( _anoma-
lib.models.image.ganomaly.lightning_model.Ganomaly
method_ ), 137
TestSplitMode( _class in anomalib.data.utils.split_ ), 75
text_embeddings ( _anoma-
lib.models.image.winclip.torch_model.WinClipModel
property_ ), 167
threshold_callback ( _anomalib.engine.Engine prop-
erty_ ), 182
tile()( _anomalib.data.utils.tiler.Tiler method_ ), 77
Tiler( _class in anomalib.data.utils.tiler_ ), 76
TimerCallback( _class in anomalib.callbacks_ ), 198
TimmFeatureExtractor ( _class in anoma-
lib.models.components.feature_extractors_ ),
82
TorchFXFeatureExtractor ( _class in anoma-
lib.models.components.feature_extractors_ ),
83
TorchInferencer( _class in anomalib.deploy_ ), 204
train( _anomalib.cli.AnomalibCLI property_ ), 201
train()( _anomalib.engine.Engine method_ ), 182
train_dataloader() ( _anoma-
lib.data.base.datamodule.AnomalibDataModule
method_ ), 20
train_transform ( _anoma-
lib.data.base.datamodule.AnomalibDataModule
property_ ), 20
trainer( _anomalib.engine.Engine property_ ), 183
trainer_arguments ( _anoma-
lib.models.image.cfa.lightning_model.Cfa
property_ ), 100
trainer_arguments ( _anoma-
lib.models.image.cflow.lightning_model.Cflow
property_ ), 103

```
trainer_arguments ( anoma-
lib.models.image.csflow.lightning_model.Csflow
property ), 106
trainer_arguments ( anoma-
lib.models.image.dfkde.lightning_model.Dfkde
property ), 109
trainer_arguments ( anoma-
lib.models.image.dfm.lightning_model.Dfm
property ), 112
trainer_arguments ( anoma-
lib.models.image.draem.lightning_model.Draem
property ), 115
trainer_arguments ( anoma-
lib.models.image.dsr.lightning_model.Dsr
property ), 127
trainer_arguments ( anoma-
lib.models.image.efficient_ad.lightning_model.EfficientAd
property ), 132
trainer_arguments ( anoma-
lib.models.image.fastflow.lightning_model.Fastflow
property ), 134
trainer_arguments ( anoma-
lib.models.image.ganomaly.lightning_model.Ganomaly
property ), 137
trainer_arguments ( anoma-
lib.models.image.padim.lightning_model.Padim
property ), 141
trainer_arguments ( anoma-
lib.models.image.patchcore.lightning_model.Patchcore
property ), 143
trainer_arguments ( anoma-
lib.models.image.reverse_distillation.lightning_model.ReverseDistillation
property ), 147
trainer_arguments ( anoma-
lib.models.image.rkde.lightning_model.Rkde
property ), 151
trainer_arguments ( anoma-
lib.models.image.stfpm.lightning_model.Stfpm
property ), 155
trainer_arguments ( anoma-
lib.models.image.uflow.lightning_model.Uflow
property ), 160
trainer_arguments ( anoma-
lib.models.image.winclip.lightning_model.WinClip
property ), 164
trainer_arguments ( anoma-
lib.models.video.ai_vad.lightning_model.AiVad
property ), 169
training_step() ( anoma-
lib.models.image.cfa.lightning_model.Cfa
method ), 100
training_step() ( anoma-
lib.models.image.cflow.lightning_model.Cflow
method ), 104
```
**Index 251**


training_step() ( _anoma-
lib.models.image.csflow.lightning_model.Csflow
method_ ), 107
training_step() ( _anoma-
lib.models.image.dfkde.lightning_model.Dfkde
method_ ), 110
training_step() ( _anoma-
lib.models.image.dfm.lightning_model.Dfm
method_ ), 112
training_step() ( _anoma-
lib.models.image.draem.lightning_model.Draem
method_ ), 115
training_step() ( _anoma-
lib.models.image.dsr.lightning_model.Dsr
method_ ), 127
training_step() ( _anoma-
lib.models.image.efficient_ad.lightning_model.EfficientAd
method_ ), 132
training_step() ( _anoma-
lib.models.image.fastflow.lightning_model.Fastflow
method_ ), 134
training_step() ( _anoma-
lib.models.image.ganomaly.lightning_model.Ganomaly
method_ ), 138
training_step() ( _anoma-
lib.models.image.padim.lightning_model.Padim
method_ ), 141
training_step() ( _anoma-
lib.models.image.patchcore.lightning_model.Patchcore
method_ ), 143
training_step() ( _anoma-
lib.models.image.reverse_distillation.lightning_model.ReverseDistillation
method_ ), 147
training_step() ( _anoma-
lib.models.image.rkde.lightning_model.Rkde
method_ ), 151
training_step() ( _anoma-
lib.models.image.stfpm.lightning_model.Stfpm
method_ ), 155
training_step() ( _anoma-
lib.models.image.uflow.lightning_model.Uflow
method_ ), 160
training_step() ( _anoma-
lib.models.video.ai_vad.lightning_model.AiVad
method_ ), 169
transform( _anomalib.data.base.datamodule.AnomalibDataModule
property_ ), 21
transform( _anomalib.models.image.winclip.torch_model.WinClipModel
property_ ), 167
transform() ( _anoma-
lib.models.components.dimensionality_reduction.PCA
method_ ), 88
transform() ( _anoma-
lib.models.components.dimensionality_reduction.SparseRandomProjection_

```
method ), 89
```
```
U
UCSDped( class in anomalib.data.video.ucsd_ped ), 50
UCSDpedClipsIndexer ( class in anoma-
lib.data.video.ucsd_ped ), 51
UCSDpedDataset ( class in anoma-
lib.data.video.ucsd_ped ), 52
Uflow ( class in anoma-
lib.models.image.uflow.lightning_model ),
160
UflowModel ( class in anoma-
lib.models.image.uflow.torch_model ), 159
UnetDecoder ( class in anoma-
lib.models.image.dsr.torch_model ), 124
UnetEncoder ( class in anoma-
lib.models.image.dsr.torch_model ), 125
UnetModel ( class in anoma-
lib.models.image.dsr.torch_model ), 125
untile()( anomalib.data.utils.tiler.Tiler method ), 77
update()( anomalib.metrics.AnomalyScoreDistribution
method ), 188
update()( anomalib.metrics.AUPR method ), 185
update()( anomalib.metrics.AUPRO method ), 187
update()( anomalib.metrics.AUROC method ), 188
update()( anomalib.metrics.BinaryPrecisionRecallCurve
method ), 188
update()( anomalib.metrics.ManualThreshold method ),
190
update()( anomalib.metrics.MinMax method ), 191
update()( anomalib.metrics.PRO method ), 192
update()( anomalib.models.video.ai_vad.density.BaseDensityEstimator
method ), 175
update()( anomalib.models.video.ai_vad.density.CombinedDensityEstimator
method ), 176
update()( anomalib.models.video.ai_vad.density.GMMEstimator
method ), 177
update()( anomalib.models.video.ai_vad.density.GroupedKNNEstimator
method ), 177
update_distances() ( anoma-
lib.models.components.sampling.KCenterGreedy
method ), 92
update_to() ( anoma-
lib.data.utils.download.DownloadProgressBar
method ), 64
UpsamplingModule ( class in anoma-
lib.models.image.dsr.torch_model ), 126
upscale_image()( in module anomalib.data.utils.tiler ),
79
```
```
V
val_dataloader() ( anoma-
lib.data.base.datamodule.AnomalibDataModule
method ), 21
```
**252 Index**


validate( _anomalib.cli.AnomalibCLI property_ ), 202
validate()( _anomalib.engine.Engine method_ ), 183
validate_and_resolve_path() ( _in module anoma-
lib.data.utils.path_ ), 60
validate_path()( _in module anomalib.data.utils.path_ ),
61
validation_step() ( _anoma-
lib.models.image.cfa.lightning_model.Cfa
method_ ), 100
validation_step() ( _anoma-
lib.models.image.cflow.lightning_model.Cflow
method_ ), 104
validation_step() ( _anoma-
lib.models.image.csflow.lightning_model.Csflow
method_ ), 107
validation_step() ( _anoma-
lib.models.image.dfkde.lightning_model.Dfkde
method_ ), 110
validation_step() ( _anoma-
lib.models.image.dfm.lightning_model.Dfm
method_ ), 112
validation_step() ( _anoma-
lib.models.image.draem.lightning_model.Draem
method_ ), 115
validation_step() ( _anoma-
lib.models.image.dsr.lightning_model.Dsr
method_ ), 128
validation_step() ( _anoma-
lib.models.image.efficient_ad.lightning_model.EfficientAd
method_ ), 132
validation_step() ( _anoma-
lib.models.image.fastflow.lightning_model.Fastflow
method_ ), 134
validation_step() ( _anoma-
lib.models.image.ganomaly.lightning_model.Ganomaly
method_ ), 138
validation_step() ( _anoma-
lib.models.image.padim.lightning_model.Padim
method_ ), 141
validation_step() ( _anoma-
lib.models.image.patchcore.lightning_model.Patchcore
method_ ), 144
validation_step() ( _anoma-
lib.models.image.reverse_distillation.lightning_model.ReverseDistillation
method_ ), 147
validation_step() ( _anoma-
lib.models.image.rkde.lightning_model.Rkde
method_ ), 151
validation_step() ( _anoma-
lib.models.image.stfpm.lightning_model.Stfpm
method_ ), 155
validation_step() ( _anoma-
lib.models.image.uflow.lightning_model.Uflow
method_ ), 161

```
validation_step() ( anoma-
lib.models.image.winclip.lightning_model.WinClip
method ), 164
validation_step() ( anoma-
lib.models.video.ai_vad.lightning_model.AiVad
method ), 169
ValSplitMode( class in anomalib.data.utils.split ), 75
VectorQuantizer ( class in anoma-
lib.models.image.dsr.torch_model ), 126
VelocityExtractor ( class in anoma-
lib.models.video.ai_vad.features ), 172
VideoTargetFrame( class in anomalib.data.base.video ),
22
Visa( class in anomalib.data.image.visa ), 39
VisaDataset( class in anomalib.data.image.visa ), 40
visual_embeddings ( anoma-
lib.models.image.winclip.torch_model.WinClipModel
property ), 167
vq_vae_bot( anomalib.models.image.dsr.torch_model.DiscreteLatentModel
property ), 119
vq_vae_top( anomalib.models.image.dsr.torch_model.DiscreteLatentModel
property ), 119
```
```
W
weights_init() ( anoma-
lib.models.image.ganomaly.torch_model.GanomalyModel
static method ), 139
WinClip ( class in anoma-
lib.models.image.winclip.lightning_model ),
163
WinClipModel ( class in anoma-
lib.models.image.winclip.torch_model ), 164
```
**Index 253**


