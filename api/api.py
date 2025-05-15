import pickle
from typing import Optional, Union, List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from pathlib import Path
from api.utilities.utilities import infer_wf, test_wf, validate_model, train_model, create_folder_datasets, \
    visualize_data, setup_dataset, export_model

app = FastAPI(
    title="Anomaly Detection API",
    description="API for training, testing, and inference using the PaDiM model for anomaly detection. This API allows users to train a model, perform inference, and test the model's performance on various datasets."
)


class TrainRequest(BaseModel):
    name: str = Field(..., example="hazelnut_toy_train_1", description="**Name** of the directory used to store training and testing results. This directory will hold model checkpoints, logs, and other related files.")
    task_type: str = Field(..., example="SEGMENTATION", description="The **task type** defines the kind of problem the model will solve, such as 'SEGMENTATION' for pixel-level classification or 'CLASSIFICATION' for image-level prediction.")
    dataset_root: str = Field(..., example="datasets/hazelnut_toy", description="Root directory where the **dataset** is stored. It should contain subdirectories for normal and abnormal images, as well as optional mask directories.")
    normal_dir: Union[str, List[str]] = Field(..., example="good", description="Subdirectory containing **normal images** (i.e., images without anomalies).")
    abnormal_dir: Union[str, List[str]] = Field(..., example="crack", description="Subdirectory containing **abnormal images** (i.e., images with anomalies).")
    mask_dir: str = Field(..., example="mask", description="Optional subdirectory containing **mask images** that highlight the areas of anomalies within the abnormal images. Useful for segmentation tasks.")
    image_size: tuple[int, int] = Field((256, 256), example=(256, 256), description="Target **size** to which all images should be resized before processing. Provided as a tuple of `(width, height)`.")
    num_workers: int = Field(..., example=0, description="Number of **workers** to use for data loading. Increasing this value can improve performance by parallelizing data processing tasks.")
    model_kwargs: Optional[dict] = Field(
        default_factory={},
        example={"backbone": "resnet18"},
        description="Additional keyword arguments for the model."
    )
    output_file_path: Optional[str] = Field(
        None,
        example="path/to/output/train_output.pkl",
        description="File path where the output object will be saved as a pickle file."
    )

class InferenceRequest(BaseModel):
    image_path: str = Field(..., example="datasets/hazelnut_toy/crack/01.jpg", description="The file path to the **image** on which to perform inference. The image should be located within the dataset root directory.")
    root_dir: Optional[str] = Field(None, example="results/Padim/hazelnut_toy/latest", description="Root directory where **inference results** will be stored. If not provided, a default directory will be used.")
    openvino_model_path: Optional[str] = Field(None, example="results/Padim/hazelnut_toy/latest/weights/openvino/model.bin", description="Path to the **OpenVINO model** file, used for accelerated inference. Optional, only needed if using OpenVINO for inference.")
    metadata: Optional[str] = Field(None, example="results/Padim/hazelnut_toy/latest/weights/openvino/metadata.json", description="Path to the **metadata file** associated with the OpenVINO model. This file typically contains model configuration and other important details.")
    output_image_path: Optional[str] = Field(None, example="path/to/output/image.png", description="File path where the **output image** with the anomaly detection results should be saved. Optional; if not provided, results may not be visualized.")
    output_file_path: Optional[str] = Field(
        None,
        example="path/to/output/train_output.pkl",
        description="File path where the output object will be saved as a pickle file."
    )

class TestRequest(BaseModel):
    name: str = Field(..., example="hazelnut_toy_test_1", description="**Name** of the directory used to store test results. It will contain logs, output images, and evaluation metrics.")
    task_type: str = Field(..., example="SEGMENTATION", description="The **task type** for testing, such as 'SEGMENTATION' or 'CLASSIFICATION'. This should match the task type used during training.")
    dataset_root: str = Field(..., example="datasets/hazelnut_toy", description="Root directory of the **dataset** used for testing. This directory should include normal and abnormal images as well as optional masks.")
    normal_dir: Union[str, List[str]] = Field(..., example="good", description="Subdirectory containing **normal images** for the testing phase.")
    abnormal_dir: Union[str, List[str]] = Field(..., example="crack", description="Subdirectory containing **abnormal images** for the testing phase.")
    mask_dir: str = Field(..., example="mask", description="Optional subdirectory containing **mask images** used for evaluating pixel-level segmentation accuracy.")
    ckpt_path: str = Field(..., example="results/Padim/hazelnut_toy/v2/weights/lightning/model.ckpt", description="Path to the model **checkpoint** file used during testing. This file contains the weights of the trained model.")
    output_file_path: Optional[str] = Field(
        None,
        example="path/to/output/train_output.pkl",
        description="File path where the output object will be saved as a pickle file."
    )

class TrainResponse(BaseModel):
    status: str = Field(..., example="Training completed successfully", description="Indicates the **status** of the training process. A successful status message confirms that the model has been trained without errors.")
    validation_result: list = Field(..., example=[
        {
            "pixel_AUROC": 0.954082727432251,
            "pixel_F1Score": 0.5962215662002563,
            "image_AUROC": 1,
            "image_F1Score": 0.800000011920929
        }
    ], description="A list of dictionaries containing **validation metrics** such as `pixel_AUROC`, `pixel_F1Score`, `image_AUROC`, and `image_F1Score`. These metrics provide an assessment of the model's performance during training.")
    output_images_dir: str = Field(..., example="results/Padim/metallic_surfaces/latest/images/", description="The directory path where the **output images** generated during training are stored. This directory contains visual representations of the model's performance.")
    output_models_dir: str = Field(..., example="results/Padim/metallic_surfaces/latest/weights/", description="The directory path where the **model weights** and other related files are stored after training. These files are used for inference and further testing.")


class InferenceResponse(BaseModel):
    status: str = Field(..., example="Inference completed successfully", description="Indicates the **status** of the inference process. A successful status confirms that the inference was completed without errors.")
    inference_result: dict = Field(..., example={
        "pred_score": 0.6108382353104678,
        "pred_label": "ABNORMAL",
        "output_image_path": "C:\\Users\\Golden Bit\\Desktop\\projects_in_progress\\protom\\anomaly-detection-API\\results\\Padim\\hazelnut_toy\\latest\\image.png"
    }, description="A dictionary containing the **inference results**, including the `pred_score` which quantifies the confidence of the prediction, the `pred_label` which indicates whether the image is `NORMAL` or `ABNORMAL`, and the `output_image_path` where the annotated image is saved.")


class TestResponse(BaseModel):
    status: str = Field(..., example="Testing completed successfully", description="Indicates the **status** of the testing process. A successful status message confirms that the model testing was completed without errors.")
    test_result: list = Field(..., example=[
        {
            "pixel_AUROC": 0.9817789793014526,
            "pixel_F1Score": 0.6180762648582458,
            "image_AUROC": 1,
            "image_F1Score": 0.5
        }
    ], description="A list of dictionaries containing **test metrics** such as `pixel_AUROC`, `pixel_F1Score`, `image_AUROC`, and `image_F1Score`. These metrics provide an evaluation of the model's performance on the test dataset.")
    output_images_dir: str = Field(..., example="results/Padim/metallic_surfaces/latest/images/",
                                   description="The directory path where the **output images** generated during training are stored. This directory contains visual representations of the model's performance.")


@app.post("/train", response_model=TrainResponse, summary="Train the PaDiM model", description="""
Train the PaDiM model using the provided dataset.

### Parameters:
- **request**: The `TrainRequest` object containing:
  - `name`: Directory name for storing results.
  - `task_type`: Type of task (e.g., SEGMENTATION).
  - `dataset_root`: Root directory of the dataset.
  - `normal_dir`: Directory containing normal images.
  - `abnormal_dir`: Directory containing abnormal images.
  - `mask_dir`: Directory containing masks for abnormal images.
  - `image_size`: Target size for image resizing.
  - `num_workers`: Number of workers for data loading.
  - `model_kwargs`: Additional model parameters.
  - `output_file_path`: Path to save the output object as a pickle file.

### Returns:
- **TrainResponse**: JSON object containing:
  - `status`: Status of the training process.
  - `validation_result`: Validation metrics of the model.
  - `output_images_dir`: Directory where the output images are stored.
  - `output_models_dir`: Directory where the model weights are stored.
  
---

This endpoint trains the PaDiM model using the dataset specified in the request.

The PaDiM model is designed for anomaly detection and can be trained for different tasks such as segmentation or classification. During the training process, the model learns to distinguish between normal and abnormal images, leveraging the provided dataset.

### Example
```json
{
    "name": "metallic_surfaces",
    "task_type": "SEGMENTATION",
    "dataset_root": "datasets/hazelnut_toy",
    "normal_dir": "good",
    "abnormal_dir": "crack",
    "mask_dir": "mask",
    "image_size": [256, 256],
    "num_workers": 4,
    "model_kwargs": {
        "learning_rate": 0.001,
        "epochs": 10
    },
    "output_file_path": "path/to/output/train_output.pkl"
}
```

### Response
```json
{
    "status": "Training completed successfully",
    "validation_result": [
        {
            "pixel_AUROC": 0.954,
            "pixel_F1Score": 0.596,
            "image_AUROC": 1,
            "image_F1Score": 0.8
        }
    ],
    "output_images_dir": "results/Padim/metallic_surfaces/latest/images/",
    "output_models_dir": "results/Padim/metallic_surfaces/latest/weights/"
}
```
""")
def train(request: TrainRequest):
    """
    This endpoint trains the PaDiM model using the dataset specified in the request.

    The PaDiM model is designed for anomaly detection and can be trained for different tasks such as segmentation or classification. During the training process, the model learns to distinguish between normal and abnormal images, leveraging the provided dataset.

    ### Example
    ```json
    {
        "name": "metallic_surfaces",
        "task_type": "SEGMENTATION",
        "dataset_root": "datasets/hazelnut_toy",
        "normal_dir": "good",
        "abnormal_dir": "crack",
        "mask_dir": "mask",
        "image_size": [256, 256],
        "num_workers": 4
    }
    ```

    ### Response
    ```json
    {
        "status": "Training completed successfully",
        "validation_result": [
            {
                "pixel_AUROC": 0.954,
                "pixel_F1Score": 0.596,
                "image_AUROC": 1,
                "image_F1Score": 0.8
            }
        ]
    }
    ```
    """
    name = request.name
    task_type = request.task_type
    dataset_root = Path.cwd() / Path(request.dataset_root)
    normal_dir = request.normal_dir
    abnormal_dir = request.abnormal_dir
    mask_dir = request.mask_dir
    image_size = request.image_size
    num_workers = request.num_workers
    model_kwargs = request.model_kwargs or {}
    output_file_path = request.output_file_path

    try:
        folder_datamodule = setup_dataset(
            name,
            task_type,
            dataset_root,
            normal_dir,
            abnormal_dir,
            mask_dir,
            image_size,
            num_workers
        )

        # visualize_data(folder_datamodule)

        engine, model = train_model(folder_datamodule, model_kwargs)

        validation_result = validate_model(engine, model, folder_datamodule, engine.best_model_path)

        export_model(engine, model)

        output = {

            "status": "Training completed successfully",

            "validation_result": validation_result,

            "output_images_dir": f"results/Padim/{name}/latest/images/",

            "output_models_dir": f"results/Padim/{name}/latest/weights/"

        }

        # Save the output object if output_file_path is provided
        if output_file_path:
            output_file_path = Path(output_file_path)
            output_dir = output_file_path.parent
            output_dir.mkdir(parents=True, exist_ok=True)
            with open(output_file_path, 'wb') as f:
                pickle.dump(output, f)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return output


@app.post("/inference", response_model=InferenceResponse, summary="Perform inference using the PaDiM model", description="""
Perform inference using the trained PaDiM model.

### Parameters:
- **request**: The `InferenceRequest` object containing:
  - `image_path`: Path to the image for inference.
  - `root_dir`: Directory for storing inference results.
  - `openvino_model_path`: Path to the OpenVINO model (optional).
  - `metadata`: Path to the metadata file for the OpenVINO model (optional).
  - `output_image_path`: Path to save the output image.
  - `output_file_path`: Path to save the output object as a pickle file.

### Returns:
- **InferenceResponse**: JSON object containing:
  - `status`: Status of the inference process.
  - `inference_result`: Results of the inference process.
  
---

This endpoint performs inference on a single image using the trained PaDiM model.

The inference process predicts whether the image contains anomalies, and if so, it identifies the regions of the image where the anomalies are located. The results include the prediction score, the label (NORMAL or ABNORMAL), and an optional output image showing the anomaly regions.

### Example
```json
{
    "image_path": "datasets/hazelnut_toy/crack/01.jpg",
    "root_dir": "results/Padim/hazelnut_toy/latest",
    "output_image_path": "path/to/output/image.png"
    "output_file_path": "path/to/output/infer_output.pkl"
}
```

### Response
```json
{
    "status": "Inference completed successfully",
    "inference_result": {
        "pred_score": 0.611,
        "pred_label": "ABNORMAL",
        "output_image_path": "C:\\results\\Padim\\hazelnut_toy\\latest\\image.png"
    }
}
```
""")
def inference(request: InferenceRequest):
    """
    This endpoint performs inference on a single image using the trained PaDiM model.

    The inference process predicts whether the image contains anomalies, and if so, it identifies the regions of the image where the anomalies are located. The results include the prediction score, the label (NORMAL or ABNORMAL), and an optional output image showing the anomaly regions.

    ### Example
    ```json
    {
        "image_path": "datasets/hazelnut_toy/crack/01.jpg",
        "root_dir": "results/Padim/hazelnut_toy/latest",
        "output_image_path": "path/to/output/image.png"
    }
    ```

    ### Response
    ```json
    {
        "status": "Inference completed successfully",
        "inference_result": {
            "pred_score": 0.611,
            "pred_label": "ABNORMAL",
            "output_image_path": "C:\\results\\Padim\\hazelnut_toy\\latest\\image.png"
        }
    }
    ```
    """
    image_path = request.image_path
    root_dir = Path.cwd() / Path(request.root_dir)
    openvino_model_path = request.openvino_model_path
    metadata = request.metadata
    output_image_path = request.output_image_path
    output_file_path = request.output_file_path

    try:
        response = infer_wf(
            engine=None,
            image_path=image_path,
            root_dir=root_dir,
            openvino_model_path=openvino_model_path,
            metadata=metadata,
            output_image_path=output_image_path
        )

        # Save the output object if output_file_path is provided
        if output_file_path:
            output_file_path = Path(output_file_path)
            output_dir = output_file_path.parent
            output_dir.mkdir(parents=True, exist_ok=True)
            with open(output_file_path, 'wb') as f:
                pickle.dump(response, f)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {
        "status": "Inference completed successfully",
        "inference_result": response
    }


@app.post("/test", response_model=TestResponse, summary="Test the PaDiM model", description="""
Test the trained PaDiM model using a specified dataset.

### Parameters:
- **request**: The `TestRequest` object containing:
  - `name`: Directory name for storing test results.
  - `task_type`: Type of task (e.g., SEGMENTATION).
  - `dataset_root`: Root directory of the dataset.
  - `normal_dir`: Directory containing normal images.
  - `abnormal_dir`: Directory containing abnormal images.
  - `mask_dir`: Directory containing masks for abnormal images.
  - `ckpt_path`: Path to the model checkpoint.

### Returns:
- **TestResponse**: JSON object containing:
  - `status`: Status of the testing process.
  - `test_result`: Results of the testing process.
  - `output_images_dir`: Directory where the output images are stored.
  
---

This endpoint tests the PaDiM model on a specific dataset using the provided checkpoint.

Testing involves running the model on the entire test dataset to evaluate its performance. The results include metrics such as AUROC and F1 score at both the pixel and image levels.

### Example
```json
{
    "name": "metallic_surfaces",
    "task_type": "SEGMENTATION",
    "dataset_root": "datasets/hazelnut_toy",
    "normal_dir": "good",
    "abnormal_dir": "crack",
    "mask_dir": "mask",
    "ckpt_path": "results/Padim/hazelnut_toy/v2/weights/lightning/model.ckpt"
}
```

### Response
```json
{
    "status": "Testing completed successfully",
    "test_result": [
        {
            "pixel_AUROC": 0.982,
            "pixel_F1Score": 0.618,
            "image_AUROC": 1,
            "image_F1Score": 0.5
        }
    ],
    "output_images_dir": "results/Padim/metallic_surfaces/latest/images/"
}
```
""")
def test(request: TestRequest):
    """
    This endpoint tests the PaDiM model on a specific dataset using the provided checkpoint.

    Testing involves running the model on the entire test dataset to evaluate its performance. The results include metrics such as AUROC and F1 score at both the pixel and image levels.

    ### Example
    ```json
    {
        "name": "metallic_surfaces",
        "task_type": "SEGMENTATION",
        "dataset_root": "datasets/hazelnut_toy",
        "normal_dir": "good",
        "abnormal_dir": "crack",
        "mask_dir": "mask",
        "ckpt_path": "results/Padim/hazelnut_toy/v2/weights/lightning/model.ckpt"
    }
    ```

    ### Response
    ```json
    {
        "status": "Testing completed successfully",
        "test_result": [
            {
                "pixel_AUROC": 0.982,
                "pixel_F1Score": 0.618,
                "image_AUROC": 1,
                "image_F1Score": 0.5
            }
        ]
    }
    ```
    """
    name = request.name
    task_type = request.task_type
    dataset_root = Path.cwd() / Path(request.dataset_root)
    normal_dir = request.normal_dir
    abnormal_dir = request.abnormal_dir
    mask_dir = request.mask_dir
    ckpt_path = request.ckpt_path

    try:
        test_response = test_wf(name=name,
                                task_type=task_type,
                                dataset_root=dataset_root,
                                normal_dir=normal_dir,
                                abnormal_dir=abnormal_dir,
                                mask_dir=mask_dir,
                                ckpt_path=ckpt_path)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {
        "status": "Testing completed successfully",
        "test_result": test_response,
        "output_images_dir": f"results/Padim/{name}/latest/images/",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8102)
