from typing import Optional, Union
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from pathlib import Path
from utilities.utilities import (infer_wf, test_wf, validate_model, train_model, create_folder_datasets, \
    visualize_data, setup_dataset, export_model)

app = FastAPI(
    title="Anomaly Detection API",
    description="API for training, testing, and inference using Padim model"
)


class TrainRequest(BaseModel):
    dataset_name: str = Field(..., example="hazelnut_toy", description="Name of the dataset")
    dataset_root: str = Field(..., example="datasets/hazelnut_toy", description="Root directory of the dataset")
    normal_dir: str = Field(..., example="good", description="Directory containing normal images")
    abnormal_dir: str = Field(..., example="crack", description="Directory containing abnormal images")
    mask_dir: str = Field(..., example="mask", description="Directory containing masks for abnormal images")
    image_size: tuple[int, int] = Field((256, 256), example=(256, 256),
                                        description="Size to which images should be resized")


class InferenceRequest(BaseModel):
    image_path: str = Field(..., example="datasets/hazelnut_toy/crack/01.jpg",
                            description="Path to the image for inference")
    root_dir: Optional[str] = Field(None, example="results/Padim/hazelnut_toy/latest",
                                    description="Root directory for inference")
    openvino_model_path: Optional[str] = Field(None,
                                               example="results/Padim/hazelnut_toy/latest/weights/openvino/model.bin",
                                               description="Path to the OpenVINO model")
    metadata: Optional[str] = Field(None, example="results/Padim/hazelnut_toy/latest/weights/openvino/metadata.json",
                                    description="Path to the metadata file for the OpenVINO model")


class TestRequest(BaseModel):
    dataset_name: str = Field(..., example="hazelnut_toy", description="Name of the dataset")
    dataset_root: str = Field(..., example="datasets/hazelnut_toy", description="Root directory of the dataset")
    normal_dir: str = Field(..., example="good", description="Directory containing normal images")
    abnormal_dir: str = Field(..., example="crack", description="Directory containing abnormal images")
    mask_dir: str = Field(..., example="mask", description="Directory containing masks for abnormal images")
    ckpt_path: str = Field(..., example="results/Padim/hazelnut_toy/v2/weights/lightning/model.ckpt",
                           description="Path to the model checkpoint")


@app.post("/train", summary="Train the model", description="Train the Padim model using the provided dataset")
def train(request: TrainRequest):
    """
    This endpoint trains the Padim model using the provided dataset.

    Parameters:
    - request: TrainRequest
        - dataset_root: Root directory of the dataset.
        - normal_dir: Directory containing normal images.
        - abnormal_dir: Directory containing abnormal images.
        - mask_dir: Directory containing masks for abnormal images.
        - image_size: Size to which images should be resized.

    Returns:
    - JSON response with status of the training process.
    """
    dataset_name = request.dataset_name
    dataset_root = Path.cwd() / Path(request.dataset_root)
    normal_dir = request.normal_dir
    abnormal_dir = request.abnormal_dir
    mask_dir = request.mask_dir
    image_size = request.image_size

    #try:
    if True:
        dataset_root = Path(dataset_root)
        folder_datamodule = setup_dataset(
            dataset_name,
            dataset_root,
            normal_dir,
            abnormal_dir,
            mask_dir
        )
        visualize_data(folder_datamodule)

        create_folder_datasets(
            dataset_name,
            dataset_root,
            normal_dir,
            abnormal_dir,
            mask_dir,
            image_size
        )

        engine, model = train_model(folder_datamodule)

        validation_result = validate_model(engine, model, folder_datamodule, engine.best_model_path)

        export_model(engine, model)
    #except Exception as e:
    #    raise HTTPException(status_code=500, detail=str(e))

    # TODO
    #  - valutare come serializzare informazioni in output
    return {"status": "Training completed successfully"}


@app.post("/inference", summary="Perform inference", description="Perform inference using the trained model")
def inference(request: InferenceRequest):
    """
    This endpoint performs inference using the trained Padim model.

    Parameters:
    - request: InferenceRequest
        - image_path: Path to the image for inference.
        - root_dir: Root directory for inference (optional).
        - openvino_model_path: Path to the OpenVINO model (optional).
        - metadata: Path to the metadata file for the OpenVINO model (optional).

    Returns:
    - JSON response with status of the inference process and the results.
    """
    image_path = request.image_path
    root_dir = Path.cwd() / Path(request.root_dir)
    openvino_model_path = request.openvino_model_path
    metadata = request.metadata

    try:
        inference_result = infer_wf(
            engine=None,
            image_path=image_path,
            root_dir=root_dir,
            openvino_model_path=openvino_model_path,
            metadata=metadata
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # TODO
    #  - valutare come serializzare informazioni in output
    return {"status": "Inference completed successfully"}#, "results": response.__dict__}


@app.post("/test", summary="Test the model", description="Test the model using the provided dataset and checkpoint")
def test(request: TestRequest):
    """
    This endpoint tests the Padim model using the provided dataset and checkpoint.

    Parameters:
    - request: TestRequest
        - dataset_root: Root directory of the dataset.
        - normal_dir: Directory containing normal images.
        - abnormal_dir: Directory containing abnormal images.
        - mask_dir: Directory containing masks for abnormal images.
        - ckpt_path: Path to the model checkpoint.

    Returns:
    - JSON response with status of the testing process.
    """
    datast_name = request.datast_name
    dataset_root = Path.cwd() / Path(request.dataset_root)
    normal_dir = request.normal_dir
    abnormal_dir = request.abnormal_dir
    mask_dir = request.mask_dir
    ckpt_path = request.ckpt_path

    try:
        test_result = test_wf(
            datast_name,
            dataset_root,
            normal_dir,
            abnormal_dir,
            mask_dir,
            ckpt_path
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # TODO
    #  - valutare come serializzare informazioni in output
    return {"status": "Testing completed successfully"}


if __name__ == "__main__":

    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8001)

