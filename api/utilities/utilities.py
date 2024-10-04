from typing import Optional, Union
from pathlib import Path
import numpy as np
from PIL import Image
from torchvision.transforms.v2 import Resize
from torchvision.transforms.v2.functional import to_pil_image
from anomalib.data.image.folder import Folder, FolderDataset
from anomalib import TaskType
from anomalib.models import Padim
from anomalib.engine import Engine
from anomalib.deploy import ExportType, OpenVINOInferencer
from matplotlib import pyplot as plt


def setup_dataset(dataset_root: Path, normal_dir: str, abnormal_dir: str, mask_dir: str) -> Folder:
    folder_datamodule = Folder(
        name="white_cabinet_surface",
        root=dataset_root,
        normal_dir=normal_dir,
        abnormal_dir=abnormal_dir,
        task=TaskType.SEGMENTATION,
        mask_dir=dataset_root / mask_dir / abnormal_dir,
        image_size=(256, 256),
        num_workers=0
    )
    folder_datamodule.setup()
    return folder_datamodule


def visualize_data(folder_datamodule: Folder):
    i, data = next(enumerate(folder_datamodule.train_dataloader()))
    print(data.keys(), data["image"].shape)

    img = to_pil_image(data["image"][0].clone())
    msk = to_pil_image(data["mask"][0]).convert("RGB")
    Image.fromarray(np.hstack((np.array(img), np.array(msk)))).show()


def create_folder_datasets(dataset_root: Path, normal_dir: str, abnormal_dir: str, mask_dir: str,
                           image_size: tuple) -> tuple:
    transform = Resize(image_size, antialias=True)

    folder_dataset_segmentation_train = FolderDataset(
        name="white_cabinet_surface",
        normal_dir=dataset_root / normal_dir,
        abnormal_dir=dataset_root / abnormal_dir,
        split="train",
        transform=transform,
        mask_dir=dataset_root / mask_dir / abnormal_dir,
        task=TaskType.SEGMENTATION,
    )

    folder_dataset_segmentation_test = FolderDataset(
        name="white_cabinet_surface",
        normal_dir=dataset_root / normal_dir,
        abnormal_dir=dataset_root / abnormal_dir,
        split="test",
        transform=transform,
        mask_dir=dataset_root / mask_dir / abnormal_dir,
        task=TaskType.SEGMENTATION,
    )

    return folder_dataset_segmentation_train, folder_dataset_segmentation_test


def train_model(folder_datamodule: Folder):
    model = Padim()
    engine = Engine(task=TaskType.SEGMENTATION)
    engine.fit(model=model, datamodule=folder_datamodule)
    return engine, model


def validate_model(engine: Engine, model: Padim, folder_datamodule: Folder, ckpt_path: str):
    test_results = engine.test(
        model=model,
        datamodule=folder_datamodule,
        ckpt_path=ckpt_path
    )
    print(test_results)
    return test_results


def export_model(engine: Engine, model: Padim):
    engine.export(model=model, export_type=ExportType.OPENVINO)


def perform_inference(engine: Optional[Engine],
                      image_path: Optional[Union[str, Path]],
                      root_dir: Optional[Union[str, Path]],
                      openvino_model_path: Optional[Union[str, Path]],
                      metadata: Optional[Union[str, Path]]):
    if engine:
        root_dir = Path(engine.trainer.default_root_dir)

    if not openvino_model_path:
        openvino_model_path = root_dir / "weights" / "openvino" / "model.bin"

    if not metadata:
        metadata = root_dir / "weights" / "openvino" / "metadata.json"

    assert openvino_model_path.exists()
    assert metadata.exists()

    inferencer = OpenVINOInferencer(
        path=openvino_model_path,
        metadata=metadata,
        device="CPU",
    )

    predictions = inferencer.predict(image=image_path)

    print(predictions.pred_score, predictions.pred_label)
    plt.imshow(predictions.image)
    plt.imshow(predictions.anomaly_map)
    plt.imshow(predictions.heat_map)
    plt.imshow(predictions.pred_mask)
    plt.imshow(predictions.segmentations)

    return predictions


def infer_wf(engine: Optional[Engine],
             image_path: Optional[Union[str, Path]],
             root_dir: Optional[Union[str, Path]],
             openvino_model_path: Optional[Union[str, Path]] = None,
             metadata: Optional[Union[str, Path]] = None):
    inference_response = perform_inference(engine=engine,
                                           image_path=Path(image_path),
                                           root_dir=root_dir,
                                           openvino_model_path=openvino_model_path,
                                           metadata=metadata)

    return inference_response


def test_wf(dataset_root: Optional[Union[str, Path]],
            normal_dir: Optional[Union[str, Path]],
            abnormal_dir: Optional[Union[str, Path]],
            mask_dir: Optional[Union[str, Path]],
            ckpt_path: Optional[Union[str, Path]]):
    dataset_root = Path(dataset_root)
    folder_datamodule = setup_dataset(dataset_root, normal_dir, abnormal_dir, mask_dir)
    model = Padim()
    engine = Engine(task=TaskType.SEGMENTATION)
    test_response = validate_model(engine, model, folder_datamodule, ckpt_path)
    print(test_response)


if __name__ == "__main__":

    pass

