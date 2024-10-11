import time
import base64
from io import BytesIO
from typing import Optional, Union, List, Tuple, Dict, Any
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
import enum
from anomalib.utils.visualization.image import ImageVisualizer, VisualizationMode, ImageResult
from PIL import Image


def setup_dataset(name: str = None,
                  task_type: str = "SEGMENTATION",
                  dataset_root: Path = None,
                  normal_dir: str = None,
                  abnormal_dir: str = None,
                  mask_dir: str = None,
                  image_size: Tuple[int, int] = (256, 256),
                  num_workers: int = 0) -> Folder:

    available_task_types = {
        "SEGMENTATION": TaskType.SEGMENTATION,
        "CLASSIFICATION": TaskType.CLASSIFICATION,
        "DETECTION": TaskType.DETECTION
    }

    folder_datamodule = Folder(
        name=name,
        root=dataset_root,
        normal_dir=normal_dir,
        abnormal_dir=abnormal_dir,
        task=available_task_types[task_type],
        mask_dir=str(dataset_root / mask_dir / abnormal_dir),
        image_size=image_size,
        num_workers=num_workers
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
        name="hazelnut_toy",
        normal_dir=dataset_root / normal_dir,
        abnormal_dir=dataset_root / abnormal_dir,
        split="train",
        transform=transform,
        mask_dir=dataset_root / mask_dir / abnormal_dir,
        task=TaskType.SEGMENTATION,
    )

    folder_dataset_segmentation_test = FolderDataset(
        name="hazelnut_toy",
        normal_dir=dataset_root / normal_dir,
        abnormal_dir=dataset_root / abnormal_dir,
        split="test",
        transform=transform,
        mask_dir=dataset_root / mask_dir / abnormal_dir,
        task=TaskType.SEGMENTATION,
    )

    return folder_dataset_segmentation_train, folder_dataset_segmentation_test


# TODO:
#  - Enable user to customize model and task parameters
def train_model(folder_datamodule: Folder,
                model_kwargs: Dict[str, Any]):

    model = Padim(**model_kwargs)
    engine = Engine(task=TaskType.SEGMENTATION)
    engine.fit(model=model, datamodule=folder_datamodule)
    return engine, model


def validate_model(engine: Engine, model: Padim, folder_datamodule: Folder, ckpt_path: str):
    test_results = engine.test(
        model=model,
        datamodule=folder_datamodule,
        ckpt_path=ckpt_path
    )

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
    elif not isinstance(openvino_model_path, Path):
        openvino_model_path = Path(openvino_model_path)

    if not metadata:
        metadata = root_dir / "weights" / "openvino" / "metadata.json"
    elif not isinstance(metadata, Path):
        metadata = Path(metadata)

    assert openvino_model_path.exists()
    assert metadata.exists()

    inferencer = OpenVINOInferencer(
        path=openvino_model_path,
        metadata=metadata,
        device="CPU",
    )

    predictions = inferencer.predict(image=image_path)

    return predictions


def infer_wf(engine: Optional[Engine],
             image_path: Optional[Union[str, Path]],
             root_dir: Optional[Union[str, Path]],
             openvino_model_path: Optional[Union[str, Path]] = None,
             metadata: Optional[Union[str, Path]] = None,
             output_image_path: Optional[Union[str, Path]] = None):

    # TODO:
    #  - salva immagini contenute in inference_response

    # Carica l'immagine per verificare la forma
    image = Image.open(image_path)
    image_np = np.array(image)

    # Controlla la forma dell'immagine e adatta il numero di canali se necessario
    if image_np.shape[2] == 4:  # Se l'immagine ha 4 canali (es: RGBA)
        # Rimuovi il canale alfa per ottenere un'immagine con 3 canali (RGB)
        image_np = image_np[:, :, :3]
        # Salva l'immagine modificata (opzionale)
        image = Image.fromarray(image_np)
        image.save(image_path)  # Sovrascrive l'immagine originale con quella modificata

    elif image_np.shape[2] != 3:
        raise ValueError(
            "Il modello si aspetta un'immagine con 3 canali (RGB), ma l'immagine fornita ha {} canali.".format(
                image_np.shape[2]))

    # Esegui l'inferenza
    inference_response = perform_inference(engine=engine,
                                           image_path=Path(image_path),
                                           root_dir=root_dir,
                                           openvino_model_path=openvino_model_path,
                                           metadata=metadata)

    # TODO:
    #  - salva immagini contenute in inference_response

    output_image_path = root_dir / output_image_path

    save_output_images(predictions=inference_response,
                       output_image_path=output_image_path)

    # base64_output_image = save_output_images_as_base64(predictions=inference_response)

    # print(base64_output_image)

    return serialize_for_api({
        "pred_score": inference_response.pred_score,
        "pred_label": inference_response.pred_label,
        "output_image_path": str(output_image_path)
    })


def test_wf(name: str,
            task_type: str,
            dataset_root: Optional[Union[str, Path]],
            normal_dir: Optional[Union[str, Path]],
            abnormal_dir: Optional[Union[str, Path]],
            mask_dir: Optional[Union[str, Path]],
            ckpt_path: Optional[Union[str, Path]]):

    # Convert inputs to Path objects
    dataset_root = Path(dataset_root)
    normal_dir = Path(normal_dir)
    abnormal_dir = Path(abnormal_dir)
    mask_dir = Path(mask_dir)
    ckpt_path = Path(ckpt_path)

    # Set up the dataset with correct path handling
    folder_datamodule = setup_dataset(name=name,
                                      task_type=task_type,
                                      dataset_root=dataset_root,
                                      normal_dir=normal_dir.name,
                                      abnormal_dir=abnormal_dir.name,
                                      mask_dir=mask_dir.name)

    model = Padim()
    engine = Engine(task=TaskType.SEGMENTATION)

    # Validate the model
    test_response = validate_model(engine, model, folder_datamodule, str(ckpt_path))

    return test_response


def serialize_for_api(data):
    def serialize_item(item):
        if isinstance(item, np.ndarray):
            return item.tolist()
        elif isinstance(item, (np.float32, np.float64)):
            return float(item)
        elif isinstance(item, (np.int32, np.int64)):
            return int(item)
        elif isinstance(item, dict):
            return {k: serialize_item(v) for k, v in item.items()}
        elif isinstance(item, list):
            return [serialize_item(i) for i in item]
        elif isinstance(item, enum.Enum):
            return item.name
        else:
            return item

    return serialize_item(data)


def save_output_images(predictions: ImageResult = None,
                       task_type: TaskType = TaskType.SEGMENTATION,
                       output_image_path: str | Path = None):

    if not output_image_path:
        return

    visualizer = ImageVisualizer(mode=VisualizationMode.FULL, task=task_type)
    output_image = visualizer.visualize_image(predictions)
    print(output_image_path)
    Image.fromarray(output_image).save(output_image_path, format="PNG")


def save_output_images_as_base64(predictions: ImageResult = None,
                                 task_type: TaskType = TaskType.SEGMENTATION) -> str:
    # Visualize the image
    visualizer = ImageVisualizer(mode=VisualizationMode.FULL, task=task_type)
    output_image = visualizer.visualize_image(predictions)

    # Convert the output image to a PIL Image
    image = Image.fromarray(output_image)

    # Save the image to a BytesIO object
    buffered = BytesIO()
    image.save(buffered, format=None)

    # Encode the BytesIO object to a Base64 string
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

    return img_str


if __name__ == "__main__":

    pass

