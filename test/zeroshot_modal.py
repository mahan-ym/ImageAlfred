import os
from io import BytesIO

import cv2
import modal
import numpy as np
from PIL import Image
from rapidfuzz import process

app = modal.App("zeroshot-test")

PYTHON_VERSION = "3.12"
CUDA_VERSION = "12.4.0"
FLAVOR = "devel"
OPERATING_SYS = "ubuntu22.04"
tag = f"{CUDA_VERSION}-{FLAVOR}-{OPERATING_SYS}"
volume = modal.Volume.from_name("zeroshot-test-volume", create_if_missing=True)
volume_path = "/vol"

MODEL_CACHE_DIR = f"{volume_path}/models/cache"
TORCH_HOME = f"{volume_path}/torch/home"
HF_HOME = f"{volume_path}/huggingface"

image = (
    modal.Image.from_registry(f"nvidia/cuda:{tag}", add_python=PYTHON_VERSION)
    .env(
        {
            "HF_HUB_ENABLE_HF_TRANSFER": "1",  # faster downloads
            "HF_HUB_CACHE": HF_HOME,
            "TORCH_HOME": TORCH_HOME,
        }
    )
    .apt_install("git")
    .pip_install(
        "huggingface-hub",
        "hf_transfer",
        "Pillow",
        "numpy",
        "transformers",
        "opencv-contrib-python-headless",
        "RapidFuzz",
        gpu="A10G",
    )
    .pip_install(
        "torch==2.4.1",
        "torchvision==0.19.1",
        index_url="https://download.pytorch.org/whl/cu124",
        gpu="A10G",
    )
    .pip_install(
        "git+https://github.com/luca-medeiros/lang-segment-anything.git",
        gpu="A10G",
    )
)

@app.function(
    image=image,
    gpu="A10G",
    volumes={volume_path: volume},
)
def zeroshot_modal(
    image_pil: Image.Image,
    labels: list[str],
) -> list[dict]:
    """
    Perform zero-shot segmentation on an image using specified labels.
    Args:
        image_pil (Image.Image): The input image as a PIL Image.
        labels (list[str]): List of labels for zero-shot segmentation.

    Returns:
        list[dict]: List of dictionaries containing label and bounding box information.
    """    
    from transformers import pipeline
    checkpoint = "google/owlv2-base-patch16-ensemble"
    detector = pipeline(model=checkpoint, task="zero-shot-object-detection")
    # Load the image
    predictions = detector(
        image_pil,
        candidate_labels=labels,
    )
    return predictions