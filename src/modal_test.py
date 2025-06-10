import os
from io import BytesIO

import cv2
import modal
import numpy as np
from PIL import Image
from enum import Enum

app = modal.App("ImageAlfredTest")

PYTHON_VERSION = "3.12"
CUDA_VERSION = "12.4.0"
FLAVOR = "devel"
OPERATING_SYS = "ubuntu22.04"
tag = f"{CUDA_VERSION}-{FLAVOR}-{OPERATING_SYS}"
volume = modal.Volume.from_name("image-alfred-volume", create_if_missing=True)
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
        "transformers",
        "Pillow",
        "numpy",
        "opencv-contrib-python-headless",
        gpu="A10G",
    )
    .pip_install(
        "torch==2.4.1",
        "torchvision==0.19.1",
        index_url="https://download.pytorch.org/whl/cu124",
        gpu="A10G",
    )
)


class segformer_labels(Enum):
    BACKGROUND = 0
    HAT = 1
    HAIR = 2
    SUNGLASSES = 3
    UPPER_CLOTHES = 4
    SKIRT = 5
    PANTS = 6
    DRESS = 7
    BELT = 8
    LEFT_SHOE = 9
    RIGHT_SHOE = 10
    FACE = 11
    LEFT_LEG = 12
    RIGHT_LEG = 13
    LEFT_ARM = 14
    RIGHT_ARM = 15
    BAG = 16
    SCARF = 17


@app.function(
    image=image,
    gpu="A10G",
    volumes={volume_path: volume},
)
def segformer(image_pil: Image.Image, labels: list[segformer_labels]) -> Image.Image:
    from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation
    import torch.nn as nn

    tokenizer = SegformerImageProcessor.from_pretrained(
        "mattmdjaga/segformer_b2_clothes",
        cache_dir=HF_HOME,
    )
    model = AutoModelForSemanticSegmentation.from_pretrained(
        "mattmdjaga/segformer_b2_clothes",
        cache_dir=HF_HOME,
    )

    inputs = tokenizer(images=image_pil, return_tensors="pt")
    outputs = model(**inputs)
    logits = outputs.logits.cpu()

    upsampled_logits = nn.functional.interpolate(
        logits,
        size=image_pil.size[::-1],
        mode="bilinear",
        align_corners=False,
    )

    pred_seg = upsampled_logits.argmax(dim=1)[0].numpy()

    pants_mask = (pred_seg == segformer_labels.UPPER_CLOTHES.value).astype(np.uint8)

    # Convert the mask to a PIL image
    mask_image = Image.fromarray(pants_mask * 255, mode="L")
    mask_image = mask_image.convert("RGB")
    return mask_image
