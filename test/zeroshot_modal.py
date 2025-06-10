import os
from io import BytesIO

import cv2
import modal
import numpy as np
from PIL import Image

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
    .apt_install(
        "git",
    )
    .pip_install(
        "huggingface-hub",
        "hf_transfer",
        "Pillow",
        "numpy",
        "transformers",
        "opencv-contrib-python-headless",
        gpu="A10G",
    )
    .pip_install(
        "torch==2.4.1",
        "torchvision==0.19.1",
        index_url="https://download.pytorch.org/whl/cu124",
        gpu="A10G",
    )
    .pip_install("git+https://github.com/openai/CLIP.git", gpu="A10G")
    .pip_install("git+https://github.com/facebookresearch/sam2.git", gpu="A10G")
)



@app.function(
    image=image,
    gpu="A10G",
    volumes={volume_path: volume},
)
def sam2(
    image_pil: Image.Image,
    boxes: list[np.ndarray]
) -> list[dict]:
    import torch
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    predictor = SAM2ImagePredictor.from_pretrained("facebook/sam2-hiera-large")

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        predictor.set_image(image_pil)
        masks, scores, _ = predictor.predict(
            point_coords=None,
            point_labels=None,
            box=boxes,
            multimask_output=False,
        )
    return masks, scores

@app.function(
    image=image,
    gpu="A10G",
    volumes={volume_path: volume},
)
def clip(
    image_pil: Image.Image,
    prompts: list[str],
) -> list[dict]:
    from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
    import torch

    processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
    model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")

    # Get original image dimensions
    orig_width, orig_height = image_pil.size

    inputs = processor(
        text=prompts,
        images=[image_pil] * len(prompts),
        padding="max_length",
        return_tensors="pt",
    )
    # predict
    with torch.no_grad():
        outputs = model(**inputs)
    preds = outputs.logits.unsqueeze(1)

    # Get the dimensions of the prediction output
    pred_height, pred_width = preds.shape[-2:]

    # Calculate scaling factors
    width_scale = orig_width / pred_width
    height_scale = orig_height / pred_height

    results = []

    # Process each prediction to find bounding boxes in high probability regions
    for i, prompt in enumerate(prompts):
        # Apply sigmoid to get probability map
        pred_tensor = torch.sigmoid(preds[i][0])
        # Convert tensor to numpy array
        pred_np = pred_tensor.cpu().numpy()

        # Convert to uint8 for OpenCV processing
        heatmap = (pred_np * 255).astype(np.uint8)

        # Apply threshold to find high probability regions
        _, binary = cv2.threshold(heatmap, 127, 255, cv2.THRESH_BINARY)

        # Find contours in thresholded image
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # Process each contour to get bounding boxes
        for contour in contours:
            # Skip very small contours that might be noise
            if cv2.contourArea(contour) < 100:  # Minimum area threshold
                continue

            # Get bounding box coordinates in prediction space
            x, y, w, h = cv2.boundingRect(contour)

            # Scale coordinates to original image dimensions
            x_orig = int(x * width_scale)
            y_orig = int(y * height_scale)
            w_orig = int(w * width_scale)
            h_orig = int(h * height_scale)

            # Calculate confidence score based on average probability in the region
            mask = np.zeros_like(pred_np)
            cv2.drawContours(mask, [contour], 0, 1, -1)
            confidence = float(np.mean(pred_np[mask == 1]))

            results.append(
                {
                    "label": prompt,
                    "score": confidence,
                    "box": {
                        "xmin": x_orig,
                        "ymin": y_orig,
                        "xmax": x_orig + w_orig,
                        "ymax": y_orig + h_orig,
                    },
                }
            )

    results = sorted(results, key=lambda x: x["score"], reverse=True)
    return results