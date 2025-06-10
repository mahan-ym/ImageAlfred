import os
from io import BytesIO

import cv2
import modal
import numpy as np
from PIL import Image

app = modal.App("ImageAlfred")

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
    .pip_install(
        "git+https://github.com/PramaLLC/BEN2.git#egg=ben2",
        gpu="A10G",
    )
)


@app.function(
    image=image,
    gpu="A10G",
    volumes={volume_path: volume},
    timeout=60 * 3,
)
def prompt_segment(
    image_pil: Image.Image,
    prompts: list[str],
) -> list[dict]:
    clip_results = clip.remote(image_pil, prompts)

    if not clip_results:
        print("No boxes returned from CLIP.")
        return None

    boxes = np.array(clip_results["boxes"])

    sam_result_masks, sam_result_scores = sam2.remote(image_pil=image_pil, boxes=boxes)

    print(f"sam_result_mask {sam_result_masks}")

    if not sam_result_masks.any():
        print("No masks or scores returned from SAM2.")
        return None

    if sam_result_masks.ndim == 3:
        # If the masks are in 3D, we need to convert them to 4D
        sam_result_masks = [sam_result_masks]

    results = {
        "labels": clip_results["labels"],
        "boxes": boxes,
        "clip_scores": clip_results["scores"],
        "sam_masking_scores": sam_result_scores,
        "masks": sam_result_masks,
    }
    return results


@app.function(
    image=image,
    gpu="A10G",
    volumes={volume_path: volume},
    timeout=60 * 3,
)
def sam2(image_pil: Image.Image, boxes: list[np.ndarray]) -> list[dict]:
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
    timeout=60 * 3,
)
def clip(
    image_pil: Image.Image,
    prompts: list[str],
) -> list[dict]:
    """
    returns:
        dict with keys each are lists:
            - labels: str, the prompt used for the prediction
            - scores: float, confidence score of the prediction
            - boxes: np.array representing bounding box coordinates
    """

    from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
    import torch

    processor = CLIPSegProcessor.from_pretrained(
        "CIDAS/clipseg-rd64-refined",
        use_fast=True,
    )
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

    labels = []
    scores = []
    boxes = []

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

            labels.append(prompt)
            scores.append(confidence)
            boxes.append(
                np.array(
                    [
                        x_orig,
                        y_orig,
                        x_orig + w_orig,
                        y_orig + h_orig,
                    ]
                )
            )

    if labels == []:
        return None

    results = {
        "labels": labels,
        "scores": scores,
        "boxes": boxes,
    }
    return results


@app.function(
    gpu="T4",
    image=image,
    volumes={volume_path: volume},
    timeout=60 * 3,
)
def change_image_objects_hsv(
    image_pil: Image.Image,
    targets_config: list[list[str | int | float]],
) -> Image.Image:
    if not isinstance(targets_config, list) or not all(
        (
            isinstance(target, list)
            and len(target) == 4
            and isinstance(target[0], str)
            and isinstance(target[1], (int))
            and isinstance(target[2], (int))
            and isinstance(target[3], (int))
            and target[1] >= 0
            and target[1] <= 255
            and target[2] >= 0
            and target[2] <= 255
            and target[3] >= 0
            and target[3] <= 255
        )
        for target in targets_config
    ):
        raise ValueError(
            "targets_config must be a list of lists, each containing [target_name, hue, saturation_scale]."  # noqa: E501
        )
    print("Change image objects hsv targets config:", targets_config)
    prompts = [target[0].strip() for target in targets_config]

    prompt_segment_results = prompt_segment.remote(
        image_pil=image_pil,
        prompts=prompts,
    )
    if not prompt_segment_results:
        return image_pil

    output_labels = prompt_segment_results["labels"]

    img_array = np.array(image_pil)
    img_hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV).astype(np.float32)

    for idx, label in enumerate(output_labels):
        if not label or label == "":
            print("Skipping empty label.")
            continue
        if label not in prompts:
            print(f"Label '{label}' not found in prompts. Skipping.")
            continue
        input_label_idx = prompts.index(label)
        target_rgb = targets_config[input_label_idx][1:]
        target_hsv = cv2.cvtColor(np.uint8([[target_rgb]]), cv2.COLOR_RGB2HSV)[0][0]

        mask = prompt_segment_results["masks"][idx][0].astype(bool)
        h, s, v = cv2.split(img_hsv)
        # Convert all channels to float32 for consistent processing
        h = h.astype(np.float32)
        s = s.astype(np.float32)
        v = v.astype(np.float32)

        # Compute original S and V means inside the mask
        mean_s = np.mean(s[mask])
        mean_v = np.mean(v[mask])

        # Target S and V
        target_hue, target_s, target_v = target_hsv

        # Compute scaling factors (avoid div by zero)
        scale_s = target_s / mean_s if mean_s > 0 else 1.0
        scale_v = target_v / mean_v if mean_v > 0 else 1.0

        scale_s = np.clip(scale_s, 0.8, 1.2)
        scale_v = np.clip(scale_v, 0.8, 1.2)

        # Apply changes only in mask
        h[mask] = target_hue
        s = s.astype(np.float32)
        v = v.astype(np.float32)
        s[mask] = np.clip(s[mask] * scale_s, 0, 255)
        v[mask] = np.clip(v[mask] * scale_v, 0, 255)

        # Merge and convert back
        img_hsv = cv2.merge(
            [
                h.astype(np.uint8),
                s.astype(np.uint8),
                v.astype(np.uint8),
            ]
        )

    output_img = cv2.cvtColor(img_hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
    output_img_pil = Image.fromarray(output_img)
    return output_img_pil


@app.function(
    gpu="T4",
    image=image,
    volumes={volume_path: volume},
    timeout=60 * 3,
)
def change_image_objects_lab(
    image_pil: Image.Image,
    targets_config: list[list[str | int | float]],
) -> Image.Image:
    """Changes the color of specified objects in an image.
    This function uses LangSAM to segment objects in the image based on provided prompts,
    and then modifies the color of those objects in the LAB color space.
    """  # noqa: E501
    if not isinstance(targets_config, list) or not all(
        (
            isinstance(target, list)
            and len(target) == 3
            and isinstance(target[0], str)
            and isinstance(target[1], int)
            and isinstance(target[2], int)
            and 0 <= target[1] <= 255
            and 0 <= target[2] <= 255
        )
        for target in targets_config
    ):
        raise ValueError(
            "targets_config must be a list of lists, each containing [target_name, new_a, new_b]."  # noqa: E501
        )

    print("change image objects lab targets config:", targets_config)

    prompts = [target[0].strip() for target in targets_config]

    prompt_segment_results = prompt_segment.remote(
        image_pil=image_pil,
        prompts=prompts,
    )
    if not prompt_segment_results:
        return image_pil

    output_labels = prompt_segment_results["labels"]

    img_array = np.array(image_pil)
    img_lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2Lab).astype(np.float32)

    for idx, label in enumerate(output_labels):
        if not label or label == "":
            print("Skipping empty label.")
            continue

        if label not in prompts:
            print(f"Label '{label}' not found in prompts. Skipping.")
            continue

        input_label_idx = prompts.index(label)

        new_a = targets_config[input_label_idx][1]
        new_b = targets_config[input_label_idx][2]

        mask = prompt_segment_results["masks"][idx][0]
        mask_bool = mask.astype(bool)

        img_lab[mask_bool, 1] = new_a
        img_lab[mask_bool, 2] = new_b

    output_img = cv2.cvtColor(img_lab.astype(np.uint8), cv2.COLOR_Lab2RGB)
    output_img_pil = Image.fromarray(output_img)

    return output_img_pil


@app.function(
    gpu="T4",
    image=image,
    volumes={volume_path: volume},
    timeout=60 * 3,
)
def apply_mosaic_with_bool_mask(
    image: np.ndarray,
    mask: np.ndarray,
    privacy_strength: int,
) -> np.ndarray:
    h, w = image.shape[:2]
    image_size_factor = min(h, w) / 1000
    block_size = int(max(1, (privacy_strength * image_size_factor)))

    # Ensure block_size is at least 1 and doesn't exceed half of image dimensions
    block_size = max(1, min(block_size, min(h, w) // 2))

    small = cv2.resize(
        image, (w // block_size, h // block_size), interpolation=cv2.INTER_LINEAR
    )
    mosaic = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)

    result = image.copy()
    result[mask] = mosaic[mask]
    return result


@app.function(
    gpu="T4",
    image=image,
    volumes={volume_path: volume},
    timeout=60 * 3,
)
def preserve_privacy(
    image_pil: Image.Image,
    prompts: str,
    privacy_strength: int = 15,
) -> Image.Image:
    """
    Preserves privacy in an image by applying a mosaic effect to specified objects.
    """
    print(f"Preserving privacy for prompt: {prompts} with strength {privacy_strength}")
    if isinstance(prompts, str):
        prompts = [prompt.strip() for prompt in prompts.split(".")]
        print(f"Parsed prompts: {prompts}")
    prompt_segment_results = prompt_segment.remote(
        image_pil=image_pil,
        prompts=prompts,
    )
    if not prompt_segment_results:
        return image_pil

    img_array = np.array(image_pil)

    for i, mask in enumerate(prompt_segment_results["masks"]):
        mask_bool = mask[0].astype(bool)

        # Create kernel for morphological operations
        kernel_size = 100
        kernel = np.ones((kernel_size, kernel_size), np.uint8)

        # Convert bool mask to uint8 for OpenCV operations
        mask_uint8 = mask_bool.astype(np.uint8) * 255

        # Apply dilation to slightly expand the mask area
        mask_uint8 = cv2.dilate(mask_uint8, kernel, iterations=2)
        # Optional: Apply erosion again to refine the mask
        mask_uint8 = cv2.erode(mask_uint8, kernel, iterations=2)

        # Convert back to boolean mask
        mask_bool = mask_uint8 > 127

        img_array = apply_mosaic_with_bool_mask.remote(
            img_array, mask_bool, privacy_strength
        )

    output_image_pil = Image.fromarray(img_array)

    return output_image_pil


@app.function(
    gpu="A10G",
    image=image,
    volumes={volume_path: volume},
    timeout=60 * 2,
)
def remove_background(image_pil: Image.Image) -> Image.Image:
    import torch  # type: ignore
    from ben2 import BEN_Base  # type: ignore

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print("type of image_pil:", type(image_pil))
    model = BEN_Base.from_pretrained("PramaLLC/BEN2")
    model.to(device).eval()  # todo check if this should be outside the function

    output_image = model.inference(
        image_pil,
        refine_foreground=True,
    )
    print(f"output type: {type(output_image)}")
    return output_image
