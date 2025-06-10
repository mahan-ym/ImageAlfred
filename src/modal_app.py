import os
from io import BytesIO

import cv2
import modal
import numpy as np
from PIL import Image
from rapidfuzz import process

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
    .apt_install("git")
    .pip_install(
        "huggingface-hub",
        "hf_transfer",
        "Pillow",
        "numpy",
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
    .pip_install(
        "git+https://github.com/PramaLLC/BEN2.git#egg=ben2",
        gpu="A10G",
    )
)


@app.function(
    gpu="A10G",
    image=image,
    volumes={volume_path: volume},
    # min_containers=1,
    timeout=60 * 3,
)
def lang_sam_segment(
    image_pil: Image.Image,
    prompt: str,
    box_threshold=0.3,
    text_threshold=0.25,
) -> list:
    """Segments an image using LangSAM based on a text prompt.
    This function uses LangSAM to segment objects in the image based on the provided prompt.
    """  # noqa: E501
    from lang_sam import LangSAM  # type: ignore

    model = LangSAM(sam_type="sam2.1_hiera_large")
    langsam_results = model.predict(
        images_pil=[image_pil],
        texts_prompt=[prompt],
        box_threshold=box_threshold,
        text_threshold=text_threshold,
    )
    if len(langsam_results[0]["labels"]) == 0:
        print("No masks found for the given prompt.")
        return None

    print(f"found {len(langsam_results[0]['labels'])} masks for prompt: {prompt}")
    print("labels:", langsam_results[0]["labels"])
    print("scores:", langsam_results[0]["scores"])
    print(
        "masks scores:",
        langsam_results[0].get("mask_scores", "No mask scores available"),
    )  # noqa: E501

    return langsam_results


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
    prompts = ". ".join(target[0] for target in targets_config)

    langsam_results = lang_sam_segment.remote(image_pil=image_pil, prompt=prompts)
    if not langsam_results:
        return image_pil
    input_labels = [target[0] for target in targets_config]
    output_labels = langsam_results[0]["labels"]
    scores = langsam_results[0]["scores"]

    img_array = np.array(image_pil)
    img_hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV).astype(np.float32)

    for idx, label in enumerate(output_labels):
        if not label or label == "":
            print("Skipping empty label.")
            continue
        input_label, score, _ = process.extractOne(label, input_labels)
        input_label_idx = input_labels.index(input_label)

        target_rgb = targets_config[input_label_idx][1:]
        target_hsv = cv2.cvtColor(np.uint8([[target_rgb]]), cv2.COLOR_RGB2HSV)[0][0]

        mask = langsam_results[0]["masks"][idx].astype(bool)
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

    prompts = ". ".join(target[0] for target in targets_config)

    langsam_results = lang_sam_segment.remote(
        image_pil=image_pil,
        prompt=prompts,
    )
    if not langsam_results:
        return image_pil

    input_labels = [target[0] for target in targets_config]
    output_labels = langsam_results[0]["labels"]
    scores = langsam_results[0]["scores"]

    img_array = np.array(image_pil)
    img_lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2Lab).astype(np.float32)

    for idx, label in enumerate(output_labels):
        if not label or label == "":
            print("Skipping empty label.")
            continue
        input_label, score, _ = process.extractOne(label, input_labels)
        input_label_idx = input_labels.index(input_label)

        new_a = targets_config[input_label_idx][1]
        new_b = targets_config[input_label_idx][2]

        mask = langsam_results[0]["masks"][idx]
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
    prompt: str,
    privacy_strength: int = 15,
) -> Image.Image:
    """
    Preserves privacy in an image by applying a mosaic effect to specified objects.
    """
    print(f"Preserving privacy for prompt: {prompt} with strength {privacy_strength}")

    langsam_results = lang_sam_segment.remote(
        image_pil=image_pil,
        prompt=prompt,
        box_threshold=0.35,
        text_threshold=0.40,
    )
    if not langsam_results:
        return image_pil

    img_array = np.array(image_pil)

    for result in langsam_results:
        print(f"result: {result}")

        for i, mask in enumerate(result["masks"]):
            if "mask_scores" in result:
                if (
                    hasattr(result["mask_scores"], "shape")
                    and result["mask_scores"].ndim > 0
                ):
                    mask_score = result["mask_scores"][i]
                else:
                    mask_score = result["mask_scores"]
            if mask_score < 0.6:
                print(f"Skipping mask {i + 1}/{len(result['masks'])} -> low score.")
                continue
            print(
                f"Processing mask {i + 1}/{len(result['masks'])} Mask score: {mask_score}"  # noqa: E501
            )

            mask_bool = mask.astype(bool)

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
