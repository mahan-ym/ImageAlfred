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
    .apt_install("git")
    .pip_install(
        "huggingface-hub",
        "hf_transfer",
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
    .pip_install(
        "git+https://github.com/luca-medeiros/lang-segment-anything.git",
        gpu="A10G",
    )
)


@app.function(
    gpu="A10G",
    image=image,
    volumes={volume_path: volume},
)
def lang_sam_segment(
    image_pil: Image.Image,
    prompt: str,  # type: ignore
) -> list:
    """Segments an image using LangSAM based on a text prompt.
    This function uses LangSAM to segment objects in the image based on the provided prompt.
    """  # noqa: E501
    from lang_sam import LangSAM  # type: ignore

    os.environ["TORCH_HOME"] = TORCH_HOME
    os.environ["HF_HOME"] = HF_HOME
    os.makedirs(HF_HOME, exist_ok=True)
    os.makedirs(TORCH_HOME, exist_ok=True)

    model = LangSAM(sam_type="sam2.1_hiera_large")
    langsam_results = model.predict(
        images_pil=[image_pil],
        texts_prompt=[prompt],
    )

    return langsam_results


@app.function(
    gpu="T4",
    image=image,
    volumes={volume_path: volume},
)
def change_image_objects_hsv(
    image_bytes: bytes,
    targets_config: list[list[str | int | float]],
) -> bytes:
    """Changes the hue and saturation of specified objects in an image.
    This function uses LangSAM to segment objects in the image based on provided prompts,
    and then modifies the hue and saturation of those objects in the HSV color space.

    Parameters
    ----------
    image_bytes : bytes
        image data in bytes format
    targets_config : list[list[str  |  int  |  float]]
        list of lists, where each inner list contains:
        - target object name (str)
        - hue value (int or float): openCV HSV range: 0-179, where 0 is red, 30 is yellow, 60 is green, 120 is cyan, 179 is blue
        - saturation scale (float): 1 means no change, <1 reduces saturation, >1 increases saturation

    Returns
    -------
    bytes
        modified image data in bytes format

    Example
    -------
    >>> targets_config = [
    ...     ["hair", 30, 1.2],
    ...     ["tshirt", 60, 1.0],
    ...     ["pants", 90, 0.8],
    ... ]
    >>> change_image_objects_hsv(image_bytes, targets_config)
    """  # noqa: E501
    if not isinstance(targets_config, list) or not all(
        (
            isinstance(target, list)
            and len(target) == 3
            and isinstance(target[0], str)
            and isinstance(target[1], (int, float))
            and isinstance(target[2], (int, float))
            and 0 <= target[1] <= 179
            and target[2] >= 0
        )
        for target in targets_config
    ):
        raise ValueError(
            "targets_config must be a list of lists, each containing [target_name, hue, saturation_scale]."  # noqa: E501
        )

    image_pil = Image.open(BytesIO(image_bytes)).convert("RGB")

    prompts = ". ".join(target[0] for target in targets_config)

    os.environ["TORCH_HOME"] = TORCH_HOME
    os.environ["HF_HOME"] = HF_HOME
    os.makedirs(HF_HOME, exist_ok=True)
    os.makedirs(TORCH_HOME, exist_ok=True)

    langsam_results = lang_sam_segment.remote(image_pil=image_pil, prompt=prompts)
    labels = langsam_results[0]["labels"]
    scores = langsam_results[0]["scores"]

    img_array = np.array(image_pil)
    img_hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV).astype(np.float32)

    for target_spec in targets_config:
        target_obj = target_spec[0]
        hue = target_spec[1]
        saturation_scale = target_spec[2]

        try:
            mask_idx = labels.index(target_obj)
        except ValueError:
            print(
                f"Warning: Label '{target_obj}' not found in the image. Skipping this target."  # noqa: E501
            )
            continue

        mask = langsam_results[0]["masks"][mask_idx]
        mask_bool = mask.astype(bool)

        img_hsv[mask_bool, 0] = float(hue)
        img_hsv[mask_bool, 1] = np.minimum(
            img_hsv[mask_bool, 1] * saturation_scale,
            255.0,
        )

    output_img = cv2.cvtColor(img_hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
    output_img_pil = Image.fromarray(output_img)
    output_buffer = BytesIO()
    output_img_pil.save(output_buffer, format="PNG")
    return output_buffer.getvalue()


@app.function(
    gpu="T4",
    image=image,
    volumes={volume_path: volume},
)
def change_image_objects_lab(
    image_bytes: bytes,
    targets_config: list[list[str | int | float]],
) -> bytes:
    """Changes the color of specified objects in an image.
    This function uses LangSAM to segment objects in the image based on provided prompts,
    and then modifies the color of those objects in the LAB color space.

    Define new color in LAB space (OpenCV LAB ranges):
        - L: 0-255 (lightness)
        - A: 0-255 (green-red, 128 is neutral)
        - B: 0-255 (blue-yellow, 128 is neutral)
        - Color examples:
        - Green: a=80, b=128
        - Red: a=180, b=160
        - Blue: a=128, b=80
        - Yellow: a=120, b=180
        - Purple: a=180, b=100

    Parameters
    ----------
    image_bytes : bytes
        binary image data
    targets_config : list[list[str  |  int  |  float]]
        list of lists, where each inner list contains:
        - target object name (str)
        - new_a (int): 0-255, green-red channel in LAB color space
        - new_b (int): 0-255, blue-yellow channel in LAB color space


    Returns
    -------
    bytes
        binary image data of the modified image

    Example
    -------
    >>> targets_config = [
    ...     ["hair", 80, 128],
    ...     ["shirt", 180, 160],
    ...     ["pants", 120, 180],
    ... ]
    >>> change_image_objects_lab(image_bytes, targets_config)
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

    image_pil = Image.open(BytesIO(image_bytes)).convert("RGB")
    prompts = ". ".join(target[0] for target in targets_config)

    os.environ["TORCH_HOME"] = TORCH_HOME
    os.environ["HF_HOME"] = HF_HOME
    os.makedirs(HF_HOME, exist_ok=True)
    os.makedirs(TORCH_HOME, exist_ok=True)

    langsam_results = lang_sam_segment.remote(
        image_pil=image_pil,
        prompt=prompts,
    )
    labels = langsam_results[0]["labels"]
    scores = langsam_results[0]["scores"]
    img_array = np.array(image_pil)
    img_lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2Lab).astype(np.float32)
    for target_spec in targets_config:
        target_obj = target_spec[0]
        new_a = target_spec[1]
        new_b = target_spec[2]

        try:
            mask_idx = labels.index(target_obj)
        except ValueError:
            print(
                f"Warning: Label '{target_obj}' not found in the image. Skipping this target."  # noqa: E501
            )
            continue

        mask = langsam_results[0]["masks"][mask_idx]
        mask_bool = mask.astype(bool)

        img_lab[mask_bool, 1] = new_a
        img_lab[mask_bool, 2] = new_b

    output_img = cv2.cvtColor(img_lab.astype(np.uint8), cv2.COLOR_Lab2RGB)
    output_img_pil = Image.fromarray(output_img)
    output_buffer = BytesIO()
    output_img_pil.save(output_buffer, format="PNG")
    return output_buffer.getvalue()


@app.function(
    gpu="T4",
    image=image,
)
def apply_mosaic_with_bool_mask(image, mask, intensity: int = 20):
    h, w = image.shape[:2]
    block_size = max(1, min(intensity, min(h, w)))

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
)
def preserve_privacy_test(image_bytes: bytes, prompt: str) -> bytes:
    image_pil = Image.open(BytesIO(image_bytes)).convert("RGB")

    langsam_results = lang_sam_segment.remote(
        image_pil=image_pil,
        prompt=prompt,
    )

    img_array = np.array(image_pil)

    for result in langsam_results:
        print(f"Found {len(result['masks'])} masks for label: {result['labels']}")
        if len(result["masks"]) == 0:
            print("No masks found for the given prompt.")
            return image_bytes
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
            print(f"Processing mask {i + 1}/{len(result['masks'])}")
            print(f"Mask score: {mask_score}")
            box = result["boxes"][i]
            cv2.rectangle(
                img_array,
                (int(box[0]), int(box[1])),
                (int(box[2]), int(box[3])),
                (255, 0, 0),  # Blue color in BGR
                2,  # Thickness of the rectangle
            )
            label = result["labels"][i]
            cv2.putText(
                img_array,
                label,
                (int(box[0]), int(box[1] - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,  # Font size
                (255, 0, 0),  # Blue color in BGR
                2,  # Thickness of the text
            )
            mask_bool = mask.astype(bool)

            img_array = apply_mosaic_with_bool_mask.remote(img_array, mask_bool)

    modified_image = Image.fromarray(img_array)
    output_buffer = BytesIO()
    modified_image.save(output_buffer, format="JPEG")
    return output_buffer.getvalue()


if __name__ == "__main__":
    input_dir = "./src/assets/input"
    output_dir = "./src/assets/output"
    img_name = "test_1.jpg"
    with open(f"{input_dir}/{img_name}", "rb") as f:
        img_bytes = f.read()

    with modal.enable_output():
        with app.run():
            result_bytes = change_image_objects_hsv.remote(
                img_bytes,
                [["hair", 30, 1.2], ["tshirt", 60, 1.0], ["pants", 90, 0.8]],
            )
            os.makedirs(output_dir, exist_ok=True)
            with open(f"{output_dir}/colored_hsv_{img_name}", "wb") as f:
                f.write(result_bytes)
