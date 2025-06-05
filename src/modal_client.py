import os
from io import BytesIO

import modal
import numpy as np
from PIL import Image

app = modal.App("ImageAlfread")

cuda_version = "12.4.0"
flavor = "devel"
operating_sys = "ubuntu22.04"
tag = f"{cuda_version}-{flavor}-{operating_sys}"

image = (
    modal.Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.12")
    .apt_install("git")
    .pip_install(
        "opencv-contrib-python",
        "huggingface-hub",
        "Pillow",
        "numpy",
        # "opencv-contrib-python-headless",
    )
    .pip_install(
        "torch==2.4.1",
        "torchvision==0.19.1",
        index_url="https://download.pytorch.org/whl/cu124",
    )
    .pip_install("git+https://github.com/luca-medeiros/lang-segment-anything.git")
)

@app.function(
    gpu="T4",
    image=image,
)
def change_image_object_hsv(
    image_bytes: bytes,
    prompt: str,
    new_hue: int,
    saturation_scale: float = 1.2,
) -> bytes:
    """
    Changes the hue of an object in an image using a mask.

        Parameters:
            image_bytes (bytes): binary image data
            prompt (str): semantic prompt for the object, e.g., "hair.", "shirt."
            new_hue (int): 0-179, OpenCV hue range
            saturation_scale (float): 1.0 means no change, <1.0 desaturates, >1.0 saturates

        Returns:
            bytes (binary image data of the modified image)
    """  # noqa: E501
    from io import BytesIO

    import cv2  # type: ignore
    from lang_sam import LangSAM  # type: ignore
    from PIL import Image

    image = Image.open(BytesIO(image_bytes)).convert("RGB")

    model = LangSAM(sam_type="sam2.1_hiera_large")
    results = model.predict(
        images_pil=[image],
        texts_prompt=[prompt],
        # box_threshold=0.3,
        # text_threshold=0.25,
    )

    # Create class_id for each unique label
    # unique_labels = list(set(results[0]["labels"]))
    # class_id_map = {label: idx for idx, label in enumerate(unique_labels)}
    # class_id = [class_id_map[label] for label in results[0]["labels"]]

    img_array = np.array(image)
    img_hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV).astype(np.float32)

    mask = results[0]["masks"][0]
    mask_bool = mask.astype(bool)

    img_hsv[mask_bool, 0] = new_hue
    img_hsv[mask_bool, 1] = np.minimum(img_hsv[mask_bool, 1] * saturation_scale, 255)

    img_hsv_result = cv2.cvtColor(img_hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
    img_hsv_colored = Image.fromarray(img_hsv_result)

    output_buffer = BytesIO()
    img_hsv_colored.save(output_buffer, format="PNG")
    return output_buffer.getvalue()


# need to check 
@app.function(
    gpu="T4",
    image=image,
)
def change_image_objects_hsv(
    image_bytes: bytes,
    objects_config: dict[str, dict[str, float]],
    default_saturation_scale: float = 1.2,
) -> bytes:
    """
        Changes the hue of multiple objects in an image using masks.

            Parameters:
                image_bytes (bytes): binary image data
                objects_config (dict): Dictionary mapping object prompts to their parameters
                                      Format: {"object prompt": {"hue": value, "saturation_scale": value}}
                default_saturation_scale (float): Default saturation scale if not specified per object

            Returns:
                bytes (binary image data of the modified image)
    """
    from io import BytesIO

    import cv2
    from lang_sam import LangSAM  # type: ignore
    from PIL import Image

    image = Image.open(BytesIO(image_bytes)).convert("RGB")

    model = LangSAM(sam_type="sam2.1_hiera_large")
    results = model.predict(
        images_pil=[image],
        texts_prompt=list(objects_config.keys()),
        # box_threshold=0.3,
        # text_threshold=0.25,
    )

    img_array = np.array(image)

    output_images = []

    for idx, (prompt, config) in enumerate(objects_config.items()):
        hue = config.get("hue", 0)
        saturation_scale = config.get("saturation_scale", default_saturation_scale)

        img_hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV).astype(np.float32)

        mask = results[idx]["masks"][0]
        mask_bool = mask.astype(bool)

        img_hsv[mask_bool, 0] = hue
        img_hsv[mask_bool, 1] = np.minimum(
            img_hsv[mask_bool, 1] * saturation_scale, 255
        )

        img_hsv_result = cv2.cvtColor(img_hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
        img_hsv_colored = Image.fromarray(img_hsv_result)
        output_images.append(np.array(img_hsv_colored))

    # Combine images: Assuming all images are the same size
    combined_image = np.sum(output_images, axis=0).astype(np.uint8)

    output_image_pil = Image.fromarray(combined_image)
    output_buffer = BytesIO()
    output_image_pil.save(output_buffer, format="PNG")
    return output_buffer.getvalue()


@app.function(
    gpu="T4",
    image=image,
)
def change_image_object_lab(
    image_bytes: bytes,
    prompt: str,
    new_a: int,
    new_b: int,
) -> bytes:
    """
    Changes the color of an object in an image using a mask.

        Parameters:
            image_bytes (bytes): binary image data
            prompt (str): semantic prompt for the object, e.g., "hair.", "shirt."
            new_a (int): 0-255, green-red channel in LAB color space
            new_b (int): 0-255, blue-yellow channel in LAB color space

        Returns:
            image (bytes): binary image data of the modified image

        Define new color in LAB space (OpenCV LAB ranges)
        L: 0-255 (lightness)
        A: 0-255 (green-red, 128 is neutral)
        B: 0-255 (blue-yellow, 128 is neutral)
        Color examples:
        Green: a=80, b=128
        Red: a=180, b=160
        Blue: a=128, b=80
        Yellow: a=120, b=180
        Purple: a=180, b=100
    """
    from io import BytesIO

    import cv2
    from lang_sam import LangSAM  # type: ignore
    from PIL import Image

    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    model = LangSAM(sam_type="sam2.1_hiera_large")
    results = model.predict(
        images_pil=[image],
        texts_prompt=[prompt],
        # box_threshold=0.3,
        # text_threshold=0.25,
    )

    img_array = np.array(image)
    img_lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2Lab).astype(np.float32)

    mask = results[0]["masks"][0]
    mask_bool = mask.astype(bool)

    img_lab[mask_bool, 1] = new_a
    img_lab[mask_bool, 2] = new_b

    img_lab_result = cv2.cvtColor(img_lab.astype(np.uint8), cv2.COLOR_Lab2RGB)
    img_lab_colored = Image.fromarray(img_lab_result)

    output_buffer = BytesIO()
    img_lab_colored.save(output_buffer, format="PNG")
    return output_buffer.getvalue()


# not sure about input and output types, need to check
@app.function(
    gpu="T4",
    image=image,
)
def change_image_objects_lab(
    image_bytes: bytes,
    objects_config: dict[str, dict[str, int]],
) -> bytes:
    """
    Changes the color of multiple objects in an image using masks.

        Parameters:
            image_bytes (bytes): binary image data
            objects_config (dict): Dictionary mapping object prompts to their parameters
                                  Format: {"object prompt": {"a": value, "b": value}}
        Returns:
            bytes (binary image data of the modified image)
    """
    from io import BytesIO

    import cv2
    from lang_sam import LangSAM
    from PIL import Image


# for running with modal CLI
@app.local_entrypoint()
def main():
    input_dir = "./src/assets/input"
    output_dir = "./src/assets/output"
    img_name = "test_1.jpg"
    with open(f"{input_dir}/{img_name}", "rb") as f:
        img_bytes = f.read()

    result_bytes = change_image_object_hsv.remote(
        img_bytes,
        "shirt.",
        120,
    )
    os.makedirs(output_dir, exist_ok=True)
    with open(f"{output_dir}/colored_hsv_{img_name}", "wb") as f:
        f.write(result_bytes)

    result_bytes = change_image_object_lab.remote(
        img_bytes,
        "shirt.",
        100,  # new_a
        150,  # new_b
    )
    with open(f"{output_dir}/colored_lab_{img_name}", "wb") as f:
        f.write(result_bytes)

    print("Image processing completed.")


if __name__ == "__main__":
    input_dir = "./src/assets/input"
    output_dir = "./src/assets/output"
    img_name = "test_1.jpg"
    with open(f"{input_dir}/{img_name}", "rb") as f:
        img_bytes = f.read()

    with modal.enable_output():
        with app.run():
            result_bytes = change_image_object_hsv.remote(
                img_bytes,
                "shirt.",
                60,
            )

            os.makedirs(output_dir, exist_ok=True)
            with open(f"{output_dir}/colored_hsv_{img_name}", "wb") as f:
                f.write(result_bytes)

        with app.run():
            result_bytes = change_image_object_lab.remote(
                img_bytes,
                "shirt.",
                100,  # new_a
                150,  # new_b
            )
            with open(f"{output_dir}/colored_lab_{img_name}", "wb") as f:
                f.write(result_bytes)

    print("Image processing completed.")
