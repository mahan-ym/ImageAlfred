from pathlib import Path

import modal
import numpy as np
from PIL import Image

from utils import upload_image_to_tmpfiles

modal_app_name = "ImageAlfred"


def preserve_privacy(input_prompt, input_img):
    """
    Obscure specified objects in the input image based on the input prompt.

    Args:
        input_prompt (list): List of [object:str].
        input_img (bytes): Input image in bytes format.

    Returns:
        bytes: Binary image data of the modified image.

    example:
        input_prompt = ["face", "license plate"]
    """

    return input_img


def change_color_objects_hsv(
    user_input,
    input_img,
) -> np.ndarray | Image.Image | str | Path | None:
    """Changes the hue and saturation of specified objects in an image.

    Segments objects based on text prompts and alters their color in the HSV
    color space. The HSV color space uses OpenCV ranges: H (0-179), S (0-255),
    V (0-255). Common color examples include Green (hue=60), Red (hue=0),
    Blue (hue=120), Yellow (hue=30), and Purple (hue=150), all with
    saturation=255.

    Args:
        user_input (list[list[str | int | float]]): A list of lists where each inner list contains three elements: target object name (str), hue value (int, 0-179), and saturation scale factor (float, >0). Example: [["hair", 30, 1.2], ["shirt", 60, 1.0]].
        input_img (np.ndarray | Image.Image | str | None): Input image as Base64-encoded string or URL string. Cannot be None.

    Returns:
        Base64-encoded string.

    Raises:
        ValueError: If user_input format is invalid, hue values are outside [0, 179] range, saturation_scale is not positive, or image format is invalid or corrupted.
        TypeError: If input_img is not a supported type or modal function returns unexpected type.
    """  # noqa: E501
    print("Received input image type:", type(input_img))
    # source, input_img = validate_image_input(input_img)
    print("before processing input:", user_input)

    for item in user_input:
        if len(item) != 3:
            raise ValueError(
                "Each item in user_input must be a list of [object, hue, saturation_scale]"  # noqa: E501
            )
        if not isinstance(item[0], str):
            item[0] = str(item[0])
        if not isinstance(item[1], (int, float)):
            item[1] = float(item[1])
            if item[1] < 0 or item[1] > 179:
                raise ValueError("Hue must be in the range [0, 179]")
        if not isinstance(item[2], (int, float)):
            item[2] = float(item[2])
            if item[2] <= 0:
                raise ValueError("Saturation scale must be greater than 0")

    print("after processing input:", user_input)

    func = modal.Function.from_name("ImageAlfred", "change_image_objects_hsv")
    output_pil = func.remote(image_pil=input_img, targets_config=user_input)

    if output_pil is None:
        raise ValueError("Received None from modal remote function.")
    if not isinstance(output_pil, Image.Image):
        raise TypeError(
            f"Expected Image.Image from modal remote function, got {type(output_pil)}"
        )
    img_link = upload_image_to_tmpfiles(output_pil)

    return output_pil


def change_color_objects_lab(user_input, input_img):
    """Changes the color of specified objects in an image using LAB color space.

    Segments objects based on text prompts and alters their color in the LAB
    color space. The LAB color space uses OpenCV ranges: L (0-255, lightness),
    A (0-255, green-red, 128 is neutral), B (0-255, blue-yellow, 128 is neutral).
    Common color examples include Green (a=80, b=128), Red (a=180, b=160),
    Blue (a=128, b=80), Yellow (a=120, b=180), and Purple (a=180, b=100).

    Args:
        user_input (list[list[str | int | float]]): A list of lists where each inner list contains three elements: target object name (str), new_a value (int, 0-255), and new_b value (int, 0-255). Example: [["hair", 80, 128], ["shirt", 180, 160]].
        input_img (np.ndarray | Image.Image | str | bytes | None): Input image as Base64-encoded string or URL string. Cannot be None.

    Returns:
        Base64-encoded string

    Raises:
        ValueError: If user_input format is invalid, a/b values are outside [0, 255] range, or image format is invalid or corrupted.
        TypeError: If input_img is not a supported type or modal function returns unexpected type.
    """  # noqa: E501
    print("Received input image type:", type(input_img))
    print("before processing input:", user_input)
    for item in user_input:
        if len(item) != 3:
            raise ValueError(
                "Each item in user_input must be a list of [object, new_a, new_b]"
            )
        if not isinstance(item[0], str):
            item[0] = str(item[0])
        if not isinstance(item[1], int):
            item[1] = int(item[1])
            if item[1] < 0 or item[1] > 255:
                raise ValueError("new A must be in the range [0, 255]")
        if not isinstance(item[2], int):
            item[2] = int(item[2])
            if item[2] < 0 or item[2] > 255:
                raise ValueError("new B must be in the range [0, 255]")

    print("after processing input:", user_input)
    func = modal.Function.from_name("ImageAlfred", "change_image_objects_lab")
    output_pil = func.remote(image_pil=input_img, targets_config=user_input)
    if output_pil is None:
        raise ValueError("Received None from modal remote function.")
    if not isinstance(output_pil, Image.Image):
        raise TypeError(
            f"Expected Image.Image from modal remote function, got {type(output_pil)}"
        )
    img_link = upload_image_to_tmpfiles(output_pil)

    return output_pil


if __name__ == "__main__":
    change_color_objects_hsv(
        user_input=[["hair", 30, 1.2], ["shirt", 60, 1.0]], input_img=b""
    )
