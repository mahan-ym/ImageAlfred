import re
from pathlib import Path

import gradio as gr
import modal
import numpy as np
from PIL import Image

modal_app_name = "ImageAlfred"


def remove_background(
    input_img,
) -> np.ndarray | Image.Image | str | Path | None:
    """
    Remove the background of the image.

    Args:
        input_img: Input image or can be URL string of the image or base64 string. Cannot be None.
    Returns:
        bytes: Binary image data of the modified image.
    """  # noqa: E501
    if not input_img:
        raise gr.Error("Input image cannot be None or empty.")

    func = modal.Function.from_name(modal_app_name, "remove_background")
    output_pil = func.remote(
        image_pil=input_img,
    )

    if output_pil is None:
        raise gr.Error("Received None from server.")
    if not isinstance(output_pil, Image.Image):
        raise gr.Error(
            f"Expected Image.Image from server function, got {type(output_pil)}"
        )

    return output_pil


def privacy_preserve_image(
    input_img,
    input_prompt,
    privacy_strength: int = 15,
) -> np.ndarray | Image.Image | str | Path | None:
    """
    Obscures specified objects in the input image based on a natural language prompt, using a privacy-preserving blur or distortion effect.

    This function segments the image to detect objects described in the `input_prompt` and applies a pixelation effect to those regions. It is useful in scenarios where sensitive content (e.g., faces, license plates, logos,
    personal belongings) needs to be hidden before sharing or publishing images.

    Args:
        input_img: Input image or can be URL string of the image or base64 string. Cannot be None.
        input_prompt (str): Object to obscure in the image. It can be a single word or multiple words, e.g., "left person face", "license plate".
        privacy_strength (int): Strength of the privacy preservation effect. Higher values result in stronger blurring. Default is 15.
    Returns:
        bytes: Binary image data of the modified image.

    example:
        input_prompt = "faces, license plates, logos"
    """  # noqa: E501
    if not input_img:
        raise gr.Error("Input image cannot be None or empty.")
    if not input_prompt or input_prompt.strip() == "":
        raise gr.Error("Input prompt cannot be None or empty.")

    func = modal.Function.from_name(modal_app_name, "preserve_privacy")
    output_pil = func.remote(
        image_pil=input_img,
        prompts=input_prompt,
        privacy_strength=privacy_strength,
    )

    if output_pil is None:
        raise gr.Error("Received None from server.")
    if not isinstance(output_pil, Image.Image):
        raise gr.Error(
            f"Expected Image.Image from server function, got {type(output_pil)}"
        )

    return output_pil


def change_color_objects_hsv(
    input_img,
    user_input,
) -> np.ndarray | Image.Image | str | Path | None:
    """
    Changes the hue and saturation of specified objects in an image using the HSV color space.
    This function segments image regions based on a user-provided text prompt and applies
    color transformations in the HSV color space. HSV separates chromatic content (hue) from
    intensity (value), making it more intuitive for color manipulation tasks.
    Use this method when:
    - You want to change the color of objects based on their hue and saturation.
    - You want to apply color transformations that are less influenced by lighting conditions or brightness variations.

    Args:
        input_img: Input image or can be URL string of the image or base64 string. Cannot be None.
        user_input : A list of target specifications for color transformation. Each inner list must contain exactly four elements in the following order: 1. target_object (str) - A short, human-readable description of the object to be modified. Multi-word, descriptions are allowed for disambiguation (e.g., "right person shirt"), but they must be concise and free of punctuation, symbols, or special characters.2. Red (int) - Desired red value in RGB color space from 0 to 255. 3. Green (int) - Desired green value in RGB color space from 0 to 255. 4. Blue (int) - Desired blue value in RGB color space from 0 to 255. Example: user_input = [["hair", 30, 55, 255], ["shirt", 70, 0 , 157]].

    Returns:
        Base64-encoded string.

    Raises:
        ValueError: If user_input format is invalid, or image format is invalid or corrupted.
        TypeError: If input_img is not a supported type or modal function returns unexpected type.
    """  # noqa: E501
    if len(user_input) == 0 or not isinstance(user_input, list):
        raise gr.Error(
            "user input must be a list of lists, each containing [object, hue, saturation_scale]."  # noqa: E501
        )
    if not input_img:
        raise gr.Error("input img cannot be None or empty.")

    print("before processing input:", user_input)
    valid_pattern = re.compile(r"^[a-zA-Z\s]+$")
    for item in user_input:
        if len(item) != 4:
            raise gr.Error(
                "Each item in user_input must be a list of [object, red, green, blue]"  # noqa: E501
            )
        if not item[0] or not valid_pattern.match(item[0]):
            raise gr.Error(
                "Object name must contain only letters and spaces and cannot be empty."
            )

        if not isinstance(item[0], str):
            item[0] = str(item[0])

        try:
            item[1] = int(item[1])
        except ValueError:
            raise gr.Error("Red must be an integer.")
        if item[1] < 0 or item[1] > 255:
            raise gr.Error("Red must be in the range [0, 255]")

        try:
            item[2] = int(item[2])
        except ValueError:
            raise gr.Error("Green must be an integer.")
        if item[2] < 0 or item[2] > 255:
            raise gr.Error("Green must be in the range [0, 255]")

        try:
            item[3] = int(item[3])
        except ValueError:
            raise gr.Error("Blue must be an integer.")
        if item[3] < 0 or item[3] > 255:
            raise gr.Error("Blue must be in the range [0, 255]")

    print("after processing input:", user_input)

    func = modal.Function.from_name(modal_app_name, "change_image_objects_hsv")
    output_pil = func.remote(image_pil=input_img, targets_config=user_input)

    if output_pil is None:
        raise ValueError("Received None from modal remote function.")
    if not isinstance(output_pil, Image.Image):
        raise TypeError(
            f"Expected Image.Image from modal remote function, got {type(output_pil)}"
        )

    return output_pil


def change_color_objects_lab(
    input_img,
    user_input,
) -> np.ndarray | Image.Image | str | Path | None:
    """
    Changes the color of specified objects in an image using the LAB color space.

    This function segments image regions based on a user-provided text prompt and applies
    color transformations in the LAB color space. LAB separates luminance (L) from color
    components (A for green-red, B for blue-yellow), making it more perceptually uniform
    and closer to how humans perceive color differences.

    Use this method when:
    - Precise perceptual color control is needed (e.g., subtle shifts in tone or matching
      specific brand colors).
    - Working in lighting-sensitive tasks where separating lightness from chroma improves quality.
    - You want color transformations that are less influenced by lighting conditions or
      brightness variations.

    OpenCV LAB Ranges:
        - L: 0-255 (lightness)
        - A: 0-255 (green-red, 128 = neutral)
        - B: 0-255 (blue-yellow, 128 = neutral)

    Common LAB color references:
        - Green: (L=?, A≈80, B≈128)
        - Red: (L=?, A≈180, B≈160)
        - Blue: (L=?, A≈128, B≈80)
        - Yellow: (L=?, A≈120, B≈180)
        - Purple: (L=?, A≈180, B≈100)

    Args:
        user_input: A list of color transformation instructions, each as a three-element list:[object_name (str), new_a (int, 0-255), new_b (int, 0-255)].- object_name: A short, unique identifier for the object to be recolored. Multi-word names are allowed for specificity (e.g., "right person shirt") but must be free of punctuation or special symbols.- new_a: The desired 'a' channel value in LAB space (green-red axis, 0-255, with 128 as neutral).- new_b: The desired 'b' channel value in LAB space (blue-yellow axis, 0-255, with 128 as neutral).Each object must appear only once in the list. Example:[["hair", 80, 128], ["right person shirt", 180, 160]]
        input_img : Input image can be URL string of the image. Cannot be None.

    Returns:
        Base64-encoded string

    Raises:
        ValueError: If user_input format is invalid, a/b values are outside [0, 255] range, or image format is invalid or corrupted.
        TypeError: If input_img is not a supported type or modal function returns unexpected type.
    """  # noqa: E501
    if len(user_input) == 0 or not isinstance(user_input, list):
        raise gr.Error(
            "user input must be a list of lists, each containing [object, new_a, new_b]."  # noqa: E501
        )
    if not input_img:
        raise gr.Error("input img cannot be None or empty.")
    valid_pattern = re.compile(r"^[a-zA-Z\s]+$")
    print("before processing input:", user_input)

    for item in user_input:
        if len(item) != 3:
            raise gr.Error(
                "Each item in user_input must be a list of [object, new_a, new_b]"
            )
        if not item[0] or not valid_pattern.match(item[0]):
            raise gr.Error(
                "Object name must contain only letters and spaces and cannot be empty."
            )
        if not isinstance(item[0], str):
            item[0] = str(item[0])
        if not item[1]:
            raise gr.Error("new A must be set and cannot be empty.")
        if not isinstance(item[1], int):
            try:
                item[1] = int(item[1])
            except ValueError:
                raise gr.Error("new A must be an integer.")
            if item[1] < 0 or item[1] > 255:
                raise gr.Error("new A must be in the range [0, 255]")
        if not item[2]:
            raise gr.Error("new B must be set and cannot be empty.")
        if not isinstance(item[2], int):
            try:
                item[2] = int(item[2])
            except ValueError:
                raise gr.Error("new B must be an integer.")
            if item[2] < 0 or item[2] > 255:
                raise gr.Error("new B must be in the range [0, 255]")

    print("after processing input:", user_input)
    func = modal.Function.from_name(modal_app_name, "change_image_objects_lab")
    output_pil = func.remote(image_pil=input_img, targets_config=user_input)
    if output_pil is None:
        raise ValueError("Received None from modal remote function.")
    if not isinstance(output_pil, Image.Image):
        raise TypeError(
            f"Expected Image.Image from modal remote function, got {type(output_pil)}"
        )

    return output_pil
