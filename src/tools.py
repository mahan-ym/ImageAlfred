import re
from pathlib import Path

import gradio as gr
import modal
import numpy as np
from PIL import Image

modal_app_name = "ImageAlfred"


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
        input_prompt (str): Object to obscure in the image has to be a dot-separated string. It can be a single word or multiple words, e.g., "left person face", "license plate" but it must be as short as possible and avoid using symbols or punctuation. Also you have to use single form of the word, e.g., "person" instead of "people", "face" instead of "faces". e.g. input_prompt = "face. right car. blue shirt."
        privacy_strength (int): Strength of the privacy preservation effect. Higher values result in stronger blurring. Default is 15.
    Returns:
        bytes: Binary image data of the modified image.

    example:
        input_prompt = ["face", "license plate"]
    """  # noqa: E501
    if not input_img:
        raise gr.Error("Input image cannot be None or empty.")
    valid_pattern = re.compile(r"^[a-zA-Z\s.]+$")
    if not input_prompt or input_prompt.strip() == "":
        raise gr.Error("Input prompt cannot be None or empty.")
    if not valid_pattern.match(input_prompt):
        raise gr.Error("Input prompt must contain only letters, spaces, and dots.")

    func = modal.Function.from_name("ImageAlfred", "preserve_privacy")
    output_pil = func.remote(
        image_pil=input_img,
        prompt=input_prompt,
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

    This function segments objects in the image based on a user-provided text prompt, then
    modifies their hue and saturation in the HSV (Hue, Saturation, Value) space. HSV is intuitive
    for color manipulation where users think in terms of basic color categories and intensity,
    making it useful for broad, vivid color shifts.

    Use this method when:
    - Performing broad color changes or visual effects (e.g., turning a shirt from red to blue).
    - Needing intuitive control over color categories (e.g., shifting everything that's red to purple).
    - Saturation and vibrancy manipulation are more important than accurate perceptual matching.

    OpenCV HSV Ranges:
        - H: 0-179 (Hue angle on color wheel, where 0 = red, 60 = green, 120 = blue, etc.)
        - S: 0-255 (Saturation)
        - V: 0-255 (Brightness)

    Common HSV color references:
        - Red: (Hue≈0), Green: (Hue≈60), Blue: (Hue≈120), Yellow: (Hue≈30), Purple: (Hue≈150)
        - Typically used with Saturation=255 for vivid colors.


    Args:
        input_img: Input image or can be URL string of the image or base64 string. Cannot be None.
        user_input : A list of target specifications for color transformation. Each inner list must contain exactly three elements in the following order: 1. target_object (str) - A short, human-readable description of the object to be modified.Multi-word descriptions are allowed for disambiguation (e.g., "right person shirt"), but they must be at most three words and concise and free of punctuation, symbols, or special characters.2. hue (int) - Desired hue value in the HSV color space, ranging from 0 to 179. Represents the color angle on the HSV color wheel (e.g., 0 = red, 60 = green, 120 = blue)3. saturation_scale (float) - A multiplicative scale factor applied to the current saturation   of the object (must be > 0). For example, 1.0 preserves current saturation, 1.2 increases vibrancy, and 0.8 slightly desaturates. Each target object must be uniquely defined in the list to avoid conflicting transformations.Example: [["hair", 30, 1.2], ["right person shirt", 60, 1.0]]

    Returns:
        Base64-encoded string.

    Raises:
        ValueError: If user_input format is invalid, hue values are outside [0, 179] range, saturation_scale is not positive, or image format is invalid or corrupted.
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
        if len(item) != 3:
            raise gr.Error(
                "Each item in user_input must be a list of [object, hue, saturation_scale]"  # noqa: E501
            )
        if not item[0] or not valid_pattern.match(item[0]):
            raise gr.Error(
                "Object name must contain only letters and spaces and cannot be empty."
            )

        if not isinstance(item[0], str):
            item[0] = str(item[0])
        if not item[1]:
            raise gr.Error("Hue must be set and cannot be empty.")
        if not isinstance(item[1], (int, float)):
            try:
                item[1] = int(item[1])
            except ValueError:
                raise gr.Error("Hue must be an integer.")
            if item[1] < 0 or item[1] > 179:
                raise gr.Error("Hue must be in the range [0, 179]")
        if not item[2]:
            raise gr.Error("Saturation scale must be set and cannot be empty.")
        if not isinstance(item[2], (int, float)):
            try:
                item[2] = float(item[2])
            except ValueError:
                raise gr.Error("Saturation scale must be a float number.")
            if item[2] <= 0:
                raise gr.Error("Saturation scale must be greater than 0")

    print("after processing input:", user_input)

    func = modal.Function.from_name("ImageAlfred", "change_image_objects_hsv")
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
        user_input: A list of color transformation instructions, each as a three-element list:[object_name (str), new_a (int, 0-255), new_b (int, 0-255)].- object_name: A short, unique identifier for the object to be recolored. Multi-word names are allowed  for specificity (e.g., "right person shirt") but must be 3 words or fewer and free of punctuation or special symbols.- new_a: The desired 'a' channel value in LAB space (green-red axis, 0-255, with 128 as neutral).- new_b: The desired 'b' channel value in LAB space (blue-yellow axis, 0-255, with 128 as neutral).Each object must appear only once in the list. Example:[["hair", 80, 128], ["right person shirt", 180, 160]]
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
    func = modal.Function.from_name("ImageAlfred", "change_image_objects_lab")
    output_pil = func.remote(image_pil=input_img, targets_config=user_input)
    if output_pil is None:
        raise ValueError("Received None from modal remote function.")
    if not isinstance(output_pil, Image.Image):
        raise TypeError(
            f"Expected Image.Image from modal remote function, got {type(output_pil)}"
        )
    # img_link = upload_image_to_tmpfiles(output_pil)

    return output_pil


if __name__ == "__main__":
    image_pil = Image.open("./src/assets/test_image.jpg")
    change_color_objects_hsv(
        user_input=[["hair", 30, 1.2], ["shirt", 60, 1.0]], input_img=image_pil
    )
