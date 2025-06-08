from pathlib import Path

import modal
import numpy as np
from PIL import Image

modal_app_name = "ImageAlfred"


def privacy_preserve_image(
    input_img,
    input_prompt,
) -> np.ndarray | Image.Image | str | Path | None:
    """
    Obscure specified objects in the input image based on the input prompt.

    Args:
        input_img (Image.Image): Input image in bytes format.
        input_prompt (str): Object to obscure in the image has to be a dot-separated string. It can be a single word or multiple words, e.g., "left person face", "license plate" but it must be as short as possible and avoid using symbols or punctuation. Also you have to use single form of the word, e.g., "person" instead of "people", "face" instead of "faces". e.g. input_prompt = "face. right car. blue shirt."

    Returns:
        bytes: Binary image data of the modified image.

    example:
        input_prompt = ["face", "license plate"]
    """  # noqa: E501
    func = modal.Function.from_name("ImageAlfred", "preserve_privacy")
    output_pil = func.remote(image_pil=input_img, prompt=input_prompt)

    if output_pil is None:
        raise ValueError("Received None from modal remote function.")
    if not isinstance(output_pil, Image.Image):
        raise TypeError(
            f"Expected Image.Image from modal remote function, got {type(output_pil)}"
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
        user_input : A list of lists where each inner list contains three elements: target object name (str), hue value (int, 0-179), and saturation scale factor (float, >0). Each target object must be unique within the list and it can be multiword but short and without punctuation or symbols. e.g.: [["hair", 30, 1.2], ["right person shirt", 60, 1.0]].
        input_img: Input image or can be URL string of the image. Cannot be None.

    Returns:
        Base64-encoded string.

    Raises:
        ValueError: If user_input format is invalid, hue values are outside [0, 179] range, saturation_scale is not positive, or image format is invalid or corrupted.
        TypeError: If input_img is not a supported type or modal function returns unexpected type.
    """  # noqa: E501
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
    # img_link = upload_image_to_tmpfiles(output_pil)

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
        user_input: A list of lists where each inner list contains three elements: target object name (str), new_a value (int, 0-255), and new_b value (int, 0-255). Target objects must be unique within the list and can be multiword but should be short and without punctuation or symbols. Also use singular form of the word, e.g., "person" instead of "people", "face" instead of "faces". Example: [["hair", 80, 128], ["right person shirt", 180, 160]].
        input_img : Input image can be URL string of the image. Cannot be None.

    Returns:
        Base64-encoded string

    Raises:
        ValueError: If user_input format is invalid, a/b values are outside [0, 255] range, or image format is invalid or corrupted.
        TypeError: If input_img is not a supported type or modal function returns unexpected type.
    """  # noqa: E501
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
    # img_link = upload_image_to_tmpfiles(output_pil)

    return output_pil


if __name__ == "__main__":
    image_pil = Image.open("./src/assets/test_image.jpg")
    change_color_objects_hsv(
        user_input=[["hair", 30, 1.2], ["shirt", 60, 1.0]], input_img=image_pil
    )
