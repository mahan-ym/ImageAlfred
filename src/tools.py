from io import BytesIO

import modal
from PIL import Image, UnidentifiedImageError

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
):
    """Changes the hue and saturation of specified objects in an image.
    This function uses LangSAM to segment objects in the image based on provided prompts,
    and then modifies the hue and saturation of those objects in the HSV color space.

    Parameters
    ----------
    user_input : list[list[str  |  int  |  float]]
        list of lists, where each inner list contains:
        - target object name (str)
        - hue value (int or float): openCV HSV range: 0-179, where 0 is red, 30 is yellow, 60 is green, 120 is cyan, 179 is blue
        - saturation scale (float): 1 means no change, <1 reduces saturation, >1 increases saturation
    input_img : bytes
        image data in bytes format

    Returns
    -------
    Image.Image
        PIL Image object of the modified image

    Example
    -------
    >>> user_input = [
    ...     ["hair", 30, 1.2],
    ...     ["tshirt", 60, 1.0],
    ...     ["pants", 90, 0.8],
    ... ]
    >>> change_image_objects_hsv(user_input, input_img)
    """  # noqa: E501
    try:
        Image.open(BytesIO(input_img))
    except UnidentifiedImageError:
        raise ValueError("Invalid image format or corrupted image data.")
    except Exception as e:
        raise ValueError(f"Could not process input image: {e}")

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
    output_bytes = func.remote(image_bytes=input_img, targets_config=user_input)

    if output_bytes is None:
        raise ValueError("Received None from modal remote function.")
    if not isinstance(output_bytes, bytes):
        raise TypeError(
            f"Expected bytes from modal remote function, got {type(output_bytes)}"
        )

    output_pil = Image.open(BytesIO(output_bytes)).convert("RGB")
    return output_pil


def change_color_objects_lab(user_input, input_img):
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
    user_input : list[list[str  |  int  |  float]]
        list of lists, where each inner list contains:
        - target object name (str)
        - new_a (int): 0-255, green-red channel in LAB color space
        - new_b (int): 0-255, blue-yellow channel in LAB color space
    input_img : bytes
        binary image data
    

    Returns
    -------
    Image.Image
        PIL Image object containing the modified image

    Example
    -------
    >>> user_input = [
    ...     ["hair", 80, 128],
    ...     ["shirt", 180, 160],
    ...     ["pants", 120, 180],
    ... ]
    >>> change_image_objects_lab(user_input, input_img)
    """  # noqa: E501
    try:
        Image.open(BytesIO(input_img))
    except UnidentifiedImageError:
        raise ValueError("Invalid image format or corrupted image data.")
    except Exception as e:
        raise ValueError(f"Could not process input image: {e}")
    
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
    output_bytes = func.remote(image_bytes=input_img, targets_config=user_input)
    if output_bytes is None:
        raise ValueError("Received None from modal remote function.")
    if not isinstance(output_bytes, bytes):
        raise TypeError(
            f"Expected bytes from modal remote function, got {type(output_bytes)}"
        )
    output_pil = Image.open(BytesIO(output_bytes)).convert("RGB")
    return output_pil


if __name__ == "__main__":
    change_color_objects_hsv(
        user_input=[["hair", 30, 1.2], ["shirt", 60, 1.0]], input_img=b""
    )
