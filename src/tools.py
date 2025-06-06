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
    """
    Recolor an image based on user input. Use this tool to recolor using HSV color space.
    Args:
        user_input (list): List of
                [object (str): semantic prompt for the object, e.g., "hair", "shirt"
                hue (int): 0-179, OpenCV hue range
                saturation_scale (float): Saturation Scale. 1.0 means no change, <1.0 desaturates, >1.0 saturates]

        input_img (bytes): Input image in bytes format.

    Returns:
        bytes: Binary image data of the modified image.

    example:
        user_input = [["hair", 30, 1.2], ["shirt", 60, 1.0]]
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
    """
    Recolor an image based on user input. Use this tool to recolor using LAB color space.
    Args:
        user_input (list): List of [object:str, new_a:int, new_b:int].
        input_img (bytes): Input image in bytes format.

    Returns:
        bytes: Binary image data of the modified image.

    example:
        user_input = [["hair", 128, 128], ["shirt", 100, 150]]
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
