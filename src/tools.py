import modal

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
    """
    langsam_prompt = ". ".join([f"{obj[0]}" for obj in user_input])
    pass


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
    """
    pass


if __name__ == "__main__":
    change_color_objects_hsv(
        user_input=[["hair", 30, 1.2], ["shirt", 60, 1.0]], input_img=b""
    )
