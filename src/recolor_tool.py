import modal


def change_color_objects_hsv(user_input, input_img):
    """
    Recolor an image based on user input. Use this tool to recolor using HSV color space.
    Args:
        user_input (list): List of [object:str, hue:int, saturationScale:float].
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
