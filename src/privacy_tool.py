from utils import cast_langsam_prompt


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
    langsam_prompt = cast_langsam_prompt(input_prompt)
    
    return input_img