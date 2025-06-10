from pathlib import Path

import gradio as gr

from tools import (
    change_color_objects_hsv,
    change_color_objects_lab,
    privacy_preserve_image,
    remove_background,
)

gr.set_static_paths(paths=[Path.cwd().absolute() / "assets"])

icon = """<link rel="icon" type="image/x-icon" href="https://raw.githubusercontent.com/mahan-ym/ImageAlfred/main/src/assets/icons/ImageAlfredIcon.png">"""

title = """Image Alfred - Recolor and Privacy Preserving Image MCP Tools
<img src="https://raw.githubusercontent.com/mahan-ym/ImageAlfred/main/src/assets/icons/ImageAlfredIcon.png" alt="Image Alfred Logo" style="width: 120px; height: auto; margin: 0 auto;">
<h4 style="text-align: center;"></h4>
"""  # noqa: E501

hsv_df_input = gr.Dataframe(
    headers=["Object", "Red", "Green", "Blue"],
    datatype=["str", "number", "number", "number"],
    col_count=(4, "fixed"),
    show_row_numbers=True,
    label="Target Objects and Their new RGB Colors",
    type="array",
)

lab_df_input = gr.Dataframe(
    headers=["Object", "New A", "New B"],
    datatype=["str", "number", "number"],
    col_count=(3, "fixed"),
    label="Target Objects and New Settings.(0-255 -- 128 = Neutral)",
    type="array",
)

change_color_objects_hsv_tool = gr.Interface(
    fn=change_color_objects_hsv,
    inputs=[
        gr.Image(label="Input Image", type="pil"),
        hsv_df_input,
    ],
    outputs=gr.Image(label="Output Image"),
    title="Image Recolor Tool (HSV)",
    description="""
    This tool allows you to recolor objects in an image using the HSV color space.
    You can specify the RGB values for each object.""",  # noqa: E501
    examples=[
        [
            "https://raw.githubusercontent.com/mahan-ym/ImageAlfred/main/src/assets/examples/test_1.jpg",
            [
                ["pants", 255, 178, 102],
            ],
        ],
        [
            "https://raw.githubusercontent.com/mahan-ym/ImageAlfred/main/src/assets/examples/test_8.jpg",
            [
                ["pants", 114, 117, 34],
                ["shirt", 51, 51, 37],
            ],
        ],
    ],
)

change_color_objects_lab_tool = gr.Interface(
    fn=change_color_objects_lab,
    inputs=[
        gr.Image(label="Input Image", type="pil"),
        lab_df_input,
    ],
    outputs=gr.Image(label="Output Image"),
    title="Image Recolor Tool (LAB)",
    description="""
    Recolor an image based on user input using the LAB color space.
    You can specify the new A and new B values for each object.
    """,  # noqa: E501
    examples=[
        [
            "https://raw.githubusercontent.com/mahan-ym/ImageAlfred/main/src/assets/examples/test_1.jpg",
            [["pants", 112, 128]],
        ],
        [
            "https://raw.githubusercontent.com/mahan-ym/ImageAlfred/main/src/assets/examples/test_4.jpg",
            [["desk", 166, 193]],
        ],
        [
            "https://raw.githubusercontent.com/mahan-ym/ImageAlfred/main/src/assets/examples/test_5.jpg",
            [["suits coat", 110, 133]],
        ],
    ],
)

privacy_preserve_tool = gr.Interface(
    fn=privacy_preserve_image,
    inputs=[
        gr.Image(label="Input Image", type="pil"),
        gr.Textbox(
            label="Objects to Mask (dot-separated)",
            placeholder="e.g., person. car. license plate",
        ),
        gr.Slider(
            label="Privacy Strength",
            minimum=1,
            maximum=50,
            value=15,
            step=1,
            info="Higher values result in stronger blurring.",
        ),
    ],
    outputs=gr.Image(label="Output Image"),
    title="Privacy Preserving Tool",
    description="Upload an image and provide a prompt for the object to enforce privacy. The tool will use blurring to obscure the specified objects in the image.",  # noqa: E501
    examples=[
        [
            "https://raw.githubusercontent.com/mahan-ym/ImageAlfred/main/src/assets/examples/test_3.jpg",
            "license plate.",
            10,
        ],
        [
            "https://raw.githubusercontent.com/mahan-ym/ImageAlfred/main/src/assets/examples/test_8.jpg",
            "face.",
            15,
        ],
        [
            "https://raw.githubusercontent.com/mahan-ym/ImageAlfred/main/src/assets/examples/test_6.jpg",
            "face.",
            20,
        ],
    ],
)

remove_background_tool = gr.Interface(
    fn=remove_background,
    inputs=[
        gr.Image(label="Input Image", type="pil"),
    ],
    outputs=gr.Image(label="Output Image"),
    title="Remove Image Background Tool",
    description="Upload an image to remove the background.",
    examples=[
        [
            "https://raw.githubusercontent.com/mahan-ym/ImageAlfred/main/src/assets/examples/test_5.jpg",
        ],
        [
            "https://raw.githubusercontent.com/mahan-ym/ImageAlfred/main/src/assets/examples/test_6.jpg",
        ],
        [
            "https://raw.githubusercontent.com/mahan-ym/ImageAlfred/main/src/assets/examples/test_8.jpg",
        ],
    ],
)

demo = gr.TabbedInterface(
    [
        privacy_preserve_tool,
        remove_background_tool,
        change_color_objects_hsv_tool,
        change_color_objects_lab_tool,
    ],
    [
        "Privacy Preserving Tool",
        "Remove Background Tool",
        "Change Color Objects HSV",
        "Change Color Objects LAB",
    ],
    title=title,
    theme=gr.themes.Default(
        primary_hue="blue",
        secondary_hue="green",
        # font="Inter",
        # font_mono="Courier New",
    ),
    head=icon,
)

# with gr.Blocks(title="Image Alfred", head=test) as demo:
#     gr.HTML(header)
#     tabs_interface.render()

if __name__ == "__main__":
    demo.launch(mcp_server=True, max_file_size="15mb")
