from pathlib import Path

import gradio as gr

from tools import (
    change_color_objects_hsv,
    change_color_objects_lab,
    privacy_preserve_image,
)

gr.set_static_paths(paths=[Path.cwd().absolute() / "assets"])

icon = """<link rel="icon" type="image/x-icon" href="https://raw.githubusercontent.com/mahan-ym/ImageAlfred/main/src/assets/icons/ImageAlfredIcon.png">"""

title = """Image Alfred - Recolor and Privacy Preserving Image Tools
<img src="https://raw.githubusercontent.com/mahan-ym/ImageAlfred/main/src/assets/icons/ImageAlfredIcon.png" alt="Image Alfred Logo" style="width: 120px; height: auto; margin: 0 auto;">
"""

hsv_df_input = gr.Dataframe(
    headers=["Object", "Hue", "Saturation Scale"],
    datatype=["str", "number", "number"],
    label="Target Objects and New Settings",
    type="array",
)

lab_df_input = gr.Dataframe(
    headers=["Object", "New A", "New B"],
    datatype=["str", "number", "number"],
    label="Target Objects and New Settings",
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
    description="This tool allows you to recolor objects in an image using the HSV color space. You can specify the hue and saturation scale for each object.",  # noqa: E501
)

change_color_objects_lab_tool = gr.Interface(
    fn=change_color_objects_lab,
    inputs=[
        gr.Image(label="Input Image", type="pil"),
        lab_df_input,
    ],
    outputs=gr.Image(label="Output Image"),
    title="Image Recolor Tool (LAB)",
    description="Recolor an image based on user input using the LAB color space. You can specify the new A and new B values for each object.",  # noqa: E501
)

privacy_preserve_tool = gr.Interface(
    fn=privacy_preserve_image,
    inputs=[
        gr.Image(label="Input Image", type="pil"),
        gr.Textbox(
            label="Objects to Mask (dot-separated)",
            placeholder="e.g., person. car. license plate",
        ),  # noqa: E501
    ],
    outputs=gr.Image(label="Output Image"),
    title="Privacy Preserving Tool",
    description="Upload an image and provide a prompt for the object to enforce privacy. The tool will use blurring to obscure the specified objects in the image.",  # noqa: E501
)

demo = gr.TabbedInterface(
    [
        change_color_objects_hsv_tool,
        change_color_objects_lab_tool,
        privacy_preserve_tool,
    ],
    ["Change Color Objects HSV", "Change Color Objects LAB", "Privacy Preserving Tool"],
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
    demo.launch(mcp_server=True, max_file_size="5mb")
