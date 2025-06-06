import gradio as gr

from recolor_tool import change_color_objects_hsv, change_color_objects_lab


def privacy_preserve_image(input_prompt, input_img):
    pass


title = """
<div style="text-align: center;">
    <img src="https://raw.githubusercontent.com/mahan-ym/ImageAlfred/main/src/assets/ImageAlfredIcon.png" alt="Image Alfred Logo" style="width: 200px; height: auto;">
    <h1>Image Alfred</h1>
    <p>Recolor and Privacy Preserving Image Tools</p>
</div>
"""  # noqa: E501

hsv_df_input = gr.Dataframe(
    headers=["object", "hue", "saturationScale"],
    datatype=["str", "number", "number"],
    label="Input Data",
    type="array",
)

lab_df_input = gr.Dataframe(
    headers=["object", "new_a", "new_b"],
    datatype=["str", "number", "number"],
    label="Input Data",
    type="array",
)

change_color_objects_hsv_tool = gr.Interface(
    fn=change_color_objects_hsv,
    inputs=[
        hsv_df_input,
        gr.File(label="Input Image", file_types=["image"], type="binary"),
    ],
    outputs=gr.Image(),
    title="Image Recolor tool (HSV)",
    description="This tool allows you to recolor objects in an image using the HSV color space. You can specify the hue and saturation scale for each object.",  # noqa: E501
)

change_color_objects_lab_tool = gr.Interface(
    fn=change_color_objects_lab,
    inputs=[
        lab_df_input,
        gr.File(label="Input Image", file_types=["image"], type="binary"),
    ],
    outputs=gr.Image(),
    title="Image Recolor tool (LAB)",
    description="Recolor an image based on user input using the LAB color space. You can specify the new_a and new_b values for each object.",  # noqa: E501
)

privacy_preserve_tool = gr.Interface(
    fn=privacy_preserve_image,
    inputs=[
        gr.Textbox("user_input"),
        gr.File(label="Input Image", file_types=["image"], type="binary"),
    ],
    outputs=gr.Image(),
    title="Privacy preserving tool",
    description="Upload an image and provide a prompt for the object to enforce privacy. The tool will use blurring to obscure the specified objects in the image.",  # noqa: E501
)

demo = gr.TabbedInterface(
    [
        change_color_objects_hsv_tool,
        change_color_objects_lab_tool,
        privacy_preserve_tool,
    ],
    ["Change Color Objects HSV", "Change Color Objects LAB", "Privacy Preserving Tool"],
    title="Image Alfred",
    theme=gr.themes.Default(
        primary_hue="blue",
        secondary_hue="blue",
        font="Inter",
        font_mono="Courier New",
    ),
)

if __name__ == "__main__":
    demo.launch(mcp_server=True)
