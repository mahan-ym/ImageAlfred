import gradio as gr


def recolor_image(input_prompt,input_img, target_color):
    """
    Recolor an image based on a user-provided prompt and a target color.

    Args:
        input_prompt (str): The prompt describing the object to recolor.
        target_color (str): The target color to apply to the object.
        input_img (numpy.ndarray): The input image as a numpy array.

    Returns:
        numpy.ndarray: The recolored image.
    """
    pass

def privacy_preserve_image(input_prompt, input_img):
    pass

recolor_tool = gr.Interface(
    fn=recolor_image,
    inputs=[gr.Textbox("user_input"),gr.Image(type="numpy"),gr.Textbox("target_color")],
    outputs=gr.Image(),
    title="Image Recolor tool",
    description="Upload an image, provide a prompt for the object to recolor, and specify the target color. The tool will recolor the specified object in the image.",
)

privacy_preserve_tool = gr.Interface(
    fn=privacy_preserve_image,
    inputs=[gr.Textbox("user_input"),gr.Image(type="numpy")],
    outputs=gr.Image(),
    title="Privacy preserving tool",
    description="Upload an image and provide a prompt for the object to enforce privacy. The tool will use blurring to obscure the specified objects in the image.",
)

demo = gr.TabbedInterface(
    [recolor_tool, privacy_preserve_tool],
    ["Recolor", "Privacy preserve"],
    title="Image Alfred",
    theme=gr.themes.Default(
        primary_hue="blue",
        secondary_hue="blue",
        font="Inter",
        font_mono="Courier New",
    )
)

if __name__ == "__main__":
    demo.launch(mcp_server=True)
