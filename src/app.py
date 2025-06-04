import gradio as gr


def process_image(input_img):
    pass


demo = gr.Interface(
    fn=process_image,
    inputs=gr.Image(type="numpy"),
    outputs=gr.Image(),
    title="Simple Image Processor",
    description="Upload an image and see the processed result.",
)

if __name__ == "__main__":
    demo.launch()
