
<div align="center">
<img src="./src/assets/icons/ImageAlfredIcon.png" alt="ImageAlfred" width=200 height=200>

<h1>Image Alfred</h1>

ImageAlfred is an image Model Context Protocol (MCP) tool designed to streamline image processing workflows

<img alt="Python Version from PEP 621 TOML" src="https://img.shields.io/python/required-version-toml?tomlFilePath=https%3A%2F%2Fraw.githubusercontent.com%2Fmahan-ym%2FImageAlfred%2Fmain%2Fpyproject.toml">
<img src="https://badge.mcpx.dev?type=server" title="MCP Server"/>
<img alt="GitHub License" src="https://img.shields.io/github/license/mahan-ym/ImageAlfred">

<a href=https://huggingface.co> <img src="src/assets/icons/hf-logo.svg" alt="huggingface" height=40> </a>
<a href="https://www.python.org"><img src="src/assets/icons/python-logo-only.svg" alt="python" height=40></a>
<!-- <a href="https://www.gradio.app" heigh=40><img src="src/assets/icons/gradio-color.svg"></a> -->
</div>

<!-- It provides a user-friendly interface for interacting with image models, leveraging the power of Gradio for the frontend and Modal for scalable backend deployment. -->

<!-- ## Features
- Intuitive web interface for image processing
- Powered by Gradio for rapid prototyping and UI
- Scalable and serverless execution with Modal
- Easily extendable for custom image models and workflows -->

## Maintainers

[Mahan-ym | Mahan Yarmohammad](https://www.mahan-ym.com/)

[Soodoo | Saaed Saadatipour](https://soodoo.me/)

## Tools

- [Gradio](https://www.gradio.app/): Serving user interface and MCP server
- [Modal.com](https://modal.com/): AI infrastructure making all the magic 🔮 possible.
- [SAM](https://segment-anything.com/): Segment Anything model by meta for image segmentation and mask generation
- [CLIPSeg](https://github.com/timojl/clipseg): Image Segmentation using CLIP. We used it as a more precise object detection model
- [HuggingFace](https://huggingface.co/): Downloading SAM and using Space for hosting.

## Getting Started

### Prerequisites

- Python 3.12+
- [uv](https://github.com/astral-sh/uv) (a fast Python package installer and virtual environment manager)

### Installation

It will create virtual environment, activate it, install dependecies and setup modal

```bash
make install
```

### Running the App

This will launch the Gradio interface for ImageAlfred.

```bash
make run
```

## License

This project is licensed under the terms of the LICENSE file in this repository.
