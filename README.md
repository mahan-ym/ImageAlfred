
![Image Alfred](./src/assets/ImageAlfredIcon.png)

# ImageAlfred

ImageAlfred is an image Model Context Protocol (MCP) tool designed to streamline image processing workflows.
<!-- It provides a user-friendly interface for interacting with image models, leveraging the power of Gradio for the frontend and Modal for scalable backend deployment. -->

<!-- ## Features
- Intuitive web interface for image processing
- Powered by Gradio for rapid prototyping and UI
- Scalable and serverless execution with Modal
- Easily extendable for custom image models and workflows -->

## Maintainers
[Mahan Yarmohammad (Mahan-ym)](https://www.mahan-ym.com/)
[Saaed Saadatipour (Soodoo)](https://soodoo.me/)

## Getting Started

### Prerequisites
- Python 3.13+
- [uv](https://github.com/astral-sh/uv) (a fast Python package installer and virtual environment manager)

### Installation

1. **Create a virtual environment using uv:**

```bash
uv venv
```

2. **Activate the virtual environment:**

```bash
source .venv/bin/activate
```

3. **Install dependencies:**

```bash
uv sync
```

4. **Setup Modal**

```bash
modal setup
```

### Running the App

```bash
python src/app.py
```

This will launch the Gradio interface for ImageAlfred.

## License

This project is licensed under the terms of the LICENSE file in this repository.
