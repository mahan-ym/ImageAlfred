import os
import modal
from PIL import Image
import numpy as np
from io import BytesIO


app = modal.App("ImageAlfread")

cuda_version = "12.4.0"  # should be no greater than host CUDA version
flavor = "devel"  #  includes full CUDA toolkit
operating_sys = "ubuntu22.04"
tag = f"{cuda_version}-{flavor}-{operating_sys}"

image = (
    modal.Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.12")
    .apt_install("git")
    .pip_install(
        "opencv-contrib-python",
        "huggingface-hub",
        "Pillow",
        "numpy",
        # "opencv-contrib-python-headless",
    )
    .pip_install(
        "torch==2.4.1",
        "torchvision==0.19.1",
        index_url="https://download.pytorch.org/whl/cu124",
    )
    .pip_install("git+https://github.com/luca-medeiros/lang-segment-anything.git")
    # .run_commands("git clone https://github.com/facebookresearch/sam2.git && cd sam2 && pip install -e .")
)


@app.function(gpu="T4", image=image)
def langsam_segment(image, prompt: str) -> list:
    from lang_sam import LangSAM  # type: ignore

    model = LangSAM(sam_type="sam2.1_hiera_large")
    results = model.predict([image], [prompt])
    return results


@app.function(gpu="T4", image=image)
def process_image(image: bytes) -> bytes:
    # Process the image using the SAM model
    pass


@app.function(gpu="T4", image=image)
def validate_cuda():
    import torch  # type: ignore

    has_cuda = torch.cuda.is_available()
    import subprocess

    output = subprocess.check_output(["nvidia-smi"], text=True)
    assert "Driver Version:" in output
    assert "CUDA Version:" in output
    return has_cuda, output



@app.local_entrypoint()
def main():
    # This is the entrypoint for local testing
    # print("Running local entrypoint")
    # has_cuda, output = validate_cuda.remote()
    # print(f"Has CUDA: {has_cuda}")
    # print(output)
    img = Image.open("./assets/example.jpg")
    masks = langsam_segment.remote(img, "give me a mask of the t-shirt")
    print("Length of Masks generated:", len(masks), type(masks))
