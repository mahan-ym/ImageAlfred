import os
from io import BytesIO

import modal
import numpy as np
from PIL import Image

app = modal.App("ImageAlfread")

cuda_version = "12.4.0"
flavor = "devel"
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
)


@app.function(gpu="T4", image=image)
def langsam_segment(image, prompt: str) -> list:
    from lang_sam import LangSAM  # type: ignore

    model = LangSAM(sam_type="sam2.1_hiera_large")
    results = model.predict([image], [prompt])
    return results


@app.function(gpu="T4", image=image)
def process_image(image: bytes) -> bytes:
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


def visualize_masks(original_image, segmentation_results: list, alpha: float = 0.5):
    if not segmentation_results:
        return []

    # Convert PIL image to numpy array if needed
    if isinstance(original_image, Image.Image):
        img_array = np.array(original_image)
    else:
        img_array = original_image

    results = []

    # Process each result item (usually there's just one for a single image input)
    for result in segmentation_results:
        masks = result.get("masks", [])

        if len(masks.shape) == 3:  # Multiple masks in a single result
            for i in range(masks.shape[0]):
                mask = masks[i]
                mask_score = 1.0
                if "mask_scores" in result:
                    # Check if mask_scores is an array or scalar
                    if (
                        hasattr(result["mask_scores"], "shape")
                        and result["mask_scores"].ndim > 0
                    ):
                        mask_score = result["mask_scores"][i]
                    else:
                        # It's a scalar, so use the value directly
                        mask_score = result["mask_scores"]
                print(f"Mask {i} score: {mask_score}")
                # Create a copy of the original image
                masked_img = img_array.copy()

                # Create colored mask overlay (using red color)
                mask_overlay = np.zeros_like(masked_img)
                mask_overlay[:, :, 0] = 255  # Red channel

                # Apply the mask to the overlay
                mask_bool = mask.astype(bool)
                masked_img[mask_bool] = (1 - alpha) * masked_img[
                    mask_bool
                ] + alpha * mask_overlay[mask_bool]

                # Convert back to PIL image
                vis_img = Image.fromarray(masked_img.astype(np.uint8))
                results.append(vis_img)

    return results


@app.local_entrypoint()
def main():
    # This is the entrypoint for local testing
    # print("Running local entrypoint")
    # has_cuda, output = validate_cuda.remote()
    # print(f"Has CUDA: {has_cuda}")
    # print(output)

    img = Image.open("./src/assets/input/test_1.jpg")
    masks = langsam_segment.remote(img, "t-shirt")
    print("Length of Masks generated:", len(masks), type(masks))
    # Visualize the masks
    visualized_images = visualize_masks(img, masks)

    # Save the visualized images
    for i, vis_img in enumerate(visualized_images):
        vis_img.save(f"./src/assets/output/output_masked_image_{i}.png")
        print(
            f"Saved masked image {i} to ./src/assets/output/output_masked_image_{i}.png"
        )
