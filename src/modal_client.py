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
def langsam_segment(
    image: Image.Image,
    prompt: str,
) -> list:
    from lang_sam import LangSAM  # type: ignore

    model = LangSAM(sam_type="sam2.1_hiera_large")
    results = model.predict([image], [prompt])
    return results


@app.function(
    gpu="T4",
    image=image,
)
def change_color_objects_image(
    image: Image.Image,
    prompt: str,
    new_hue: int,
    new_saturation=None,
) -> bytes:
    """
    Changes the hue of an object in an image using a soft mask.

    Parameters:
    - image: PIL.Image
    - results: SAM2 model output
    - new_hue: int (0-179, OpenCV hue range)
    - new_saturation: int or None — if provided, overrides object saturation

    Returns:
    - PIL.Image with modified object color
    """
    import cv2  # type: ignore
    from lang_sam import LangSAM  # type: ignore

    model = LangSAM(sam_type="sam2.1_hiera_large")
    results = model.predict(
        images_pil=[image],
        texts_prompt=[prompt],
        # box_threshold=0.3,
        # text_threshold=0.25,
    )

    # Create class_id for each unique label
    unique_labels = list(set(results[0]["labels"]))
    class_id_map = {label: idx for idx, label in enumerate(unique_labels)}
    class_id = [class_id_map[label] for label in results[0]["labels"]]

    mask = results[0]["masks"][0]
    mask = np.clip(mask, 0.0, 1.0)

    # Convert mask to uint8 for contour detection
    mask_uint8 = (mask * 255).astype(np.uint8)

    # Find contours and draw filled mask for smoother shape
    contours, _ = cv2.findContours(
        mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    clean_mask = np.zeros_like(mask_uint8)
    cv2.drawContours(
        clean_mask, contours, -1, 255, thickness=cv2.FILLED, lineType=cv2.LINE_AA
    )
    # Feather the mask with Gaussian blur for soft edges
    feathered_mask = cv2.GaussianBlur(clean_mask, (25, 25), 0) / 255.0
    mask_3ch = np.repeat(feathered_mask[:, :, np.newaxis], 3, axis=2)

    # Convert PIL image to OpenCV format
    bgr_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # Apply changes in masked regions only
    h = h.astype(np.float32)
    s = s.astype(np.float32)

    h = feathered_mask * new_hue + (1 - feathered_mask) * h
    if new_saturation is not None:
        s = feathered_mask * new_saturation + (1 - feathered_mask) * s
    # Clip and convert back to uint8
    h = np.clip(h, 0, 179).astype(np.uint8)  # Hue range in OpenCV is [0, 179]
    s = np.clip(s, 0, 255).astype(np.uint8)

    # Merge channels and convert back to BGR
    hsv_modified = cv2.merge([h, s, v])
    colored_bgr = cv2.cvtColor(hsv_modified, cv2.COLOR_HSV2BGR)

    # Convert colored_bgr and original bgr_image to RGB for blending
    colored_rgb = cv2.cvtColor(colored_bgr, cv2.COLOR_BGR2RGB)
    original_rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

    # Blend in RGB space using soft mask
    output_rgb = (
        colored_rgb.astype(np.float32) * mask_3ch
        + original_rgb.astype(np.float32) * (1 - mask_3ch)
    ).astype(np.uint8)

    # Convert back to Pillow Image
    result_pil = Image.fromarray(output_rgb)

    return result_pil


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


def test_lang_sam():
    input_dir = "./src/assets/input"
    output_dir = "./src/assets/output"
    img_name = "test_1.jpg"
    img = Image.open(f"{input_dir}/{img_name}").convert("RGB")

    masks = langsam_segment.remote(img, "head.")
    print("Length of Masks generated:", len(masks), type(masks))

    # Visualize the masks
    visualized_images = visualize_masks(img, masks)
    # Remove all previous PNG files in the output directory
    for file in os.listdir(output_dir):
        if file.endswith(".png") or file.endswith(".jpg"):
            file_path = os.path.join(output_dir, file)
            os.remove(file_path)
            print(f"Removed previous output: {file_path}")

    # Save the visualized images
    os.makedirs(output_dir, exist_ok=True)
    for i, vis_img in enumerate(visualized_images):
        vis_img.save(f"{output_dir}/mask_{i}_{img_name}")
        print(f"Saved masked image {i} to {output_dir}/mask_{i}_{img_name}")


@app.local_entrypoint()
def main():
    input_dir = "./src/assets/input"
    output_dir = "./src/assets/output"
    img_name = "test_1.jpg"
    img = Image.open(f"{input_dir}/{img_name}").convert("RGB")

    new_image = change_color_objects_image.remote(
        img,
        "hair.",
        120,
    )
    save_path = f"{output_dir}/colored_{img_name}"
    new_image.save(save_path)


if __name__ == "__main__":
    input_dir = "./src/assets/input"
    output_dir = "./src/assets/output"
    img_name = "test_1.jpg"
    img = Image.open(f"{input_dir}/{img_name}").convert("RGB")
    with modal.enable_output():
        with app.run():
            new_image = change_color_objects_image.remote(
                img,
                "shirt.",
                120,
            )
            save_path = f"{output_dir}/colored_{img_name}"
            new_image.save(save_path)