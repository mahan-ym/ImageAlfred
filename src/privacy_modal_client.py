import os
from io import BytesIO

import modal
import numpy as np
from PIL import Image
from pathlib import Path

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

@app.function(
    gpu="T4",
    image=image,
)
def apply_mosaic_with_bool_mask(image: np.ndarray, mask: np.ndarray, intensity: int = 20):
    import cv2  # type: ignore

    h, w = image.shape[:2]
    block_size = max(1, min(intensity, min(h, w)))

    # Step 1: Mosaic the entire image
    small = cv2.resize(image, (w // block_size, h // block_size), interpolation=cv2.INTER_LINEAR)
    mosaic = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)

    # Step 2: Apply mask to blend
    result = image.copy()
    result[mask] = mosaic[mask]
    return result

@app.function(
    gpu="T4",
    image=image,
)
def preserve_privacy_test(
    image_bytes: bytes,
    prompt: str
) -> bytes:
    from io import BytesIO

    import cv2  # type: ignore
    from lang_sam import LangSAM  # type: ignore
    from PIL import Image

    image = Image.open(BytesIO(image_bytes)).convert("RGB")

    model = LangSAM(sam_type="sam2.1_hiera_large")
    results = model.predict(
        images_pil=[image],
        texts_prompt=[prompt],
        box_threshold=0.35,
        text_threshold=0.10,
    )

    img_array = np.array(image)

    for result in results:
        print(f"Found {len(result['masks'])} masks for label: {result['labels']}")
        if len(result["masks"]) == 0:
            print("No masks found for the given prompt.")
            return image_bytes
        print(f"result: {result}")
        for i, mask in enumerate(result["masks"]):
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
            if mask_score < 0.6:
                print(f"Skipping mask {i + 1}/{len(result['masks'])} -> low score.")
                continue
            print(f"Processing mask {i + 1}/{len(result['masks'])}")
            print(f"Mask score: {mask_score}")
            # show bounding box from the result["boxes"]
            box = result["boxes"][i]
            # Draw a rectangle around the bounding box
            cv2.rectangle(
                img_array,
                (int(box[0]), int(box[1])),
                (int(box[2]), int(box[3])),
                (255, 0, 0),  # Blue color in BGR
                2,  # Thickness of the rectangle
            )
            # show label from the result["labels"]
            label = result["labels"][i]
            # Put the label text on the image
            cv2.putText(
                img_array,
                label,
                (int(box[0]), int(box[1] - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,  # Font size
                (255, 0, 0),  # Blue color in BGR
                2,  # Thickness of the text
            )
            # Instead of drawing rectangles, apply pixelation
            mask_bool = mask.astype(bool)
            
            # Apply pixelation to the masked area
            img_array = apply_mosaic_with_bool_mask.remote(img_array, mask_bool)

    
    modified_image = Image.fromarray(img_array)
    # Save the modified image to a BytesIO object
    output_buffer = BytesIO()
    modified_image.save(output_buffer, format="JPEG")
    return output_buffer.getvalue()

@app.local_entrypoint()
def main():
    input_dir = "./src/assets/input"
    output_dir = "./src/assets/output"
    img_name = "test_1.jpg"
    with open(f"{input_dir}/{img_name}", "rb") as f:
        img_bytes = f.read()

    result_bytes = preserve_privacy_test.remote(
        img_bytes,
        "face. license plate"
    )
    os.makedirs(output_dir, exist_ok=True)
    with open(f"{output_dir}/privacy_preserved_{img_name}", "wb") as f:
        f.write(result_bytes)

    print("Image processing completed.")
