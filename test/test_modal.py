import datetime
import os

import modal
from PIL import Image

if __name__ == "__main__":
    input_dir = "./src/assets/examples"
    output_dir = "./test/output"
    img_name = "test_1.jpg"

    image_pil = Image.open(f"{input_dir}/{img_name}")
    f = modal.Function.from_name("ImageAlfred", "change_image_objects_hsv")
    result = f.remote(
        image_pil,
        [["hair", 10, 1.2], ["shirt", 60, 1.0], ["pants", 150, 0.8]],
    )

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"{timestamp}_{img_name}"

    output_path = f"{output_dir}/{output_filename}"
    image_pil.save(output_path)

    print(f"Image saved to {output_path}")
