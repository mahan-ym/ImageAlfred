import datetime
import os

import modal

if __name__ == "__main__":
    input_dir = "./src/assets/examples"
    output_dir = "./test/output"
    img_name = "test_1.jpg"
    with open(f"{input_dir}/{img_name}", "rb") as f:
        img_bytes = f.read()

    f = modal.Function.from_name("ImageAlfred", "change_image_objects_hsv")
    result = f.remote(
        img_bytes,
        [["hair", 10, 1.2], ["shirt", 60, 1.0], ["pants", 150, 0.8]],
    )

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"{timestamp}_{img_name}"

    output_path = f"{output_dir}/{output_filename}"
    with open(output_path, "wb") as f:
        f.write(result)

    print(f"Image saved to {output_path}")
