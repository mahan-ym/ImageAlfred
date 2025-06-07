import base64
from enum import Enum
from io import BytesIO

import requests
from PIL import Image


def upload_image_to_tmpfiles(image):
    """
    Upload an image to tmpfiles.org and return the URL.
    """

    img_byte_arr = BytesIO()
    image.save(img_byte_arr, format="PNG")
    img_byte_arr.seek(0)

    files = {"file": ("image.png", img_byte_arr, "image/png")}
    response = requests.post("https://tmpfiles.org/api/v1/upload", files=files)

    if response.status_code != 200:
        raise ValueError(f"Upload failed: Status {response.status_code}")

    data = response.json()
    if "data" in data and "url" in data["data"]:
        url = data["data"]["url"]
        if not url:
            raise ValueError("Invalid URL in response")
        print(f"Uploaded image URL: {url}")
        return url
    else:
        raise ValueError(f"Invalid response: {data}")
