import base64
from enum import Enum
from io import BytesIO

import requests
from PIL import Image


class ImageSource(Enum):
    """Enum representing different sources of image input"""

    PIL = "PIL"
    BASE64 = "base64"
    URL = "url"
    FILE = "file"


def validate_image_input(url_or_data):
    """Handle different image input formats for MCP"""
    if isinstance(url_or_data, Image.Image):
        print("Received input image type: PIL.Image")
        return (ImageSource.PIL, url_or_data)

    if isinstance(url_or_data, str):
        if url_or_data.startswith("data:image"):
            try:
                # Handle base64 data URLs
                print("Received input image type: base64 data")
                header, encoded = url_or_data.split(",", 1)
                decoded_bytes = base64.b64decode(encoded)
                return (
                    ImageSource.BASE64,
                    Image.open(BytesIO(decoded_bytes)).convert("RGB"),
                )
            except Exception as e:
                raise ValueError(f"Invalid base64 data URL: {e}")
        elif url_or_data.startswith(("http://", "https://")):
            # Handle URLs
            try:
                response = requests.get(url_or_data, timeout=30)
                response.raise_for_status()
                print("Received input image type: URL")
                return (
                    ImageSource.URL,
                    Image.open(BytesIO(response.content)).convert("RGB"),
                )
            except requests.exceptions.RequestException as e:
                raise ValueError(f"Could not download image from URL: {e}")
        else:
            # Handle file paths
            try:
                with open(url_or_data, "rb") as f:
                    return (ImageSource.FILE, Image.open(f).convert("RGB"))
            except FileNotFoundError:
                raise ValueError(f"File not found: {url_or_data}")
            except Exception as e:
                raise ValueError(f"Could not read file {url_or_data}: {e}")

    raise ValueError(f"Unsupported image input format: {type(url_or_data)}")


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
    print(f"Response from tmpfiles.org: {data}")
    if "data" in data and "url" in data["data"]:
        url = data["data"]["url"]
        if not url:
            raise ValueError("Invalid URL in response")
        print(f"Uploaded image URL: {url}")
        return url
    else:
        raise ValueError(f"Invalid response: {data}")
