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
        return (ImageSource.PIL, url_or_data)

    if isinstance(url_or_data, str):
        if url_or_data.startswith("data:image"):
            try:
                # Handle base64 data URLs
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
