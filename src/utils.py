import numpy as np
from PIL import Image


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

def cast_langsam_prompt(user_input: list) -> str:
    return ". ".join([f"{obj[0]}" for obj in user_input])