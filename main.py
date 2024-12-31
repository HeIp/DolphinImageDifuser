from PIL import Image, ImageChops
import os

def find_matching_clean_image(watermarked_image_path, clean_images_dir):
    """
    Finds the corresponding clean image for a given watermarked image.

    Args:
        watermarked_image_path: Path to the watermarked image.
        clean_images_dir: Directory containing clean images.

    Returns:
        Path to the corresponding clean image if found, otherwise None.
    """
    watermarked_image_filename = os.path.basename(watermarked_image_path)
    for clean_image_filename in os.listdir(clean_images_dir):
        if watermarked_image_filename == clean_image_filename:
            return os.path.join(clean_images_dir, clean_image_filename)
    return None

def remove_watermark(watermarked_image_path, clean_image_path, output_image_path):
    """
    Removes a watermark from an image using a clean reference image.

    Args:
      watermarked_image_path: Path to the image with the watermark.
      clean_image_path: Path to the image without the watermark.
      output_image_path: Path to save the resulting image without the watermark.
    """
    try:
        # Open images using PIL
        with Image.open(watermarked_image_path) as watermarked_image, \
             Image.open(clean_image_path) as clean_image:

            # Ensure images are in the same mode (e.g., RGB)
            if watermarked_image.mode != clean_image.mode:
                clean_image = clean_image.convert(watermarked_image.mode)

            # Calculate the difference between the images
            difference = ImageChops.difference(watermarked_image, clean_image)

            # Invert the difference to get the watermark
            watermark = ImageChops.invert(difference)

            # Subtract the watermark from the original image
            result = ImageChops.subtract(watermarked_image, watermark)

            # Save the resulting image
            result.save(output_image_path)

        print(f"Watermark removed successfully. Saved as {output_image_path}")

    except FileNotFoundError:
        print("Error: One or both image files not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    watermarked_images_dir = 'path/to/watermarked_images'
    clean_images_dir = 'path/to/clean_images'
    output_dir = 'path/to/output_images'

    for watermarked_image_filename in os.listdir(watermarked_images_dir):
        watermarked_image_path = os.path.join(watermarked_images_dir, watermarked_image_filename)
        clean_image_path = find_matching_clean_image(watermarked_image_path, clean_images_dir)

        if clean_image_path:
            output_image_path = os.path.join(output_dir, watermarked_image_filename)
            remove_watermark(watermarked_image_path, clean_image_path, output_image_path)
        else:
            print(f"Warning: No matching clean image found for {watermarked_image_filename}")