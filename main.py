if __name__ == "__main__":
    clean_images_dir = '/NoWaterMark'
    watermarked_samples_dir = '/YesWaterMark'
    watermarked_images_dir = '/SealWaterMark'
    output_dir = '/NoSealWaterMark'

    # Accumulate watermark information from multiple samples
    watermark_data = []

    for filename in os.listdir(clean_images_dir):
        clean_image_path = os.path.join(clean_images_dir, filename)
        watermarked_image_path = os.path.join(watermarked_samples_dir, filename)

        # Check if corresponding files exist
        if os.path.isfile(clean_image_path) and os.path.isfile(watermarked_image_path):
            try:
                watermark = find_watermark(watermarked_image_path, clean_image_path)
                if watermark:
                    watermark_data.append(watermark)
            except Exception as e:
                print(f"Error processing sample pair {filename}: {e}")

    # If watermark data is collected, proceed with watermark removal
    if watermark_data:
        for watermarked_image_filename in os.listdir(watermarked_images_dir):
            watermarked_image_path = os.path.join(watermarked_images_dir, watermarked_image_filename)
            output_image_path = os.path.join(output_dir, watermarked_image_filename)

            # Consider averaging watermark data for potentially better results
            average_watermark = sum(image.point for image in watermark_data) / len(watermark_data)
            remove_watermark(watermarked_image_path, average_watermark, output_image_path)
    else:
        print("Error: Could not calculate watermark from any sample image pairs.")