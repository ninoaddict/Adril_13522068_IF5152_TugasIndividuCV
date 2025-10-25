# Nama: Adril Putra Merin
# NIM: 13522068
# Fitur unik: Implementasi multi-filter dengan visualisasi perbandingan dan analisis parameter

import cv2
import numpy as np
from skimage import data, filters
import matplotlib.pyplot as plt
import pandas as pd
import os


class ImageFiltering:
    def __init__(self, output_dir="generated"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.results = []

    def load_standard_images(self):
        """Load images from skimage"""
        images = {
            "cameraman": data.camera(),
            "coins": data.coins(),
            "checkerboard": data.checkerboard(),
            "astronaut": data.astronaut(),
        }
        return images

    def gaussian_filter(self, image, sigma=1.0):
        """Apply Gaussian filter"""
        if len(image.shape) == 3:
            # convert to grayscale if color
            image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            image_gray = image

        filtered = filters.gaussian(image_gray, sigma=sigma)
        return filtered

    def median_filter(self, image, size=5):
        """Apply Median filter"""
        if len(image.shape) == 3:
            image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            image_gray = image

        # convert to uint8 for cv2.medianBlur
        img_uint8 = (
            (image_gray * 255).astype(np.uint8)
            if image_gray.max() <= 1
            else image_gray.astype(np.uint8)
        )
        filtered = cv2.medianBlur(img_uint8, size)
        return filtered / 255.0

    def sobel_filter(self, image):
        """Apply Sobel filter for edge enhancement"""
        if len(image.shape) == 3:
            image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            image_gray = image

        sobel_x = filters.sobel_h(image_gray)
        sobel_y = filters.sobel_v(image_gray)
        sobel_combined = np.sqrt(sobel_x**2 + sobel_y**2)
        return sobel_combined

    def process_image(self, image, image_name, gaussian_sigma=1.5, median_size=5):
        """Process single image with all filters"""
        print(f"Processing {image_name}...")

        # apply filters
        gaussian_result = self.gaussian_filter(image, sigma=gaussian_sigma)
        median_result = self.median_filter(image, size=median_size)
        sobel_result = self.sobel_filter(image)

        # save results
        self.save_comparison(
            image, gaussian_result, median_result, sobel_result, image_name
        )

        # record parameters
        self.results.append(
            {
                "Image": image_name,
                "Filter": "Gaussian",
                "Parameter": f"sigma={gaussian_sigma}",
                "Output_Size": gaussian_result.shape,
            }
        )
        self.results.append(
            {
                "Image": image_name,
                "Filter": "Median",
                "Parameter": f"kernel_size={median_size}",
                "Output_Size": median_result.shape,
            }
        )
        self.results.append(
            {
                "Image": image_name,
                "Filter": "Sobel",
                "Parameter": "default",
                "Output_Size": sobel_result.shape,
            }
        )

        return gaussian_result, median_result, sobel_result

    def save_comparison(self, original, gaussian, median, sobel, image_name):
        """Save before-after comparison"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 12))

        # Original
        if len(original.shape) == 3:
            axes[0, 0].imshow(original)
        else:
            axes[0, 0].imshow(original, cmap="gray")
        axes[0, 0].set_title("Original")
        axes[0, 0].axis("off")

        # Gaussian
        axes[0, 1].imshow(gaussian, cmap="gray")
        axes[0, 1].set_title("Gaussian Filter")
        axes[0, 1].axis("off")

        # Median
        axes[1, 0].imshow(median, cmap="gray")
        axes[1, 0].set_title("Median Filter")
        axes[1, 0].axis("off")

        # Sobel
        axes[1, 1].imshow(sobel, cmap="gray")
        axes[1, 1].set_title("Sobel Filter")
        axes[1, 1].axis("off")

        plt.tight_layout()
        plt.savefig(
            f"{self.output_dir}/{image_name}_filtering_comparison.png",
            dpi=150,
            bbox_inches="tight",
        )
        plt.close()

        print(f"Saved: {image_name}_filtering_comparison.png")

    def save_parameters_table(self):
        """Save parameters to CSV"""
        df = pd.DataFrame(self.results)
        csv_path = f"{self.output_dir}/filtering_parameters.csv"
        df.to_csv(csv_path, index=False)
        print(f"\nParameter table saved to: {csv_path}")
        print("\nParameter Summary:")
        print(df.to_string(index=False))

    def run_pipeline(self):
        """Run complete filtering pipeline"""
        print("=" * 60)
        print("IMAGE FILTERING PIPELINE")
        print("=" * 60)

        # load standard images
        images = self.load_standard_images()

        # process each image
        for name, img in images.items():
            self.process_image(img, name)

        # save parameter table
        self.save_parameters_table()

        print("\n" + "=" * 60)
        print("FILTERING PIPELINE COMPLETED!")
        print(f"Results saved in: {self.output_dir}/")
        print("=" * 60)


def main():
    filter_pipeline = ImageFiltering()
    filter_pipeline.run_pipeline()


if __name__ == "__main__":
    main()
