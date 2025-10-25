# Nama: Adril Putra Merin
# NIM: 13522068
# Fitur unik: Multi-method edge detection dengan analisis threshold dan perbandingan

import cv2
import numpy as np
from skimage import data, feature, filters
import matplotlib.pyplot as plt
import pandas as pd
import os


class EdgeDetection:
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

    def sobel_edge(self, image):
        """Sobel edge detection"""
        if len(image.shape) == 3:
            image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            image_gray = image

        # normalize to 0-255
        if image_gray.max() <= 1:
            image_gray = (image_gray * 255).astype(np.uint8)

        sobel_x = cv2.Sobel(image_gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(image_gray, cv2.CV_64F, 0, 1, ksize=3)
        sobel_combined = np.sqrt(sobel_x**2 + sobel_y**2)
        sobel_combined = (sobel_combined / sobel_combined.max() * 255).astype(np.uint8)

        return sobel_combined

    def canny_edge(self, image, low_threshold=50, high_threshold=150):
        """Canny edge detection"""
        if len(image.shape) == 3:
            image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            image_gray = image

        # normalize to 0-255
        if image_gray.max() <= 1:
            image_gray = (image_gray * 255).astype(np.uint8)

        edges = cv2.Canny(image_gray, low_threshold, high_threshold)
        return edges

    def process_with_multiple_thresholds(self, image, image_name):
        """Process image with multiple threshold values"""
        print(f"\nProcessing {image_name} with multiple thresholds...")

        # sobel (no threshold needed)
        sobel_result = self.sobel_edge(image)

        # canny with different thresholds
        canny_results = {}
        threshold_configs = [(30, 90), (50, 150), (100, 200)]

        for low, high in threshold_configs:
            canny_results[f"Canny_L{low}_H{high}"] = self.canny_edge(image, low, high)
            self.results.append(
                {
                    "Image": image_name,
                    "Method": "Canny",
                    "Low_Threshold": low,
                    "High_Threshold": high,
                    "Edge_Pixels": np.count_nonzero(
                        canny_results[f"Canny_L{low}_H{high}"]
                    ),
                }
            )

        self.results.append(
            {
                "Image": image_name,
                "Method": "Sobel",
                "Low_Threshold": "N/A",
                "High_Threshold": "N/A",
                "Edge_Pixels": np.count_nonzero(sobel_result > 50),
            }
        )

        return sobel_result, canny_results

    def save_edge_comparison(self, original, sobel, canny_results, image_name):
        """Save edge detection comparison"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        # original
        if len(original.shape) == 3:
            axes[0, 0].imshow(original)
        else:
            axes[0, 0].imshow(original, cmap="gray")
        axes[0, 0].set_title("Original Image")
        axes[0, 0].axis("off")

        # sobel
        axes[0, 1].imshow(sobel, cmap="gray")
        axes[0, 1].set_title("Sobel Edge Detection")
        axes[0, 1].axis("off")

        # canny results
        canny_keys = list(canny_results.keys())
        positions = [(0, 2), (1, 0), (1, 1), (1, 2)]

        for idx, key in enumerate(canny_keys):
            row, col = positions[idx + 1]
            axes[row, col].imshow(canny_results[key], cmap="gray")
            axes[row, col].set_title(key.replace("_", " "))
            axes[row, col].axis("off")

        plt.tight_layout()
        plt.savefig(
            f"{self.output_dir}/{image_name}_edge_comparison.png",
            dpi=150,
            bbox_inches="tight",
        )
        plt.close()

        print(f"Saved: {image_name}_edge_comparison.png")

    def analyze_sampling_effect(self, image, image_name):
        """Analyze effect of image sampling on edge detection"""
        print(f"\nAnalyzing sampling effect on {image_name}...")

        if len(image.shape) == 3:
            image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            image_gray = image

        if image_gray.max() <= 1:
            image_gray = (image_gray * 255).astype(np.uint8)

        sampling_results = {}
        sampling_factors = [1, 2, 4]

        fig, axes = plt.subplots(len(sampling_factors), 3, figsize=(12, 12))

        for idx, factor in enumerate(sampling_factors):
            # downsample
            h, w = image_gray.shape
            sampled = cv2.resize(image_gray, (w // factor, h // factor))

            # edge detection on sampled image
            sobel_sampled = self.sobel_edge(sampled)
            canny_sampled = self.canny_edge(sampled, 50, 150)

            # upsample back for comparison
            sobel_upsampled = cv2.resize(sobel_sampled, (w, h))
            canny_upsampled = cv2.resize(canny_sampled, (w, h))

            # plot
            axes[idx, 0].imshow(sampled, cmap="gray")
            axes[idx, 0].set_title(f"Sampled 1/{factor}")
            axes[idx, 0].axis("off")

            axes[idx, 1].imshow(sobel_upsampled, cmap="gray")
            axes[idx, 1].set_title(f"Sobel (1/{factor})")
            axes[idx, 1].axis("off")

            axes[idx, 2].imshow(canny_upsampled, cmap="gray")
            axes[idx, 2].set_title(f"Canny (1/{factor})")
            axes[idx, 2].axis("off")

        plt.tight_layout()
        plt.savefig(
            f"{self.output_dir}/{image_name}_sampling_analysis.png",
            dpi=150,
            bbox_inches="tight",
        )
        plt.close()

        print(f"Saved: {image_name}_sampling_analysis.png")

    def save_parameters_table(self):
        """Save parameters and results to CSV"""
        df = pd.DataFrame(self.results)
        csv_path = f"{self.output_dir}/edge_parameters.csv"
        df.to_csv(csv_path, index=False)
        print(f"\nParameter table saved to: {csv_path}")
        print("\nEdge Detection Summary:")
        print(df.to_string(index=False))

    def run_pipeline(self):
        """Run complete edge detection pipeline"""
        print("=" * 60)
        print("EDGE DETECTION PIPELINE")
        print("=" * 60)

        # load standard images
        images = self.load_standard_images()

        # process each image
        for name, img in images.items():
            sobel, canny_results = self.process_with_multiple_thresholds(img, name)
            self.save_edge_comparison(img, sobel, canny_results, name)
            self.analyze_sampling_effect(img, name)

        # save parameter table
        self.save_parameters_table()

        print("\n" + "=" * 60)
        print("EDGE DETECTION PIPELINE COMPLETED!")
        print(f"Results saved in: {self.output_dir}/")
        print("=" * 60)


def main():
    edge_pipeline = EdgeDetection()
    edge_pipeline.run_pipeline()


if __name__ == "__main__":
    main()
