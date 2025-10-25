# Nama: Adril Putra Merin
# NIM: 13522068
# Fitur unik: Multi-detector feature points dengan analisis statistik dan visualisasi

import cv2
import numpy as np
from skimage import data, feature
import matplotlib.pyplot as plt
import pandas as pd
import os


class FeaturePointDetection:
    def __init__(self, output_dir="generated"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.statistics = []

    def load_standard_images(self):
        """Load images from skimage"""
        images = {
            "cameraman": data.camera(),
            "coins": data.coins(),
            "checkerboard": data.checkerboard(),
            "astronaut": data.astronaut(),
        }
        return images

    def harris_corner_detection(self, image, threshold=0.01):
        """Harris corner detection"""
        if len(image.shape) == 3:
            image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            image_gray = image

        if image_gray.max() <= 1:
            image_gray = (image_gray * 255).astype(np.uint8)
        else:
            image_gray = image_gray.astype(np.uint8)

        # Harris corner detection
        harris_response = cv2.cornerHarris(image_gray.astype(np.float32), 2, 3, 0.04)
        harris_response = cv2.dilate(harris_response, None)

        # Threshold for corner detection
        threshold_value = threshold * harris_response.max()
        corners = np.argwhere(harris_response > threshold_value)

        return corners, harris_response

    def fast_feature_detection(self, image, threshold=10):
        """FAST feature detection"""
        if len(image.shape) == 3:
            image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            image_gray = image

        if image_gray.max() <= 1:
            image_gray = (image_gray * 255).astype(np.uint8)
        else:
            image_gray = image_gray.astype(np.uint8)

        # FAST detector
        fast = cv2.FastFeatureDetector_create(threshold=threshold)
        keypoints = fast.detect(image_gray, None)

        # Convert keypoints to array
        points = np.array([kp.pt for kp in keypoints])

        return points, keypoints

    def sift_feature_detection(self, image):
        """SIFT feature detection"""
        if len(image.shape) == 3:
            image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            image_gray = image

        if image_gray.max() <= 1:
            image_gray = (image_gray * 255).astype(np.uint8)
        else:
            image_gray = image_gray.astype(np.uint8)

        # SIFT detector
        sift = cv2.SIFT_create()
        keypoints, descriptors = sift.detectAndCompute(image_gray, None)

        # Convert keypoints to array
        points = np.array([kp.pt for kp in keypoints])

        return points, keypoints, descriptors

    def mark_features_on_image(
        self, image, harris_corners, fast_points, sift_points, image_name
    ):
        """Visualize detected features on image"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 14))

        # Original
        if len(image.shape) == 3:
            axes[0, 0].imshow(image)
        else:
            axes[0, 0].imshow(image, cmap="gray")
        axes[0, 0].set_title("Original Image")
        axes[0, 0].axis("off")

        # Harris corners
        if len(image.shape) == 3:
            axes[0, 1].imshow(image)
        else:
            axes[0, 1].imshow(image, cmap="gray")
        if len(harris_corners) > 0:
            axes[0, 1].scatter(
                harris_corners[:, 1],
                harris_corners[:, 0],
                c="red",
                s=20,
                marker="x",
                alpha=0.6,
            )
        axes[0, 1].set_title(f"Harris Corners ({len(harris_corners)} points)")
        axes[0, 1].axis("off")

        # FAST features
        if len(image.shape) == 3:
            axes[1, 0].imshow(image)
        else:
            axes[1, 0].imshow(image, cmap="gray")
        if len(fast_points) > 0:
            axes[1, 0].scatter(
                fast_points[:, 0],
                fast_points[:, 1],
                c="green",
                s=20,
                marker="o",
                alpha=0.6,
            )
        axes[1, 0].set_title(f"FAST Features ({len(fast_points)} points)")
        axes[1, 0].axis("off")

        # SIFT features
        if len(image.shape) == 3:
            axes[1, 1].imshow(image)
        else:
            axes[1, 1].imshow(image, cmap="gray")
        if len(sift_points) > 0:
            axes[1, 1].scatter(
                sift_points[:, 0],
                sift_points[:, 1],
                c="blue",
                s=20,
                marker="+",
                alpha=0.6,
            )
        axes[1, 1].set_title(f"SIFT Features ({len(sift_points)} points)")
        axes[1, 1].axis("off")

        plt.tight_layout()
        plt.savefig(
            f"{self.output_dir}/{image_name}_feature_marking.png",
            dpi=150,
            bbox_inches="tight",
        )
        plt.close()

        print(f"Saved: {image_name}_feature_marking.png")

    def calculate_statistics(
        self,
        image_name,
        harris_corners,
        harris_response,
        fast_points,
        sift_points,
        sift_descriptors,
    ):
        """Calculate and record statistics"""
        stats = {
            "Image": image_name,
            "Harris_Count": len(harris_corners),
            "Harris_Max_Response": (
                float(harris_response.max()) if harris_response is not None else 0
            ),
            "Harris_Mean_Response": (
                float(harris_response.mean()) if harris_response is not None else 0
            ),
            "FAST_Count": len(fast_points),
            "SIFT_Count": len(sift_points),
            "SIFT_Descriptor_Dim": (
                sift_descriptors.shape[1] if sift_descriptors is not None else 0
            ),
        }

        self.statistics.append(stats)
        return stats

    def process_image(self, image, image_name):
        """Process single image with all feature detectors"""
        print(f"\nProcessing {image_name}...")

        # Detect features
        harris_corners, harris_response = self.harris_corner_detection(image)
        fast_points, fast_kp = self.fast_feature_detection(image)
        sift_points, sift_kp, sift_desc = self.sift_feature_detection(image)

        # Visualize
        self.mark_features_on_image(
            image, harris_corners, fast_points, sift_points, image_name
        )

        # Calculate statistics
        stats = self.calculate_statistics(
            image_name,
            harris_corners,
            harris_response,
            fast_points,
            sift_points,
            sift_desc,
        )

        print(f"  Harris: {stats['Harris_Count']} corners")
        print(f"  FAST: {stats['FAST_Count']} features")
        print(f"  SIFT: {stats['SIFT_Count']} keypoints")

        return harris_corners, fast_points, sift_points

    def save_statistics(self):
        """Save statistics to CSV"""
        df = pd.DataFrame(self.statistics)
        csv_path = f"{self.output_dir}/feature_statistics.csv"
        df.to_csv(csv_path, index=False)
        print(f"\nStatistics saved to: {csv_path}")
        print("\nFeature Detection Statistics:")
        print(df.to_string(index=False))

        # Create summary visualization
        self.plot_statistics_comparison(df)

    def plot_statistics_comparison(self, df):
        """Plot comparison of feature counts"""
        fig, ax = plt.subplots(figsize=(10, 6))

        x = np.arange(len(df))
        width = 0.25

        ax.bar(x - width, df["Harris_Count"], width, label="Harris", alpha=0.8)
        ax.bar(x, df["FAST_Count"], width, label="FAST", alpha=0.8)
        ax.bar(x + width, df["SIFT_Count"], width, label="SIFT", alpha=0.8)

        ax.set_xlabel("Image")
        ax.set_ylabel("Number of Features")
        ax.set_title("Feature Count Comparison")
        ax.set_xticks(x)
        ax.set_xticklabels(df["Image"], rotation=45, ha="right")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            f"{self.output_dir}/feature_comparison.png", dpi=150, bbox_inches="tight"
        )
        plt.close()

        print(f"Saved: feature_comparison.png")

    def run_pipeline(self):
        """Run complete feature detection pipeline"""
        print("=" * 60)
        print("FEATURE POINT DETECTION PIPELINE")
        print("=" * 60)

        # Load standard images
        images = self.load_standard_images()

        # Process each image
        for name, img in images.items():
            self.process_image(img, name)

        # Save statistics
        self.save_statistics()

        print("\n" + "=" * 60)
        print("FEATURE DETECTION PIPELINE COMPLETED!")
        print(f"Results saved in: {self.output_dir}/")
        print("=" * 60)


def main():
    feature_pipeline = FeaturePointDetection()
    feature_pipeline.run_pipeline()


if __name__ == "__main__":
    main()
