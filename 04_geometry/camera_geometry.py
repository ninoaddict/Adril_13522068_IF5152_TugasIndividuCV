# Nama: Adril Putra Merin
# NIM: 13522068
# Fitur unik: Camera calibration dengan checkerboard dan transformasi geometri

import cv2
import numpy as np
from skimage import data
import matplotlib.pyplot as plt
import pandas as pd
import os


class CameraGeometry:
    def __init__(self, output_dir="generated"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.calibration_results = []

    def load_standard_images(self):
        """Load images from skimage"""
        images = {
            "cameraman": {"image": data.camera(), "pattern_size": (7, 7)},
            "coins": {"image": data.coins(), "pattern_size": (5, 3)},
            "checkerboard": {"image": data.checkerboard(), "pattern_size": (7, 7)},
            "astronaut": {"image": data.astronaut(), "pattern_size": (7, 7)},
        }
        return images

    def to_gray_scale(self, image):
        if len(image.shape) == 3:
            final_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            final_image = image
        return final_image

    def to_uint8_img(self, image):
        img_uint8 = (
            (image * 255).astype(np.uint8)
            if image.max() <= 1
            else image.astype(np.uint8)
        )
        return img_uint8

    def detect_checkerboard_corners(self, image, pattern_size=(7, 7)):
        """Detect corners in checkerboard pattern"""
        image_gray = self.to_gray_scale(image)
        image_gray = self.to_uint8_img(image_gray)

        # Find checkerboard corners
        ret, corners = cv2.findChessboardCorners(image_gray, pattern_size, None)

        if ret:
            # Refine corner locations
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners_refined = cv2.cornerSubPix(
                image_gray, corners, (11, 11), (-1, -1), criteria
            )
            return ret, corners_refined

        return ret, None

    def camera_calibration(self, image, pattern_size=(7, 7), square_size=1.0):
        """Perform camera calibration using checkerboard"""
        ret, corners = self.detect_checkerboard_corners(image, pattern_size)

        if not ret:
            print("  Checkerboard corners not detected")
            return None, None, None

        # Prepare object points (3D points in real world space)
        objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0 : pattern_size[0], 0 : pattern_size[1]].T.reshape(
            -1, 2
        )
        objp *= square_size

        # Prepare points for calibration
        obj_points = [objp]
        img_points = [corners]

        # Camera calibration
        image_gray = self.to_gray_scale(image)
        image_gray = self.to_uint8_img(image_gray)

        ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
            obj_points, img_points, image_gray.shape[::-1], None, None
        )

        return camera_matrix, dist_coeffs, corners

    def apply_perspective_transform(self, image, transform_type="rotation"):
        """Apply perspective transformation"""
        img = self.to_gray_scale(image)
        img = self.to_uint8_img(img)

        h, w = img.shape

        if transform_type == "rotation":
            # Rotation matrix
            center = (w // 2, h // 2)
            angle = 30
            scale = 1.0
            M = cv2.getRotationMatrix2D(center, angle, scale)
            transformed = cv2.warpAffine(img, M, (w, h))

        elif transform_type == "perspective":
            # Perspective transformation
            pts1 = np.float32([[50, 50], [w - 50, 50], [50, h - 50], [w - 50, h - 50]])
            pts2 = np.float32(
                [[10, 100], [w - 10, 50], [100, h - 10], [w - 50, h - 100]]
            )
            M = cv2.getPerspectiveTransform(pts1, pts2)
            transformed = cv2.warpPerspective(img, M, (w, h))

        else:  # affine
            pts1 = np.float32([[50, 50], [w - 50, 50], [50, h - 50]])
            pts2 = np.float32([[10, 100], [w - 10, 50], [100, h - 10]])
            M = cv2.getAffineTransform(pts1, pts2)
            transformed = cv2.warpAffine(img, M, (w, h))

        return transformed, M

    def visualize_calibration(
        self, image, corners, camera_matrix, image_name, pattern_size=(7, 7)
    ):
        """Visualize calibration results"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))

        # Original with detected corners
        if len(image.shape) == 3:
            img_with_corners = image.copy()
        else:
            img_with_corners = self.to_uint8_img(image)
            img_with_corners = cv2.cvtColor(img_with_corners, cv2.COLOR_GRAY2RGB)

        if corners is not None:
            cv2.drawChessboardCorners(img_with_corners, pattern_size, corners, True)

        axes[0].imshow(img_with_corners, cmap="gray")
        axes[0].set_title("Detected Checkerboard Corners")
        axes[0].axis("off")

        # Camera matrix visualization
        if camera_matrix is not None:
            axes[1].text(0.1, 0.7, "Camera Matrix:", fontsize=12, weight="bold")
            axes[1].text(0.1, 0.5, str(camera_matrix), fontsize=10, family="monospace")
            axes[1].text(
                0.1, 0.2, f"Focal Length (fx): {camera_matrix[0, 0]:.2f}", fontsize=10
            )
            axes[1].text(
                0.1, 0.1, f"Focal Length (fy): {camera_matrix[1, 1]:.2f}", fontsize=10
            )
        else:
            axes[1].text(
                0.5, 0.5, "Calibration Failed", ha="center", va="center", fontsize=14
            )
        axes[1].axis("off")

        plt.tight_layout()
        plt.savefig(
            f"{self.output_dir}/{image_name}_calibration.png",
            dpi=150,
            bbox_inches="tight",
        )
        plt.close()

        print(f"Saved: {image_name}_calibration.png")

    def visualize_transformations(
        self, original, rotated, perspective, affine, image_name
    ):
        """Visualize geometric transformations"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 12))

        axes[0, 0].imshow(original, cmap="gray")
        axes[0, 0].set_title("Original Image")
        axes[0, 0].axis("off")

        axes[0, 1].imshow(rotated, cmap="gray")
        axes[0, 1].set_title("Rotation (30°)")
        axes[0, 1].axis("off")

        axes[1, 0].imshow(perspective, cmap="gray")
        axes[1, 0].set_title("Perspective Transform")
        axes[1, 0].axis("off")

        axes[1, 1].imshow(affine, cmap="gray")
        axes[1, 1].set_title("Affine Transform")
        axes[1, 1].axis("off")

        plt.tight_layout()
        plt.savefig(
            f"{self.output_dir}/{image_name}_transformations.png",
            dpi=150,
            bbox_inches="tight",
        )
        plt.close()

        print(f"Saved: {image_name}_transformations.png")

    def process_image(self, image, image_name, pattern_size):
        """Process image for calibration and transformation"""
        print(f"\nProcessing {image_name}...")

        camera_matrix, dist_coeffs, corners = self.camera_calibration(
            image, pattern_size=pattern_size
        )

        # Visualize calibration
        self.visualize_calibration(
            image, corners, camera_matrix, image_name, pattern_size
        )

        # Apply transformations
        rotated, rot_matrix = self.apply_perspective_transform(image, "rotation")
        perspective, persp_matrix = self.apply_perspective_transform(
            image, "perspective"
        )
        affine_trans, affine_matrix = self.apply_perspective_transform(image, "affine")

        # Visualize transformations
        img_gray = self.to_gray_scale(image)
        self.visualize_transformations(
            img_gray, rotated, perspective, affine_trans, image_name
        )

        # Record results
        result = {
            "Image": image_name,
            "Calibration_Success": camera_matrix is not None,
            "Corners_Detected": corners is not None,
            "Focal_X": camera_matrix[0, 0] if camera_matrix is not None else "N/A",
            "Focal_Y": camera_matrix[1, 1] if camera_matrix is not None else "N/A",
            "Principal_Point_X": (
                camera_matrix[0, 2] if camera_matrix is not None else "N/A"
            ),
            "Principal_Point_Y": (
                camera_matrix[1, 2] if camera_matrix is not None else "N/A"
            ),
        }
        self.calibration_results.append(result)

        # Save transformation matrices
        self.save_transformation_matrices(
            image_name, rot_matrix, persp_matrix, affine_matrix, camera_matrix
        )

        return camera_matrix, corners

    def save_transformation_matrices(
        self, image_name, rot_matrix, persp_matrix, affine_matrix, camera_matrix
    ):
        """Save transformation matrices to file"""
        filepath = f"{self.output_dir}/{image_name}_matrices.txt"

        with open(filepath, "w") as f:
            f.write(f"Transformation Matrices for {image_name}\n")
            f.write("=" * 60 + "\n\n")

            f.write("Rotation Matrix (30 degrees):\n")
            f.write(str(rot_matrix) + "\n\n")

            f.write("Perspective Transform Matrix:\n")
            f.write(str(persp_matrix) + "\n\n")

            f.write("Affine Transform Matrix:\n")
            f.write(str(affine_matrix) + "\n\n")

            if camera_matrix is not None:
                f.write("Camera Matrix:\n")
                f.write(str(camera_matrix) + "\n\n")
            else:
                f.write("Camera Matrix: Not computed (no checkerboard detected)\n\n")

        print(f"Saved: {image_name}_matrices.txt")

    def save_calibration_summary(self):
        """Save calibration summary to CSV"""
        df = pd.DataFrame(self.calibration_results)
        csv_path = f"{self.output_dir}/geometry_parameters.csv"
        df.to_csv(csv_path, index=False)
        print(f"\nCalibration summary saved to: {csv_path}")
        print("\nGeometry & Calibration Summary:")
        print(df.to_string(index=False))

    def run_pipeline(self):
        """Run complete geometry and calibration pipeline"""
        print("=" * 60)
        print("CAMERA GEOMETRY & CALIBRATION PIPELINE")
        print("=" * 60)

        # Load standard images
        images = self.load_standard_images()

        # Process each image
        for name, info in images.items():
            self.process_image(info["image"], name, info["pattern_size"])

        # Save calibration summary
        self.save_calibration_summary()

        print("\n" + "=" * 60)
        print("GEOMETRY PIPELINE COMPLETED!")
        print(f"Results saved in: {self.output_dir}/")
        print("=" * 60)


def main():
    geometry_pipeline = CameraGeometry()
    geometry_pipeline.run_pipeline()


if __name__ == "__main__":
    main()
