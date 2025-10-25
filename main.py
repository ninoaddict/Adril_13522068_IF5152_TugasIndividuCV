# Nama: Adril Putra Merin
# NIM: 13522068
# Fitur unik: Integrated Computer Vision Pipeline dengan automated workflow

import importlib
import sys
import os

# can import module automatically since the folder names in in format <number_featurename>/
# the workaround is to manually add to path and use importlib
script_dir = os.path.dirname(os.path.abspath(__file__))

# import ImageFiltering
filtering_path = os.path.join(script_dir, "01_filtering")
sys.path.append(filtering_path)
image_filtering_module = importlib.import_module("image_filtering")
ImageFiltering = image_filtering_module.ImageFiltering

# import EdgeDetection
edge_path = os.path.join(script_dir, "02_edge")
sys.path.append(edge_path)
edge_detection_module = importlib.import_module("edge_detection")
EdgeDetection = edge_detection_module.EdgeDetection

# import FeaturePointDetection
feature_path = os.path.join(script_dir, "03_featurepoints")
sys.path.append(feature_path)
feature_point_detection_module = importlib.import_module("feature_point_detection")
FeaturePointDetection = feature_point_detection_module.FeaturePointDetection

# import CameraGeometry
geometry_path = os.path.join(script_dir, "04_geometry")
sys.path.append(geometry_path)
camera_geometry_module = importlib.import_module("camera_geometry")
CameraGeometry = camera_geometry_module.CameraGeometry


def run_complete_pipeline():
    """Run all four components of the CV pipeline"""
    print("\n" + "=" * 70)
    print(" " * 15 + "COMPUTER VISION PIPELINE")
    print(" " * 10 + "Tugas Individu IF5152 - Minggu 3-6")
    print("=" * 70)

    components = [
        ("01_filtering", "Image Filtering"),
        ("02_edge", "Edge Detection"),
        ("03_featurepoints", "Feature Point Detection"),
        ("04_geometry", "Camera Geometry & Calibration"),
    ]

    print("\nPipeline Components:")
    for idx, (folder, name) in enumerate(components, 1):
        print(f"  {idx}. {name} → {folder}/")

    print("\n" + "-" * 70)
    input("Press Enter to start the pipeline...")
    print()

    # Component 1: Image Filtering
    try:
        print("\n[1/4] Running Image Filtering...")
        filter_pipeline = ImageFiltering(output_dir="01_filtering/generated")
        filter_pipeline.run_pipeline()
        print("Filtering completed successfully")
    except Exception as e:
        print(f"Error in filtering: {e}")

    # Component 2: Edge Detection
    try:
        print("\n[2/4] Running Edge Detection...")
        edge_pipeline = EdgeDetection(output_dir="02_edge/generated")
        edge_pipeline.run_pipeline()
        print("Edge detection completed successfully")
    except Exception as e:
        print(f"Error in edge detection: {e}")

    # Component 3: Feature Point Detection
    try:
        print("\n[3/4] Running Feature Point Detection...")
        feature_pipeline = FeaturePointDetection(
            output_dir="03_featurepoints/generated"
        )
        feature_pipeline.run_pipeline()
        print("Feature detection completed successfully")
    except Exception as e:
        print(f"Error in feature detection: {e}")

    # Component 4: Camera Geometry & Calibration
    try:
        print("\n[4/4] Running Camera Geometry & Calibration...")
        geometry_pipeline = CameraGeometry(output_dir="04_geometry/generated")
        geometry_pipeline.run_pipeline()
        print("Geometry pipeline completed successfully")
    except Exception as e:
        print(f"Error in geometry: {e}")

    # Final summary
    print("\n" + "=" * 70)
    print(" " * 20 + "PIPELINE COMPLETED!")
    print("=" * 70)
    print("\nGenerated folders:")
    for folder, name in components:
        if os.path.exists(folder):
            file_count = len(
                [
                    f
                    for f in os.listdir(folder)
                    if os.path.isfile(os.path.join(folder, f))
                ]
            )
            print(f"  {folder}/ ({file_count} files)")
        else:
            print(f"  {folder}/ (not created)")


def run_individual_component(component_number):
    """Run a specific component"""
    components = {
        1: ("Image Filtering", ImageFiltering, "01_filtering/generated"),
        2: ("Edge Detection", EdgeDetection, "02_edge/generated"),
        3: ("Feature Points", FeaturePointDetection, "03_featurepoints/generated"),
        4: ("Camera Geometry", CameraGeometry, "04_geometry/generated"),
    }

    if component_number not in components:
        print(f"Invalid component number: {component_number}")
        return

    name, cls, folder = components[component_number]
    print(f"\nRunning {name}...")
    pipeline = cls(output_dir=folder)
    pipeline.run_pipeline()
    print(f"\n{name} completed!")


def main():
    """Main entry point"""
    print("\n" + "=" * 70)
    print("  Computer Vision Pipeline - Tugas Individu IF5152")
    print("=" * 70)
    print("\nOptions:")
    print("  1. Run complete pipeline (all 4 components)")
    print("  2. Run individual component")
    print("  0. Exit")

    try:
        choice = input("\nSelect option [1]: ").strip() or "1"

        if choice == "0":
            print("Exiting...")
            return

        elif choice == "1":
            run_complete_pipeline()

        elif choice == "2":
            print("\nComponents:")
            print("  1. Image Filtering")
            print("  2. Edge Detection")
            print("  3. Feature Point Detection")
            print("  4. Camera Geometry & Calibration")

            comp = input("\nSelect component [1-4]: ").strip()
            if comp.isdigit() and 1 <= int(comp) <= 4:
                run_individual_component(int(comp))
            else:
                print("Invalid component number")

        else:
            print("Invalid option")

    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\nError: {e}")


if __name__ == "__main__":
    main()
