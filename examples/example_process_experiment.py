"""Example usage of the new experiment processing functions.

This example shows how to process a directory structure like:
\\\\multilab-na.ad.salk.edu\\hpi_dev\\users\\eberrigan\\circumnutation\\20250819_Suyash_Patil_CMTN_Kitx_vs_Hk1-3_07-30-25\\004\\_set1_day1_20250730-212759_004.tif
"""

from pathlib import Path
from sleap_roots_predict import (
    check_timelapse_image_directory,
    find_image_directories,
    process_timelapse_experiment,
)


# Example 1: Check a single directory
def check_single_directory_example():
    """Check a single plate directory for validity."""
    # Path to a single plate directory
    plate_dir = Path(
        r"\\multilab-na.ad.salk.edu\hpi_dev\users\eberrigan\circumnutation"
        r"\20250819_Suyash_Patil_CMTN_Kitx_vs_Hk1-3_07-30-25\004"
    )

    # Check the directory with specific requirements
    results = check_timelapse_image_directory(
        plate_dir,
        expected_suffix_pattern=r"^\d{3}$",  # Expect 3-digit suffixes like '004'
        min_images=10,  # Minimum 10 images for a valid timelapse
        max_images=1000,  # Maximum 1000 images
        check_datetime=True,  # Ensure datetime is present
        check_suffix_consistency=True,  # All files should have same suffix (plate number)
    )

    if results["valid"]:
        print(f"✓ Directory is valid: {plate_dir}")
        print(f"  - Found {results['image_count']} images")
        print(f"  - Plate suffix: {results['suffixes']}")
    else:
        print(f"✗ Directory validation failed: {plate_dir}")
        for error in results["errors"]:
            print(f"  ERROR: {error}")

    for warning in results["warnings"]:
        print(f"  WARNING: {warning}")

    return results


# Example 2: Process an entire experiment
def process_full_experiment_example():
    """Process all plates in an experiment directory using CSV metadata.

    This example shows how to:
    1. Load metadata from a CSV file with per-plate information
    2. Process multiple plates with their specific metadata
    3. Handle plate number matching (single digit in CSV -> 3-digit in filenames)
    """
    # Base experiment directory
    experiment_dir = Path(
        r"\\multilab-na.ad.salk.edu\hpi_dev\users\eberrigan\circumnutation"
        r"\20250819_Suyash_Patil_CMTN_Kitx_vs_Hk1-3_07-30-25"
    )

    # Path to CSV file with plate metadata
    # CSV should have columns: plate_number, treatment, num_plants
    # Optional columns: accesion, num_images, experiment_start, growth_media
    metadata_csv = Path(r"C:\experiments\metadata\CMTN_KITXvsHK1-3_META.csv")

    # Output directory for processed files
    output_dir = Path(r"C:\processed_experiments\circumnutation_output")

    # First, do a dry run to check what will be processed
    print("\n=== DRY RUN ===")
    dry_results = process_experiment(
        base_dir=experiment_dir,
        metadata_csv=metadata_csv,
        experiment_name="CMTN_Kitx_vs_Hk1-3",
        output_dir=output_dir,
        # Processing parameters
        greyscale=True,  # Convert to greyscale to save space
        image_pattern="*.tif",
        # Validation parameters
        expected_suffix_pattern=r"^\d{3}$",  # Default ensures 3-digit suffix matching
        min_images=10,
        check_datetime=True,
        check_suffix_consistency=True,
        # Control
        dry_run=True,
    )

    print(f"Would process {len(dry_results['skipped'])} directories")
    print(f"Would skip {len(dry_results['failed'])} failed directories")

    # Show which plates were matched to which metadata
    for item in dry_results["skipped"]:
        if item["reason"] == "dry_run" and "plate_metadata" in item:
            print(f"  Plate {item['directory']}: {item['plate_metadata']['treatment']}")

    # If dry run looks good, process for real
    response = input("\nProceed with processing? (y/n): ")
    if response.lower() == "y":
        print("\n=== PROCESSING ===")
        results = process_timelapse_experiment(
            base_dir=experiment_dir,
            metadata_csv=metadata_csv,
            experiment_name="CMTN_Kitx_vs_Hk1-3",
            output_dir=output_dir,
            # Processing parameters
            greyscale=True,
            image_pattern="*.tif",
            # Validation parameters
            expected_suffix_pattern=r"^\d{3}$",
            min_images=10,
            check_datetime=True,
            check_suffix_consistency=True,
            # Control
            dry_run=False,
        )

        print(f"\n✓ Processed {len(results['processed'])} directories successfully")
        for item in results["processed"]:
            plate_meta = item.get("plate_metadata", {})
            print(f"  - {item['directory']}")
            print(f"    Treatment: {plate_meta.get('treatment')}")
            print(f"    Num plants: {plate_meta.get('num_plants')}")
            print(f"    H5: {item['h5_path']}")
            print(f"    CSV: {item['csv_path']}")

        if results["failed"]:
            print(f"\n✗ Failed to process {len(results['failed'])} directories")
            for item in results["failed"]:
                print(f"  - {item['directory']}")
                for error in item["check_results"]["errors"]:
                    print(f"    ERROR: {error}")

        if results["skipped"]:
            print(f"\n⚠ Skipped {len(results['skipped'])} directories")
            for item in results["skipped"]:
                if item["reason"] != "dry_run":
                    print(f"  - {item['directory']}: {item['reason']}")

    return results


# Example 3: Find and list all image directories
def find_directories_example():
    """Find all directories containing TIFF images."""
    base_dir = Path(r"\\multilab-na.ad.salk.edu\hpi_dev\users\eberrigan\circumnutation")

    print("Searching for image directories...")
    image_dirs = find_image_directories(base_dir)

    print(f"Found {len(image_dirs)} directories with TIFF images:")
    for dir_path in image_dirs:
        # Get relative path for cleaner display
        try:
            rel_path = dir_path.relative_to(base_dir)
            print(f"  - {rel_path}")
        except ValueError:
            print(f"  - {dir_path}")

    return image_dirs


if __name__ == "__main__":
    print("=== Experiment Processing Examples ===\n")

    # Example 1: Check a single directory
    print("1. Checking a single directory:")
    # check_single_directory_example()

    # Example 2: Find all image directories
    print("\n2. Finding all image directories:")
    # find_directories_example()

    # Example 3: Process full experiment
    print("\n3. Processing full experiment:")
    # process_full_experiment_example()

    print("\nUncomment the examples you want to run!")
