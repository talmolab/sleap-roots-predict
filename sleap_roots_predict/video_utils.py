import numpy as np
import h5py
from pathlib import Path
import re
import pandas as pd


def natural_sort(l):
    """Sort a list of strings in a way that considers numerical values within the strings.

    For example, natural_sort(["img2.png", "img10.png", "img1.png"])
    will return ["img1.png", "img2.png", "img10.png"].

    Args:
        l (list): List of strings to sort.

    Returns:
        list: List of sorted strings.
    """
    l = [x.as_posix() if isinstance(x, Path) else x for x in l]
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split("([0-9]+)", key)]
    return sorted(l, key=alphanum_key)


def process_image_directory(
    source_dir, experiment_name, treatment, num_plants, output_dir=None
):
    """Processes a directory of images for a plate over time into an h5 file and a metadata file.

    Args:
        source_dir (Path): Path to the source directory containing images.
        experiment_name (str): The name of the experiment.
        treatment (str): The chemical or physical alterations to the plate media.
        num_plants (int): The number of plants expected on a plate image.
        greyscale (bool): Whether or not to convert images to greyscale.
        metadata (bool): Whether or not to save a dataframe of metadata.
        output_dir (Any): The directory to store the h5 file and metadata file. If none is specified, it is saved to the source directory.
    """

    # Check if the source directory exists
    if not source_dir.exists():
        print(f"Source directory {source_dir} does not exist.")
        return

    # Check if the source directory is a directory
    if not source_dir.is_dir():
        print(f"Source path {source_dir} is not a directory.")

    # Use source_dir as default output_dir if not provided
    if output_dir is None:
        output_dir = source_dir

    # Make sure output_dir exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get all image files in the source directory
    image_files = list(source_dir.glob("*.tif"))

    if not image_files:
        print(f"No image files found in {source_dir}.")
        return

    print(f"Found {len(image_files)} image files in {source_dir}.")

    # Sort the image files naturally
    image_files = natural_sort(image_files)

    images = []
    df_rows = []

    # Process each image file
    print("Reading images...")
    frame_idx = 0
    for img_file in image_files:

        print(f"Reading {img_file}...")
        # Read the image
        img = iio.imread(img_file)

        if greyscale:
            # Convert image to greyscale
            img = np.dot(img[..., :3], [0.2989, 0.5870, 0.1140])
            img = img.astype(np.uint8)

        # Extract plate number, day, and time from the filename
        # Assuming the filename format is "_set1_day1_timestamp_platenumber.tif"

        timestamp = img_file.split("_")[-2]
        plate_number = img_file.split("_")[-1].split(".")[0]
        filename = img_file.split("/")[-1]

        df_rows.append(
            {
                "experiment": experiment_name,
                "filename": filename,
                "treatment": treatment,
                "plate_number": plate_number,
                "expected_num_plants": num_plants,
                "timestamp": timestamp,
                "frame": frame_idx,
            }
        )

        images.append(img)
        frame_idx += 1

    print("Finished reading images.")

    print("Stacking images...")
    vol = np.stack(images, axis=0)
    print("Finished stacking images.")

    if greyscale:
        h5_name = f"plate_{source_dir.name}_greyscale.h5"
    else:
        h5_name = f"plate_{source_dir.name}_color.h5"

    h5_path = output_dir / h5_name

    # Save the volume as a .h5 file
    with h5py.File(h5_path, "w") as f:
        # Create a dataset in the HDF5 file
        print(f"Creating dataset in {h5_name}...")

        if greyscale:
            # Expand dimensions to (frames, height, width, 1)
            vol = np.expand_dims(vol, axis=-1)

            print(f"Vol shape: {vol.shape}")

            f.create_dataset("vol", data=vol, compression="gzip", compression_opts=1)
        else:
            # Expected shape: (frames, height, width, 3)
            print(f"Vol shape: {vol.shape}")

            f.create_dataset("vol", data=vol, compression="gzip", compression_opts=1)

        print(f"Saved vol to {h5_path}")

    # Save the DataFrame to a .csv file
    df_path = output_dir / ("plate_" + source_dir.name + "_metadata.csv")
    df = pd.DataFrame.from_records(df_rows)
    df.to_csv(df_path, index=False)
    print(f"Saved DataFrame {df_path} to {df_path}")
