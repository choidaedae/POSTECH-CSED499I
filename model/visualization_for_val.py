import os
import numpy as np
from PIL import Image

def save_npy_as_image(npy_path, output_dir, image_format='JPEG'):
    # Load NumPy array
    numpy_array = np.load(npy_path)

    # Convert NumPy array to PIL Image
    image = Image.fromarray(numpy_array)

    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Get the filename without extension
    file_name = os.path.splitext(os.path.basename(npy_path))[0]

    # Save image as JPEG (or other specified format)
    image.save(os.path.join(output_dir, f"{file_name}.jpg"), format=image_format)

def convert_npy_files(directory_path, output_dir):
    # List all files in the directory
    files = os.listdir(directory_path)

    # Filter files with .npy extension
    npy_files = [file for file in files if file.endswith('.npy')]

    # Process each .npy file
    for npy_file in npy_files:
        npy_path = os.path.join(directory_path, npy_file)
        save_npy_as_image(npy_path, output_dir)

if __name__ == "__main__":
    # Specify the directory containing .npy files
    input_directory = "/path/to/npy/files"

    # Specify the output directory for images
    output_directory = "/path/to/output/images"

    # Convert .npy files to images
    convert_npy_files(input_directory, output_directory)