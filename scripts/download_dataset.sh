#!/bin/bash

# Script to download and extract the Colored MNIST dataset (tar.gz)
# from the provided Google Drive link.

# Exit on any error
set -e

# --- Configuration ---
# Google Drive File ID for 'mnist_10color_jitter.tar.gz'
# (from https://drive.google.com/file/d/1NSv4RCSHjcHois3dXjYw_PaLIoVlLgXu/view)
FILE_ID="1NSv4RCSHjcHois3dXjYw_PaLIoVlLgXu"
OUTPUT_FILENAME="mnist_10color_jitter.tar.gz"

# Target directory for extraction, relative to the project root.
# This should match what's expected by your data_loader.py and YAML config's 'data.path'.
# For example, if your YAML config data.path is "./data/colored_mnist/",
# then TARGET_DIR should be "./data/colored_mnist".
# The script will create this directory if it doesn't exist.
DEFAULT_TARGET_DIR="./data/colored_mnist"

# You can pass a target directory as an argument to the script
TARGET_DIR=${1:-$DEFAULT_TARGET_DIR}


# --- Helper Functions ---
check_command() {
    if ! command -v "$1" &> /dev/null; then
        echo "Error: '$1' command not found."
        echo "Please install '$1'. For gdown, try: pip install gdown"
        exit 1
    fi
}

# --- Main Script ---
echo "-------------------------------------------------------"
echo "Colored MNIST Dataset Download and Extraction Script"
echo "-------------------------------------------------------"

# 1. Check for gdown
check_command "gdown"

# 2. Create target directory if it doesn't exist
echo "Ensuring target directory exists: $TARGET_DIR"
mkdir -p "$TARGET_DIR"

# Define the full path for the downloaded tar.gz file
DOWNLOADED_TAR_PATH="$TARGET_DIR/$OUTPUT_FILENAME"

# 3. Download the file using gdown
# Check if the tar.gz file already exists to avoid re-downloading
if [ -f "$DOWNLOADED_TAR_PATH" ]; then
    echo "Dataset archive ($OUTPUT_FILENAME) already exists in $TARGET_DIR. Skipping download."
else
    echo "Downloading $OUTPUT_FILENAME to $TARGET_DIR..."
    if gdown --id "$FILE_ID" -O "$DOWNLOADED_TAR_PATH"; then
        echo "Download successful."
    else
        echo "Error during download. Please check network or gdown issues."
        # Clean up partially downloaded file if gdown creates one on failure
        [ -f "$DOWNLOADED_TAR_PATH" ] && rm "$DOWNLOADED_TAR_PATH"
        exit 1
    fi
fi

# 4. Extract the tar.gz file
# We need to know what the .npy files are named to check if extraction is needed.
# If this file (or any .npy file) exists, we can assume extraction was done.
EXPECTED_NPY_FILE_EXAMPLE="$TARGET_DIR/mnist_10color_jitter_var_0.030.npy" # Adjust if needed

if [ -f "$EXPECTED_NPY_FILE_EXAMPLE" ]; then
    echo "It appears the dataset has already been extracted in $TARGET_DIR. Skipping extraction."
else
    if [ -f "$DOWNLOADED_TAR_PATH" ]; then # Check if tar file exists before trying to extract
        echo "Extracting $DOWNLOADED_TAR_PATH to $TARGET_DIR..."
        # The -C flag tells tar to change to the TARGET_DIR before extracting.
        if tar -xzf "$DOWNLOADED_TAR_PATH" -C "$TARGET_DIR"; then
            echo "Extraction successful."
            
            # 5. Optional: Remove the tar.gz file after successful extraction
            echo "Removing downloaded archive: $DOWNLOADED_TAR_PATH"
            rm "$DOWNLOADED_TAR_PATH"
        else
            echo "Error during extraction. The archive might be corrupted or incomplete."
            exit 1
        fi
    else
        echo "Error: Archive file $DOWNLOADED_TAR_PATH not found for extraction. Please download it first."
        exit 1
    fi
fi

echo "-------------------------------------------------------"
echo "Dataset setup complete."
echo "The .npy files should now be available in: $TARGET_DIR"
echo "Ensure your YAML configuration ('data.path') points to this directory."
echo "And 'data.color_var' matches one of the available .npy files (e.g., 0.030)."
echo "-------------------------------------------------------"

exit 0