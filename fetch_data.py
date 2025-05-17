import os
import sys
import zipfile
import tarfile
import requests # For downloading from URL
from torchvision import datasets as tv_datasets
from PIL import Image
import shutil # For moving files

# Attempt to import kaggle API
try:
    import kaggle
    KAGGLE_API_AVAILABLE = True
except ImportError:
    KAGGLE_API_AVAILABLE = False
    print("WARNING: Kaggle API not found. To download Dogs vs. Cats automatically, "
          "please install it (`pip install kaggle`) and set up your API credentials "
          "(~/.kaggle/kaggle.json).")
except Exception as e:
    KAGGLE_API_AVAILABLE = False
    print(f"WARNING: Error importing Kaggle API: {e}. Dogs vs. Cats download might fail.")

# Attempt to import gdown
try:
    import gdown
    GDOWN_AVAILABLE = True
except ImportError:
    GDOWN_AVAILABLE = False
    print("WARNING: gdown not found. To download IMDB-Face from Google Drive automatically, "
          "please install it (`pip install gdown`).")


def ensure_dir(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Created directory: {directory_path}")
    else:
        print(f"Directory already exists: {directory_path}")

def download_file_from_url(url, destination_path, chunk_size=8192):
    # ... (same as before)
    print(f"Downloading from {url} to {destination_path}...")
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        total_size = int(response.headers.get('content-length', 0))
        
        with open(destination_path, 'wb') as f:
            try:
                from tqdm import tqdm
                with tqdm(total=total_size, unit='iB', unit_scale=True, desc=os.path.basename(destination_path)) as pbar:
                    for chunk in response.iter_content(chunk_size=chunk_size):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))
            except ImportError:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk: f.write(chunk)
        print(f"Successfully downloaded {os.path.basename(destination_path)}")
        return True
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        if os.path.exists(destination_path): os.remove(destination_path)
        return False


def fetch_mnist(data_root_base):
    # ... (same as before)
    mnist_data_path = os.path.join(data_root_base, "mnist_colored_data")
    ensure_dir(mnist_data_path)
    print(f"Attempting to download MNIST to: {mnist_data_path}")
    try:
        tv_datasets.MNIST(root=mnist_data_path, train=True, download=True)
        tv_datasets.MNIST(root=mnist_data_path, train=False, download=True)
        print("MNIST download attempt complete.")
    except Exception as e:
        print(f"Error downloading MNIST: {e}")


def setup_dogs_cats_dirs(data_root_base):
    dogs_cats_root = os.path.join(data_root_base, "dogs_vs_cats")
    images_dir = os.path.join(dogs_cats_root, "images") # For final cat.*.jpg, dog.*.jpg
    raw_kaggle_dir = os.path.join(dogs_cats_root, "raw_kaggle_downloads") # To store downloaded zips/files
    lists_dir = os.path.join(dogs_cats_root, "lists")

    ensure_dir(dogs_cats_root)
    ensure_dir(images_dir)
    ensure_dir(raw_kaggle_dir)
    ensure_dir(lists_dir)

    print("\n--- Dogs vs. Cats Dataset Setup ---")
    
    download_successful = False
    extracted_to_images_dir = os.listdir(images_dir) # Check if images_dir already populated

    if not extracted_to_images_dir and KAGGLE_API_AVAILABLE:
        # Try original competition first
        kaggle_competition_slug = 'dogs-vs-cats'
        print(f"Attempt 1: Downloading '{kaggle_competition_slug}' competition data from Kaggle to '{raw_kaggle_dir}'...")
        try:
            kaggle.api.competition_download_files(kaggle_competition_slug, path=raw_kaggle_dir, force=False, quiet=False)
            print(f"Kaggle competition download attempt complete. Files should be in '{raw_kaggle_dir}'.")
            
            train_zip_path = os.path.join(raw_kaggle_dir, "train.zip")
            if os.path.exists(train_zip_path):
                print(f"Found 'train.zip'. Attempting to extract its contents to '{images_dir}'...")
                with zipfile.ZipFile(train_zip_path, 'r') as zip_ref:
                    for member in zip_ref.namelist():
                        if member.lower().endswith(('.jpg', '.jpeg', '.png')) and \
                           (member.lower().startswith('cat.') or member.lower().startswith('dog.')):
                            # Ensure the target path is directly in images_dir, not with original zip path
                            target_path = os.path.join(images_dir, os.path.basename(member))
                            with open(target_path, 'wb') as outfile:
                                outfile.write(zip_ref.read(member))
                print(f"Successfully extracted images from 'train.zip' to '{images_dir}'.")
                download_successful = True
            else:
                print(f"'train.zip' not found from competition download in '{raw_kaggle_dir}'.")
        except Exception as e:
            print(f"Error during Kaggle competition download ('{kaggle_competition_slug}'): {e}")
            if "403" in str(e) and "accept this competition" in str(e).lower():
                print("This often means competition rules need to be accepted, or the competition is too old for API rule acceptance.")
            else:
                print("Please ensure Kaggle API is configured correctly.")
        
        # If competition download failed (especially due to 403) or train.zip not found, try alternative dataset
        if not download_successful:
            kaggle_dataset_slug = 'shaunthesheep/microsoft-catsvsdogs-dataset'
            print(f"\nAttempt 2: Downloading '{kaggle_dataset_slug}' dataset from Kaggle to '{raw_kaggle_dir}'...")
            try:
                kaggle.api.dataset_download_files(kaggle_dataset_slug, path=raw_kaggle_dir, unzip=False, force=False, quiet=False)
                # This usually downloads as 'microsoft-catsvsdogs-dataset.zip'
                dataset_zip_name = "microsoft-catsvsdogs-dataset.zip" # Common name
                dataset_zip_path = os.path.join(raw_kaggle_dir, dataset_zip_name)

                if not os.path.exists(dataset_zip_path): # Try to find the actual zip name
                    found_zips = [f for f in os.listdir(raw_kaggle_dir) if f.endswith('.zip')]
                    if found_zips:
                        dataset_zip_path = os.path.join(raw_kaggle_dir, found_zips[0])
                        print(f"Found zip file: {found_zips[0]}")
                    else:
                        print(f"No zip file found for dataset '{kaggle_dataset_slug}' in '{raw_kaggle_dir}'.")
                        raise FileNotFoundError("Dataset zip not found.")

                print(f"Found dataset zip: '{os.path.basename(dataset_zip_path)}'. Attempting to extract...")
                temp_extract_path = os.path.join(raw_kaggle_dir, "temp_extract_msft")
                ensure_dir(temp_extract_path)
                with zipfile.ZipFile(dataset_zip_path, 'r') as zip_ref:
                    zip_ref.extractall(temp_extract_path)
                
                print(f"Extracted to '{temp_extract_path}'. Now moving images to '{images_dir}'...")
                # This dataset often has structure like PetImages/Cat/*.jpg and PetImages/Dog/*.jpg
                # Or sometimes directly Cat/*.jpg, Dog/*.jpg in the zip root after 'PetImages'
                # We need to find .jpg files and copy them with cat. or dog. prefix if not already there.
                moved_count = 0
                for root_dir, _, files in os.walk(temp_extract_path):
                    for file_name in files:
                        if file_name.lower().endswith(('.jpg', '.jpeg')):
                            source_path = os.path.join(root_dir, file_name)
                            # Determine if it's a cat or dog based on folder or filename
                            is_cat = "cat" in root_dir.lower() or "cat" in file_name.lower()
                            is_dog = "dog" in root_dir.lower() or "dog" in file_name.lower()

                            if is_cat or is_dog:
                                # Create a new filename like cat.xxxx.jpg or dog.xxxx.jpg to match original competition format
                                # This part might need adjustment if filenames are already good
                                prefix = "cat" if is_cat else "dog"
                                # Try to keep original numbering if possible, or generate new ones
                                # For simplicity, let's use a generic counter if names are not already 'cat/dog.num.jpg'
                                # However, the microsoft dataset usually has good names already inside Cat/Dog folders.
                                # We just need to move them.
                                target_filename = os.path.basename(source_path) # Use original filename from Cat/Dog folder
                                if not (target_filename.lower().startswith("cat.") or target_filename.lower().startswith("dog.")):
                                    # If filename is not like "cat.123.jpg", prepend based on folder
                                    # This is a basic heuristic
                                    target_filename = f"{prefix}.{target_filename}"


                                target_path = os.path.join(images_dir, target_filename)
                                # Avoid overwriting if a file with same name (e.g. from a different subfolder) exists
                                if os.path.exists(target_path):
                                    base, ext = os.path.splitext(target_filename)
                                    target_filename = f"{base}_{moved_count}{ext}"
                                    target_path = os.path.join(images_dir, target_filename)

                                try:
                                    # Some images in this dataset are corrupted, try to open and save
                                    img_pil = Image.open(source_path)
                                    img_pil.save(target_path) # This re-saves and can fix some minor issues
                                    moved_count += 1
                                except (IOError, SyntaxError, Image.DecompressionBombError) as img_err:
                                    print(f"Skipping corrupted or problematic image: {source_path} due to {img_err}")
                                    continue
                
                if moved_count > 0:
                    print(f"Successfully processed and moved {moved_count} images to '{images_dir}'.")
                    download_successful = True
                else:
                    print(f"No images were successfully moved from '{temp_extract_path}'. Check its structure.")

                shutil.rmtree(temp_extract_path) # Clean up temp extraction
            except Exception as e:
                print(f"Error during Kaggle dataset download ('{kaggle_dataset_slug}') or processing: {e}")
    
    elif extracted_to_images_dir:
        print(f"Images directory '{images_dir}' is already populated. Skipping download/extraction for Dogs vs. Cats.")
        download_successful = True # Assume it's correctly set up
    else: # Not extracted and Kaggle API not available
        print("Kaggle API not available. Skipping automatic download for Dogs vs. Cats.")

    if not download_successful:
        print("\nAutomatic download/extraction for Dogs vs. Cats failed or was skipped.")
        print("Please manually ensure the 'images' directory is populated:")
        print(f"  {images_dir}")
        print("You can download 'train.zip' from the original Kaggle competition page or use another source.")

    print(f"\nDirectory structure for Dogs vs. Cats is in: {dogs_cats_root}")
    print("Next steps for Dogs vs. Cats:")
    print(f"1. Ensure all cat.*.jpg and dog.*.jpg images are in: {images_dir}")
    print(f"2. Manually categorize images by color ('bright', 'dark') as per the paper's methodology.")
    print(f"3. Create 'list_bright.txt' and 'list_dark.txt' in: {lists_dir}")
    print("   (Each file: one image filename per line, e.g., 'dog.123.jpg')")
    print(f"4. Create your test image list (e.g., 'list_test_unbiased.txt') in: {lists_dir}")

    if not os.path.exists(os.path.join(lists_dir, "EXAMPLE_list_bright.txt")):
        with open(os.path.join(lists_dir, "EXAMPLE_list_bright.txt"), 'w') as f: f.write("dog.1.jpg\ncat.2.jpg\n")
        print(f"Created example list file: {os.path.join(lists_dir, 'EXAMPLE_list_bright.txt')}")


def setup_imdb_face_dirs(data_root_base):
    # ... (same as before, using gdown)
    imdb_face_root = os.path.join(data_root_base, "imdb_face")
    filtered_data_dir = os.path.join(imdb_face_root, "filtered_images") 
    manifests_dir = os.path.join(imdb_face_root, "manifests")

    ensure_dir(imdb_face_root); ensure_dir(filtered_data_dir); ensure_dir(manifests_dir)
    print("\n--- IMDB-Face Dataset Setup (using authors' filtered data) ---")
    gdrive_file_id = "1ZFZ2tUjq3BBRw3rcXB0LkBvaqKD4_8WK"
    downloaded_archive_name = "imdb_face_filtered.zip" 
    downloaded_archive_path = os.path.join(imdb_face_root, downloaded_archive_name)

    if not os.listdir(filtered_data_dir):
        if GDOWN_AVAILABLE:
            print(f"Attempting to download filtered IMDB-Face data from Google Drive (ID: {gdrive_file_id})...")
            try:
                gdown.download(id=gdrive_file_id, output=downloaded_archive_path, quiet=False)
                print(f"Successfully downloaded to '{downloaded_archive_path}'.")
                print(f"Attempting to extract '{downloaded_archive_path}' into '{filtered_data_dir}'...")
                if downloaded_archive_path.endswith(".zip"):
                    with zipfile.ZipFile(downloaded_archive_path, 'r') as zip_ref: zip_ref.extractall(filtered_data_dir)
                elif downloaded_archive_path.endswith((".tar.gz", ".tgz")):
                    with tarfile.open(downloaded_archive_path, "r:gz") as tar: tar.extractall(path=filtered_data_dir)
                else: print(f"Downloaded file '{downloaded_archive_name}' not .zip or .tar.gz. Please extract manually.")
                print(f"Successfully extracted to '{filtered_data_dir}'.")
                # os.remove(downloaded_archive_path) # Optional cleanup
            except Exception as e:
                print(f"Error during Google Drive download or extraction: {e}")
                print(f"Manual download: https://drive.google.com/file/d/{gdrive_file_id}/view -> {filtered_data_dir}")
        else:
            print(f"gdown not available. Manual download: https://drive.google.com/file/d/{gdrive_file_id}/view -> {filtered_data_dir}")
    else:
        print(f"Filtered IMDB-Face data seems to exist in '{filtered_data_dir}'. Skipping.")

    print(f"\nDirectory structure for IMDB-Face in: {imdb_face_root}")
    print("Next steps for IMDB-Face:")
    print(f"1. Ensure filtered images are in: {filtered_data_dir}")
    print(f"2. Create manifest CSV files (e.g., train_eb1_gender.csv) in: {manifests_dir}")
    print("   Manifest columns: relative_image_path_in_filtered_dir,gender_label,age_value")
    if not os.path.exists(os.path.join(manifests_dir, "EXAMPLE_train_manifest_eb1_gender.csv")):
        with open(os.path.join(manifests_dir, "EXAMPLE_train_manifest_eb1_gender.csv"), 'w') as f:
            f.write("relative_image_path,gender_label,age_value\nimage_001.jpg,0,25\n")
        print(f"Created example manifest: {os.path.join(manifests_dir, 'EXAMPLE_train_manifest_eb1_gender.csv')}")


if __name__ == "__main__":
    data_root_base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    ensure_dir(data_root_base_dir)
    print("--- Starting Data Fetch/Setup Process ---")
    print("\nSetting up MNIST...")
    fetch_mnist(data_root_base_dir)
    print("\nSetting up Dogs and Cats directories and attempting download...")
    setup_dogs_cats_dirs(data_root_base_dir)
    print("\nSetting up IMDB-Face directories and attempting filtered data download...")
    setup_imdb_face_dirs(data_root_base_dir)
    print("\n--- Data Fetch/Setup Process Finished ---")
    print("Please review instructions for each dataset to complete setup (manifests/lists).")

