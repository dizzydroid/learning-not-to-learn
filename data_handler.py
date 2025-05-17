import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets as tv_datasets
from torchvision import transforms
from PIL import Image
import numpy as np
import os
import random

# --- ColoredMNISTDataset ---
class ColoredMNISTDataset(Dataset):
    def __init__(self, root='./data', train=True, sigma_bias=0.05,
                 biased_training_target=True, download=True, transform=None,
                 target_transform=None, color_map=None, bias_label_type='mean_color_idx'):
        self.root = root
        self.train = train
        self.sigma_bias = sigma_bias
        self.biased_training_target = biased_training_target
        self.provided_transform = transform
        self.target_transform = target_transform
        self.bias_label_type = bias_label_type

        if color_map is None:
            self.color_map = [
                (0.8, 0.2, 0.2), (0.2, 0.8, 0.2), (0.2, 0.2, 0.8), (0.8, 0.8, 0.2),
                (0.8, 0.2, 0.8), (0.2, 0.8, 0.8), (0.5, 0.5, 0.2), (0.5, 0.2, 0.5),
                (0.2, 0.5, 0.5), (0.6, 0.4, 0.3)
            ]
        else:
            self.color_map = color_map
        
        assert len(self.color_map) == 10, "Color map must have 10 colors."
        self.mnist_dataset = tv_datasets.MNIST(self.root, train=self.train, download=download)
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    def _get_color_and_bias_label(self, digit_label):
        if self.biased_training_target:
            mean_color_rgb = self.color_map[digit_label]
            bias_mean_color_idx = digit_label
        else:
            random_color_idx = random.randint(0, 9)
            mean_color_rgb = self.color_map[random_color_idx]
            bias_mean_color_idx = random_color_idx
        sampled_color_r = np.clip(np.random.normal(mean_color_rgb[0], self.sigma_bias), 0, 1)
        sampled_color_g = np.clip(np.random.normal(mean_color_rgb[1], self.sigma_bias), 0, 1)
        sampled_color_b = np.clip(np.random.normal(mean_color_rgb[2], self.sigma_bias), 0, 1)
        final_color_rgb = (sampled_color_r, sampled_color_g, sampled_color_b)
        bias_label = bias_mean_color_idx
        return final_color_rgb, torch.tensor(bias_label).long()

    def __getitem__(self, index):
        img_pil, digit_label = self.mnist_dataset[index]
        if self.target_transform is not None:
            digit_label = self.target_transform(digit_label)
        final_color_rgb, bias_label = self._get_color_and_bias_label(digit_label)
        gray_img_tensor = self.to_tensor(img_pil)
        mask = (gray_img_tensor > 0.1).squeeze(0)
        colored_image_tensor = torch.zeros((3, gray_img_tensor.shape[1], gray_img_tensor.shape[2]))
        for i in range(3):
            colored_image_tensor[i, mask] = final_color_rgb[i]
        colored_image_tensor = self.normalize(colored_image_tensor)
        if self.provided_transform:
            colored_image_tensor = self.provided_transform(colored_image_tensor)
        return colored_image_tensor, torch.tensor(digit_label).long(), bias_label

    def __len__(self):
        return len(self.mnist_dataset)

# --- Updated DogsAndCatsDataset ---
class DogsAndCatsDataset(Dataset):
    def __init__(self, image_root, transform=None,
                 list_bright_file=None, list_dark_file=None, list_test_file=None,
                 biased_train_type=None):
        """
        Args:
            image_root (str): Root directory where cat.*.jpg and dog.*.jpg images are.
            transform (callable, optional): Transform to apply to images.
            list_bright_file (str, optional): Path to file listing bright image filenames.
            list_dark_file (str, optional): Path to file listing dark image filenames.
            list_test_file (str, optional): Path to file listing test image filenames (unbiased).
            biased_train_type (str, optional): "TB1" (bright dogs, dark cats),
                                               "TB2" (dark dogs, bright cats),
                                               or None (for test set using list_test_file).
        """
        self.image_root = image_root
        self.transform = transform
        self.samples = []

        # Define labels
        LABEL_CAT = 0
        LABEL_DOG = 1
        COLOR_BRIGHT = 0
        COLOR_DARK = 1

        def load_images_from_list(list_file, color_category):
            images = []
            if list_file and os.path.exists(list_file):
                with open(list_file, 'r') as f:
                    for filename in f:
                        filename = filename.strip()
                        if not filename: continue
                        img_path = os.path.join(self.image_root, filename)
                        true_label = LABEL_DOG if 'dog.' in filename.lower() else LABEL_CAT
                        images.append({'path': img_path, 'label': true_label, 'color': color_category, 'filename': filename})
            return images

        bright_images = load_images_from_list(list_bright_file, COLOR_BRIGHT)
        dark_images = load_images_from_list(list_dark_file, COLOR_DARK)

        if biased_train_type == "TB1": # Bright dogs, Dark cats
            for img_info in bright_images:
                if img_info['label'] == LABEL_DOG: # Bright Dog
                    self.samples.append((img_info['path'], img_info['label'], img_info['color']))
            for img_info in dark_images:
                if img_info['label'] == LABEL_CAT: # Dark Cat
                    self.samples.append((img_info['path'], img_info['label'], img_info['color']))
        elif biased_train_type == "TB2": # Dark dogs, Bright cats
            for img_info in dark_images:
                if img_info['label'] == LABEL_DOG: # Dark Dog
                    self.samples.append((img_info['path'], img_info['label'], img_info['color']))
            for img_info in bright_images:
                if img_info['label'] == LABEL_CAT: # Bright Cat
                    self.samples.append((img_info['path'], img_info['label'], img_info['color']))
        elif biased_train_type is None and list_test_file: # Unbiased test set
            # For test set, color category from list_bright/dark is not strictly the "bias label"
            # but can be used if h is evaluated on color.
            # The paper's test is on the original unbiased images.
            # We assume list_test_file contains filenames, and we infer their original color if needed.
            # For simplicity, bias_h_label for test can be a dummy or inferred if available.
            # Let's assume list_test_file just gives us images and their true labels.
            # The bias label for 'h' on test set should be the actual color characteristic.
            # We'll need to determine color for test images if list_test_file doesn't provide it.
            # For now, let's assume the bias_label is the "natural" color category if we could determine it.
            # Or, if the authors' list_test.txt also has some color info or it's part of a general manifest.
            # The simplest is to load all images from list_test_file and assign a dummy/inferred bias label.
            if os.path.exists(list_test_file):
                with open(list_test_file, 'r') as f:
                    for filename in f:
                        filename = filename.strip()
                        if not filename: continue
                        img_path = os.path.join(self.image_root, filename)
                        true_label = LABEL_DOG if 'dog.' in filename.lower() else LABEL_CAT
                        # Determine color for test images:
                        # This is tricky without the original categorization logic or explicit labels.
                        # For now, assign a placeholder bias label for test, or use the one from list_bright/dark if image is in them.
                        # This part needs to align with how the authors constructed their test set evaluation for bias.
                        # A simple approach: if image is in bright_images, its color is BRIGHT, else if in dark_images, its DARK, else -1.
                        img_color_test = -1
                        if any(entry['filename'] == filename for entry in bright_images): img_color_test = COLOR_BRIGHT
                        elif any(entry['filename'] == filename for entry in dark_images): img_color_test = COLOR_DARK
                        
                        self.samples.append((img_path, true_label, img_color_test)) # Using inferred color as bias label
            else:
                print(f"Warning: Test file list {list_test_file} not found.")
        else:
            if train: # If training but no valid biased_train_type, this is an issue
                print(f"Warning: DogsAndCatsDataset called for training with invalid bias_train_type '{biased_train_type}' or missing list files.")


    def __getitem__(self, index):
        img_path, true_label, bias_h_label = self.samples[index]
        try:
            image = Image.open(img_path).convert('RGB')
        except FileNotFoundError:
            print(f"Error: Image not found at {img_path}")
            return torch.randn(3, 224, 224), torch.tensor(-1).long(), torch.tensor(-1).long()

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(true_label).long(), torch.tensor(bias_h_label).long()

    def __len__(self):
        return len(self.samples)


# --- IMDBFaceDataset (same as before, assuming filtered data is used) ---
class IMDBFaceDataset(Dataset):
    def __init__(self, root, image_list_file, task='classify_gender',
                 biased_train_type=None, transform=None):
        # 'root' is where the filtered images are (e.g., ./data/imdb_face/filtered_images/)
        # 'image_list_file' is the manifest CSV (e.g., ./data/imdb_face/manifests/train_eb1.csv)
        # Paths in manifest are relative to 'root'.
        self.root = root
        self.transform = transform
        self.samples = []
        AGE_GROUP_YOUNG = 0; AGE_GROUP_OLD = 1

        with open(image_list_file, 'r') as f:
            next(f) # Skip header if manifest has one
            for line in f:
                parts = line.strip().split(',')
                img_rel_path = parts[0]
                img_abs_path = os.path.join(self.root, img_rel_path)
                gender_label = int(parts[1])
                age = int(parts[2])
                age_group = -1
                if 0 <= age <= 29: age_group = AGE_GROUP_YOUNG
                elif age >= 40: age_group = AGE_GROUP_OLD

                main_label, bias_h_label = -1, -1
                if task == 'classify_gender':
                    main_label = gender_label; bias_h_label = age_group
                elif task == 'classify_age':
                    if age_group == -1: continue
                    main_label = age_group; bias_h_label = gender_label
                else: raise ValueError(f"Unknown task: {task}")

                if biased_train_type == "EB1":
                    if not ((gender_label == 0 and age_group == AGE_GROUP_YOUNG) or \
                            (gender_label == 1 and age_group == AGE_GROUP_OLD)): continue
                elif biased_train_type == "EB2":
                    if not ((gender_label == 0 and age_group == AGE_GROUP_OLD) or \
                            (gender_label == 1 and age_group == AGE_GROUP_YOUNG)): continue
                elif biased_train_type is None: # Test set
                    if age_group == -1 : continue # Paper's test set is 0-29 or 40+
                
                if bias_h_label == -1 and biased_train_type is not None : continue # Ensure valid bias for training

                self.samples.append((img_abs_path, main_label, bias_h_label))

    def __getitem__(self, index):
        img_path, main_label, bias_h_label = self.samples[index]
        try: image = Image.open(img_path).convert('RGB')
        except FileNotFoundError:
            print(f"Error: Image not found at {img_path}"); return torch.randn(3,224,224), torch.tensor(-1L), torch.tensor(-1L)
        if self.transform: image = self.transform(image)
        return image, torch.tensor(main_label).long(), torch.tensor(bias_h_label).long()

    def __len__(self): return len(self.samples)


# --- Updated get_data_loader ---
def get_data_loader(args, train, biased_training_target_for_mnist=True):
    dataset_name = args.dataset_name
    data_root_base = args.data_root_base # Base data dir (e.g. ./data)
    batch_size = args.batch_size
    num_workers = args.num_workers if hasattr(args, 'num_workers') else 4

    if dataset_name == 'ColoredMNIST':
        custom_transform = None
        dataset = ColoredMNISTDataset(
            root=os.path.join(data_root_base, 'mnist_colored_data'),
            train=train,
            sigma_bias=args.sigma_bias,
            biased_training_target=biased_training_target_for_mnist if train else False,
            download=args.download_data,
            transform=custom_transform,
            bias_label_type=args.cmnist_bias_label_type
        )
    elif dataset_name == 'DogsAndCats':
        img_transform = transforms.Compose([
            transforms.Resize((args.image_size, args.image_size)),
            transforms.RandomHorizontalFlip() if train else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        # image_root is where cat.x.jpg and dog.x.jpg are
        dac_image_root = os.path.join(data_root_base, 'dogs_vs_cats', 'images')
        dac_lists_root = os.path.join(data_root_base, 'dogs_vs_cats', 'lists')

        list_bright = os.path.join(dac_lists_root, args.dac_list_bright) if hasattr(args, 'dac_list_bright') and args.dac_list_bright else None
        list_dark = os.path.join(dac_lists_root, args.dac_list_dark) if hasattr(args, 'dac_list_dark') and args.dac_list_dark else None
        list_test = os.path.join(dac_lists_root, args.dac_list_test) if hasattr(args, 'dac_list_test') and args.dac_list_test else None
        
        bias_type_train_arg = args.bias_type_train if train else None

        dataset = DogsAndCatsDataset(
            image_root=dac_image_root,
            transform=img_transform,
            list_bright_file=list_bright,
            list_dark_file=list_dark,
            list_test_file=list_test,
            biased_train_type=bias_type_train_arg
        )
    elif dataset_name == 'IMDBFace':
        img_transform = transforms.Compose([
            transforms.Resize((args.image_size, args.image_size)),
            transforms.RandomHorizontalFlip() if train else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        # image_root for IMDB is where the filtered images are
        imdb_image_root = os.path.join(data_root_base, 'imdb_face', 'filtered_images')
        # manifest_file path is now relative to data_root_base/imdb_face/manifests
        manifest_dir = os.path.join(data_root_base, 'imdb_face', 'manifests')
        manifest_filename = args.train_manifest if train else args.test_manifest
        full_manifest_path = os.path.join(manifest_dir, manifest_filename) if manifest_filename else None

        bias_type_train_arg = args.bias_type_train if train else None
        
        if not full_manifest_path or not os.path.exists(full_manifest_path):
            raise FileNotFoundError(f"IMDB Manifest file not found: {full_manifest_path}. Ensure it's created in {manifest_dir}")

        dataset = IMDBFaceDataset(
            root=imdb_image_root, # This is where images like 'image_001.jpg' are
            image_list_file=full_manifest_path, # This contains 'image_001.jpg,...'
            task=args.imdb_task,
            biased_train_type=bias_type_train_arg,
            transform=img_transform
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=train, num_workers=num_workers, pin_memory=True)
    return data_loader

if __name__ == '__main__':
    # Simplified main for testing new DogsAndCatsDataset
    print("Testing DogsAndCatsDataset with list files...")
    
    # Create dummy image and list files for D&C
    dummy_data_root = './data_test_handler'
    dac_img_path = os.path.join(dummy_data_root, 'dogs_vs_cats', 'images')
    dac_list_path = os.path.join(dummy_data_root, 'dogs_vs_cats', 'lists')
    ensure_dir(dac_img_path); ensure_dir(dac_list_path)

    Image.new('RGB', (60,30)).save(os.path.join(dac_img_path, 'dog.1.jpg'))
    Image.new('RGB', (60,30)).save(os.path.join(dac_img_path, 'dog.2.jpg'))
    Image.new('RGB', (60,30)).save(os.path.join(dac_img_path, 'cat.1.jpg'))
    Image.new('RGB', (60,30)).save(os.path.join(dac_img_path, 'cat.2.jpg'))

    with open(os.path.join(dac_list_path, 'bright.txt'), 'w') as f: f.write('dog.1.jpg\ncat.1.jpg\n')
    with open(os.path.join(dac_list_path, 'dark.txt'), 'w') as f: f.write('dog.2.jpg\ncat.2.jpg\n')
    with open(os.path.join(dac_list_path, 'test.txt'), 'w') as f: f.write('dog.1.jpg\ncat.2.jpg\n')

    class DummyArgsDAC:
        dataset_name = 'DogsAndCats'
        data_root_base = dummy_data_root # Base data dir
        batch_size = 2; num_workers = 0; image_size = 64
        dac_list_bright = "bright.txt"
        dac_list_dark = "dark.txt"
        dac_list_test = "test.txt" # For test loader
        bias_type_train = "TB1" # Bright dogs, Dark cats

    args_dac_tb1 = DummyArgsDAC()
    dac_train_loader_tb1 = get_data_loader(args_dac_tb1, train=True)
    print(f"DAC TB1 Train samples: {len(dac_train_loader_tb1.dataset)}") # Expected 2 (bright dog1, dark cat2)
    for img,lbl,bias in dac_train_loader_tb1: print(f"TB1 sample: lbl {lbl}, bias {bias}"); break
    
    args_dac_tb1.bias_type_train = "TB2" # Dark dogs, Bright cats
    dac_train_loader_tb2 = get_data_loader(args_dac_tb1, train=True)
    print(f"DAC TB2 Train samples: {len(dac_train_loader_tb2.dataset)}") # Expected 2 (dark dog2, bright cat1)
    for img,lbl,bias in dac_train_loader_tb2: print(f"TB2 sample: lbl {lbl}, bias {bias}"); break

    dac_test_loader = get_data_loader(args_dac_tb1, train=False) # Uses dac_list_test
    print(f"DAC Test samples: {len(dac_test_loader.dataset)}") # Expected 2 from test.txt
    for img,lbl,bias in dac_test_loader: print(f"Test sample: lbl {lbl}, bias {bias}"); break

    # Clean up dummy files
    import shutil
    if os.path.exists(dummy_data_root): shutil.rmtree(dummy_data_root)
