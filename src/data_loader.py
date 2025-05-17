import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import logging

from .utils import ensure_dir # Assuming utils.py is in the same directory (src)

class ColoredMNISTDataset(Dataset):
    """
    Colored MNIST Dataset.
    Loads images and labels from pre-generated .npy files as used in the
    'Learning Not To Learn' (feidfoe) repository.
    The bias label (a color map) is derived directly from the image pixels.
    """
    def __init__(self, config: dict, split: str = 'train'):
        """
        Args:
            config (dict): The global configuration dictionary, expecting data sub-config.
            split (str): 'train' or 'test'.
        """
        self.config_data = config['data']
        self.split = split.lower()
        
        if self.split not in ['train', 'test']:
            raise ValueError(f"Invalid split '{split}'. Must be 'train' or 'test'.")

        self.root_dir = self.config_data['path']
        self.color_var = self.config_data['color_var']
        
        # Construct .npy file path
        self.npy_file_path = os.path.join(
            self.root_dir, 
            f"mnist_10color_jitter_var_{self.color_var:.3f}.npy"
        )

        if not os.path.exists(self.npy_file_path):
            error_msg = (
                f"Dataset file not found: {self.npy_file_path}\n"
                f"Please download it from the link provided in the 'feidfoe/learning-not-to-learn' "
                f"repository (Google Drive) or ensure your 'data.path' and 'data.color_var' "
                f"in the config are correct. Place the .npy file in the specified path."
            )
            logging.error(error_msg)
            raise FileNotFoundError(error_msg)

        logging.info(f"Loading data from: {self.npy_file_path}")
        # latin1 encoding is used in the original feidfoe data_loader.py
        data_dic = np.load(self.npy_file_path, allow_pickle=True, encoding='latin1').item()
        
        self.images = data_dic[f'{self.split}_image'] # Expected shape (N, H, W, C), e.g., (N, 28, 28, 3)
        self.main_labels = data_dic[f'{self.split}_label'] # Expected shape (N,), digit labels

        # Image transformation: ToTensor and Normalize with feidfoe's stats
        # These stats are ((0.4914,0.4822,0.4465), (0.2023,0.1994,0.2010)) which are typically CIFAR-10 stats.
        # This implies the images in the .npy file are indeed 3-channel and these stats are appropriate.
        self.img_transform = transforms.Compose([
            transforms.ToTensor(), # Converts PIL Image or numpy.ndarray (H x W x C) in range [0, 255] to a torch.FloatTensor of shape (C x H x W) in range [0.0, 1.0]
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
        ])
        
        # This is used internally for the bias label extraction process
        self._to_pil = transforms.ToPILImage()

        logging.info(f"Loaded {self.split} dataset: {len(self.images)} samples.")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        # Image data (numpy array, HxWxC, uint8)
        img_data_np = self.images[index] 
        
        # Main task label (digit)
        main_label = self.main_labels[index]
        main_label_tensor = torch.tensor(main_label, dtype=torch.long)

        # Convert numpy image data to PIL Image for transformations
        # The .npy files store images as (H,W,C) numpy arrays.
        # ToPILImage expects (H,W,C) for multi-channel.
        pil_img = self._to_pil(img_data_np)

        # Apply transformations for the main image input to the network
        transformed_img = self.img_transform(pil_img) # Output: (C, H, W) tensor

        # Bias Label Extraction (replicating feidfoe's logic for `label_image`)
        # This process derives a pixel-wise color target map.
        # 1. Resize image to 14x14
        bias_label_map_pil = pil_img.resize((14, 14), Image.NEAREST)
        
        # 2. Convert to numpy array (H_small x W_small x C)
        bias_label_map_np = np.array(bias_label_map_pil) 
        
        # 3. Convert to PyTorch tensor (C x H_small x W_small)
        bias_label_map_tensor = torch.from_numpy(bias_label_map_np.transpose((2, 0, 1))).contiguous()
        
        # 4. Create a mask for very dark pixels (original logic)
        # Values < 0.00001 are considered 0.
        # The mask will have 255 where original was ~0, and 0 otherwise.
        mask_image = torch.lt(bias_label_map_tensor.float() - 0.00001, 0.) * 255.
        
        # 5. Quantize color values into bins (0-7 range for each channel)
        # Original pixels are likely 0-255. Dividing by 32 maps them to 0-7.
        bias_label_map_tensor = torch.div(bias_label_map_tensor, 32, rounding_mode='floor') # Use floor division for quantization
        
        # 6. Add the mask: pixels that were originally ~0 and became 0 after quantization
        # will now become 255. This is likely an 'ignore_index' for the loss.
        bias_label_map_tensor = bias_label_map_tensor + mask_image
        
        # 7. Convert to Long type for CrossEntropyLoss
        bias_label_map_tensor = bias_label_map_tensor.long() # Shape: (C, 14, 14)

        return transformed_img, bias_label_map_tensor, main_label_tensor


def get_data_loaders(config: dict, create_val_loader: bool = False, val_split_ratio: float = 0.1) -> tuple:
    """
    Creates and returns PyTorch DataLoaders for the Colored MNIST dataset.

    Args:
        config (dict): The global configuration dictionary.
        create_val_loader (bool): Whether to create a validation loader by splitting the training set.
        val_split_ratio (float): Fraction of training data to use for validation if create_val_loader is True.


    Returns:
        tuple: (train_loader, test_loader, val_loader (or None))
    """
    data_config = config['data']
    
    train_dataset = ColoredMNISTDataset(config=config, split='train')
    test_dataset = ColoredMNISTDataset(config=config, split='test')

    val_loader = None
    if create_val_loader:
        if not (0 < val_split_ratio < 1):
            raise ValueError("val_split_ratio must be between 0 and 1.")
        
        num_train = len(train_dataset)
        indices = list(range(num_train))
        split = int(np.floor(val_split_ratio * num_train))
        
        # For reproducibility of split, ensure random state is fixed if needed
        # np.random.seed(config.get('project', {}).get('seed', 42)) # Ensure seed is set before shuffle
        np.random.shuffle(indices) # Shuffle indices
        
        train_idx, val_idx = indices[split:], indices[:split]
        
        # Create Subset datasets
        train_subset = torch.utils.data.Subset(train_dataset, train_idx)
        val_dataset = torch.utils.data.Subset(train_dataset, val_idx) # Use original train_dataset for subset

        logging.info(f"Split training data: {len(train_subset)} for training, {len(val_dataset)} for validation.")
        
        train_loader = DataLoader(
            train_subset,
            batch_size=data_config['train_batch_size'],
            shuffle=True, # Shuffle training data
            num_workers=data_config.get('num_workers', 4),
            pin_memory=data_config.get('pin_memory', True)
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=data_config.get('val_batch_size', data_config['train_batch_size']),
            shuffle=False, # No need to shuffle validation data
            num_workers=data_config.get('num_workers', 4),
            pin_memory=data_config.get('pin_memory', True)
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=data_config['train_batch_size'],
            shuffle=True,
            num_workers=data_config.get('num_workers', 4),
            pin_memory=data_config.get('pin_memory', True)
        )

    test_loader = DataLoader(
        test_dataset,
        batch_size=data_config.get('test_batch_size', data_config['train_batch_size']),
        shuffle=False,
        num_workers=data_config.get('num_workers', 4),
        pin_memory=data_config.get('pin_memory', True)
    )
    
    logging.info(f"Train DataLoader: {len(train_loader.dataset)} samples, Batch size: {data_config['train_batch_size']}")
    if val_loader:
        logging.info(f"Validation DataLoader: {len(val_loader.dataset)} samples, Batch size: {data_config.get('val_batch_size', data_config['train_batch_size'])}")
    logging.info(f"Test DataLoader: {len(test_loader.dataset)} samples, Batch size: {data_config.get('test_batch_size', data_config['train_batch_size'])}")

    return train_loader, test_loader, val_loader


# --- Example Usage (for testing this script directly) ---
if __name__ == '__main__':
    # This is a placeholder for direct testing of data_loader.py.
    # You'll need a sample config and the .npy file.
    
    # 1. Setup basic logging (from utils.py)
    # For direct testing, you might need to adjust paths if utils is not directly accessible
    # or add a simplified logging setup here.
    try:
        from utils import setup_logging, ensure_dir # If run from project root: python -m src.data_loader
    except ImportError: # If run directly: python src/data_loader.py
        # Simplified logging for direct script run if utils not found easily
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        def ensure_dir(dir_path):
            if dir_path: os.makedirs(dir_path, exist_ok=True)

    setup_logging()
    
    logging.info("Testing data_loader.py...")

    # 2. Create a dummy config for testing
    # IMPORTANT: Replace './data/colored_mnist_test/' and 'color_var' 
    # with the actual path to your downloaded .npy file and its color_var.
    
    # Create a dummy directory and a tiny dummy .npy file for the test to run
    dummy_data_path = './tmp_data/colored_mnist_dummy/'
    ensure_dir(dummy_data_path)
    dummy_color_var = 0.030 
    dummy_npy_filename = f"mnist_10color_jitter_var_{dummy_color_var:.3f}.npy"
    dummy_npy_filepath = os.path.join(dummy_data_path, dummy_npy_filename)

    if not os.path.exists(dummy_npy_filepath):
        logging.warning(f"Creating a dummy .npy file for testing: {dummy_npy_filepath}")
        num_samples = 10
        train_images = np.random.randint(0, 256, size=(num_samples, 28, 28, 3), dtype=np.uint8)
        train_labels = np.random.randint(0, 10, size=(num_samples,), dtype=np.int64)
        test_images = np.random.randint(0, 256, size=(num_samples // 2, 28, 28, 3), dtype=np.uint8)
        test_labels = np.random.randint(0, 10, size=(num_samples // 2,), dtype=np.int64)
        dummy_data_dic = {
            'train_image': train_images, 'train_label': train_labels,
            'test_image': test_images, 'test_label': test_labels
        }
        np.save(dummy_npy_filepath, dummy_data_dic)
        logging.info(f"Dummy .npy file created with {num_samples} train and {num_samples//2} test samples.")
    
    test_config = {
        'project': {'seed': 42},
        'data': {
            'name': "ColoredMNIST_Test",
            'path': dummy_data_path, # Path to the directory containing the .npy file
            'color_var': dummy_color_var,   # Matches the .npy filename
            'train_batch_size': 4,
            'test_batch_size': 2,
            'num_workers': 0, # Simpler for testing
            'pin_memory': False
        },
        # Add other minimal sections if your classes expect them
    }

    # 3. Test get_data_loaders
    try:
        train_loader, test_loader, val_loader = get_data_loaders(test_config, create_val_loader=True, val_split_ratio=0.2)
        
        logging.info(f"Successfully created data loaders.")
        logging.info(f"Number of training batches: {len(train_loader)}")
        if val_loader:
            logging.info(f"Number of validation batches: {len(val_loader)}")
        logging.info(f"Number of test batches: {len(test_loader)}")

        # 4. Fetch a sample batch from train_loader
        logging.info("Fetching a sample batch from train_loader...")
        sample_images, sample_bias_maps, sample_main_labels = next(iter(train_loader))
        
        logging.info(f"Sample Images batch shape: {sample_images.shape}")     # Expected: (batch_size, C, H, W) e.g. (4, 3, 28, 28)
        logging.info(f"Sample Bias Maps batch shape: {sample_bias_maps.shape}")# Expected: (batch_size, C, 14, 14) e.g. (4, 3, 14, 14)
        logging.info(f"Sample Main Labels batch shape: {sample_main_labels.shape}")# Expected: (batch_size,) e.g. (4,)
        
        logging.info(f"Transformed image dtype: {sample_images.dtype}")
        logging.info(f"Bias map dtype: {sample_bias_maps.dtype}, Min: {sample_bias_maps.min()}, Max: {sample_bias_maps.max()}")
        logging.info(f"Main label dtype: {sample_main_labels.dtype}")

    except FileNotFoundError as e:
        logging.error(f"Test failed: {e}")
        logging.error("Please ensure you have the correct .npy file in the specified 'data.path' "
                      "or that the dummy file generation for testing is working.")
    except Exception as e:
        logging.error(f"An error occurred during testing: {e}", exc_info=True)
    finally:
        # Clean up dummy file and directory if you want
        # if os.path.exists(dummy_npy_filepath):
        #     os.remove(dummy_npy_filepath)
        # if os.path.exists(dummy_data_path):
        #     if not os.listdir(dummy_data_path): # Check if dir is empty
        #         os.rmdir(dummy_data_path)
        #     if not os.listdir(os.path.dirname(dummy_data_path)): # Check if parent 'tmp_data' is empty
        #          os.rmdir(os.path.dirname(dummy_data_path))
        pass