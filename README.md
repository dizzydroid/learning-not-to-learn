# Learning Not to Learn: Training Deep Neural Networks with Biased Data

This repository provides a PyTorch implementation of the paper:
**"Learning Not to Learn: Training Deep Neural Networks with Biased Data"**
by Byungju Kim, Hyunwoo Kim, Sungjin Kim, Junmo Kim, and Kyungsu Kim.

**Paper:** [arXiv:1812.10352](https://arxiv.org/abs/1812.10352)
(CVPR 2019)

## Overview

Deep neural networks are powerful learners that can inadvertently pick up and rely on spurious correlations or biases present in training data. When these biases are not representative of the true underlying task, the model's performance can degrade significantly on unbiased test sets.

This paper introduces an adversarial regularization algorithm designed to train deep neural networks that are robust to such dataset biases. The core idea is to make the learned feature representations informative for the main task while being uninformative about the known bias. This is achieved by:

1.  A **Feature Extractor network ($f$)**: Learns feature embeddings from input data.
2.  A **Label Predictor network ($g$)**: Predicts the target labels from the features.
3.  A **Bias Predictor network ($h$)**: Adversarially trained to predict the bias from the features.

The feature extractor ($f$) is trained to support the label predictor ($g$) while simultaneously "fooling" the bias predictor ($h$), often using a Gradient Reversal Layer (GRL). The overall objective aims to minimize the mutual information between the learned features and the bias variables, forcing the network to "unlearn" the bias.

This implementation focuses on reproducing the experiments from the paper using datasets like Colored MNIST, Dogs and Cats (with color bias), and IMDB-Face (with age/gender bias).

## Repository Structure

```

.
├── Makefile                # For managing tasks like setup, training, cleaning
├── README.md               # This file
├── requirements.txt        # Python dependencies
├── fetch_data.py           # Script to download datasets and setup directories
├── main.py                 # Main script to run experiments (parses args, inits components)
├── solver.py               # Contains the Solver class with training and evaluation logic
├── network.py              # Defines network architectures (f, g, h, GRL)
├── data_handler.py         # Dataset classes and data loader functions
├── utils.py                # Utility functions (e.g., accuracy, seeding)
│
├── data/                   # Root directory for datasets (created by fetch_data.py)
│   ├── mnist_colored_data/ # For Colored MNIST
│   ├── dogs_vs_cats/       # For Dogs and Cats dataset
│   │   ├── images/         # Raw cat/dog images
│   │   ├── lists/          # list_bright.txt, list_dark.txt, list_test_unbiased.txt
│   │   └── raw_kaggle_downloads/ # Downloaded zips from Kaggle
│   └── imdb_face/          # For IMDB-Face dataset
│       ├── filtered_images/ # Authors' pre-filtered images (from GDrive)
│       ├── manifests/      # CSV manifest files for train/test splits
│       └── (imdb_face_filtered.zip) # Archive from GDrive
│
├── scripts/                # Shell scripts to run specific experiments
│   ├── run_cmnist.sh
│   ├── run_dogs_cats_tb1.sh
│   └── run_imdb_eb1_gender.sh
│
└── outputs/                # Directory for saving logs, checkpoints, and results
├── checkpoints/
├── logs/           # TensorBoard logs
└── (test_results.npy) # Saved predictions on test set

````

## Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/dizzydroid/learning-not-to-learn
cd learning-not-to-learn
````

### 2. Install Dependencies

It's highly recommended to use a Python virtual environment (e.g., Conda or venv).

```bash
make setup
# This will execute: pip install -r requirements.txt
```

The `requirements.txt` file should include `torch`, `torchvision`, `numpy`, `Pillow`, `pyyaml`, `tqdm`, `kaggle`, `gdown`, `requests`, `tensorboard`, etc.

### 3. Set Up Kaggle API (for Dogs vs. Cats raw data)

If you want the script to attempt downloading the Dogs vs. Cats dataset from Kaggle:

  * Ensure the Kaggle API is installed (covered by `make setup` if `kaggle` is in `requirements.txt`).
  * Download your `kaggle.json` API token from your Kaggle account page (My Account > API > Create New API Token).
  * Place it in `~/.kaggle/kaggle.json`.
  * Set appropriate permissions: `chmod 600 ~/.kaggle/kaggle.json` (on Linux/macOS).

### 4. Fetch and Prepare Datasets

Run the data fetching script. This script will:

  * Download MNIST automatically via torchvision.
  * Attempt to download Dogs vs. Cats raw data from Kaggle using the API (and a fallback public dataset).
  * Attempt to download the authors' pre-filtered IMDB-Face data from the provided Google Drive link using `gdown`.
  * Create the necessary directory structures within `./data/`.

<!-- end list -->

```bash
make fetch_data
# This executes: python fetch_data.py
```

**Crucial Manual Steps after `make fetch_data`:**

  * **Dogs vs. Cats:**
      * Verify that images are populated in `data/dogs_vs_cats/images/`.
      * **Manually categorize** these images by color ("bright", "dark") as per the paper's description.
      * Create `list_bright.txt`, `list_dark.txt` (containing filenames of bright and dark images, respectively) and your test image list (e.g., `list_test_unbiased.txt`) in the `data/dogs_vs_cats/lists/` directory. The `fetch_data.py` script prints detailed instructions and creates example list files.
  * **IMDB-Face:**
      * Verify that the filtered images (downloaded from Google Drive) are correctly extracted into `data/imdb_face/filtered_images/`.
      * **Create manifest CSV files** (e.g., `train_eb1_gender.csv`, `test_unbiased_gender.csv`) in the `data/imdb_face/manifests/` directory. These CSVs should list image filenames (relative to the `filtered_images/` directory), their true gender labels, and true age values. This allows `data_handler.py` to construct the biased training sets (EB1, EB2) and the test set according to the paper's methodology. `fetch_data.py` provides example manifest structures.

## Running Experiments

Experiments are configured and run using shell scripts located in the `scripts/` directory. These scripts invoke `main.py` with specific command-line arguments. The `Makefile` provides convenient shortcuts.

### Using Makefile

Ensure you are in the project root directory.

```bash
# Run the Colored MNIST experiment
make run_cmnist

# Run the Dogs and Cats TB1 bias experiment
# (Ensure scripts/run_dogs_cats_tb1.sh is configured and list files are ready)
make run_dogs_cats_tb1

# Run the IMDB Face EB1 (Gender task) experiment
# (Ensure scripts/run_imdb_eb1_gender.sh is configured and manifest files are ready)
make run_imdb_eb1_gen
```

You can add more targets to the `Makefile` for other scripts you create in the `scripts/` directory.

### Using Shell Scripts Directly

```bash
bash scripts/run_cmnist.sh
# or
bash scripts/run_dogs_cats_tb1.sh
```

Modify parameters directly within the `.sh` files for different experimental setups.

### Using `main.py` Directly

For custom runs or debugging:

```bash
python main.py 
    --dataset_name ColoredMNIST 
    --data_root_base ./data 
    --num_main_classes 10 
    --num_bias_classes 10 
    --f_network_name SimpleCNN 
    --g_network_name SimpleCNN 
    --h_network_name SimpleCNN_ConvBias 
    --experiment_name custom_cmnist_run 
    --epochs 10 
    # ... add other necessary arguments ...
```

Run `python main.py --help` to see all available command-line options.

## Monitoring Training and Results

### TensorBoard

Training progress (losses, accuracies for main task and bias predictor, GRL alpha schedule) is logged to TensorBoard.
To launch TensorBoard:

```bash
tensorboard --logdir ./outputs
```

Then open your browser to `http://localhost:6006` (or the URL TensorBoard provides). Logs for each experiment are stored in `outputs/[experiment_name]/logs/`.

### Model Checkpoints

Trained model checkpoints (`.pth` files containing model weights and optimizer states) are saved periodically in `outputs/[experiment_name]/checkpoints/`. These can be used to resume training or for later evaluation.

### Test Set Evaluation & Output

To evaluate a trained model on the test set and save its predictions:

1.  Identify the path to your trained model checkpoint (e.g., `outputs/your_experiment_name/checkpoints/model_epoch_X.pth`).
2.  Use `main.py` with `mode="test"` and provide the checkpoint path.
    Example (modify your script or run directly):
    ```bash
    python main.py 
        --mode "test" 
        --load_checkpoint_path "outputs/cmnist_debias_sigma0.05_mu1.0_lambda0.5/checkpoints/model_epoch_50.pth" 
        --test_output_npy "cmnist_results_epoch50.npy" 
        --dataset_name "ColoredMNIST" 
        --data_root_base "./data" 
        --num_main_classes 10 
        --num_bias_classes 10 
        --f_network_name "SimpleCNN" 
        --g_network_name "SimpleCNN" 
        --h_network_name "SimpleCNN_ConvBias" 
        # ... ensure other dataset/model args match the training config of the loaded checkpoint ...
    ```
    This command loads the model, runs inference on the test set (as defined by `get_data_loader(args, train=False, ...)`), and saves a dictionary containing true labels, predicted labels/probabilities, and bias-related information into an `.npy` file located in `outputs/[experiment_name]/`.

### Analyzing Results

You can create Jupyter notebooks (e.g., in a `notebooks/` directory that you can create) to:

  * Load the saved `test_results.npy` files.
  * Calculate detailed fairness metrics.
  * Plot confusion matrices for both main task and bias prediction.
  * Visualize qualitative results (e.g., show images where the model overcame bias).

## Expected Results

The primary goal is to achieve high accuracy on the main classification task while the bias predictor ($h$) performs poorly on the (unbiased) test set's bias attributes. This indicates that the feature extractor ($f$) has learned representations that are largely invariant to the dataset bias. Refer to Figures 4, 5, 6, 7 and Tables 1, 2 in the original paper for quantitative benchmarks on specific datasets.

## Citation

If you use this code or the methods described in the paper, please consider citing the original work:

```bibtex
@inproceedings{kim2019learning,
  title={Learning not to learn: Training deep neural networks with biased data},
  author={Kim, Byungju and Kim, Hyunwoo and Kim, Kyungsu and Kim, Sungjin and Kim, Junmo},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={9004--9012},
  year={2019}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

  * This work is an implementation based on the CVPR 2019 paper "Learning Not to Learn: Training Deep Neural Networks with Biased Data" by Kim et al.
