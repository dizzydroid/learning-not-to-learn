# Learning Not To Learn: A User-Friendly Implementation

This repository provides a user-friendly PyTorch implementation of the paper "[Learning Not to Learn: Training Deep Neural Networks with Biased Data](https://arxiv.org/abs/1812.10352)" by Kim et al. (CVPR 2019). The goal of this project is to replicate the paper's adversarial approach to mitigate dataset bias in a way that is more accessible, configurable, and easier to understand and extend.

The core idea is to train a feature extractor that "unlearns" bias information by making it difficult for a separate bias prediction network to identify the bias from the extracted features, while still allowing a main task classifier to perform well.

## Features

* **Modular Codebase:** Clear separation of concerns for data loading, model definitions, training logic, and utilities.
* **Configuration-Driven:** Experiments are managed via YAML configuration files, making it easy to define and track different setups.
* **Automated Workflow:** A `Makefile` is provided for common tasks like setup, training, evaluation, and cleaning.
* **User-Friendly Setup:** Includes scripts for dataset download and clear instructions.
* **Phased Training:** Supports optional pre-training phases for components.
* **Comprehensive Logging:** Console logging and TensorBoard integration for tracking metrics.
* **Robust Checkpointing:** Saves and loads model checkpoints, including optimizer states, for resuming training and evaluation. Includes "best model" saving.
* **Gradient Reversal Layer (GRL):** Implements GRL for the adversarial training component.
* **Jupyter Notebooks:** For data exploration, model sanity checks, and (template for) results analysis.

## Project Structure

```

learning-not-to-learn/
├── .gitignore               # Git ignore file
├── Makefile                 # Automation for common tasks
├── README.md                # This file
├── requirements.txt         # Python package dependencies
├── environment.yml          # Optional: For Conda environments
│
├── configs/                 # Configuration files for experiments
│   └── colored_mnist_default.yaml # Example config
│
├── data/                    # Default location for datasets
│   └── colored_mnist/       # Data for Colored MNIST (after download script)
│   └── README.md            # Brief note about dataset storage
│
├── notebooks/               # Jupyter notebooks
│   ├── 01_data_exploration_and_visualization.ipynb
│   ├── 02_model_sanity_checks.ipynb
│   └── 03_results_analysis.ipynb (template)
│
├── results/                 # Output directory (logs, checkpoints, plots) - in .gitignore
│
├── scripts/                 # Utility scripts
│   └── download_dataset.sh  # Script to download and extract Colored MNIST
│
└── src/                     # Main source code
├── __init__.py
├── data_loader.py       # Dataset and DataLoader classes
├── main.py              # Main script for training and evaluation
├── models.py            # Neural network architecture definitions (f, g, h, GRL)
├── trainer.py           # Trainer class with training/evaluation logic
└── utils.py             # Utility functions (config loading, seeding, logging)

````

## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/dizzydroid/learning-not-to-learn
cd learning-not-to-learn
````

### 2\. Create a Virtual Environment (Recommended)

**Using `venv`:**

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

**Using Conda:**

```bash
conda create -n lntl_env python=3.9  # Or your preferred Python version
conda activate lntl_env
```

### 3\. Setup Project using Makefile

The easiest way to install dependencies and download the dataset is by using the `Makefile`:

```bash
make setup
```

This command will:

1.  Install Python dependencies from `requirements.txt` (including `gdown` for dataset download).
2.  Download and extract the Colored MNIST dataset to `./data/colored_mnist/` using `scripts/download_dataset.sh`.

Alternatively, you can run these steps manually:

  * Install dependencies: `make install` or `pip install -r requirements.txt && pip install gdown`
  * Download data: `make download_data` or `bash scripts/download_dataset.sh`

## Configuration

Experiments are controlled by YAML configuration files located in the `configs/` directory. `configs/colored_mnist_default.yaml` is provided as a template. Modify this file or create new ones to define your experimental setup (dataset paths, model parameters, training hyperparameters, etc.).

You can list available configurations using:

```bash
make list_configs
```

## Running Experiments with Makefile

The `Makefile` provides convenient shortcuts for common operations. Run `make help` to see all available targets.

### Training

  * **Run a full training session** (uses `DEFAULT_TRAIN_EPOCHS` from Makefile, e.g., 50 epochs, and `configs/colored_mnist_default.yaml`):
    ```bash
    make train
    ```
  * **Run a quick 1-epoch test training:**
    ```bash
    make quick_train
    ```
  * **Customize training:**
    ```bash
    make train EPOCHS=100 CONFIG=configs/your_custom_config.yaml DEVICE=cuda:0
    ```

Outputs (logs, TensorBoard files, checkpoints) will be saved to the directory specified in the `logging.output_dir` and `project.experiment_name` fields of your config (e.g., `results/colored_mnist_baseline/`).

### Resuming Training

To resume training from a checkpoint, you'll need to use the `python src/main.py` command directly, specifying the checkpoint path:

```bash
python src/main.py --config configs/your_config.yaml --mode train --checkpoint_path path/to/your/checkpoint.pth
```

### Evaluation

  * **Evaluate the best model** from the default experiment run:
    ```bash
    make eval
    ```
  * **Evaluate a specific checkpoint:**
    ```bash
    make eval CHECKPOINT=./results/colored_mnist_baseline/checkpoints/checkpoint_epoch_XX.pth
    ```

### Monitoring with TensorBoard

  * Launch TensorBoard for the default experiment:
    ```bash
    make tensorboard
    ```
    Then open `http://localhost:6006` in your browser.

## Jupyter Notebooks

The `notebooks/` directory contains:

  * **`01_data_exploration_and_visualization.ipynb`**: Load and visualize the Colored MNIST dataset to understand its properties and biases.
  * **`02_model_sanity_checks.ipynb`**: Instantiate and test individual model components ($f$, g, $h$) with dummy data to verify shapes and forward passes.
  * **`03_results_analysis.ipynb`**: A template for loading metrics from TensorBoard, plotting learning curves, and performing other analyses on completed training runs.

Start Jupyter Lab/Notebook using:

```bash
make notebook
```

## Core Logic ("Learning Not To Learn")

1.  **Feature Extractor ($f$)**: A network (e.g., CNN) that learns a representation of the input data.
2.  **Task Classifier ($g$)**: A network (e.g., MLP) that takes features from $f$ and predicts the main task label.
3.  **Bias Predictor ($h$)**: A network that takes features from $f$ and tries to predict a known bias in the data.
4.  **Adversarial Training**:
      * $f$ and $g$ are trained to minimize the main task classification loss.
      * $h$ is trained to minimize its bias prediction loss.
      * $f$ is *also* trained to *maximize* $h$'s bias prediction loss (typically via a Gradient Reversal Layer - GRL).
      * The `training.adversarial_lambda` in the config controls the strength of this adversarial component.

The goal is for $f$ to learn features that are discriminative for the main task but contain minimal information about the unwanted bias.

## Citation

```bibtex
@inproceedings{kim2019learning,
  title={Learning not to learn: Training deep neural networks with biased data},
  author={Kim, Byungju and Kim, Hyunwoo and Kim, Kyungsu and Kim, Sungjin and Kim, Junmo},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={9012--9020},
  year={2019}
}
```

## License

This project is licensed under the MIT License - see the [`LICENSE`](LICENSE) file for details.
