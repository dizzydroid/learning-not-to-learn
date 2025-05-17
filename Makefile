# Makefile for the Learning Not To Learn project

# --- Variables ---
# Python interpreter (uses the one in the current environment)
PYTHON = python
PIP = pip

# Project directories
SRC_DIR = src
CONFIGS_DIR = configs
SCRIPTS_DIR = scripts
RESULTS_DIR = ./results
NOTEBOOKS_DIR = notebooks
DATA_DIR = ./data/colored_mnist # Default data directory

# Default configuration file
DEFAULT_CONFIG = $(CONFIGS_DIR)/colored_mnist_default.yaml
DEFAULT_EXPERIMENT_NAME = colored_mnist_baseline # Must match experiment_name in DEFAULT_CONFIG
DEFAULT_RESULTS_SUBDIR = $(RESULTS_DIR)/$(DEFAULT_EXPERIMENT_NAME)
DEFAULT_CHECKPOINTS_DIR = $(DEFAULT_RESULTS_SUBDIR)/checkpoints
DEFAULT_BEST_MODEL = $(DEFAULT_CHECKPOINTS_DIR)/best_model.pth
DEFAULT_TENSORBOARD_LOGDIR = $(DEFAULT_RESULTS_SUBDIR)/tensorboard_logs

# Default number of epochs for a quick test train, can be overridden
DEFAULT_TRAIN_EPOCHS = 50
QUICK_TRAIN_EPOCHS = 1


# Phony targets (targets that don't represent actual files)
.PHONY: help install setup download_data train quick_train eval tensorboard clean clean_results clean_data list_configs

# --- Main Targets ---

help:
	@echo "Makefile for Learning Not To Learn Project"
	@echo ""
	@echo "Usage: make [target]"
	@echo ""
	@echo "Available targets:"
	@echo "  install          Install/update Python dependencies from requirements.txt."
	@echo "  setup            Run install and download_data."
	@echo "  download_data    Download and extract the Colored MNIST dataset."
	@echo "  train            Run training with the default configuration ($(DEFAULT_CONFIG))."
	@echo "                   Override epochs: make train EPOCHS=100"
	@echo "                   Override config: make train CONFIG=configs/another_config.yaml"
	@echo "                   Override device: make train DEVICE=cuda:0"
	@echo "  quick_train      Run a short training (1 epoch) with default config for testing."
	@echo "  eval             Evaluate the best model from the default training run."
	@echo "                   Override checkpoint: make eval CHECKPOINT=path/to/model.pth"
	@echo "                   Override config: make eval CONFIG=configs/another_config.yaml"
	@echo "  tensorboard      Launch TensorBoard for the default experiment."
	@echo "                   Override logdir: make tensorboard LOGDIR=./results/another_exp/tensorboard_logs"
	@echo "  notebook         Start Jupyter Lab/Notebook (assumes jupyter is installed)."
	@echo "  clean            Remove Python cache, build artifacts, and generated results."
	@echo "  clean_results    Remove only the results directory."
	@echo "  clean_data       Remove only the downloaded data directory ($(DATA_DIR))."
	@echo "  list_configs     List available YAML configuration files."
	@echo ""

# Default target (if user just types `make`)
all: help

# --- Setup and Installation ---

install: requirements.txt
	@echo "Installing/updating Python dependencies..."
	$(PIP) install -r requirements.txt
	@echo "Verifying gdown installation (for dataset download)..."
	$(PIP) show gdown > /dev/null || (echo "gdown not found, installing..." && $(PIP) install gdown)
	@echo "Dependencies installed."

download_data: $(SCRIPTS_DIR)/download_dataset.sh
	@echo "Downloading and extracting dataset..."
	@if [ ! -d "$(DATA_DIR)" ]; then \
		echo "Data directory $(DATA_DIR) not found, creating..."; \
		mkdir -p $(DATA_DIR); \
	fi
	# Pass the target directory to the script
	bash $(SCRIPTS_DIR)/download_dataset.sh $(DATA_DIR)
	@echo "Dataset download script executed."

setup: install download_data
	@echo "Project setup complete."

# --- Training and Evaluation ---

# Variables for overriding in train/eval targets
EPOCHS ?= $(DEFAULT_TRAIN_EPOCHS)
CONFIG ?= $(DEFAULT_CONFIG)
DEVICE ?= auto # Default to 'auto', can be 'cpu', 'cuda', 'cuda:0', etc.
CHECKPOINT ?= $(DEFAULT_BEST_MODEL)

train:
	@echo "Starting training with config: $(CONFIG), epochs: $(EPOCHS), device: $(DEVICE)..."
	$(PYTHON) $(SRC_DIR)/main.py --config $(CONFIG) --mode train --num_epochs $(EPOCHS) --device $(DEVICE)

quick_train:
	@echo "Starting quick training (1 epoch) with config: $(CONFIG), device: $(DEVICE)..."
	$(PYTHON) $(SRC_DIR)/main.py --config $(CONFIG) --mode train --num_epochs $(QUICK_TRAIN_EPOCHS) --device $(DEVICE)

eval:
	@echo "Evaluating model with config: $(CONFIG), checkpoint: $(CHECKPOINT), device: $(DEVICE)..."
	@if [ ! -f "$(CHECKPOINT)" ] && [ "$(CHECKPOINT)" = "$(DEFAULT_BEST_MODEL)" ]; then \
		echo "Warning: Default best model $(DEFAULT_BEST_MODEL) not found. Ensure training was run first or specify a CHECKPOINT."; \
	fi
	$(PYTHON) $(SRC_DIR)/main.py --config $(CONFIG) --mode evaluate --checkpoint_path $(CHECKPOINT) --device $(DEVICE)

# --- TensorBoard ---
LOGDIR ?= $(DEFAULT_TENSORBOARD_LOGDIR)

tensorboard:
	@echo "Launching TensorBoard with log directory: $(LOGDIR)..."
	@echo "Open your browser to http://localhost:6006 (or the URL provided by TensorBoard)"
	tensorboard --logdir $(LOGDIR)

# --- Jupyter Notebook ---
notebook:
	@echo "Starting Jupyter Lab/Notebook..."
	@echo "Ensure you have jupyter lab or notebook installed (e.g., pip install jupyterlab)."
	# Tries jupyter lab first, then jupyter notebook
	jupyter lab --notebook-dir=$(NOTEBOOKS_DIR) || jupyter notebook --notebook-dir=$(NOTEBOOKS_DIR)


# --- Cleaning ---

clean:
	@echo "Cleaning up project..."
	# Remove Python cache files
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete
	# Remove build artifacts (if any)
	rm -rf build/ dist/ *.egg-info/
	# Remove results
	$(MAKE) clean_results
	@echo "Cleanup complete."

clean_results:
	@echo "Removing results directory: $(RESULTS_DIR)..."
	rm -rf $(RESULTS_DIR)
	@echo "Results directory removed."

clean_data:
	@echo "Removing data directory: $(DATA_DIR)..."
	rm -rf $(DATA_DIR)
	@echo "Data directory removed."

# --- Utilities ---
list_configs:
	@echo "Available configuration files in $(CONFIGS_DIR):"
	@ls -1 $(CONFIGS_DIR)/*.yaml | xargs -n 1 basename
	@echo ""
	@echo "Example usage: make train CONFIG=configs/your_chosen_config.yaml"

