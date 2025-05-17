# Makefile for "Learning Not to Learn" Project (Refactored)

PYTHON = python3
REQS = requirements.txt
OUTPUT_BASE_DIR = outputs # Base for all experiment outputs
DATA_DIR = data # General data directory
FETCH_SCRIPT = fetch_data.py

# --- Default Target ---
.PHONY: help
help:
	@echo "Makefile for 'Learning Not to Learn' Project (Refactored)"
	@echo ""
	@echo "Usage:"
	@echo "  make setup             Install dependencies from $(REQS)"
	@echo "  make fetch_data        Download MNIST and setup directories for other datasets"
	@echo "  make run_cmnist        Run Colored MNIST experiment (uses scripts/run_cmnist.sh)"
	@echo "  make run_dogs_cats_tb1 Run Dogs and Cats TB1 experiment (uses scripts/run_dogs_cats_tb1.sh - you need to create this script)"
	@echo "  make run_imdb_eb1_gen  Run IMDB Face EB1 Gender experiment (uses scripts/run_imdb_eb1_gender.sh - you need to create this script)"
	@echo "  make lint              Run linting (requires flake8)"
	@echo "  make clean             Remove Python cache files and ALL contents of $(OUTPUT_BASE_DIR)"
	@echo "  make clean_pycache     Remove only Python cache files"
	@echo ""
	@echo "Note: Specific experiment parameters are now managed within the .sh files in the 'scripts/' directory."

# --- Setup ---
.PHONY: setup
setup:
	@echo "Installing dependencies from $(REQS)..."
	$(PYTHON) -m pip install -r $(REQS)
	@echo "Setup complete."

# --- Data Fetching ---
.PHONY: fetch_data
fetch_data:
	@echo "Running data fetching and setup script: $(FETCH_SCRIPT)..."
	@if [ ! -f $(FETCH_SCRIPT) ]; then \
		echo "Error: $(FETCH_SCRIPT) not found. Please ensure it's in the project root."; \
		exit 1; \
	fi
	$(PYTHON) $(FETCH_SCRIPT)

# --- Training Targets (using shell scripts) ---
.PHONY: run_cmnist
run_cmnist:
	@echo "Executing Colored MNIST experiment script..."
	bash scripts/run_cmnist.sh

.PHONY: run_dogs_cats_tb1
run_dogs_cats_tb1:
	@echo "Executing Dogs and Cats TB1 experiment script..."
	@if [ ! -f scripts/run_dogs_cats_tb1.sh ]; then \
		echo "Error: scripts/run_dogs_cats_tb1.sh not found. Please create it."; \
		exit 1; \
	fi
	bash scripts/run_dogs_cats_tb1.sh

.PHONY: run_imdb_eb1_gen
run_imdb_eb1_gen:
	@echo "Executing IMDB Face EB1 Gender experiment script..."
	@if [ ! -f scripts/run_imdb_eb1_gender.sh ]; then \
		echo "Error: scripts/run_imdb_eb1_gender.sh not found. Please create it."; \
		exit 1; \
	fi
	bash scripts/run_imdb_eb1_gender.sh

# --- Linting ---
.PHONY: lint
lint:
	@echo "Running flake8 linter on root .py files..."
	@command -v flake8 >/dev/null 2>&1 || { echo >&2 "flake8 not found. Please install it (pip install flake8)."; exit 1; }
	flake8 *.py --count --select=E9,F63,F7,F82 --show-source --statistics
	flake8 *.py --count --exit-zero --max-complexity=12 --max-line-length=120 --statistics

# --- Cleaning ---
.PHONY: clean_pycache
clean_pycache:
	@echo "Cleaning Python cache files..."
	find . -type f -name '*.py[co]' -delete
	find . -type d -name '__pycache__' -exec rm -rf {} +
	rm -rf .pytest_cache
	@echo "Python cache cleanup complete."

.PHONY: clean
clean: clean_pycache
	@echo "Cleaning ALL experiment outputs in $(OUTPUT_BASE_DIR)..."
	rm -rf $(OUTPUT_BASE_DIR)/* # Careful: this removes all subdirectories and files in outputs
	@echo "Experiment outputs cleanup complete."

