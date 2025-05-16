# Data Directory

This folder contains all datasets used in this project, organized for reproducibility.

---

## Directory Structure

```
data/
├── raw/         # Downloaded/original datasets (do not edit)
├── processed/   # Preprocessed, ready-to-use datasets (colorized, etc.)
└── README.md    # (this file)
````

---

## 1. Colored MNIST

### **Step 1: Download MNIST**

You can let the PyTorch/Torchvision dataset loader download the [MNIST](http://yann.lecun.com/exdb/mnist/) dataset automatically, or you can download it manually into `data/raw/`.

*Automatic (recommended):*
- When you run the code, if MNIST is not found in `data/raw/`, it will be automatically downloaded.

*Manual:*
- Download from [here](http://yann.lecun.com/exdb/mnist/).
- Place all files (e.g., `train-images-idx3-ubyte.gz`, etc.) into `data/raw/`.

### **Step 2: Generate Colored MNIST**

The original paper introduces a **color bias** by colorizing grayscale MNIST digits, assigning a mean color per digit and adding variance (see `src/datasets.py`).  
When you run the training script, colored MNIST will be generated automatically and saved in `data/processed/`.

- If you want to **pre-generate** it, you can run:
  ```bash
  python src/datasets.py --generate_colored_mnist --outdir data/processed/
    ```

## 2. Other Datasets

**Cats & Dogs and IMDB-Face**
These are optional extensions. For most research and course projects, *Colored MNIST* is enough.

* **Cats & Dogs:** [Kaggle Dogs vs. Cats](https://www.kaggle.com/c/dogs-vs-cats/data)

  * Download and unzip into `data/raw/dogs_vs_cats/`
  * Follow similar steps for processing as needed.
* **IMDB-Face:** See [IMDB-Face](https://github.com/fwang91/IMDb-Face/) or [official source](https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/)

  * Place raw images in `data/raw/imdb_face/`
  * Preprocessing scripts not included by default.

---

## Troubleshooting

* If you have issues with dataset downloading or generation, check `src/datasets.py` or open an issue in the project repository.

---

**Contact:**
For dataset or preprocessing questions, contact the repo maintainers or open an issue on GitHub.


