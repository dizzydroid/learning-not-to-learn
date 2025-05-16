# Learning Not to Learn: Training Deep Neural Networks with Biased Data

> Implementation and research review of ["Learning Not to Learn: Training Deep Neural Networks with Biased Data"](https://arxiv.org/abs/1812.10352)  
> By Byungju Kim et al.

---

## Overview

This repository contains a clean, modular, and reproducible implementation of the mutual-information-based bias unlearning method proposed in the above paper.  
Key features:
- Modular PyTorch code for feature extractor, classifier, and bias predictor.
- Colored MNIST bias experiments, with dataset colorization.
- Support for further experiments on Cats & Dogs and IMDB-Face datasets.
- Experiment configs for easy reproducibility.
- All code organized for clarity and extension.
- Presentation slides summarizing our findings and implementation.

---

## Project Structure

```
learning-not-to-learn/
├── README.md
├── requirements.txt
├── environment.yml         # (optional, for conda)
├── LICENSE
├── .gitignore
├── data/
│   ├── raw/                # Downloaded/original datasets
│   └── processed/          # Preprocessed/colored datasets
├── notebooks/
│   └── exploratory.ipynb   # EDA & visualization
├── src/
│   ├── __init__.py
│   ├── datasets.py         # Dataset code, colored MNIST, etc.
│   ├── models.py           # Feature extractor, classifier, bias predictor
│   ├── train.py            # Training logic, loss, adversarial game
│   ├── utils.py            # Plotting, logging, helpers
├── experiments/
│   └── exp1\_colored\_mnist.yaml
├── results/
│   ├── logs/
│   ├── figures/
│   └── checkpoints/
├── slides/
│   └── presentation.pptx
└── tests/
    ├── test\_datasets.py
    ├── test\_models.py
    └── test\_utils.py

````

---

## Installation & Setup

### 1. Clone the repository

```bash
git clone https://github.com/dizzydroid/learning-not-to-learn.git
cd learning-not-to-learn
````

### 2. Install Python dependencies

```bash
pip install -r requirements.txt
```

or, for conda users:

```bash
conda env create -f environment.yml
conda activate learning-not-to-learn
```

### 3. Download Datasets

Instructions for downloading and preprocessing datasets (e.g., colored MNIST) are in `data/README.md`.

---

## Usage

* **Training Example**
  (After configuring your experiment YAML in `experiments/`):

  ```bash
  python src/train.py --config experiments/exp1_colored_mnist.yaml
  ```

* **Jupyter Notebook**
  Use the `notebooks/` directory for quick experiments and visualizations.

* **Results**
  Trained models, logs, and plots will be saved under `results/`.

---

## Slides

The project and findings are summarized in [`slides/presentation.pptx`](slides/presentation.pptx).

---

## References

* Kim, Byungju, et al. "Learning Not to Learn: Training Deep Neural Networks with Biased Data." [arXiv:1812.10352](https://arxiv.org/abs/1812.10352), 2018.
* [PyTorch](https://pytorch.org/) - The deep learning framework used for this implementation.
* [scikit-learn](https://scikit-learn.org/stable/) - For data preprocessing and evaluation metrics.
* [Matplotlib](https://matplotlib.org/) - For plotting and visualization.
* [NumPy](https://numpy.org/) - For numerical operations.
* [Pandas](https://pandas.pydata.org/) - For data manipulation and analysis.
* [Seaborn](https://seaborn.pydata.org/) - For statistical data visualization.
---

## License

This project is for academic purposes. For any other usage, refer to the LICENSE file.

---

