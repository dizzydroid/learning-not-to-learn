# Datasets

This directory is the default location for storing datasets used by the project.

## Colored MNIST

The Colored MNIST dataset (`.npy` files) can be downloaded by running the script:
```bash
../../scripts/download_dataset.sh 
```
(Assuming you are in the project root when running the script, it will place data in `./data/colored_mnist/`)

The specific `.npy` file used for an experiment (e.g., `mnist_10color_jitter_var_0.030.npy`) is determined by the `data.path` and `data.color_var` settings in the YAML configuration file.
