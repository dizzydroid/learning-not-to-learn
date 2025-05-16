import torch
import torch.nn as nn
import torch.optim as optim
from src.models import FeatureExtractor, Classifier, BiasPredictor
from src.datasets import get_colored_mnist

def train(config):
    # Set up models
    feature_extractor = FeatureExtractor()
    classifier = Classifier()
    bias_predictor = BiasPredictor()
    # TODO: Implement training loop as in the paper (including adversarial loss)
    pass

if __name__ == '__main__':
    # TODO: Load config from YAML or argparse
    config = {}
    train(config)
