import torch
from src.models import FeatureExtractor, Classifier, BiasPredictor

def test_forward_pass():
    x = torch.randn(8, 3, 28, 28)
    f = FeatureExtractor()
    c = Classifier()
    b = BiasPredictor()
    feat = f(x)
    out_cls = c(feat)
    out_bias = b(feat)
    assert out_cls.shape == (8, 10)
    assert out_bias.shape == (8, 10)
