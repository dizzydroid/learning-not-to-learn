import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models as torchvision_models
from torch.autograd import Function

# --- 1. Gradient Reversal Layer (GRL) ---
class GRL(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None

def grad_reverse(x, alpha=1.0):
    return GRL.apply(x, alpha)

# --- 2. Feature Extractor Architectures (f) ---

class FeatExtractor(nn.Module):
    """ Feature Extractor (f) """
    def __init__(self, network_name='SimpleCNN', input_channels=3, pretrained=True):
        super(FeatExtractor, self).__init__()
        self.network_name = network_name

        if network_name == 'SimpleCNN':
            # For Colored MNIST
            self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=5, stride=1, padding=2)
            self.bn1 = nn.BatchNorm2d(32)
            self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2)
            self.bn2 = nn.BatchNorm2d(64)
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
            # Output: 64 x 7 x 7 for 28x28 input
            self.output_channels = 64
            self.output_dim_spatial = 7

        elif network_name == 'ResNet18':
            # For Dogs & Cats, IMDB Face
            resnet18 = torchvision_models.resnet18(weights=torchvision_models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
            self.conv1 = resnet18.conv1
            self.bn1 = resnet18.bn1
            self.relu = resnet18.relu
            self.maxpool = resnet18.maxpool
            self.layer1 = resnet18.layer1
            self.layer2 = resnet18.layer2
            self.layer3 = resnet18.layer3 # f goes up to layer3
            # Output channels after layer3 of ResNet-18 is 256
            self.output_channels = 256
            # Spatial dim depends on input, e.g., 14 for 224x224 input (224/16)
            # We can use AdaptiveAvgPool later if a fixed vector is needed by classifier
        else:
            raise ValueError(f"Unknown network_name for FeatExtractor: {network_name}")

    def forward(self, x):
        if self.network_name == 'SimpleCNN':
            x = F.relu(self.bn1(self.conv1(x)))
            x = self.pool(x)
            x = F.relu(self.bn2(self.conv2(x)))
            x = self.pool(x)
        elif self.network_name == 'ResNet18':
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
        return x

# --- 3. Label Predictor Architectures (g) ---
class LabelClassifier(nn.Module):
    """ Label Predictor/Classifier (g) """
    def __init__(self, network_name='SimpleCNN', num_classes=10, 
                 input_channels_f=64, input_dim_spatial_f=7): # Params from f
        super(LabelClassifier, self).__init__()
        self.network_name = network_name

        if network_name == 'SimpleCNN':
            # Takes features from FeatExtractor (SimpleCNN: 64 x 7 x 7)
            self.conv3 = nn.Conv2d(input_channels_f, 128, kernel_size=3, stride=1, padding=1)
            self.bn3 = nn.BatchNorm2d(128)
            # self.conv4 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1) # Optional 4th conv
            # self.bn4 = nn.BatchNorm2d(128)
            self.pool = nn.AdaptiveAvgPool2d((1,1))
            self.fc = nn.Linear(128, num_classes)

        elif network_name == 'ResNet18':
            # Takes features from FeatExtractor (ResNet18 layer3 output: 256 channels)
            # g uses layer4 of ResNet18
            resnet18_for_g = torchvision_models.resnet18(weights=None) # Structure only
            self.layer4 = resnet18_for_g.layer4 # Input 256 (from f), Output 512
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(512, num_classes)
        else:
            raise ValueError(f"Unknown network_name for LabelClassifier: {network_name}")

    def forward(self, features_from_f):
        if self.network_name == 'SimpleCNN':
            x = F.relu(self.bn3(self.conv3(features_from_f)))
            # x = F.relu(self.bn4(self.conv4(x))) # If using 4th conv
            x = self.pool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)
        elif self.network_name == 'ResNet18':
            x = self.layer4(features_from_f)
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)
        return x

# --- 4. Bias Predictor Architectures (h) ---
class BiasClassifier(nn.Module):
    """ Bias Predictor/Classifier (h) """
    def __init__(self, network_name='SimpleCNN_ConvBias', num_bias_classes=10,
                 input_channels_f=64, input_dim_spatial_f=7): # Params from f
        super(BiasClassifier, self).__init__()
        self.network_name = network_name # e.g. 'SimpleCNN_ConvBias', 'ResNet18_FCBias'

        if network_name == 'SimpleCNN_ConvBias': # For Colored MNIST color bias
            # Takes features from FeatExtractor (SimpleCNN: 64 x 7 x 7)
            # Paper: "h was implemented with two convolutional layers"
            self.conv1 = nn.Conv2d(input_channels_f, 32, kernel_size=3, padding=1)
            self.bn1 = nn.BatchNorm2d(32)
            self.conv2 = nn.Conv2d(32, 16, kernel_size=3, padding=1)
            self.bn2 = nn.BatchNorm2d(16)
            self.pool = nn.AdaptiveAvgPool2d((1,1))
            self.fc = nn.Linear(16, num_bias_classes)

        elif network_name == 'ResNet18_FCBias': # For IMDB Face gender/age bias from ResNet features
            # Paper: "h consisted of a single fully connected layer"
            # Takes features from FeatExtractor (ResNet18 layer3 output: 256 channels)
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(input_channels_f, num_bias_classes) # input_channels_f is 256 here

        elif network_name == 'ResNet18_ConvBias': # For Dogs&Cats color bias from ResNet features
            # This is more complex if f is deep ResNet. Paper might simplify h or f for this.
            # Using an FC layer after pooling is often more robust for ResNet features.
            # The official repo uses a simpler Feature Extractor for BAR (Dogs&Cats) than full ResNet for h.
            print(f"Warning: ResNet18_ConvBias selected for BiasClassifier. Consider if FCBias is more appropriate or if FeatExtractor for bias path should be shallower.")
            self.conv1 = nn.Conv2d(input_channels_f, 128, kernel_size=3, padding=1) # Example
            self.bn1 = nn.BatchNorm2d(128)
            self.pool = nn.AdaptiveAvgPool2d((1,1))
            self.fc = nn.Linear(128, num_bias_classes)
        else:
            raise ValueError(f"Unknown network_name for BiasClassifier: {network_name}")

    def forward(self, features_from_f):
        x = features_from_f
        if self.network_name == 'SimpleCNN_ConvBias':
            x = F.relu(self.bn1(self.conv1(x)))
            x = F.relu(self.bn2(self.conv2(x)))
            x = self.pool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)
        elif self.network_name == 'ResNet18_FCBias':
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)
        elif self.network_name == 'ResNet18_ConvBias':
            x = F.relu(self.bn1(self.conv1(x)))
            x = self.pool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)
        return x


# --- Combined Model ---
class DebiasingNetwork(nn.Module):
    def __init__(self, feat_extractor, label_classifier, bias_classifier):
        super(DebiasingNetwork, self).__init__()
        self.f = feat_extractor
        self.g = label_classifier
        self.h = bias_classifier

    def forward(self, x, alpha_grl=1.0, mode='train'):
        """
        mode: 'train', 'predict_label', 'predict_bias_no_grl'
        """
        features = self.f(x)

        if mode == 'train':
            label_output = self.g(features)
            reversed_features = grad_reverse(features, alpha_grl)
            bias_output = self.h(reversed_features)
            return label_output, bias_output
        elif mode == 'predict_label':
            label_output = self.g(features)
            return label_output
        elif mode == 'predict_bias_no_grl': # For evaluating h's true ability
            bias_output = self.h(features)
            return bias_output
        else:
            raise ValueError(f"Unknown mode: {mode}")


if __name__ == '__main__':
    print("Testing Network Architectures...")

    # --- MNIST Test ---
    f_mnist = FeatExtractor(network_name='SimpleCNN', input_channels=3)
    dummy_mnist_input = torch.randn(4, 3, 28, 28)
    mnist_feats = f_mnist(dummy_mnist_input)
    print(f"MNIST Feats shape: {mnist_feats.shape}") # Expected: [4, 64, 7, 7]

    g_mnist = LabelClassifier(network_name='SimpleCNN', num_classes=10,
                              input_channels_f=f_mnist.output_channels,
                              input_dim_spatial_f=f_mnist.output_dim_spatial)
    mnist_label_preds = g_mnist(mnist_feats)
    print(f"MNIST Label Preds shape: {mnist_label_preds.shape}") # Expected: [4, 10]

    h_mnist = BiasClassifier(network_name='SimpleCNN_ConvBias', num_bias_classes=10,
                             input_channels_f=f_mnist.output_channels,
                             input_dim_spatial_f=f_mnist.output_dim_spatial)
    mnist_bias_preds = h_mnist(mnist_feats) # Test without GRL for shape
    print(f"MNIST Bias Preds shape: {mnist_bias_preds.shape}") # Expected: [4, 10]

    # --- ResNet Test ---
    f_resnet = FeatExtractor(network_name='ResNet18', pretrained=False)
    dummy_imgnet_input = torch.randn(2, 3, 224, 224)
    resnet_feats = f_resnet(dummy_imgnet_input)
    print(f"ResNet Feats shape: {resnet_feats.shape}") # e.g., [2, 256, 14, 14]

    g_resnet = LabelClassifier(network_name='ResNet18', num_classes=2,
                               input_channels_f=f_resnet.output_channels)
    resnet_label_preds = g_resnet(resnet_feats)
    print(f"ResNet Label Preds shape: {resnet_label_preds.shape}") # Expected: [2, 2]

    h_resnet = BiasClassifier(network_name='ResNet18_FCBias', num_bias_classes=2,
                              input_channels_f=f_resnet.output_channels)
    resnet_bias_preds = h_resnet(resnet_feats)
    print(f"ResNet Bias Preds shape: {resnet_bias_preds.shape}") # Expected: [2, 2]

    # Test combined model
    combined_model = DebiasingNetwork(f_mnist, g_mnist, h_mnist)
    label_out, bias_out = combined_model(dummy_mnist_input, alpha_grl=0.1, mode='train')
    print(f"Combined Model (train): Label out {label_out.shape}, Bias out {bias_out.shape}")
    label_only_out = combined_model(dummy_mnist_input, mode='predict_label')
    print(f"Combined Model (predict_label): Label out {label_only_out.shape}")
