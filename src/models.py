import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import logging # Import logging

# --- Gradient Reversal Layer ---
class GradientReversalFunction(Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        # Store lambda_ for backward pass
        ctx.lambda_ = lambda_
        # Forward pass is identity
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        # Reverse the gradient and scale it by lambda_
        # grad_output is the gradient from the subsequent layer
        # The first returned value is the gradient for x, the second for lambda_ (None)
        return (grad_output.neg() * ctx.lambda_), None

class GradientReversalLayer(nn.Module):
    def __init__(self, lambda_val=1.0):
        super(GradientReversalLayer, self).__init__()
        self.lambda_val = lambda_val

    def set_lambda(self, lambda_val):
        self.lambda_val = lambda_val
        
    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_val)

# --- Feature Extractor (f) ---
class LeNet_F(nn.Module):
    """
    LeNet-style feature extractor for Colored MNIST (28x28x3 images).
    Outputs a flat feature vector.
    """
    def __init__(self, input_channels=3, feature_dim=128): # Changed output_dim to feature_dim for clarity
        super(LeNet_F, self).__init__()
        self.input_channels = input_channels
        self.feature_dim = feature_dim

        # CNN layers: (Input: B, 3, 28, 28)
        self.conv1 = nn.Conv2d(self.input_channels, 32, kernel_size=5, stride=1, padding=2) # (B, 32, 28, 28)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0) # (B, 32, 14, 14)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2) # (B, 64, 14, 14)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0) # (B, 64, 7, 7)
        
        # Calculate the flattened size after convolutions
        # For 28x28 input -> conv1 (28x28) -> pool1 (14x14) -> conv2 (14x14) -> pool2 (7x7)
        self.flattened_size = 64 * 7 * 7
        
        self.fc = nn.Linear(self.flattened_size, self.feature_dim)
        logging.info(f"LeNet_F initialized: Input Channels={input_channels}, Feature Dim={feature_dim}")

    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = x.view(-1, self.flattened_size) # Flatten
        x = F.relu(self.fc(x)) # Feature vector
        return x

# --- Task Classifier (g) ---
class MLP_Classifier_G(nn.Module):
    """
    MLP classifier for the main task.
    Takes features from F and predicts main task classes.
    """
    def __init__(self, input_dim, hidden_dims, output_dim):
        super(MLP_Classifier_G, self).__init__()
        layers = []
        current_dim = input_dim
        if hidden_dims is None: # Handle case where hidden_dims might be None or empty
            hidden_dims = []
            
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.ReLU())
            # layers.append(nn.Dropout(0.5)) # Optional dropout
            current_dim = hidden_dim
        
        layers.append(nn.Linear(current_dim, output_dim))
        # Logits are returned; Softmax/LogSoftmax will be part of CrossEntropyLoss
        
        self.classifier = nn.Sequential(*layers)
        logging.info(f"MLP_Classifier_G initialized: Input Dim={input_dim}, Hidden Dims={hidden_dims}, Output Dim={output_dim}")

    def forward(self, x):
        return self.classifier(x)

# --- Bias Predictor (h) ---
class ConvBiasPredictorH(nn.Module):
    """
    Convolutional Bias Predictor.
    Takes features from F and predicts a spatial bias map.
    The output shape is (Batch, NumBiasBins, NumBiasChannels, H_out, W_out),
    e.g., (B, 8, 3, 14, 14) for predicting 8 quantization bins for each of 3 color channels
    on a 14x14 grid.
    """
    def __init__(self, input_feature_dim, num_bias_channels=3, num_bias_quantization_bins=8, 
                 output_h=14, output_w=14, intermediate_channels=64):
        super(ConvBiasPredictorH, self).__init__()
        self.input_feature_dim = input_feature_dim
        self.num_bias_channels = num_bias_channels
        self.num_bias_quantization_bins = num_bias_quantization_bins
        self.output_h = output_h
        self.output_w = output_w
        self.intermediate_channels = intermediate_channels # Channels after reshaping features

        # We need to project the flat feature vector to something spatial that can be upsampled.
        # Let's aim for a base spatial size, e.g., 7x7, to upsample to 14x14 in one step.
        self.base_spatial_dim = output_h // 2 # Assuming output_h is even, e.g., 14 -> 7
        
        self.fc_project = nn.Linear(input_feature_dim, intermediate_channels * self.base_spatial_dim * self.base_spatial_dim)
        
        # Transposed convolution to upsample and produce the desired number of output channels
        # Output channels = num_bias_channels * num_bias_quantization_bins
        self.upsample_conv = nn.ConvTranspose2d(
            in_channels=intermediate_channels,
            out_channels=self.num_bias_channels * self.num_bias_quantization_bins,
            kernel_size=4, # Kernel to go from base_spatial_dim to output_h/w
            stride=2,      # Stride 2 for doubling spatial size
            padding=1      # Padding: (kernel_size - stride) / 2 = (4-2)/2 = 1
        )
        # This ConvTranspose2d should take (B, intermediate_channels, base_spatial_dim, base_spatial_dim)
        # to (B, num_bias_channels * num_bias_quantization_bins, output_h, output_w)

        logging.info(
            f"ConvBiasPredictorH initialized: InputFeatureDim={input_feature_dim}, "
            f"NumBiasChannels={num_bias_channels}, NumBiasQuantizationBins={num_bias_quantization_bins}, "
            f"Output Spatial=({output_h},{output_w}), IntermediateChannels={intermediate_channels}, "
            f"BaseSpatialDimForUpsample={self.base_spatial_dim}"
        )

    def forward(self, x_features):
        # x_features is the flat feature vector from F, e.g., (B, input_feature_dim)
        
        # Project and reshape to a spatial representation
        x = self.fc_project(x_features)
        x = x.view(-1, self.intermediate_channels, self.base_spatial_dim, self.base_spatial_dim)
        x = F.relu(x)
        
        # Upsample and generate final channel dimensions
        x = self.upsample_conv(x) # Output: (B, num_bias_channels * num_bias_quantization_bins, output_h, output_w)
        
        # Reshape to (B, num_bias_quantization_bins, num_bias_channels, output_h, output_w)
        # This is so that CrossEntropyLoss can treat num_bias_quantization_bins as the 'Class' dimension.
        # Target from data_loader is (B, num_bias_channels, output_h, output_w) with class indices 0..7
        batch_size = x.shape[0]
        x = x.view(batch_size, 
                     self.num_bias_quantization_bins, # This will be the 'K' classes for CrossEntropy
                     self.num_bias_channels, 
                     self.output_h, 
                     self.output_w)
        return x


# --- Model Factory ---
def get_models(config: dict, device: torch.device) -> dict:
    """
    Factory function to create and return the models based on config.
    """
    model_config = config['model']
    data_config = config['data'] # Needed for num_classes, etc.
    
    # 1. Feature Extractor (f)
    f_params = model_config['feature_extractor_f']['params']
    feature_extractor_f = LeNet_F(
        input_channels=f_params.get('input_channels', data_config.get('img_channels', 3)),
        feature_dim=f_params['feature_dim']
    ).to(device)

    # 2. Task Classifier (g)
    g_params = model_config['task_classifier_g']['params']
    task_classifier_g = MLP_Classifier_G(
        input_dim=g_params.get('input_dim', f_params['feature_dim']), # Default to f's output
        hidden_dims=g_params.get('hidden_dims', [64]),
        output_dim=g_params.get('output_dim', data_config['num_main_classes'])
    ).to(device)

    # 3. Bias Predictor (h)
    h_config = model_config['bias_predictor_h']
    h_params = h_config['params']
    
    # IMPORTANT: num_bias_quantization_bins should come from config.
    # Let's assume it's defined in h_params, e.g., h_params['num_bias_quantization_bins']
    # Defaulting to 8 if not specified, as per our discussion for the color map target.
    num_bias_quantization_bins = h_params.get('num_bias_quantization_bins', 8)
    
    # For ConvBiasPredictorH, output_dim is implicitly defined by bins*channels*H*W
    # We'll use the name defined in config if it matches, otherwise default to ConvBiasPredictorH
    if h_config['name'].lower() == 'convbiaspredictorh':
        bias_predictor_h = ConvBiasPredictorH(
            input_feature_dim=h_params.get('input_dim', f_params['feature_dim']),
            num_bias_channels=data_config.get('img_channels', 3), # Bias channels should match image channels
            num_bias_quantization_bins=num_bias_quantization_bins,
            output_h=h_params.get('output_h', 14), # Should match data_loader's bias map size
            output_w=h_params.get('output_w', 14), # Should match data_loader's bias map size
            intermediate_channels=h_params.get('intermediate_channels', 64)
        ).to(device)
    else:
        # Fallback or error if other H types are expected but not defined
        raise ValueError(f"Unsupported bias_predictor_h name: {h_config['name']}. Expected ConvBiasPredictorH or implement others.")

    # 4. Gradient Reversal Layer (optional, based on config)
    grl_lambda = config['training'].get('gradient_reversal_layer', {}).get('grl_lambda_fixed', 1.0)
    grl = GradientReversalLayer(lambda_val=grl_lambda) 
    # GRL itself doesn't need .to(device) as it's a Function container

    models_dict = {
        'feature_extractor_f': feature_extractor_f,
        'task_classifier_g': task_classifier_g,
        'bias_predictor_h': bias_predictor_h,
        'grl': grl # Include GRL if it's to be used by the trainer
    }
    
    logging.info("All models created and moved to device.")
    return models_dict


if __name__ == '__main__':
    # --- Example Usage for testing models.py ---
    setup_logging() # Assuming utils.py is accessible

    # Create a dummy config for testing
    # (Make sure this matches the structure your get_models function expects)
    test_config = {
        'data': {
            'img_channels': 3,
            'num_main_classes': 10,
            # 'num_bias_classes': 10, # Conceptual high-level bias classes
        },
        'model': {
            'feature_extractor_f': {
                'name': 'LeNet_F',
                'params': {'input_channels': 3, 'feature_dim': 128}
            },
            'task_classifier_g': {
                'name': 'MLP_Classifier_G',
                'params': {'input_dim': 128, 'hidden_dims': [64], 'output_dim': 10}
            },
            'bias_predictor_h': {
                'name': 'ConvBiasPredictorH', # Crucial: matches our implemented H
                'params': {
                    'input_dim': 128,
                    'num_bias_quantization_bins': 8, # For 0-7 pixel value bins
                    # num_bias_channels will be taken from data.img_channels (3) by get_models
                    'output_h': 14, # Target spatial dim for bias map
                    'output_w': 14, # Target spatial dim for bias map
                    'intermediate_channels': 64
                }
            },
            'adversarial_method': 'gradient_reversal' # To ensure GRL might be relevant
        },
        'training': { # For GRL lambda
            'gradient_reversal_layer': {'grl_lambda_fixed': 0.5}
        }
    }
    device = torch.device("cpu") # Test on CPU

    # Test get_models
    logging.info("Testing get_models...")
    all_models = get_models(test_config, device)
    f_net = all_models['feature_extractor_f']
    g_net = all_models['task_classifier_g']
    h_net = all_models['bias_predictor_h']
    grl_layer = all_models['grl']

    logging.info(f"GRL lambda: {grl_layer.lambda_val}")

    # Create dummy input
    batch_size = 4
    dummy_image = torch.randn(batch_size, 3, 28, 28).to(device) # B, C, H, W

    # Test forward passes
    logging.info("Testing forward passes...")
    features = f_net(dummy_image)
    logging.info(f"Features shape: {features.shape}") # Expected: (B, feature_dim) e.g., (4, 128)

    task_output = g_net(features)
    logging.info(f"Task classifier output shape: {task_output.shape}") # Expected: (B, num_main_classes) e.g., (4, 10)

    # For H_net, features might be passed through GRL first in an adversarial setup
    # features_for_h = grl_layer(features) # If GRL is applied before H
    # bias_output = h_net(features_for_h)
    bias_output = h_net(features) # Direct test of H
    logging.info(f"Bias predictor output shape: {bias_output.shape}") 
    # Expected for ConvBiasPredictorH: (B, num_bias_quantization_bins, num_bias_channels, H_out, W_out)
    # e.g., (4, 8, 3, 14, 14)

    # Dummy target for bias loss (as from data_loader)
    # Target shape: (B, num_bias_channels, H_out, W_out) with class indices 0..7 (+ ignore_index)
    dummy_bias_target = torch.randint(0, 8, (batch_size, 3, 14, 14), dtype=torch.long).to(device) 
    
    # Test loss compatibility (conceptual)
    criterion_bias = nn.CrossEntropyLoss(ignore_index=255) # Assuming 255 is an ignore index
    try:
        # input for CrossEntropyLoss: (N, K, d1, d2, ...) where K = num_classes
        # target for CrossEntropyLoss: (N, d1, d2, ...) with values in [0, K-1]
        # Our h_net output: (B, NumBins=8, C=3, H=14, W=14)
        # Our target: (B, C=3, H=14, W=14) with values 0-7
        # This means for each item in (C,H,W) dimensions, we predict one of NumBins classes.
        # So, permute h_net output to (B, C, H, W, NumBins) and target to (B, C, H, W)
        # Then apply loss, or ensure shapes match CE expectations.
        # nn.CrossEntropyLoss expects input (N, K, ...) and target (N, ...).
        # Here, N can be B*C*H*W, and K is NumBins.
        # So, reshape:
        # h_output_reshaped = bias_output.permute(0, 2, 3, 4, 1).contiguous().view(-1, test_config['model']['bias_predictor_h']['params']['num_bias_quantization_bins'])
        # bias_target_reshaped = dummy_bias_target.view(-1)
        # loss_h = criterion_bias(h_output_reshaped, bias_target_reshaped)
        
        # Simpler for PyTorch >= 1.0: input (B, K, C, H, W), target (B, C, H, W)
        # where K is num_classes (our num_bias_quantization_bins)
        loss_h = criterion_bias(bias_output, dummy_bias_target)

        logging.info(f"Successfully calculated conceptual bias loss: {loss_h.item()}")
    except Exception as e:
        logging.error(f"Error calculating conceptual bias loss: {e}", exc_info=True)

    logging.info("models.py test finished.")