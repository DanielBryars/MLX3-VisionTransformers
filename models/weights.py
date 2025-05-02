import torch

def init_projection_weights(linear_layer, sigma_ratio=0.5):
    """
    Initialize weights of a linear layer to transform between dimensions while preserving
    image structure as much as possible.
    
    Args:
        linear_layer: A torch.nn.Linear layer
        sigma_ratio: Controls the width of Gaussian in the general case (default: 0.5)
    """
    with torch.no_grad():
        in_dim = linear_layer.in_features
        out_dim = linear_layer.out_features
        W = torch.zeros(out_dim, in_dim)
        
        if in_dim == out_dim:
            # Identity - perfect preservation
            W = torch.eye(in_dim)
            
        elif in_dim % out_dim == 0:
            # Downsampling: average input chunks
            step = in_dim // out_dim
            for i in range(out_dim):
                W[i, i*step:(i+1)*step] = 1.0 / step
                
        elif out_dim % in_dim == 0:
            # Upsampling: duplicate inputs
            factor = out_dim // in_dim
            for i in range(in_dim):
                W[i*factor:(i+1)*factor, i] = 1.0 / factor
                
        else:
            # General case: use Gaussian interpolation
            # Create evenly spaced positions for the output dimension
            out_positions = torch.linspace(0, 1, out_dim)
            in_positions = torch.linspace(0, 1, in_dim)
            
            # Calculate sigma based on relative dimensions
            sigma = sigma_ratio * (1.0 / out_dim)
            
            # For each output position, calculate weights for all input positions
            for out_i in range(out_dim):
                out_pos = out_positions[out_i]
                # Calculate squared distances to all input positions
                sq_distances = (in_positions - out_pos)**2
                # Apply Gaussian kernel
                weights = torch.exp(-0.5 * sq_distances / (sigma**2))
                # Normalize weights to sum to 1
                weights = weights / weights.sum()
                W[out_i] = weights
        
        # Set the weights in the linear layer
        linear_layer.weight.copy_(W)
        
        # Zero out any bias
        if linear_layer.bias is not None:
            linear_layer.bias.zero_()
            
    return linear_layer  # Return the layer for convenience


# Example usage
if __name__ == "__main__":
    # Test with different dimension configurations
    test_cases = [
        (10, 10),    # Identity case
        (10, 5),     # Downsampling (2:1)
        (5, 10),     # Upsampling (1:2)
        (12, 7),     # General case
        (7, 16)      # General case
    ]
    
    for in_dim, out_dim in test_cases:
        layer = torch.nn.Linear(in_dim, out_dim, bias=True)
        init_projection_weights(layer)
        
        print(f"\nTransform from {in_dim} to {out_dim}:")
        print(f"Weight matrix shape: {layer.weight.shape}")
        print(f"Row sums: {layer.weight.sum(dim=1)}")  # Each row should sum to ~1.0
        
        # Test with a sample input
        x = torch.randn(1, in_dim)
        x_normalized = x / x.abs().sum()
        y = layer(x_normalized)
        
        print(f"Input sum: {x_normalized.sum().item():.4f}")
        print(f"Output sum: {y.sum().item():.4f}")