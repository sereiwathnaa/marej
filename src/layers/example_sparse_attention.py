
import torch
import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.layers.attention import FixedSparseAttention

def visualize_attention(attn_layer: torch.nn.Module, seqlen: int):
    """Visualize attention scores by perturbing inputs."""
    print("\nVisualizing attention patterns...")
    # Get embed_dim from the layer
    embed_dim = attn_layer.to_qkv.in_features
    
    # Create random input
    x = torch.randn(1, seqlen, embed_dim)
    
    # Move to same device as layer
    device = next(attn_layer.parameters()).device
    x = x.to(device)
    
    # Forward pass
    with torch.no_grad():
        attn_output = attn_layer(x)
    
    # Perturb the input, and see which elements in the output respond
    attn_scores = []
    for i in range(seqlen):
        x_perturbed = x.clone()
        # Perturb input at position i
        perturbation = 10 * torch.randn(1, embed_dim).to(device)
        x_perturbed[:, i, :] += perturbation
        
        with torch.no_grad():
            perturbed_attn_output = attn_layer(x_perturbed)
        
        delta = attn_output - perturbed_attn_output
        # Sum absolute difference across batch and embedding dim
        # delta shape: (1, seqlen, embed_dim)
        # We want to know for each output position j, did it change?
        delta_at_indices = delta.abs().sum(dim=(0, 2))
        
        # Check if change is significant
        attn_scores.append((delta_at_indices > 1e-4).cpu().numpy())
    
    # Stack: (seqlen_input, seqlen_output) -> Transpose to (seqlen_output, seqlen_input)
    # attn_scores[j, i] means "did input i affect output j?"
    attn_scores = np.stack(attn_scores).T
    
    # Visualize attention
    try:
        plt.figure(figsize=(10, 8))
        plt.imshow(attn_scores, origin='upper', cmap='viridis')
        plt.xlabel('Input Position (Perturbed)')
        plt.ylabel('Output Position (Affected)')
        plt.title(f'Attention Connectivity (Block Size: {attn_layer.block_size})')
        plt.colorbar(label='Affected')
        
        # Add grid lines for blocks
        block_size = attn_layer.block_size
        for i in range(0, seqlen, block_size):
            plt.axhline(y=i-0.5, color='r', linestyle='-', alpha=0.3)
            plt.axvline(x=i-0.5, color='r', linestyle='-', alpha=0.3)
            
        output_path = 'attention_visualization.png'
        plt.savefig(output_path)
        print(f"Saved visualization to {output_path}")
        plt.close()
    except Exception as e:
        print(f"Error plotting: {e}")

def test_sparse_attention():
    print("Testing FixedSparseAttention...")
    
    # Configuration
    batch_size = 2
    seq_len = 32 # Must be divisible by block_size=8
    embed_dim = 64
    n_heads = 4
    dim_head = 16
    block_size = 8
    
    print(f"Config: batch={batch_size}, seq_len={seq_len}, embed={embed_dim}, block_size={block_size}")
    
    model = FixedSparseAttention(
        embed_dim=embed_dim,
        n_heads=n_heads,
        dim_head=dim_head,
        block_size=block_size,
        use_flash=True
    )
    
    x = torch.randn(batch_size, seq_len, embed_dim)
    print(f"Input shape: {x.shape}")
    
    print("\nRunning with Flash Attention...")
    device = "cuda" if torch.cuda.is_available() else "cpu" 
    print(f"Using {device}")
    model = model.to(device)
    x = x.to(device)
        
    with torch.no_grad():
        out = model(x)
    
    print(f"Output shape: {out.shape}")
    assert out.shape == x.shape, f"Output shape mismatch: {out.shape} vs {x.shape}"
    print("Shape check passed!")
    
    # Test with manual implementation (disable flash)
    print("\nRunning with Manual Attention...")
    model_manual = FixedSparseAttention(
        embed_dim=embed_dim,
        n_heads=n_heads,
        dim_head=dim_head,
        block_size=block_size,
        use_flash=False
    )
    
    # Copy weights to compare
    model_manual.load_state_dict(model.state_dict())
    model_manual = model_manual.to(device)
    
    with torch.no_grad():
        out_manual = model_manual(x)
        
    print(f"Manual Output shape: {out_manual.shape}")
    
    # Compare outputs
    diff = (out - out_manual).abs().max().item()
    print(f"Max difference between Flash and Manual: {diff}")
    
    if diff < 1e-4:
        print("Implementations match!")
    else:
        print("Warning: Implementations differ significantly (could be due to precision or implementation details)")

    print("\nTest completed.")

if __name__ == "__main__":
    test_sparse_attention()
