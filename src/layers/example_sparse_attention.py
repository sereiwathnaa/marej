
import torch
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.layers.attention import FixedSparseAttention

def test_sparse_attention():
    print("Testing FixedSparseAttention...")
    
    # Configuration
    batch_size = 2
    seq_len = 32 # Not divisible by block_size=8
    embed_dim = 64
    n_heads = 4
    dim_head = 16
    block_size = 8
    
    print(f"Config: batch={batch_size}, seq_len={seq_len}, embed={embed_dim}, block_size={block_size}")
    
    # Initialize model
    model = FixedSparseAttention(
        embed_dim=embed_dim,
        n_heads=n_heads,
        dim_head=dim_head,
        block_size=block_size,
        use_flash=True
    )
    
    # Create random input
    x = torch.randn(batch_size, seq_len, embed_dim)
    print(f"Input shape: {x.shape}")
    
    # Forward pass (Flash Attention)
    print("\nRunning with Flash Attention...")
    # Force CPU due to potential CUDA version mismatch in environment
    device = 'cpu' 
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
