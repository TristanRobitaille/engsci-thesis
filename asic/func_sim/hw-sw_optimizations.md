## Optimizations that should be tried once a minimal viable product is function
- Weight quantization
- Try training without gamma and beta in LayerNorm to save data movement and computation (61*64 multiplications and additions)
- Try training without biases in linear Dense to save computation
- Try without dividing QK_T by sqrt(NUM_HEADS)