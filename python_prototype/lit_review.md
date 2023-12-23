# Literature review

### An Image Is Worth 16X16 Words: Transformers for Image Recognition at Scale
- **Main ideas**:
    - Split image into N patches of (P x P) -> Flatten patches -> Map to D dimensions with trainable linear projections ("patch embeddings") and add positional encoding
    - Prepend learnable embedding whose output serves as y (true class)

### Accommodating Transformer onto FPGA: Coupling the Balanced Model Compression and FPGA-Implementation Optimization
- **Results**:
    - Used very aggressive sparsity of 90%
    - ~2x the speed-up of using GPU
- **Main ideas**:
    - Pruning sparse matrix with Block-Balanced Pruning (BBP) instead of Block-wise Pruning (BWP) to maintain accuracy while lowering memory requirements
    - Develop efficient Compressed Block Row (CBR) to store spare matrices using 2 arrays:
        - W matrix (3D): 1st dim is block index, 2nd and 3rd are non-zero weights
        - W-Index matrix (2D): 1st dim is block index, 2nd dim is index of rows remaining in block after pruning
    - Make a design space solver by defining legal parameter values, and run simulations for each