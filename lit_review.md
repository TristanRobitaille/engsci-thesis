# Literature review

### An Image Is Worth 16X16 Words: Transformers for Image Recognition at Scale
- **Main ideas**:
    - Split image into N patches of (P x P) -> Flatten patches -> Map to D dimensions with trainable linear projections ("patch embeddings") and add positional encoding
    - Prepend learnable embedding whose output serves as y (true class)

### Accommodating Transformer onto FPGA: Coupling the Balanced Model Compression and FPGA-Implementation Optimization
- **Main ideas**:
    - Pruning sparse matrix with Block-Balanced Pruning (BBP) instead of Block-wise Pruning (BWP) to maintain accuracy while lowering memory requirements
    - Develop efficient Compressed Block Row (CBR) to store spare matrices using 2 arrays:
        - W matrix (3D): 1st dim is block index, 2nd and 3rd are non-zero weights
        - W-Index matrix (2D): 1st dim is block index, 2nd dim is index of rows remaining in block after pruning
    - Make a design space solver by defining legal parameter values, and run simulations for each
- **Results**:
    - Used very aggressive sparsity of 90%
    - ~2x the speed-up of using GPU

###  FTRANS: Energy-Efficient Acceleration of Transformers using FPGA
- **Main ideas**:
    - Enhanced Block Circular Matrix model compression: Reduce weight storage by representing weight matrices as a vector of small circulant matrices of dimensions b x b
        - Augment the work of Ding et al. by redefining the index vector, where each element circulant matrix is the average of all matrices over the row j (this is not precise, need to review)
        - Saves significant storage space
        - Matrix-vector multiplies can be done in O(blogb) instead of O(b^2)
    - Optimized fine-grained scheduling algorithm:
        - Maximize utilization (and minimize inference latency) by having each layer in the pipeline (not necessarily model layer) take the same time
    - Segregated weight storage: Weights for embedding layer are stored off chip since they are used once but weights for the encoder/decoder stacks are stored on chip since they are the more compute-heavy parts
    - Develop 3 different PEs (processing element): 2 standard matrix multiplies PEs (of different sizes) and an FFT/IFFT-based (radix-2 Cooley-Turkey algo for FFT) PE with accumulator and adder for FC layers
    - Exponential in softmax is piecewise linear
- **Results**:
    - Reduced model size by ~16x (using BCM and conversion from 32b-float to 16b-fixed)
    - 27x and 81x improvement in performance and energy efficiency, respectively, over CPU
    - 8.8x and 1.77x improvement in performance and energy efficiency, respectively, over GPU
    - Minimal loss in accuracy for BCM block sizes of 4 and 8 for shallower transformer, and loss in accuracy of 4.2% and 4.3% for deep transformer.
    - No accuracy loss when converting weights to 16b fixed-point from 32b floating point

###  CirCNN: Accelerating and Compressing Deep Neural Networks Using Block-Circulant Weight Matrices
- **Results**:
    - Matrix-vector multiplies can be done in O(blogb) instead of O(b^2)
    - Compared to NVIDIA Jetson TX1, 570x less energy. Compared to previous FPGA, 60-70x higher energy efficiency. Due to:
        - BCM algorithmic improvements
        - Weight storage reduction -> Significant reduction of off-chip DRAM access
        - Efficient basic computing block design
- **Main ideas**:
    - Cons of more traditional weight pruning:
        - Irregular network structure -> Requires storing indices, which decreases benefits.
        - Increased training complexity
        - No rigourous guarantee of compression ratio and inference accuracy
    - DRAM access is 200x more energy than on-chip memory
    - BCM for weight storage: 
        - Partition weight matrix in array of circulant matrices. Can use "circulant convolution theorem" and FFT to perform matrix-vector multiply in O(nlogn) instead of O(n^2).
        - Needs special backpropagation in training.
        - 400x-4000x storage reduction (when combined with 32b float -> 16b fixed quantization)
        - 5x-9x speedup in training
- **Limitations**:
    - Requires change to training algorithm, not merely compressing existing weight matrices.

### APTx: better activation function than MISH, SWISH, and ReLU’s variants used in deep learning
- **Results**:
    - APTx requires fewer operations for forward and backwards propagation than MISH, SWISH
    - Can approximate MISH(x), SWISH(x, p) by changing parameters
- **Main ideas**:
    - Developed "Alpha Plus Tanh Times" (APTx) as approximation to MISH and SWISH: act.(x) = (a + tanh(bx)) * phi*x

### LLM in a flash: Efficient Large Language Model Inference with Limited Memory
- **Results**:
    -Run models 2x RAM size up to 4-5x (on CPU) and 20-25x (on GPU) compared to naive implementation.
- **Main ideas**:
    -Large models don't fit in RAM --> Minimize data transfers to speed up inference
    -To ammortize data transfer overheads, read data in large contiguous chunks as much as possible
    -Feed-Forward Network in Transformers is typically >90% sparse --> Keep embeddings in memory but select which weights to load for ReLU layers with a predictor
    -Only maintain activated neurons associated with the past k tokens s.t. you only need to change some small amount of incremental data 
    -Bundle columns and rows from up projection and down projection in memory so they can be loaded contiguously in large chunks
    -Closest friend: Neurons tend to be co-activated together so load your closest friend together with you. In practice, not good because you tend to load the same data a lot.