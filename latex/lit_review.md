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
    - ~2x speedup over of using GPU

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
    - Run models 2x RAM size up to 4-5x (on CPU) and 20-25x (on GPU) compared to naive implementation.
- **Main ideas**:
    - Large models don't fit in RAM --> Minimize data transfers to speed up inference
    - To ammortize data transfer overheads, read data in large contiguous chunks as much as possible
    - Feed-Forward Network in Transformers is typically >90% sparse --> Keep embeddings in memory but select which weights to load for ReLU layers with a predictor
    - Only maintain activated neurons associated with the past k tokens s.t. you only need to change some small amount of incremental data
    - Bundle columns and rows from up projection and down projection in memory so they can be loaded contiguously in large chunks
    - Closest friend: Neurons tend to be co-activated together so load your closest friend together with you. In practice, not good because you tend to load the same data a lot.

### Opportunities and Limitations of in-Memory Multiply-and-Accumulate Arrays
- **Results**:
    - Their architecture (vs. von Neumann) is ~5x less energy per inference and ~100-1000x less memory I/O
- **Main ideas**:
    - Power demands for AI increased 300,000x since 2012
    - Place MACs in DRAM (1 MAC shared between 2 bitcell arrays)
    - Implemented "Clipped staircase ReLU" as activation function because 1) Very cheap and 2) Prevents overflow (MACs are 8b mult. and 32b acc.), which is an issue for deep NN
    - Inference workflow summary:
        - Input is stored off-array, weights are local to MAC's bitcell arrays.
        - During compute, the (local) weights are loaded into MAC input and the input is broadcast serially on "global bit lines" and each MAC that needs it grabs it (good for parallelization)
        - MAC stack + activation function compute embedded in memory increase memory size by 12%
    - Future work: Thermal and metal area constraints, OS and compiler support

### CoMeFa: Compute-in-Memory Blocks for FPGA
- **Results**
    - Throughput increases 1.3x-2x depending on benchmark
    - Energy reduction of ~55%
- **Main ideas**
    - Design new block RAM (20kbit) with configurable processing element (160 or 80 PEs per BRAM) to allow for CiM. BRAM has bank of operands 1, operands 2 and results
    - Perform operations bit-serial instead of bit-parallel for higher throughput (but higher per-operation latency)
    - Key: The processing elements are 1b and the BRAM is passed a 40b instruction. The precision is adjustable simply by passing different sequence of instruction.
    - Latencies: (Integer): Add = n+1 cycles, Mult = n^2+3n-2 cycles. (Float): Add = 2ME+9M+7E+12, Mult = M^2+7M+3E+5 (M=# of mantissa bits, W=# of exponent bits)
    - Also developed a "swizzle" module to interface data from DRAM to CoFeMa RAM
    - Tools: Verilog-to-Routing, COFFE for area and delay value and FreePDK45 for SPICE

### Compute-Capable Block RAMs for Efficient Deep Leanring Acceleration on FPGAs
- **Results**
    - 1.6x and 2.3x speedup for int8 and bfp8 MACs at 1.8% increased area
    - An IP CiM BRAM that can be instantiated with vendor's BRAM IP
- **Main ideas**
    - Present CiM BRAM design for FPGA (argue advantages of CiM are: high parallelism, tight integration and elimination of data movement)
    - Their CiM BRAM module
        - Bit-serial operation on transposed data (data bits are along bitline instead of wordline)
        - Can do adds, multiplies, reduction and AND/NOR (on each bitline). Other bit-serial operations in [11]. Advantage of bit-serial is lower energy but at higher latency.
        - Can function as traditional memory
    - Explain block floating point: A block of floats share the same exponent such that only the mantissa needs to be computed (as integer operations).
    - Their 64Kb BRAM for FPGA is ~11800um^2 (7.4% area increase for CiM overhead --> Decreases to 1.8% overall since BRAM is ~25% of FPGA area)
    - BFP8 MAC: 113 cycles, INT8 MAC: 23 cycles
    - Use elegant heatmaps for design space exploration
    - Simulation with COFFE