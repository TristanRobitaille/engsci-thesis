## Step-by-step inference

### Notes
    - Faster inference implies lower buffer of ADC measurement, so inference latency (not so important) can save area (important!)
    - To ensure that the chip can keep up with clip, will need to buffer some measurements at the beginning of the clip while it runs inference on previous clip. Then,
    to "catch up", will need to send data at a faster rate than sampling frequency. There MAC must be much faster than sampling frequency.

### Step 0: Normalize input to [0, 1]
    - Optional, depends on whether rescaling improves model accuracy

### Step 1: Patch projection
    - Each patch is broadcast to all CiM sequentially as it is sampled.
    - Each CiM has patch_projection_dense.kernel[:][id] and patch_projection_dense.bias[id] in local memory
    - (MAC operation + 1 addition) must be shorter than sampling period to avoid having to buffer and save on temporary storage
    - Once full clip is sampled, each CiM is left with a (NUM_PATCHES, 1) vector in temporary memory

### Step 2: Concatenation with class embedding token
    - Internal step only
    - Simply move class embedding token from parameter memory to temporary memory to get a (NUM_PATCHES+1, 1) vector

### Step 3: Addition with position embedding
    - Internal step only
    - Element-wise addition with positional embedding

### Step 4: [Encoder] LayerNorm
    - 4.1: Reshape broadcast to transpose matrix prior to summations
    - 4.2: Normalizations summations
    - 4.3: Reshape broadcast to transpose matrix to use gamma and beta. Each CiM stores its beta and gamma parameter

### Step 5: [Encoder][MHSA] Q, K, V linear
    - 5.1: Reshape broadcast to transpose matrix back to horizontal vectors prior to matrix multiplication
    - Matrix multiplication

### Step 6: [Encoder][MHSA] QK^T
    - Need reshape broadcast, after which each CiM contains Q[:][id][:] and V^T[:][:][:] (too much data!) to end up with a "Z-slice" (QV^T[:][:][id])

### Step 7: [Encoder][MHSA] Divide by # of heads
    - Element-wise division, so internal only

### Step 8: [Encoder][MHSA] Softmax
    - Need reshape broadcast and apply softmax over last dimension (Z)

### Step 9: [Encoder][MHSA] Multiply with V
    - Need reshape broadcast and multiply with V

### Step 10: [Encoder][MHSA] Multiply with Dense linear
    - Need reshape broadcast and multiply with Dense linear    
