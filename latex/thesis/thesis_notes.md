# Thesis report content notes

## Literature review

## Model design

## Architecture design

## Methods
- Tools used
    - Python TF prototype, scratch C++ model, SV ASIC, CocoTB, etc.
    - Canada Compute, EECG and Synopsys tools, remote PC, local MacBook
    - Dataset

## Results
- Model: Number of params, number of operations, accuracy
- ASIC: Inference time, Fmax, area usage, energy per inference, power (dynamic and leakage), memory usage

## Architecture discussion
- Centralized vs. distributed intermediate results
    - What are the clock cycles used for (number of clock cycles for data transfer vs computation for 64 CiM vs single module)
    - Extra space used by 64 CiM vs single module
    - Seeing the end results, was a multi-CiM architecture beneficial (essentially, would single CiM be fast enough)?
- Communication bus
    - Data carried
    - Op codes supported
- Compute modules
    - Area, latency, approximations, features, energy per compute
- Software-hardware co-design
    - Number of CiM == EMB_DEPTH (which is a dimension in most matrices) --> Each CiM only has to perform one vector multiplication per matrix multiply, saving overhead and data movement
    - Number of patches + classification embedding token == 61 --> This is close to # of CiM, giving very high utilization.
    - EMB_DEPTH is a power of two and am using fixed-point arithmetic, meaning we can simply bit-shift during LayerNorms instead of performing a full divide, which reduces LayerNorm latency (and energy consumption) by xyz %.
    - [NOT DONE, should investigate]: Square root of number of heads should be a power of two so we again don't have to divide but can rather just bit-shift.
- Fixed-point accuracy study
    - Number of clock cyles per operation
- Discussion about memory
    - Amount of memory (intermediate results and params storage)
    - Overhead in storing LuTs for memory locations, lengths, etc.
- Clock gating
    - Use "-gate_clock" in command, but should be done automatically by default. Try to disable and see difference

## Results
- Area, power, latency (w/ and w/o parameters load)
- Qualify the above results:
    - Assumption on external memory access latency

## Next steps
- Potentially reduce number of data transpose needed (don't train beta and gamma in LayerNorm, etc.)
- Architecture changes
- Training on different datasets
- Evaluate dynamic fixed-point (different Q and N based on layer)
- What else could we run this on?
    - ARM-Cortex MCU and Edge TPU
    - Power consumption of STM32L4

## Interesting project metrics
- Number of lines of code (Python, C++, SystemVerilog, Tex, Shell)
- Number of files (Python, C++, SystemVerilog, Tex, Shell)
- Number of commits

## Reflection on learnings and experience gained
