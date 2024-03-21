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
- Fixed-point accuracy study
    - Number of clock cyles per operation
- Discussion about memory
    - Amount of memory (intermediate results and params storage)
    - Overhead in storing LuTs for memory locations, lengths, etc.

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
