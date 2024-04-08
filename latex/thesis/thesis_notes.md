# Thesis report content notes

## Acknowledgments
- Prof. Xilin Li
- Claude for remote PC and voluntary code review
- Profs. Jeffrey, Anderson, Moshovos, Enright Jerger, Halupka for courses
- Compute Canada, CMC
- Profs. Romkey and Chong for the resources and structure of the EngSci thesis (ESC499)
- Friends

## Introduction

## Background
- Insmonia problem
- Sleep staging (5 stages, definition from sleep foundation)
- PSG
- Neuromodulation
- Pitch the idea: Sleep staging AirPod
    - Patent search (say that wearables are increasingly smart)
        - Apple's recent AirPod patent
- Sleep staging using AI
    - Lit. review from interim report
- Technical goals and requirements
    - Equal to PSG
    - ASIC specs
- Pose research question: Can we make an accelerator that will meet the requirements

## How to Design and AI Accelerator
- The three step plan
- Tools used
    - Python TF prototype, scratch C++ model, SV ASIC, CocoTB, etc.
    - Canada Compute, EECG and Synopsys tools, remote PC, local MacBook
    - Dataset

## Vision transformer design
- Why vision transformer?
    - No need for decoder since it's sequence-to-one
- Describe each operation (LayerNorm, Softmax, MAC, Swish)

## ASIC Architecture
- Centralized vs. distributed intermediate results
    - What are the clock cycles used for (number of clock cycles for data transfer vs computation for 64 CiM vs single module)
    - Extra space used by 64 CiM vs single module
- Master
    - 
- Memory
    -
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
- Model: Number of params, number of operations, accuracy
- ASIC:
    -Inference time, Fmax, area usage, energy per inference, power (dynamic and leakage), memory usage
    -PE utilization: 50.8%
- Area, power, latency (w/ and w/o parameters load)
- Qualify the above results:
    - Assumption on external memory access latency
- Seeing the end results, was a multi-CiM architecture beneficial (essentially, would single CiM be fast enough)?

## Future Work
- Architecture changes: Switch to a centralized architecture
    - Area reduction: Less memory overhead, eliminate 64/63 compute units, eliminate a lot of redundant logic, eliminate tranpose operations
    - To speed up, split memory into 3 (or have 3 read ports) to reduce by ~3 transpose/dense broadcast delays. Quantify how much time is spent in comms vs useful work.
        - This also minimizes overhead
        - In a centralized archtiecture, this is needed anyways because the maximum memory size is 16384 # of words and we have ~32k parameters
    - Although inference time will increase, it will remain far under the requirement of 3s. In fact, the inference time is expected to be less than 64x times longer than the distributed architecture.
    Indeed, as mentionned in section {distributed architecture}, the CiM spend roughly x percent of their time in compute. The rest is mainly data broadcast. In data broadcast, the majority of the time (x/7 cycles) is spent waiting for data from the single-port memory. With three memory banks, the time is further reduced.
    - Estimated relative energy per inference
    - Estimated area savings
    - Evaluate dynamic fixed-point (different Q and N based on layer)
    - Fine tune fixed-point format
        - Save significant area and leakage power in adder/multiplier
        - Save significant area in memory
        - Save some latency in divides, which are the slowest operation
    - Compute blocks can be improved
        - Is the MUX needed to share temporary compute variables worth the area?
        - Is Gaussian rounding needed?
        - Should add_/mult_refresh be used?
        - Should multiplier be pipelined
    - Balance the Fmax --> This can actually save power since maximizing the Fmax will reduce inference latency, meaning we can shut down the accelerator for longer, meaning the effective leakage current (by far the dominant power loss) can be reduced
    - Look into cutting power to accelerator when not in use to avoid leakage current
- Model changes:
    - Potentially reduce number of data transpose needed (don't train beta and gamma in LayerNorm, etc.)
    - Training on different datasets
        - Including in-ear EEG PSG
- What else could we run this on?
    - ARM-Cortex MCU and Edge TPU
    - Power consumption of STM32L4

## Interesting project metrics
- Number of lines of code (Python, C++, SystemVerilog, TeX, Shell)
- Number of files (Python, C++, SystemVerilog, TeX, Shell)
- Number of commits

## Reflection on learnings and experience gained
- Also wanted to be able to spend more time on the memory compiler and its integration in the project
- What I would do differently:
    - Use the functional simulation to help inform more of the hardware design. Perhaps design the compute units, estimate the PPA and reconsider the architecture before fully writing it.
- What I learned:
    - Owning the full-stack is powerful and gives significant design freedom. [Like what?] In turn, this can prove destabilizing as essentially all aspects of the design has compounding pros and cons. It is thus critical to spend enough time evaluating ideas in simulations of varying complexity before jumping to a HDL. I am glad to have done that to some extent with the C++ model and various Python studies, but, in retrospect, more time should have been spent desiging and obtaining proxy metrics to determine the ideal high-level architecture. However, I think such learning can only be appreciated once an architect goes through the full design cycle at least once, so I am glad to have had the opportunity to earn this wisdom, which I will carry in future projects.
    - ^ I'm realizing that a informed, complete, analysis of a design without necessarily implementing it is more insightful and
    and impactful than a somewhat underdiscussed but working design
- What I wish I could have learned more of:
    - Spent more time with the ARM Memory Compiler and Synopsys DC.
    - Gone further in the ASIC design process: Layout, post-layout simulations, etc.