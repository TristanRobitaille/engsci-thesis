# Thesis interim report notes
Notes for writing interim report, due January 14, 2024.

## Introduction

## Literature review

## Progress to date
### TensorFlow model
- Model architecture
- SLURM tooling
- Hyperparameter search

### Edge TPU
- Ran it, got latency metrics (present in table).
- [Need to do] Researched potential integration with MCU for prototyping purposes.

### ASIC RTL
- Instrumenting model, running quantization, sampling frequency and clip length experiments
- Explain rationale for design decisions
    - Low power -> Switch less charge
    - Low area -> Maximize area 
- Determined workflow

## Future work
### TensorFlow model
- [Need to do] Integrate PSG data from Stanford Technology Analytics and Genomics in Sleep (STAGES), pending access approval
- RTL and logic simulation
- Physical design synthesis and simulation (power, area, frequency)