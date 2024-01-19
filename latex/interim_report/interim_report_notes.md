# Thesis interim report notes
Notes for writing interim report, due January 19, 2024.

## Introduction (adjusted version of thesis proposal)
- Motivate design: Insomnia is bad --> Insomnia treatments --> Neuromodulation --> In-ear ASIC
- Change references from FPGA to ASIC.
- Broad objective: Demonstrate feasibility of transformer-based single-channel (EEG) sleep stage classification on an in-ear ASIC.
    - Need to get figures of merit for energy per inference, total area, inference latency and inference accuracy.
- Although this work relies on deep software-hardware co-design to reach the desired figures of merit, the focus of its novelty is in the hardware implementation.

## Literature review I: AI for sleep stage classification (2 pages)
- CNN, LSTM and transformers have been used.
- What is a transformer
- Why current research uses transformers
    - Input data is sequence and transformer is particularly suite for sequence data because
    - Transformer are known as lightweight --> Directly reduced power and area
- Existing transformers in sleep stage classification literature:
    - Transformers in the literature are much too heavy (>3.7M parameters according to SleepTransformer: Automatic Sleep Staging With Interpretability and Uncertainty Quantification)
    - Some aren't suitable for live, in-situ sleep stage detection as they require multiple clips simultaneously (i.e. SleepTransformer, L-SeqSleepNet)
- Why a new transformer?
    - Need lightweight model for low-power and low area while keeping accuracy.
    - Also, most training is done on SleepEDF and MASS; we try augmenting dataset with Stanford STAGES due to its massive scale (1906 recordings vs 62).

## Precise design objectives and direction (2 pages)
- Table with
    - < 1mW average
    - < 200kB param.
    - > 80% accuracy
    - < 1mm^2 area for accelerator + weigths
    - < 30s inference time
- Explain rationale for design decisions
    - Low power -> Switch less charge
    - Low area -> Maximize area

## Literature review II: Transformer ASIC accelerator (2 pages)
- CiM

## Progress to date (2 pages)
### TensorFlow model
- Data processing/extraction
- Model architecture
- SLURM tooling
- Quantization
- Hyperparameter search to increase accuracy and reduce size
- Further methods to reduce model size:
    - [Need to do] Pruning

### Edge TPU
- Ran it, got latency metrics (present in table).
- Researched potential integration with MCU for prototyping purposes.

### ASIC RTL
- Instrumenting model, running quantization, sampling frequency and clip length experiments
- Determined workflow

## Future work (2 pages)
### TensorFlow model
- [Need to do] Integrate PSG data from Stanford Technology Analytics and Genomics in Sleep (STAGES), pending access approval
- Learn Synopsis Design Compiler
- RTL and logic simulation
- Physical design synthesis and simulation (power, area, frequency)