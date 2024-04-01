## Thesis proposal document notes
### Context
* 23.8% of Canadians above 18 suffer from insomnia [insomnia_prevalence]
* Neuromodulation have been found to be an effective treatment against insomnia [yoon2021neuromodulation]
#### Problem with state of the art polysomnography:
* Number of sensors needed:
    * Polysomnography requires (at least) 8 EEG, 5 EMG, 4 sensors for respiration physiological signals, 1 pulse oximetry, 1 ECG = 19 sensors [RUNDO2019381]
* There is thus a need to develop a self-contained brain-machine interface (BMI) device for sleep stage detection and neuromodulation. The proposed thesis project focusses on the development of an AI model and its FPGA prototype implementation as part of the larger BMI device.

* Research question:
    * Refers

### Objectives
##### Constraints:
* Inference time < 30s (clip length)
* Whole architecture fits on Xilinx XYZ dev board and model fits on Google Coral

##### Metrics:
* Beat accuracy of state of the art non-AI sleep-stage detection systems (to justify new technology/approach to the problem)
* Beat area and power usage of Google's TPU (to justify custom hardware)
* Beat accuracy of other FPGA transformers (to justify novel publication)

### Methods
#### Steps
* Two ways of reaching the constraints are within the scope of this model: improving the AI model, and improving the AI hardware.
* 1) Develop and optimize transformer on TensorFlow
* 2) Write intermidiary C model of above transformer to aid in development of FPGA by providing more visibility of the calculations
* 3) Write model for Google Coral TPU. Run benchmarks.
* 4) Draw and write FPGA (Xilinx Red dev board) high-level HDL (Verilog) design.
* 5) Implement hardware improvements and compare with "basic" hardware implementation
* 6) Iterate on step 5, and step 1 (time-permitting) until Jan 15.
* 7) Final benchmark against Google TPU and literature

#### References for designing improvements
* AI:
* Hardware:

### Questions
* Is it appropriate to put the research question after Objectives, given that it will essentially be: "Can we beat what's available commercially and in research"
* Do I need specific technical improvements to model or hardware yet?

Can I target submitting paper to journals on these dates:
* Journal 1:
* Journal 2: 