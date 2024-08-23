## IP Verification
The purpose of these files is to exercise the functionality of core IPs required for this project.
It contains:
* ```adder```: Basic, general-purpose, fixed-point adder.
* ```counter```: Basic, general-purpose counter with parametrized width and posedge-triggered or level-trigged modes of operation.
* ```divider```: General-purpose fixed-point divider.
* ```exp```: Module for approximate fixed-point computation of e^x. It uses exponential identity and cubic Taylor series to approximate e^x. With 10b fractional fixed-point, it achieves 0.098% accuracy vs. Python's built-in fixed-point method and 2.54% accuracy compared to native Python float exponential.
* ```multiplier```: Basic, general-purpose fixed-point multiplier.
* ```sqrt```: Module to compute the square root of a fixed-point number.

### How to run
Run the testbench for a given IP with ```make && gtkwave dump.vcd``` in the IP's folder. This compiles the code, simulates it and opens the simulation waveform in GTKWave.