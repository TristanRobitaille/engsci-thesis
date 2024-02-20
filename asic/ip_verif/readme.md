## IP Verification
The purpose of these files is to exercise the functionality of core IPs required for this project.
It contains:
* ```counter```: Basic, general-purpose counter with parametrized width and posedge-triggered or level-trigged modes of operation

### How to run
Run the testbench for a given IP with ```make && gtkwave dump.vcd``` in the IP's folder. This compiles the code, simulates it and opens the simulation waveform in GTKWave.