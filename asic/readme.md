### Note
The synthesized design cannot be shared as it contains information protected under NDA. I am using TSMC 65nm. I can only share the RTL and high-level simulation outputs.

### Running CocoTB testbenches
CocoTB is used for testbenches for the RTL modules. 
Running the testbench for the distributed vs. centralized (simply because I improved the workflow for the centralized architecture).

#### Testbench for distributed architecture
Run the testbench simply by executing `make` in the directory of interest. You might want to `make clean` if running into a compilation issue.

#### Testbench for the centralized architecture
Run the testbench simply by executing `source asic/rtl/centralized/sim/run_sim.sh`.