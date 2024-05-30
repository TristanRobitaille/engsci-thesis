## ASIC C++ functional simulation
The purpose of the functional simulation is to ensure the control logic for the ASIC accelerator is functional.
The simulation can also be used to run inference on a given set of input data to compare with the TensorFlow reference. It reads a .h5 dataset contain a table of EEG recordings (clips).
You can define a start and end index for the simulation to run on select clips. It saves the inferred sleep stage in a CSV. The number of decimal bits in the fixed-point format is
configurable in `include/Misc.hpp`. The simulation writes the inferred sleep stage in a column respective to `NUM_FRAC_BITS` in a given CSV, which can be used to run a study on the
accuracy of different fixed-point formats.

Note: The functional simulation makes use of Xilinx's HLS Arbitrary Precision Types library to accurately model fixed-point arithmetic of arbitrary width. Documentation can be found here: https://docs.amd.com/r/en-US/ug1399-vitis-hls/C-Arbitrary-Precision-Fixed-Point-Types.

### Setup
Clone repo.\
From the `asic/func_sim/` directory:
1. To initialize the submodules: `git submodule update --init --recursive`
2. To generate the Makefile, run `cmake CMakeLists.txt`

### How to run
Run from the `asic/func_sim/` directory.\
To compile and run the executable without writing the inferred sleep stage to the CSV, run `make && ./Func_Sim`
To compile and run the executable on a given set of clips in `reference_data/eeg.h5` and save to CSV, run `make && ./Func_Sim --start_index <index> --end_index <index> --results_csv_fp "<filepath>"`