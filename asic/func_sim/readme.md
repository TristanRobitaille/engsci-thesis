## ASIC C++ functional simulation
The purpose of the functional simulation is to ensure the control logic for the ASIC accelerator is functional.
The simulation can also be used to run inference on a given set of input data to compare with the TensorFlow reference. It reads a .h5 dataset containing a table of EEG recordings (clips).You can define a start and end index for the simulation to run on select clips. It saves the inferred sleep stage in a CSV. The number of decimal bits in the fixed-point format is configurable in `include/Misc.hpp`. The simulation uses fixed-point format that is variable on a per-layer basis. The total number of storage bits is configurable at compile time by running `export N_STO_INT_RES=<xyz>` before compilation. It defaults to `N_STO_INT_RES=20`, which should cause no accuracy error. You can also change the number of storage bits for the parameters with `N_STO_PARAMS` instead.

Note: The functional simulation makes use of Xilinx's HLS Arbitrary Precision Types library to accurately model fixed-point arithmetic of arbitrary width. Documentation can be found here: https://docs.amd.com/r/en-US/ug1399-vitis-hls/C-Arbitrary-Precision-Fixed-Point-Types.

### How to run
Run from the `asic/func_sim/` directory.\
Prior to compiling, run `cmake CMakeLists.txt`.
To compile and run the executable without writing the inferred sleep stage to the CSV, run `make build && ./build/Func_Sim`
To compile and run the executable on a given set of clips in `../fixed_point_accuracy_study/eeg.h5` and save to CSV, run `make build && ./build/Func_Sim --start_index <index> --end_index <index> --results_csv_fp "<filepath>" --bit_type=INT_RES`. The last parameter, `--bit_type` is described in `fixed_point_accuracy_study/readme.md`.