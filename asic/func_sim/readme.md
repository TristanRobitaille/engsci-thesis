## ASIC C++ functional simulation
The purpose of the functional simulation is to ensure the control logic for the ASIC accelerator is functional.

### Dependencies
To generate the Makefile, `CMake` is required. Install it with `brew install cmake`. \
To compile the simulation, `gcc`, `hdf5` and `boost` are required. Install them with `brew install gcc hdf5 boost`.

### How to run
Run from the `asic/func_sim/` directory. \
To generate the Makefile, run `cmake CMakeLists.txt` \
To compile and run the executable, run `make && ./func_sim`