# AI accelerator ASIC for sleep stage classification

Part of my Engineering Science (ECE option) degree at the University of Toronto. Supervised by Prof. Xilin Liu of UofT X-Lab.\
Project description: https://tristan.cbcr.me/?type=a#edge_AI

## Dependencies
#### Tensorflow
Create a virtualenv named `asic_sleep_trans` by running `python -m venv asic_sleep_trans` inside the `python_prototype/` directory
Activate the venv and install Python dependencies with: `source asic_sleep_trans/bin/activate && pip3 install -r requirements.txt`.\
Run the Python file from VS Code or look at `.vscode/launch.json` to for the required arguments if running from command line (don't forget to `source asic_sleep_trans/bin/activate` if running from command line).
(Optional) Install EdgeTPU runtime and PyCoral by following this [guide](https://coral.ai/docs/accelerator/get-started/).

#### ASIC Function Simulation
To generate the Makefile, `CMake` is required. Install it with `brew install cmake`. \
To compile the simulation, `gcc`, `hdf5`, `boost` and `armadillo` are required. Install them with `brew install gcc hdf5 boost armadillo`.

#### RTL
Install `verilator` for Verilog compilation (requires MacOS Homebrew): `brew install verilator`.\
Install `gtkwave` to view HDL simulation waveforms (requires MacOS Homebrew): `brew install --cask gtkwave`.\
Install Python dependencies for IP verification with `pip3 install -r asic/ip_verif/requirements.txt`

#### LaTeX
To compile LaTeX documents, install `latexml`, `texlive` and `biber`: `brew install latexml texlive biber`.\
You'll also need the Python package `SciencePlots` to export plots used in the thesis. Install it with: `python -m pip install git+https://github.com/garrettj403/SciencePlots.git`
I recommend using the `LaTeX Workshop` extension in VSCode.

## Map
In this repository, you will find most files for this thesis.\
In `/python_prototype`, you'll find all files for the vision transformer model, written using TensorFlow. There all also files for signal extract and data processing from raw polysomnography datasets. Finally, it contains shell scripts to train on the Compute Canada cluster and some utilities files.\
In `/latex`, you'll find all deliverables for this course (ESC499).\
In `/asic`, you'll find a C++ functional model of the ASIC accelerator and most files for its synthesis (SystemVerilog, testbenches, etc.). Some files are excluded as they may contain information under NDA.