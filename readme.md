# AI accelerator ASIC for sleep stage classification

Part of my Engineering Science (ECE option) degree at the University of Toronto. Supervised by Prof. Xilin Liu of UofT X-Lab.\
Project description: https://tristan.cbcr.me/?type=a#edge_AI

## Cloning the repo
After you've cloned the repo, be sure to `git submodule init` and `git submodule update` to clone in the submodules for the ASIC functional simulation.

## Docker
You can now run everything inside a Docker container. Ensure you have the Docker app running.
Build it with `docker build -t engsci-thesis .`. Be prepared to wait around 20 minutes for the build process to complete. Unfortunately, there are a lot of components to this thesis, and a lot of dependencies.
Run it with ``docker run -it -v `pwd`:/tmp engsci-thesis``. 
You will also probably want to use VSCode with the container:
1. Install `Dev Containers` in VSCode.
2. Run the container: ``docker run -it -v `pwd`:/tmp engsci-thesis``.
3. Execute the command: "Dev Containers: Attach to Running Container" and select the `engsci-thesis` container.
4. [First time only] In the container, install the VSCode extensions `C/C++` and `debugpy`.

#### Apptainer on Compute Canada
Compute Canada does not use Docker, but rather Apptainer. We must generate an Apptainer image (`.sif`) from the Docker image and transfer it to the server in order to run the container.
1. Install Apptainer: https://apptainer.org/docs/admin/main/installation.html#
2. Follow Compute Canada's instruction to build an Apptainer image from a Dockerfile: https://docs.alliancecan.ca/wiki/Apptainer#Creating_an_Apptainer_container_from_a_Dockerfile
3. SCP the `.sif` file over: `scp -v engsci-thesis.sif <yourusername>@cedar.computecanada.ca:/home/<yourusername>/projects/def-xilinliu/<yourusername>/engsci-thesis`
4. Load Apptainer: `module load apptainer/1.2.4`
5. Run Apptainer container: `apptainer shell engsci-thesis.sif`

Also, although `.sh` files should already have the execute permission, run `source add_execute_permission.sh .` to add it, which lets us run the script from the container externally without sourcing it. 
You can run commands directly, without having to open an interactive shell (this is what the slurm job files use): `apptainer run engsci-thesis.sif <command>`.

## Dependencies
### Tensorflow
Consult `/python_prototype/requirements.txt` for the Python dependencies. These are installed when building the Docker container.\
You will also need `graphviz`; install it with `brew install graphviz`.\
(Optional and not in container) Install EdgeTPU runtime and PyCoral by following this [guide](https://coral.ai/docs/accelerator/get-started/).

### ASIC Function Simulation
The extra libraries with use are `hdf5`, `boost` and `armadillo`. These are installed when building the Docker container.\
To generate the Makefile, run `cmake CMakeLists.txt` from `/asic/func_sim`. Compile with `make -j12`. Run with `./build/Func_Sim`.

### RTL
For RTL verification, we use `verilator` and `CocoTB`. These are installed when building the Docker container.\
You can inspect dumpfiles with `gtkwave`. Install the program with Homebrew: `brew install --cask gtkwave`.\

### LaTeX
To compile LaTeX documents, we use `latexmk`. We also need `biber` and `texlive`. To generate pretty EPS plots, we use the Python package `SciencePlots`.
The VSCode extension `LaTeX Workshop` is decent to browse the files and insert symbols. You can compile the PDF from the Terminal: `latexmk <filepath>.tex`.

## Map
In this repository, you will find most files for this thesis.\
In `/python_prototype`, you'll find all files for the vision transformer model, written using TensorFlow. There all also files for signal extraction and data processing from raw polysomnography datasets. Finally, it contains shell scripts to train on the Compute Canada cluster and some utilities files.\
In `/latex`, you'll find all deliverables for this course (ESC499).\
In `/asic`, you'll find a C++ functional model of the ASIC accelerator and most files for its synthesis (SystemVerilog, testbenches, etc.). Some files are excluded as they may contain information under NDA. You will also find files for a fixed-point accuracy study use to determine the optimal fixed-point format.

## Note
Always execute the SLURM scripts from the root of this repository