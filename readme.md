# Implementation of AI model on FPGA for sleep stage detection

Part of my Engineering Science (ECE option) at the University of Toronto. Supervised by Prof. Xilin Liu of UofT X-Lab.\
Project description: https://tristan.cbcr.me/?type=a#edge_AI

## Dependencies
#### Tensorflow
Install Python dependencies with: ```pip3 install -r requirements.txt```.\
Install EdgeTPU runtime and PyCoral by following this [guide](https://coral.ai/docs/accelerator/get-started/) .

#### RTL
Install `verilator` for Verilog compilation (requires MacOS Homebrew): ```brew install verilator```.\
Install `gtkwave` to view HDL simulation waveforms (requires MacOS Homebrew): ```brew install --cask gtkwave```.

#### LaTeX
To compile LaTeX documents, install `latexml`, `texlive` and `biber`: ```brew install latexml texlive biber```.\
I recommend using the `LaTeX Workshop` extension in VSCode.