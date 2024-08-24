import os
import sys
from cocotb_test.simulator import run

# Paths
src_dir = "asic/rtl/centralized"
home_dir = os.path.expanduser("~/../tmp")
src_dir = os.path.join(home_dir, src_dir)

sys.path.append(f"{home_dir}/asic/rtl/centralized/sim") # For constants from Centralized CiM testbench
sys.path.append(f"{home_dir}/asic/rtl/") # For testbench utilities

# Arguments
ARGS = [
    # Simulator
    "--trace",
    "--trace-fst",
    "--trace-structs",
    "--trace-max-array","1024", # Max array depth
    "-j","16",
    "--timescale","1ns/10ps",
]

# Run the simulation
def test_dff_verilog():
    run(
        verilog_sources=[
            os.path.join(src_dir, "defines.svh"),
            os.path.join(src_dir, "ip/ComputeIPInterface.sv"),
            os.path.join(src_dir, "ip/adder/adder.sv"),
            os.path.join(src_dir, "ip/adder/adder_tb.sv"),
        ],
        toplevel="adder_tb",
        module="adder_tb",
        compile_args=ARGS,
        sim_args=ARGS,
        make_args=["-j16"],
        waves=True
    )
