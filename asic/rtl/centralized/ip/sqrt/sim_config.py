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
            os.path.join(src_dir, "Defines.svh"),
            os.path.join(src_dir, "ip/ComputeIPInterface.sv"),
            os.path.join(src_dir, "ip/sqrt/sqrt.sv"),
            os.path.join(src_dir, "ip/sqrt/sqrt_tb.sv"),
        ],
        toplevel="sqrt_tb",
        module="sqrt_tb",
        compile_args=ARGS,
        sim_args=ARGS,
        make_args=["-j16"],
        waves=True
    )
