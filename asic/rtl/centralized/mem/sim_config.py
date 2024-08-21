import os
import sys
from cocotb_test.simulator import run

# Paths
src_dir = "asic/rtl/centralized"
home_dir = os.path.expanduser("~/../tmp")
src_dir = os.path.join(home_dir, src_dir)

# Constants from Centralized CiM testbench
current_dir = os.path.dirname(__file__)
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(f"{current_dir}/../sim")

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
            os.path.join(src_dir, "mem/MemoryAccessSignals.sv"),
            os.path.join(src_dir, "mem/mem_model.sv"),
            os.path.join(src_dir, "mem/params_mem.sv"),
            os.path.join(src_dir, "mem/int_res_mem.sv"),
            os.path.join(src_dir, "mem/mem_tb.sv")
        ],
        toplevel="mem_tb",
        module="mem_tb",
        compile_args=ARGS,
        sim_args=ARGS,
        make_args=["-j16"],
        waves=True
    )
