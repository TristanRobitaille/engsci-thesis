import os
import sys
from cocotb_test.simulator import run

RANDOM_SEED = 123

# Paths
src_dir = "asic/rtl/centralized"
home_dir = os.path.expanduser("~/../tmp")
src_dir = os.path.join(home_dir, src_dir)

# Constants from Centralized CiM testbench
sys.path.append(f"{home_dir}/asic/rtl/centralized/sim") # For constants from Centralized CiM testbench
sys.path.append(f"{home_dir}/asic/rtl/") # For testbench utilities

ASSERTIONS_ENABLE = True

# Arguments
ARGS = [
    # Simulator
    "--trace",
    "--trace-fst",
    "--trace-structs",
    "--trace-max-array","1024", # Max array depth
    "-j","16",
    "--timescale","1ns/10ps",

    # SV compilation
    "-DUSE_MEM_MODEL=1",
]

if ASSERTIONS_ENABLE: ARGS.append("-DASSERTIONS_ENABLE")

# Run the simulation
def test_dff_verilog():
    run(
        verilog_sources=[
            os.path.join(src_dir, "Defines.svh"),
            os.path.join(src_dir, "mem/MemoryInterface.sv"),
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
        waves=True,
        seed=RANDOM_SEED
    )
