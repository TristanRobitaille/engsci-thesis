import os
from cocotb_test.simulator import run

# Paths
ip_dir = "asic/rtl/ip"
src_dir = "asic/rtl/centralized"
home_dir = os.path.expanduser("~/../tmp")
ip_dir = os.path.join(home_dir, ip_dir)
src_dir = os.path.join(home_dir, src_dir)

# Arguments
COMPILE_ARGS = [
    "-DCENTRALIZED_ARCH=1",
    "-DSTANDALONE_TB=1"
]

SIM_ARGS = [
    "--trace-fst",
    "--trace-structs",
    "-j16",
    "--timescale=1ns/10ps",
]

# Run the simulation
def test_dff_verilog():
    run(
        verilog_sources=[
            os.path.join(src_dir, "cim_centralized.sv"),
            os.path.join(ip_dir, "counter/counter.sv"),
        ],
        toplevel="cim_centralized",
        module="cim_centralized_tb",
        compile_args=COMPILE_ARGS,
        sim_args=SIM_ARGS,
        make_args=["-j16"],
        waves=False
    )