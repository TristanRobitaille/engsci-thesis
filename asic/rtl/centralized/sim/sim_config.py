import os
from cocotb_test.simulator import run

# Paths
ip_dir = "asic/rtl/centralized/ip"
src_dir = "asic/rtl/centralized"
home_dir = os.path.expanduser("~/../tmp")
ip_dir = os.path.join(home_dir, ip_dir)
src_dir = os.path.join(home_dir, src_dir)

# Arguments
ARGS = [
    # Simulator
    "--trace",
    "--trace-fst",
    "--trace-structs",
    "-j","16",
    "--timescale","1ns/10ps",

    # SV compilation
    "-DCENTRALIZED_ARCH=1",
    "-DSTANDALONE_TB=1",
]

# Run the simulation
def test_dff_verilog():
    run(
        verilog_sources=[
            os.path.join(src_dir, "defines.svh"),
            # IP
            os.path.join(ip_dir, "counter/counter.sv"),
            os.path.join(ip_dir, "counter/CounterInterface.sv"),
            # Memory
            os.path.join(src_dir, "mem/int_res_mem.sv"),
            os.path.join(src_dir, "mem/params_mem.sv"),
            os.path.join(src_dir, "mem/mem_model.sv"),
            os.path.join(src_dir, "mem/MemoryInterface.sv"),
            # Main
            os.path.join(src_dir, "SoCInterface.sv"),
            os.path.join(src_dir, "cim_centralized.sv"),
            os.path.join(src_dir, "cim_centralized_tb.sv"),
        ],
        toplevel="cim_centralized_tb",
        module="inference_tb",
        compile_args=ARGS,
        sim_args=ARGS,
        make_args=["-j16"],
        waves=True
    )
