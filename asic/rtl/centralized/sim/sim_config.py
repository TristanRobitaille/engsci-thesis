import os
import sys
import socket
from cocotb_test.simulator import run

RANDOM_SEED = 1

# Paths
hostname = socket.gethostname()
if "cedar" in hostname: repo_root = os.path.expanduser("~/projects/def-xilinliu/tristanr/engsci-thesis")
else: repo_root = os.path.expanduser("~/../tmp")

src_dir = "asic/rtl/centralized"
ip_dir = "asic/rtl/centralized/ip"
src_dir = os.path.join(repo_root, src_dir)
ip_dir = os.path.join(repo_root, ip_dir)

sys.path.append(f"{repo_root}/asic/rtl/centralized/sim") # For constants from Centralized CiM testbench
sys.path.append(f"{repo_root}/asic/rtl/") # For testbench utilities

OVERFLOW_WARNING = False
ALLOW_NEG_RADICAND_SQRT = True # Allow negative radicand in sqrt
ASSERTIONS_ENABLE = False

# Arguments
ARGS = [
    # Verilator
    "--trace",
    "--trace-fst",
    "--trace-structs",
    "-j","16",
    "--timescale","1ns/10ps",
    "-output-split","15000",
    "-O3",

    # SV compilation
    "-DCENTRALIZED_ARCH=1",
    "-DUSE_MEM_MODEL=1",
]

if OVERFLOW_WARNING: ARGS.append("-DOVERFLOW_WARNING")
if ALLOW_NEG_RADICAND_SQRT: ARGS.append("-DALLOW_NEG_RAD_SQRT")
if ASSERTIONS_ENABLE: ARGS.append("-DASSERTIONS_ENABLE")

# Run the simulation
def test_dff_verilog():
    run(
        verilog_sources=[
            os.path.join(src_dir, "Defines.svh"),
            # Memory
            os.path.join(src_dir, "mem/int_res_mem.sv"),
            os.path.join(src_dir, "mem/params_mem.sv"),
            os.path.join(src_dir, "mem/mem_model.sv"),
            os.path.join(src_dir, "mem/MemoryInterface.sv"),
            # IP
            os.path.join(ip_dir, "counter/counter.sv"),
            os.path.join(ip_dir, "counter/CounterInterface.sv"),
            os.path.join(ip_dir, "ComputeIPInterface.sv"),
            os.path.join(ip_dir, "adder/adder.sv"),
            os.path.join(ip_dir, "multiplier/multiplier.sv"),
            os.path.join(ip_dir, "divider/divider.sv"),
            os.path.join(ip_dir, "exp/exp.sv"),
            os.path.join(ip_dir, "mac/mac.sv"),
            os.path.join(ip_dir, "sqrt/sqrt.sv"),
            os.path.join(ip_dir, "softmax/softmax.sv"),
            os.path.join(ip_dir, "layernorm/layernorm.sv"),
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
        waves=True,
        seed=RANDOM_SEED
    )
