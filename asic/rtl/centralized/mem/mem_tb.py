import os
import sys
import cocotb
import random
from cocotb.triggers import RisingEdge
from cocotb.clock import Clock

# current_dir = os.path.dirname(__file__)
# parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
# sys.path.append(f"{current_dir}/../sim")
import Constants

# ----- CONSTANTS ----- #
CLK_FREQ_MHZ = 100
NUM_WRITES = 1000

# ----- HELPERS ----- #
async def reset(dut):
    dut.rst_n.value = 0
    await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)

async def write_one_word(dut, addr, data):
    dut.param_chip_en.value = 1
    dut.param_write_en.value = 1
    dut.param_write_data_width.value = 0 # Single-width
    dut.param_write_addr.value = addr
    dut.param_write_data.value = data
    await RisingEdge(dut.clk)
    dut.param_write_en.value = 0

# ----- TEST ----- #
@cocotb.test()
async def mem_tb(dut):
    cocotb.start_soon(Clock(dut.clk, 1/CLK_FREQ_MHZ, 'us').start())

    await reset(dut)

    for _ in range(NUM_WRITES):
        addr = random.randint(0, Constants.CIM_PARAMS_NUM_BANKS * Constants.CIM_PARAMS_BANK_SIZE_NUM_WORD)
        data = random.randint(0, 2**Constants.N_STO_PARAMS-1)
        await write_one_word(dut, addr, data)

        # Check if the data was written correctly
        dut.param_read_en.value = 1
        dut.param_read_addr.value = addr
        for _ in range(2): await RisingEdge(dut.clk)
        assert (dut.param_read_data.value == data), f"Expected {data}, got {dut.param_read_data.value}"
        dut.param_read_en.value = 0
