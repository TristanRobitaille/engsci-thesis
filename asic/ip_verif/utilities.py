# Some common functions that are used in the testbenches
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge

#----- CONSTANTS -----#
NUM_INT_BITS = 22
NUM_FRACT_BITS = 10
MAX_INT = (2**(NUM_INT_BITS-1))/2 - 1

#----- HELPERS -----#
async def reset(dut):
    dut.rst_n.value = 0
    await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)