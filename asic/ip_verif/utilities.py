# Some common functions that are used in the testbenches
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge
from FixedPoint import FXfamily

#----- CONSTANTS -----#
NUM_INT_BITS = 12
NUM_FRACT_BITS = 10
MAX_INT_ADD = 2**(NUM_INT_BITS-1)/2 - 1
MAX_INT_MULT = 2**(NUM_INT_BITS//2-1) - 1

num_Q = FXfamily(NUM_FRACT_BITS, NUM_INT_BITS)

#----- HELPERS -----#
async def reset(dut):
    dut.rst_n.value = 0
    await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)

def BinToDec(dec:float):
    z2 = num_Q(dec)
    z2_str = z2.toBinaryString(logBase=1, twosComp=True).replace(".","")
    return int(z2_str, base=2)

async def start_routine_basic_arithmetic(dut):
    cocotb.start_soon(Clock(dut.clk, 1, units="ns").start())
    await RisingEdge(dut.clk)
    await reset(dut)
    dut.refresh.value = 1