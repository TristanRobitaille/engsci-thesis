# Some common functions that are used in the testbenches
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge
from FixedPoint import FXfamily

#----- CONSTANTS -----#
NUM_INT_BITS_STORAGE = 6
NUM_INT_BITS_COMP = 12
NUM_FRACT_BITS = 10
MAX_INT_ADD = 2**(NUM_INT_BITS_COMP-1)/2 - 1
MAX_INT_MULT = 2**(NUM_INT_BITS_COMP//2-1) - 1
NUM_HEADS = 8
ASIC_FREQUENCY_MHZ = 100
SAMPLING_FREQ_HZ = 128
CLIP_LENGTH_S = 30
EEG_SAMPLING_PERIOD_CLOCK_CYCLE = 50 # Cannot simulate the real sampling period as it would take too long

num_Q_storage = FXfamily(NUM_FRACT_BITS, NUM_INT_BITS_STORAGE)
num_Q_comp = FXfamily(NUM_FRACT_BITS, NUM_INT_BITS_COMP)

#----- HELPERS -----#
async def reset(dut):
    dut.rst_n.value = 0
    await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)

async def start_pulse(dut):
    dut.start.value = 1
    await RisingEdge(dut.clk)
    dut.start.value = 0
    await RisingEdge(dut.clk)

def BinToDec(dec:float, num_type:FXfamily):
    z2 = num_type(dec)
    z2_str = z2.toBinaryString(logBase=1, twosComp=True).replace(".","")
    return int(z2_str, base=2)

async def start_routine_basic_arithmetic(dut):
    cocotb.start_soon(Clock(dut.clk, 1, units="ns").start())
    await RisingEdge(dut.clk)
    await reset(dut)
    dut.refresh.value = 1