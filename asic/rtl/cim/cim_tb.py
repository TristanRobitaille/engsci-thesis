# Simple tests for the fixed-point counter module
import math
import sys
sys.path.append("..")
from utilities import *

import cocotb
import cocotb.triggers
from cocotb.clock import Clock

#----- TESTS -----#
@cocotb.test()
async def basic_reset(dut):
    cocotb.start_soon(Clock(dut.clk, 1/ASIC_FREQUENCY_MHZ, units="us").start()) # 100MHz clock
    await reset(dut)
    await cocotb.triggers.ClockCycles(dut.clk, 1000)