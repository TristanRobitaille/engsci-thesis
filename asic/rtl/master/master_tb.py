# Simple tests for the fixed-point counter module
import sys
sys.path.append("..")
from utilities import *

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge

@cocotb.test()
async def basic_count(dut):
    cocotb.start_soon(Clock(dut.clk, 1, units="ns").start())
    await reset(dut)

    for _ in range(200):
        await RisingEdge(dut.clk)
    
    dut.start_param_load.value = 1
    for _ in range(200):
        await RisingEdge(dut.clk)
