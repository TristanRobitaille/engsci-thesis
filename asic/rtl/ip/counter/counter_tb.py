# Simple tests for the fixed-point counter module
import sys
sys.path.append("../../")
from utilities import *

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge

#----- HELPERS -----#
async def inc_pulse(dut, inc_value):
    dut.inc.value = inc_value
    await RisingEdge(dut.clk)
    dut.inc.value = 0
    await RisingEdge(dut.clk)
    
@cocotb.test()
async def basic_count(dut):
    cocotb.start_soon(Clock(dut.clk, 1, units="ns").start())
    await reset(dut)

    for cnt in range(1,51):
        await inc_pulse(dut, 1)
        assert dut.cnt.value == cnt, "Counter result is incorrect: %d != %d" % (dut.cnt.value, cnt)

    await reset(dut)

    for cnt in range(3,51,3):
        await inc_pulse(dut, 3)
        assert dut.cnt.value == cnt, "Counter result is incorrect: %d != %d" % (dut.cnt.value, cnt)
