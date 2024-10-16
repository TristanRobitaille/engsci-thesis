# Simple tests for the fixed-point square root module
import sys
sys.path.append("../../")
import random
from utilities import *
from FixedPoint import FXnum

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge

#----- CONSTANTS -----#
NUM_TESTS = 1000

#----- HELPERS -----#
async def start_pulse(dut):
    dut.start.value = 1
    await RisingEdge(dut.clk)
    dut.start.value = 0
    await RisingEdge(dut.clk)

@cocotb.test()
async def random_test(dut):
    cocotb.start_soon(Clock(dut.clk, 10, units="ns").start())
    await reset(dut)

    for _ in range(NUM_TESTS):
        await RisingEdge(dut.clk)
        radicand = random.uniform(0, 200)
        dut.rad_q.value = BinToDec(radicand, num_Q_comp)
        expected = FXnum(radicand, num_Q_comp).sqrt()
        expected_str = expected.toBinaryString(logBase=1).replace(".","")

        await start_pulse(dut)
        while dut.done.value != 1: await RisingEdge(dut.clk)
        assert ((int(expected_str, base=2)-1) <= int(dut.root_q.value) <= (int(expected_str, base=2)+1)), f"Expected: {int(expected_str, base=2)}, received: {int(str(dut.root_q.value), base=2)} (radicand: {radicand:.6f})"
