# Simple tests for the fixed-point divider module
import sys
sys.path.append("../../")
import random
from utilities import *
from FixedPoint import FXnum

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge

NUM_TESTS = 1000

@cocotb.test()
async def random_test(dut):
    cocotb.start_soon(Clock(dut.clk, 10, units="ns").start())
    await reset(dut)

    for _ in range(NUM_TESTS):
        await RisingEdge(dut.clk)
        dividend = random.uniform(-200, -200)
        divisor = random.uniform(-2000, -2000)
        if (abs(divisor) < 0.1): divisor = 0.1
        dut.dividend.value = BinToDec(dividend, num_Q_comp)
        dut.divisor.value = BinToDec(divisor, num_Q_comp)
        expected = FXnum(dividend, num_Q_comp)/FXnum(divisor, num_Q_comp)
        expected_str = expected.toBinaryString(logBase=1).replace(".","")

        await start_pulse(dut)
        while dut.done.value != 1: await RisingEdge(dut.clk)
        assert ((int(expected_str, base=2)-1) <= int(dut.output_q.value) <= (int(expected_str, base=2)+1)), f"Expected: {int(expected_str, base=2)}, received: {int(str(dut.output_q.value), base=2)} (dividend: {dividend:.6f}, divisor: {divisor:.6f})"
