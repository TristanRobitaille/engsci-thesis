# Simple tests for the fixed-point divider module
import random
import pytest
import utilities
import Constants as const
from FixedPoint import FXnum

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge

NUM_TESTS = 1000

@cocotb.test()
async def random_test(dut):
    cocotb.start_soon(Clock(dut.clk, 10, units="ns").start())
    await utilities.reset(dut)

    for _ in range(NUM_TESTS):
        await RisingEdge(dut.clk)
        dividend = random.uniform(-1000, 1000)
        divisor = random.uniform(-2000, 2000)
        if (abs(divisor) < 0.1): divisor = 0.1
        dut.dividend.value = utilities.BinToDec(dividend, const.num_Q_comp)
        dut.divisor.value = utilities.BinToDec(divisor, const.num_Q_comp)
        expected = FXnum(dividend, const.num_Q_comp) / FXnum(divisor, const.num_Q_comp)
        expected_str = expected.toBinaryString(logBase=1).replace(".","")

        dut.start.value = 1
        await RisingEdge(dut.clk)
        dut.start.value = 0
        while dut.done.value != 1: await RisingEdge(dut.clk)

        assert dut.output_q.value == pytest.approx(int(expected_str, base=2), rel=1e-3), \
            f"Expected: {int(expected_str, base=2)}, received: {int(str(dut.output_q.value), base=2)} (dividend: {const.num_Q_comp(dividend)}, divisor: {const.num_Q_comp(divisor)})"
