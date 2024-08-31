# Simple tests for the fixed-point adder module
import random
import utilities
import Constants as const
from FixedPoint import FXnum

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge

#----- CONSTANTS -----#
TOLERANCE_REL = 0.025 # Percent
TOLERANCE_ABS = 0.01 # Absolute
# Reasonable values for the input range according to functional simulation
MIN_INPUT = -4
MAX_INPUT = 4
NUM_TESTS = 1000

#----- TESTS -----#
@cocotb.test()
async def random_test(dut):
    cocotb.start_soon(Clock(dut.clk, 10, units="ns").start()) # 100MHz
    await utilities.reset(dut)

    for _ in range(NUM_TESTS):
        await RisingEdge(dut.clk)
        input = random.uniform(MIN_INPUT, MAX_INPUT)
        # input = random.uniform(0, MAX_INPUT)

        dut.input_q.value = utilities.BinToDec(input, const.num_Q_comp)
        await utilities.start_pulse(dut)
        while dut.done.value != 1: await RisingEdge(dut.clk)
        
        expected = FXnum(input, const.num_Q_comp).exp()
        expected_str = expected.toBinaryString(logBase=1).replace(".","")
        below_tolerance_rel = (int(expected_str, base=2)*(1-TOLERANCE_REL)) <= int(dut.output_q.value) <= (int(expected_str, base=2)*(1+TOLERANCE_REL))
        below_tolerance_abs = abs(int(expected_str, base=2) - int(dut.output_q.value)) <= TOLERANCE_ABS * 2**const.num_Q_comp.fraction_bits
        assert ((below_tolerance_abs and below_tolerance_abs) or (below_tolerance_rel and not below_tolerance_abs)), f"Expected: {int(expected_str, base=2)}, received: {int(str(dut.output_q.value), base=2)} (input: {input:.6f})"
