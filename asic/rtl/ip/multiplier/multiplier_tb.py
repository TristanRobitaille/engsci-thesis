# Simple tests for the fixed-point multiplier module
import sys
sys.path.append("..")
import random
from asic.rtl.utilities import *
from FixedPoint import FXnum

import cocotb
from cocotb.triggers import RisingEdge, FallingEdge

#----- HELPERS -----#
async def output_check(dut, in1:float, in2:float, expected=None):
    if expected is None:
        expected = FXnum(in1, num_Q)*FXnum(in2, num_Q)
    expected_str = expected.toBinaryString(logBase=1).replace(".","")

    dut.input_q_1.value = BinToDec(in1)
    dut.input_q_2.value = BinToDec(in2)
    for _ in range(2): await RisingEdge(dut.clk)
    assert (dut.output_q.value == int(expected_str, base=2) or dut.output_q.value == (int(expected_str, base=2)-1)), f"Expected: {int(expected_str, base=2)}, received: {int(str(dut.output_q.value), base=2)} (in1: {in1:.6f}, in2: {in2:.6f})"

#----- TESTS -----#
@cocotb.test()
async def basic_reset(dut):
    await start_routine_basic_arithmetic(dut)
    input1 = -0.5
    input2 = 10.25
    dut.input_q_1.value = BinToDec(input1)
    dut.input_q_2.value = BinToDec(input2)

    # Reset
    for _ in range(2): await RisingEdge(dut.clk)
    await reset(dut)
    assert dut.output_q.value == 0, "Expected: %d, received: %d" % (0, dut.output_q.value)

    # Refresh
    await FallingEdge(dut.clk)
    dut.refresh.value = 0
    for _ in range(1): await RisingEdge(dut.clk)
    await output_check(dut, 2.5, 13.25, num_Q(input1)*num_Q(input2))

@cocotb.test()
async def basic_mult(dut):
    await start_routine_basic_arithmetic(dut)
    input_q_1 = [0, 1, 1, 0]
    input_q_2 = [0, 1, -1, -1]

    # Random inputs
    for _ in range(1000):
        input1 = random.uniform(-MAX_INT_MULT, MAX_INT_MULT)
        input2 = random.uniform(-MAX_INT_MULT, MAX_INT_MULT)
        await output_check(dut, input1, input2)

    # Edge cases
    for i in range(len(input_q_1)):
        await output_check(dut, input_q_1[i], input_q_2[i])

@cocotb.test()
async def overflow(dut):
    await start_routine_basic_arithmetic(dut)

    input_q_1 = [2*MAX_INT_MULT+1]
    input_q_2 = [2*MAX_INT_MULT+1]

    for _ in range(2): await RisingEdge(dut.clk)

    for i in range(len(input_q_1)):
        dut.input_q_1.value = BinToDec(input_q_1[i])
        dut.input_q_2.value = BinToDec(input_q_2[i])
        for _ in range(2): await RisingEdge(dut.clk)
        assert dut.overflow.value == 1, "Overflow not set as expected!"