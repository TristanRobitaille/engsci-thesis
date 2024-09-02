# Simple tests for the fixed-point multiplier module
import pytest
import random
import utilities
import Constants as const

import cocotb
from cocotb.triggers import RisingEdge

NUM_TESTS = 1000

#----- HELPERS -----#
async def output_check(dut, in_1:float, in_2:float):
    # if expected is None:
    #     expected = FXnum(in_1, const.num_Q_comp)*FXnum(in_2, const.num_Q_comp)
    expected = const.num_Q_comp(in_1) * const.num_Q_comp(in_2)
    expected_str = expected.toBinaryString(logBase=1).replace(".","")

    dut.input_q_1.value = utilities.BinToDec(in_1, const.num_Q_comp)
    dut.input_q_2.value = utilities.BinToDec(in_2, const.num_Q_comp)
    dut.start.value = 1
    await RisingEdge(dut.clk)
    dut.start.value = 0
    await RisingEdge(dut.clk)

    # assert dut.output_q.value == pytest.approx(int(expected_str, base=2), rel=0), \
    #     f"Expected: {int(expected_str, base=2)}, received: {int(str(dut.output_q.value), base=2)} (in_1: {const.num_Q_comp(input_1)}, in_2: {const.num_Q_comp(input_2)})"

    # assert (dut.output_q.value == int(expected_str, base=2) or dut.output_q.value == (int(expected_str, base=2)-1)), f"Expected: {int(expected_str, base=2)}, received: {int(str(dut.output_q.value), base=2)} (in1: {in1:.6f}, in2: {in2:.6f})"
    assert dut.output_q.value == pytest.approx(int(expected_str, base=2), rel=0), \
        f"Expected: {int(expected_str, base=2)}, received: {int(str(dut.output_q.value), base=2)} (in_1: {in_1:.6f}, in_2: {in_2:.6f})"

#----- TESTS -----#
@cocotb.test()
async def basic_test(dut):
    await utilities.start_routine_basic_arithmetic(dut)
    input_1 = -0.5
    input_2 = 10.25
    dut.input_q_1.value = utilities.BinToDec(input_1, const.num_Q_comp)
    dut.input_q_2.value = utilities.BinToDec(input_2, const.num_Q_comp)

    # Reset
    for _ in range(2): await RisingEdge(dut.clk)
    await utilities.reset(dut)
    assert dut.output_q.value == 0, "Expected: %d, received: %d" % (0, dut.output_q.value)

    # Refresh
    await output_check(dut, input_1, input_2)

@cocotb.test()
async def random_test(dut):
    await utilities.start_routine_basic_arithmetic(dut)

    # Random inputs
    for _ in range(NUM_TESTS):
        input_1 = random.uniform(-utilities.MAX_INT_MULT, utilities.MAX_INT_MULT)
        input_2 = random.uniform(-utilities.MAX_INT_MULT, utilities.MAX_INT_MULT)
        await output_check(dut, input_1, input_2)

@cocotb.test()
async def unary_test_cases(dut):
    await utilities.start_routine_basic_arithmetic(dut)

    input_q_1 = [0, 1, 1, 0, -1]
    input_q_2 = [0, 1, -1, -1, -1]

    # Edge cases
    for i in range(len(input_q_1)):
        await output_check(dut, input_q_1[i], input_q_2[i])