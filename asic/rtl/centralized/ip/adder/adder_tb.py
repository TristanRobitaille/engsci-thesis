# Simple tests for the fixed-point adder module
import random
import pytest
import utilities
import Constants as const

import cocotb
from cocotb.triggers import RisingEdge, FallingEdge

#----- CONSTANTS -----#
NUM_TESTS = 1000

#----- TESTS -----#
@cocotb.test()
async def basic_reset(dut):
    await utilities.start_routine_basic_arithmetic(dut)
    input_1 = 0.5
    input_2 = 10.25
    dut.input_q_1.value = utilities.BinToDec(input_1, const.num_Q_comp)
    dut.input_q_2.value = utilities.BinToDec(input_2, const.num_Q_comp)

    # Reset
    for _ in range(2): await RisingEdge(dut.clk)
    await utilities.reset(dut)
    assert dut.output_q.value == 0, "Expected: %d, received: %d" % (0, dut.output_q.value)

    # Refresh prevents outputs from being updated
    await FallingEdge(dut.clk)
    dut.start.value = 0
    for _ in range(1): await RisingEdge(dut.clk)
    dut.input_q_1.value = utilities.BinToDec(2.5, const.num_Q_comp)
    dut.input_q_2.value = utilities.BinToDec(13.25, const.num_Q_comp)
    for _ in range(2): await RisingEdge(dut.clk)
    expected = const.num_Q_comp(input_1) + const.num_Q_comp(input_2)
    expected_str = expected.toBinaryString(logBase=1).replace(".","")
    assert dut.output_q.value == pytest.approx(int(expected_str, base=2), rel=0), \
        f"Expected: {int(expected_str, base=2)}, received: {int(dut.output_q.value, base=2)} (in_1: {input_1}, in_2: {input_2})"

@cocotb.test()
async def basic_count(dut):
    await utilities.start_routine_basic_arithmetic(dut)
    
    await RisingEdge(dut.clk)
    for _ in range(NUM_TESTS):
        input_1 = random.uniform(-const.MAX_INT_ADD, const.MAX_INT_ADD)
        input_2 = random.uniform(-const.MAX_INT_ADD, const.MAX_INT_ADD)
        expected = const.num_Q_comp(input_1) + const.num_Q_comp(input_2)
        expected_str = expected.toBinaryString(logBase=1).replace(".","")
        dut.input_q_1.value = utilities.BinToDec(input_1, const.num_Q_comp)
        dut.input_q_2.value = utilities.BinToDec(input_2, const.num_Q_comp)
        for _ in range(2): await RisingEdge(dut.clk)

        # Check that expected_str is a string
        assert dut.output_q.value == pytest.approx(int(expected_str, base=2), rel=0), \
            f"Expected: {int(expected_str, base=2)}, received: {int(str(dut.output_q.value), base=2)} (in_1: {const.num_Q_comp(input_1)}, in_2: {const.num_Q_comp(input_2)})"

@cocotb.test()
async def overflow(dut):
    await utilities.start_routine_basic_arithmetic(dut)

    input_q_1 = [2**(const.N_COMP-const.Q_COMP-1)-1]
    input_q_2 = [2**(const.N_COMP-const.Q_COMP-1)-1]
    expected  = [0]

    for i in range(len(input_q_1)):
        dut.input_q_1.value = utilities.BinToDec(input_q_1[i], const.num_Q_comp_overflow)
        dut.input_q_2.value = utilities.BinToDec(input_q_2[i], const.num_Q_comp_overflow)
        expected = const.num_Q_comp_overflow(expected[i])
        for _ in range(2): await RisingEdge(dut.clk)
        assert dut.overflow.value == 1, "Overflow not set as expected!"
