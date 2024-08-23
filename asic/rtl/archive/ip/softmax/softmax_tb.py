# Simple tests for the fixed-point softmax module
import sys
sys.path.append("../../")
import random
from utilities import *
from FixedPoint import FXnum

import cocotb
from cocotb.triggers import RisingEdge

#----- CONSTANTS -----#
LEN = 64
NUM_TESTS = 10
ABS_TOLERANCE = 50

#----- FUNCTIONS -----#
async def test_run(dut):
    start_addr = 10
    dut.start_addr.value = start_addr
    dut.len.value = LEN

    mem_copy = []
    mem_copy_exp = []
    exp_sum = BinToDec(0, num_Q_storage)

    # Store value in memory and partially compute expected values
    for i in range(LEN):
        data = random_input(-2, 2)
        data_q = FXnum(data, num_Q_comp)
        mem_copy.append(data_q)
        dut.int_res[start_addr+i].value = BinToDec(data, num_Q_storage)
        mem_copy_exp.append(data_q.exp())
        exp_sum += mem_copy_exp[-1]

    # Start and wait
    dut.start.value = 1
    while (dut.done == 0):
        await RisingEdge(dut.clk)
        dut.start.value = 0

    # Check values against expected
    for i in range(LEN):
        expected = mem_copy_exp[i] / exp_sum
        expected_str = FXnum(float(expected), num_Q_storage).toBinaryString(logBase=1).replace(".","")
        dut_value = int(dut.int_res[start_addr+i].value)

        within_spec = True

        expected_min = int(expected_str, base=2) - ABS_TOLERANCE
        expected_max = int(expected_str, base=2) + ABS_TOLERANCE

        if (expected_min < 0): # Need to check for underflow
            within_spec = ((dut_value > 2**(NUM_FRACT_BITS+NUM_INT_BITS_STORAGE) + expected_min) and (dut_value < 2**(NUM_FRACT_BITS+NUM_INT_BITS_STORAGE))) or (int(expected_str, base=2) <= int(expected_str, base=2) + ABS_TOLERANCE)
        else:
            within_spec = expected_min < dut_value < expected_max
        assert (within_spec), f"Softmax output value is incorrect at i={i}! Expected: {int(expected_str, base=2)}, got: {int(dut.int_res[start_addr+i].value)}"

    print("Success!")

#----- TESTS -----#
@cocotb.test()
async def random_test(dut):
    cocotb.start_soon(Clock(dut.clk, 10, units="ns").start())
    await RisingEdge(dut.clk)
    await reset(dut)

    for _ in range(NUM_TESTS):
        await test_run(dut)
        await RisingEdge(dut.clk)
