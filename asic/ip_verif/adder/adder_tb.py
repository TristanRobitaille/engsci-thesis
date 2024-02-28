# Simple tests for an counter module
import sys
sys.path.append("..")
from utilities import *

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, FallingEdge

import random
from FixedPoint import FXfamily

num_Q = FXfamily(NUM_FRACT_BITS, NUM_INT_BITS)

#----- HELPERS -----#
def BinToDec(dec:float):
    z2 = num_Q(dec)
    z2_str = z2.toBinaryString(logBase=1).replace(".","")
    return int(z2_str, base=2)

async def start_routine(dut):
    cocotb.start_soon(Clock(dut.clk, 1, units="ns").start())
    await RisingEdge(dut.clk)
    await reset(dut)
    dut.refresh.value = 1

#----- TESTS -----#
@cocotb.test()
async def basic_reset(dut):
    await start_routine(dut)
    input1 = 0.5
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
    dut.input_q_1.value = BinToDec(2.5)
    dut.input_q_2.value = BinToDec(13.25)
    for _ in range(2): await RisingEdge(dut.clk)
    expected = num_Q(input1)+num_Q(input2)
    expected_str = expected.toBinaryString(logBase=1).replace(".","")
    assert dut.output_q.value == int(expected_str, base=2), "Expected: %d, received: %d" % (int(expected_str, base=2), dut.output_q.value)

@cocotb.test()
async def basic_count(dut):
    await start_routine(dut)

    for _ in range(100):
        input1 = random.uniform(0, MAX_INT)
        input2 = random.uniform(0, MAX_INT)
        expected = num_Q(input1)+num_Q(input2)
        expected_str = expected.toBinaryString(logBase=1).replace(".","")
        dut.input_q_1.value = BinToDec(input1)
        dut.input_q_2.value = BinToDec(input2)
        for _ in range(2): await RisingEdge(dut.clk)
        assert dut.output_q.value == int(expected_str, base=2), "Expected: %d, received: %d" % (int(expected_str, base=2), dut.output_q.value)

@cocotb.test()
async def overflow(dut):
    await start_routine(dut)

    input_q_1 = [2**(NUM_INT_BITS-1)-1]
    input_q_2 = [2**(NUM_INT_BITS-1)-1]
    expected  = [0]

    for _ in range(2): await RisingEdge(dut.clk)

    for i in range(len(input_q_1)):
        dut.input_q_1.value = BinToDec(input_q_1[i])
        dut.input_q_2.value = BinToDec(input_q_2[i])
        expected = num_Q(expected[i])
        expected_str = expected.toBinaryString(logBase=1).replace(".","")
        for _ in range(2): await RisingEdge(dut.clk)
        assert dut.overflow.value == 1, "Overflow not set as expected!"
