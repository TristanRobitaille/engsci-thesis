# Simple tests for the fixed-point MAC module
import sys
sys.path.append("../../")
from utilities import *
from FixedPoint import FXnum

import cocotb
from cocotb.triggers import RisingEdge

#----- CONSTANTS -----#
len = 64
MAX_LEN = 64

#----- TESTS -----#
@cocotb.test()
async def random_test(dut):
    cocotb.start_soon(Clock(dut.clk, 1, units="ns").start())
    await RisingEdge(dut.clk)
    await reset(dut)

    # Prepare MAC
    start_addr = 10
    dut.start_addr.value = start_addr
    dut.half_select.value = LayerNormHalfSelect.FIRST_HALF.value

    # Fill memory with random values and compute expected value
    mean = FXnum(0, num_Q_comp)
    mem_copy = []
    for i in range(len):
        data = random_input(0, MAX_VAL)
        mem_copy.append(FXnum(data, num_Q_comp))
        dut.int_res[start_addr+i].value = BinToDec(data, num_Q_storage)
        mean += FXnum(data, num_Q_comp)
    mean /= FXnum(len, num_Q_comp)

    variance = FXnum(0, num_Q_comp)
    temp = FXnum(0, num_Q_comp)
    for i in range(len):
        temp = mem_copy[i] - mean
        variance += temp*temp

    variance /= FXnum(len, num_Q_comp)
    std_dev = variance.sqrt()
    for i in range(len): mem_copy[i] = (mem_copy[i] - mean) / std_dev

    dut.start.value = 1
    while dut.done.value == 0:
        await RisingEdge(dut.clk)
        dut.start.value = 0

    # Check results
    for i in range(len):
        expected_str = FXnum(float(mem_copy[i]), num_Q_storage).toBinaryString(logBase=1).replace(".","")
        assert ((int(expected_str, base=2)-50) <= int(dut.int_res[start_addr+i].value) <= (int(expected_str, base=2)+50)), f"MAC output value is incorrect! Expected: {int(expected_str, base=2)}, got: {int(dut.int_res[start_addr+i].value)}"

    await cocotb.triggers.ClockCycles(dut.clk, 1000)
