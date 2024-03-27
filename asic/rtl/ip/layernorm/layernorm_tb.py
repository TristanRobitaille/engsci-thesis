# Simple tests for the fixed-point MAC module
import sys
sys.path.append("../../")
from utilities import *
from FixedPoint import FXnum

import cocotb
from cocotb.triggers import RisingEdge

#----- CONSTANTS -----#
LEN_FIRST_HALF = 64
LEN_SECOND_HALF = NUM_PATCHES + 1
MAX_LEN = 64
NUM_TESTS = 10
ABS_TOLERANCE = 15 # In fixed-point format

#----- FUNCTIONS -----#
async def test_run(dut):
    # Prepare MAC
    start_addr = 10
    dut.start_addr.value = start_addr
    dut.half_select.value = LayerNormHalfSelect.FIRST_HALF.value

    # Fill memory with random values and compute expected value
    mean = FXnum(0, num_Q_comp)
    mem_copy = []
    for i in range(LEN_FIRST_HALF):
        data = random_input(-MAX_VAL, MAX_VAL)
        mem_copy.append(FXnum(data, num_Q_comp))
        dut.int_res[start_addr+i].value = BinToDec(data, num_Q_storage)
        mean += FXnum(data, num_Q_comp)
    mean /= FXnum(LEN_FIRST_HALF, num_Q_comp)

    variance = FXnum(0, num_Q_comp)
    temp = FXnum(0, num_Q_comp)
    for i in range(LEN_FIRST_HALF):
        temp = mem_copy[i] - mean
        variance += temp*temp

    variance /= FXnum(LEN_FIRST_HALF, num_Q_comp)
    std_dev = variance.sqrt()
    for i in range(LEN_FIRST_HALF): mem_copy[i] = (mem_copy[i] - mean) / std_dev

    dut.start.value = 1
    while dut.done.value == 0:
        await RisingEdge(dut.clk)
        dut.start.value = 0

    # Check results for 1st half
    for i in range(LEN_FIRST_HALF):
        expected_str = FXnum(float(mem_copy[i]), num_Q_storage).toBinaryString(logBase=1).replace(".","")
        assert ((int(expected_str, base=2)-ABS_TOLERANCE) <= int(dut.int_res[start_addr+i].value) <= (int(expected_str, base=2)+ABS_TOLERANCE)), f"MAC output value is incorrect (1st half) at i={i}! Expected: {int(expected_str, base=2)}, got: {int(dut.int_res[start_addr+i].value)}"

    print(f"LayerNorm 1st half passed!")
    await cocotb.triggers.ClockCycles(dut.clk, 1000)

    # 2nd half
    dut.half_select.value = LayerNormHalfSelect.SECOND_HALF.value
    beta = random_input(-MAX_VAL, MAX_VAL)
    gamma = random_input(-MAX_VAL, MAX_VAL)
    beta_addr = 300
    gamma_addr = 301
    dut.beta_addr.value = beta_addr # Random addresses
    dut.gamma_addr.value = gamma_addr
    dut.params[beta_addr].value = BinToDec(beta, num_Q_storage)
    dut.params[gamma_addr].value = BinToDec(gamma, num_Q_storage)

    for i in range(LEN_SECOND_HALF):
        mem_copy[i] = FXnum(gamma, num_Q_comp) * mem_copy[i] + FXnum(beta, num_Q_comp)

    dut.start.value = 1
    while dut.done.value == 0:
        await RisingEdge(dut.clk)
        dut.start.value = 0

    for i in range(LEN_SECOND_HALF):
        expected_str = FXnum(float(mem_copy[i]), num_Q_storage).toBinaryString(logBase=1).replace(".","")
        assert ((int(expected_str, base=2)-ABS_TOLERANCE) <= int(dut.int_res[start_addr+i].value) <= (int(expected_str, base=2)+ABS_TOLERANCE)), f"MAC output value is incorrect (2nd half) at i={i}! Expected: {int(expected_str, base=2)}, got: {int(dut.int_res[start_addr+i].value)}"

    print(f"LayerNorm 2nd half passed!")
    await cocotb.triggers.ClockCycles(dut.clk, 10)

#----- TESTS -----#
@cocotb.test()
async def random_test(dut):
    cocotb.start_soon(Clock(dut.clk, 1, units="ns").start())
    await RisingEdge(dut.clk)
    await reset(dut)

    for _ in range(NUM_TESTS):
        await test_run(dut)
