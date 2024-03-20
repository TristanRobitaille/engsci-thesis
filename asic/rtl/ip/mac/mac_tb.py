# Simple tests for the fixed-point MAC module
import sys
sys.path.append("../../")
import random
from utilities import *
from FixedPoint import FXnum

import cocotb
from cocotb.triggers import RisingEdge

#----- CONSTANTS -----#
# Rough figures from functional simulation
MAX_LEN = 64
NUM_TESTS = 100

#----- HELPERS-----#                         
async def test_MAC(dut, activation=ActivationType.NO_ACTIVATION.value):
    # Prepare MAC
    expected_result = 0
    start_addr1 = random.randint(MAX_LEN, TEMP_RES_STORAGE_SIZE_CIM-1-MAX_LEN) # Starts at MAX_LEN to make it easier to correct for overlap
    start_addr2 = random.randint(MAX_LEN, PARAMS_STORAGE_SIZE_CIM-1-MAX_LEN)
    # Fix address if there's an overlap
    if ((start_addr1+MAX_LEN) > start_addr2):
        start_addr1 = start_addr2 - MAX_LEN
    elif ((start_addr2+MAX_LEN) > start_addr1):
        start_addr2 = start_addr1 - MAX_LEN

    len = 64
    param_type = MACParamType.INTERMEDIATE_RES.value

    dut.start_addr1.value = start_addr1
    dut.start_addr2.value = start_addr2
    dut.len.value = len
    dut.param_type.value = param_type
    dut.activation.value = activation

    expected_result = 0
    for i in range(len):
        in1 = random_input()
        in2 = random_input()
        bias = random_input()
        dut.intermediate_res[start_addr1+i].value = BinToDec(in1, num_Q_storage)
        dut.intermediate_res[start_addr2+i].value = BinToDec(in2, num_Q_storage)
        dut.params[param_addr_map["patch_proj_bias"]].value = BinToDec(bias, num_Q_storage)
        expected_result += FXnum(in1, num_Q_comp)*FXnum(in2, num_Q_comp)

    if activation == ActivationType.LINEAR_ACTIVATION.value:
        expected_result += FXnum(bias, num_Q_comp)

    dut.start.value = 1
    await RisingEdge(dut.clk)
    dut.start.value = 0
    await RisingEdge(dut.clk)

    # Wait for the MAC to finish and check result
    while dut.done.value == 0: await RisingEdge(dut.clk)
    expected_str = expected_result.toBinaryString(logBase=1).replace(".","")
    assert ((int(expected_str, base=2)-50) <= int(dut.computation_result.value) <= (int(expected_str, base=2)+50)), f"MAC output value is incorrect! Expected: {BinToDec(expected_result, num_Q_comp)}, got: {int(dut.computation_result.value)}"

#----- TESTS -----#
@cocotb.test()
async def random_test(dut):
    cocotb.start_soon(Clock(dut.clk, 1, units="ns").start())
    await RisingEdge(dut.clk)
    await reset(dut)

    for _ in range(NUM_TESTS):
        await test_MAC(dut, ActivationType.NO_ACTIVATION.value)

    for _ in range(NUM_TESTS):
        await test_MAC(dut, ActivationType.LINEAR_ACTIVATION.value)
