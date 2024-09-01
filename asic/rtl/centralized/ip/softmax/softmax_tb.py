# Simple tests for the fixed-point softmax module
import random
import pytest
import utilities
import Constants as const
from FixedPoint import FXnum

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge

#----- CONSTANTS -----#
LEN = utilities.NUM_PATCHES+1
MAX_VAL = 4
NUM_TESTS = 5
TOLERANCE = 0.05

#----- FUNCTIONS -----#
async def test_run(dut, int_res_width, int_res_format):
    start_addr = random.randint(0, const.CIM_INT_RES_BANK_SIZE_NUM_WORD*const.CIM_INT_RES_NUM_BANKS - LEN - 1)
    dut.start_addr.value = start_addr
    dut.len.value = LEN

    mem_copy = []
    mem_copy_exp = []
    exp_sum = 0

    # Store value in memory and partially compute expected values
    for i in range(LEN):
        data = utilities.random_input(-MAX_VAL, MAX_VAL)
        data_q = FXnum(data, const.num_Q_comp)
        mem_copy.append(data_q)
        mem_copy_exp.append(data_q.exp())
        exp_sum += data_q.exp()
        await utilities.write_one_word_cent(dut=dut, addr=start_addr+i, data=data, device="int_res", data_width=int_res_width, data_format=int_res_format)

    # Start and wait
    dut.start.value = 1
    while (dut.done == 0):
        await RisingEdge(dut.clk)
        dut.start.value = 0

    # Check values against expected
    for i in range(LEN):
        expected = mem_copy_exp[i] / exp_sum
        received = await utilities.read_one_word_cent(dut=dut, addr=start_addr+i, device="int_res", data_width=int_res_width, data_format=int_res_format)
        expected_result = float(expected)
        assert received == pytest.approx(expected_result, rel=TOLERANCE, abs=0.02), f"Expected: {expected_result}, received: {received}"

    await RisingEdge(dut.clk)

#----- TESTS -----#
@cocotb.test()
async def random_test(dut):
    cocotb.start_soon(Clock(dut.clk, 10, units="ns").start())
    await utilities.reset(dut)

    for int_res_width in const.DataWidth:
        for _ in range(NUM_TESTS):
            if int_res_width == const.DataWidth.SINGLE_WIDTH:
                int_res_format = random.choice([const.FxFormatIntRes.INT_RES_SW_FX_5_X, const.FxFormatIntRes.INT_RES_SW_FX_6_X])
            elif int_res_width == const.DataWidth.DOUBLE_WIDTH:
                int_res_format = const.FxFormatIntRes.INT_RES_DW_FX
    
            await test_run(dut, int_res_width=int_res_width, int_res_format=int_res_format)
