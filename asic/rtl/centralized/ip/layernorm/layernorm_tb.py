# Simple tests for the fixed-point LayerNorm module
import random
import pytest
import utilities
import Constants as const
from FixedPoint import FXnum

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge

#----- CONSTANTS -----#
LEN_FIRST_HALF = 64
LEN_SECOND_HALF = utilities.NUM_PATCHES + 1
MAX_LEN = 8
NUM_TESTS = 5
MAX_VAL = 4
TOLERANCE = 0.05

#----- FUNCTIONS -----#
async def test_run(dut, int_res_format:const.FxFormatIntRes, params_format:const.FxFormatParams, int_res_width:const.DataWidth):
    # Prepare MAC
    start_addr = random.randint(0, const.CIM_INT_RES_BANK_SIZE_NUM_WORD*const.CIM_INT_RES_NUM_BANKS - MAX_LEN - 1)
    dut.start_addr.value = start_addr
    dut.half_select.value = const.LayerNormHalfSelect.FIRST_HALF.value
    dut.int_res_read_format.value = const.int_res_fx_rtl_enum[int_res_format]
    dut.int_res_read_data_width.value = int_res_width.value
    dut.param_write_format.value = const.params_fx_rtl_enum[params_format]

    # Fill memory with random values and compute expected value
    mean = FXnum(0, const.num_Q_comp)
    mem_copy = []
    for i in range(LEN_FIRST_HALF):
        data = utilities.random_input(-MAX_VAL, MAX_VAL) # Bias towards positive values to avoid having very small mean and std_dev
        mem_copy.append(FXnum(data, const.num_Q_comp))
        await utilities.write_one_word_cent(dut=dut, addr=start_addr+i, data=data, device="int_res", data_format=int_res_format, data_width=int_res_width)
        mean += FXnum(data, const.num_Q_comp)
    mean /= FXnum(LEN_FIRST_HALF, const.num_Q_comp)
    
    variance = FXnum(0, const.num_Q_comp)
    temp = FXnum(0, const.num_Q_comp)
    for i in range(LEN_FIRST_HALF):
        temp = mem_copy[i] - mean
        variance += temp*temp
    variance /= FXnum(LEN_FIRST_HALF, const.num_Q_comp)
    std_dev = variance.sqrt()

    for i in range(LEN_FIRST_HALF): mem_copy[i] = (mem_copy[i] - mean) / std_dev

    dut.start.value = 1
    while not dut.done.value:
        await RisingEdge(dut.clk)
        dut.start.value = 0

    # Check results for 1st half
    for i in range(LEN_FIRST_HALF):
        received = await utilities.read_one_word_cent(dut=dut, addr=start_addr+i, device="int_res", data_format=int_res_format, data_width=int_res_width)
        expected_result = float(mem_copy[i])
        assert received == pytest.approx(expected_result, rel=TOLERANCE, abs=0.01), f"Expected: {expected_result}, received: {received}"

    print(f"LayerNorm 1st half passed!")
    for _ in range (1000): await cocotb.triggers.ClockCycles(dut.clk, 1)

    # 2nd half
    dut.half_select.value = const.LayerNormHalfSelect.SECOND_HALF.value
    beta = utilities.random_input(-MAX_VAL, MAX_VAL)
    gamma = utilities.random_input(-MAX_VAL, MAX_VAL)
    beta_addr = random.randint(MAX_LEN, const.CIM_PARAMS_BANK_SIZE_NUM_WORD*const.CIM_PARAMS_NUM_BANKS - MAX_LEN - 1)
    gamma_addr = random.randint(MAX_LEN, const.CIM_PARAMS_BANK_SIZE_NUM_WORD*const.CIM_PARAMS_NUM_BANKS - MAX_LEN - 1)
    dut.beta_addr.value = beta_addr
    dut.gamma_addr.value = gamma_addr
    await utilities.write_one_word_cent(dut=dut, addr=beta_addr, data=beta, device="params", data_format=params_format)
    await utilities.write_one_word_cent(dut=dut, addr=gamma_addr, data=gamma, device="params", data_format=params_format)

    for i in range(LEN_SECOND_HALF): mem_copy[i] = FXnum(gamma, const.num_Q_comp) * mem_copy[i] + FXnum(beta, const.num_Q_comp)

    dut.start.value = 1
    while not dut.done.value:
        await RisingEdge(dut.clk)
        dut.start.value = 0

    for i in range(LEN_SECOND_HALF):
        received = await utilities.read_one_word_cent(dut=dut, addr=start_addr+i, device="int_res", data_format=int_res_format, data_width=int_res_width)
        expected_result = float(mem_copy[i])
        assert received == pytest.approx(expected_result, rel=TOLERANCE, abs=0.01), f"Expected: {expected_result}, received: {received}"

    print(f"LayerNorm 2nd half passed!")
    await cocotb.triggers.ClockCycles(dut.clk, 10)

#----- TESTS -----#
@cocotb.test()
async def random_test(dut):
    cocotb.start_soon(Clock(dut.clk, 10, units="ns").start())
    await utilities.reset(dut)

    for params_format in const.FxFormatParams:
        for int_res_width in const.DataWidth:
            for _ in range(NUM_TESTS):
                if int_res_width == const.DataWidth.SINGLE_WIDTH:
                    int_res_format = const.FxFormatIntRes.INT_RES_SW_FX_5_X
                elif int_res_width == const.DataWidth.DOUBLE_WIDTH:
                    int_res_format = const.FxFormatIntRes.INT_RES_DW_FX

                await test_run(dut, int_res_format=int_res_format, params_format=params_format, int_res_width=int_res_width)
