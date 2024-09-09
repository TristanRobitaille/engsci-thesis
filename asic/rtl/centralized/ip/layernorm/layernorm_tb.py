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
LEN_FIRST_HALF = utilities.EMB_DEPTH
LEN_SECOND_HALF = utilities.NUM_PATCHES + 1
MAX_LEN = utilities.EMB_DEPTH
NUM_TESTS = 3 
MAX_VAL = 4
TOLERANCE = 0.01

#----- FUNCTIONS -----#
async def fill_int_res_mem(dut, data_format=const.FxFormatIntRes, data_width=const.DataWidth):
    for addr in range(const.CIM_INT_RES_NUM_BANKS * const.CIM_INT_RES_BANK_SIZE_NUM_WORD):
        data = random.uniform(-2**(data_format.value-1)+1, 2**(data_format.value-1)-1)
        await utilities.write_one_word_cent(dut, addr=addr, data=data, device="int_res", data_format=data_format, data_width=data_width)

async def test_run(dut, int_res_format:const.FxFormatIntRes, params_format:const.FxFormatParams, int_res_width:const.DataWidth):
    # Prepare MAC
    start_addr = random.randint(0, const.CIM_INT_RES_BANK_SIZE_NUM_WORD*const.CIM_INT_RES_NUM_BANKS - MAX_LEN - 1)
    output_addr = random.randint(0, const.CIM_INT_RES_BANK_SIZE_NUM_WORD*const.CIM_INT_RES_NUM_BANKS - MAX_LEN - 1)

    # Fix address if there's an overlap
    if ((start_addr + MAX_LEN) > output_addr): start_addr = output_addr - MAX_LEN
    elif ((output_addr + MAX_LEN) > start_addr): output_addr = start_addr - MAX_LEN

    dut.start_addr.value = start_addr
    dut.output_addr.value = output_addr
    dut.half_select.value = const.LayerNormHalfSelect.FIRST_HALF.value
    dut.int_res_read_format.value = const.int_res_fx_rtl_enum[int_res_format]
    dut.int_res_read_data_width.value = int_res_width.value
    dut.param_write_format.value = const.params_fx_rtl_enum[params_format]

    # Fill memory with random values and compute expected value
    mean = FXnum(0, const.num_Q_comp)
    mem_copy = []
    await fill_int_res_mem(dut, data_format=int_res_format, data_width=int_res_width)
    for i in range(LEN_FIRST_HALF):
        data = await utilities.read_one_word_cent(dut=dut, addr=start_addr+i, device="int_res", data_format=int_res_format, data_width=int_res_width)
        mem_copy.append(data)
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
        received = await utilities.read_one_word_cent(dut=dut, addr=output_addr+i, device="int_res", data_format=int_res_format, data_width=int_res_width)
        expected_result = float(mem_copy[i])
        assert received == pytest.approx(expected_result, rel=TOLERANCE, abs=0.01), f"Expected: {expected_result}, received: {received} at index {i}"

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
    await utilities.write_one_word_cent(dut=dut, addr=beta_addr, data=beta, device="params", data_format=params_format, data_width=const.DataWidth.SINGLE_WIDTH)
    await utilities.write_one_word_cent(dut=dut, addr=gamma_addr, data=gamma, device="params", data_format=params_format, data_width=const.DataWidth.SINGLE_WIDTH)

    for i in range(LEN_SECOND_HALF):
        data = await utilities.read_one_word_cent(dut=dut, addr=output_addr+i*utilities.EMB_DEPTH, device="int_res", data_format=int_res_format, data_width=int_res_width)
        mem_copy[i] = FXnum(gamma, const.num_Q_comp) * data + FXnum(beta, const.num_Q_comp)

    dut.start.value = 1
    while not dut.done.value:
        await RisingEdge(dut.clk)
        dut.start.value = 0

    for i in range(LEN_SECOND_HALF):
        received = await utilities.read_one_word_cent(dut=dut, addr=output_addr+i*utilities.EMB_DEPTH, device="int_res", data_format=int_res_format, data_width=int_res_width)
        expected_result = float(mem_copy[i])
        assert received == pytest.approx(expected_result, rel=TOLERANCE, abs=0.01), f"Expected: {expected_result}, received: {received} at index {i}"

    print(f"LayerNorm 2nd half passed!")
    await cocotb.triggers.ClockCycles(dut.clk, 10)

#----- TESTS -----#
@cocotb.test()
async def random_test(dut):
    cocotb.start_soon(Clock(dut.clk, 10, units="ns").start())
    await utilities.reset(dut)
    await test_run(dut, int_res_format=const.FxFormatIntRes.INT_RES_SW_FX_5_X, params_format=const.FxFormatParams.PARAMS_FX_3_X, int_res_width=const.DataWidth.SINGLE_WIDTH)
