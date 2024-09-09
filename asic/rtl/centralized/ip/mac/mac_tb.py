# Simple tests for the fixed-point MAC module
import random
import pytest
import utilities
import Constants as const
from FixedPoint import FXnum

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge

#----- CONSTANTS -----#
MAX_LEN = 64
MAX_VAL_MAC = 4
NUM_TESTS = 3
TOLERANCE = 0.075

#----- FUNCTIONS -----#
async def test_MAC(dut, activation, param_type, int_res_width, int_res_format, param_format):
    # Prepare MAC
    expected_result = 0
    compute_temp_fp = 0
    start_addr_1 = random.randint(MAX_LEN, const.CIM_INT_RES_BANK_SIZE_NUM_WORD*const.CIM_INT_RES_NUM_BANKS - MAX_LEN - 1) # Starts at MAX_LEN to make it easier to correct for overlap
    start_addr_2 = random.randint(MAX_LEN, const.CIM_PARAMS_BANK_SIZE_NUM_WORD*const.CIM_PARAMS_NUM_BANKS - MAX_LEN - 1)

    # Fix address if there's an overlap
    if ((start_addr_1+MAX_LEN) > start_addr_2): start_addr_1 = start_addr_2 - MAX_LEN
    elif ((start_addr_2+MAX_LEN) > start_addr_1): start_addr_2 = start_addr_1 - MAX_LEN

    dut.start_addr_1.value = start_addr_1
    dut.start_addr_2.value = start_addr_2
    dut.len.value = MAX_LEN
    dut.param_type.value = param_type.value
    dut.activation.value = activation.value

    expected_result = 0
    for i in range(MAX_LEN):
        in_1 = utilities.random_input(-MAX_VAL_MAC, MAX_VAL_MAC)
        in_2 = utilities.random_input(-MAX_VAL_MAC, MAX_VAL_MAC)
        await utilities.write_one_word_cent(dut=dut, addr=start_addr_1+i, data=in_1, device="int_res", data_width=int_res_width, data_format=int_res_format)
        if (param_type == const.MACParamType.INTERMEDIATE_RES):
            await utilities.write_one_word_cent(dut=dut, addr=start_addr_2+i, data=in_2, device="int_res", data_width=int_res_width, data_format=int_res_format)
        elif (param_type == const.MACParamType.MODEL_PARAM):
            await utilities.write_one_word_cent(dut=dut, addr=start_addr_2+i, data=in_2, device="params", data_format=param_format, data_width=const.DataWidth.SINGLE_WIDTH)
        
        expected_result += FXnum(in_1, const.num_Q_comp)*FXnum(in_2, const.num_Q_comp)

    # Write bias
    if not activation == const.ActivationType.NO_ACTIVATION:
        bias = utilities.random_input(-MAX_VAL_MAC, MAX_VAL_MAC)
        bias_addr = random.randint(0, const.CIM_PARAMS_BANK_SIZE_NUM_WORD*const.CIM_PARAMS_NUM_BANKS - 1 - MAX_LEN)    
        await utilities.write_one_word_cent(dut=dut, addr=bias_addr, data=bias, device="params", data_format=param_format, data_width=const.DataWidth.SINGLE_WIDTH)
        dut.bias_addr.value = bias_addr

    if activation == const.ActivationType.LINEAR_ACTIVATION:
        expected_result += FXnum(bias, const.num_Q_comp)
    elif activation == const.ActivationType.SWISH_ACTIVATION:
        compute_temp_fp = expected_result + FXnum(bias, const.num_Q_comp)
        compute_temp_fp_neg = -compute_temp_fp
        expected_result += compute_temp_fp / (FXnum(1, const.num_Q_comp) + compute_temp_fp_neg.exp())

    await utilities.start_pulse(dut)

    # Wait for the MAC to finish and check result
    while dut.done.value == 0: await RisingEdge(dut.clk)
    received = utilities.twos_complement_to_float(dut.computation_result.value)
    expected_result = float(expected_result)
    assert received == pytest.approx(expected_result, rel=TOLERANCE), f"Expected: {expected_result}, received: {received}"

#----- TESTS -----#
@cocotb.test()
async def MAC_test_no_activation(dut):
    cocotb.start_soon(Clock(dut.clk, 10, units="ns").start())
    await utilities.reset(dut)
    for activation in const.ActivationType:
        for mac_param_type in const.MACParamType:
            for data_width in const.DataWidth:
                for int_res_format in [const.FxFormatIntRes.INT_RES_SW_FX_5_X, const.FxFormatIntRes.INT_RES_SW_FX_6_X]:
                    for param_format in [const.FxFormatParams.PARAMS_FX_4_X, const.FxFormatParams.PARAMS_FX_5_X]:
                        if (data_width == const.DataWidth.DOUBLE_WIDTH): int_res_format = const.FxFormatIntRes.INT_RES_DW_FX
                        for _ in range(NUM_TESTS): await test_MAC(dut,
                                                                  activation=activation,
                                                                  param_type=mac_param_type,
                                                                  int_res_width=data_width,
                                                                  int_res_format=int_res_format,
                                                                  param_format=param_format)
