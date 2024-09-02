# Testbench for the centralized compute CiM module

import cocotb
import random
import utilities
import Constants as const
from cocotb.triggers import RisingEdge
from cocotb.clock import Clock

# ----- CONSTANTS ----- #
CLK_FREQ_MHZ = 100

# ----- HELPERS ----- #
async def fill_params_mem(dut):
    for addr in range(const.CIM_PARAMS_NUM_BANKS * const.CIM_PARAMS_BANK_SIZE_NUM_WORD):
        data_format = const.FxFormatParams.PARAMS_FX_4_X # Default for all params
        data = random.uniform(-2**(data_format.value-1)+1, 2**(data_format.value-1)-1)
        await utilities.write_one_word_cent(dut, addr=addr, data=data, device="params", data_format=data_format, data_width=const.DataWidth.SINGLE_WIDTH)

async def fill_int_res_mem(dut):
    for addr in range(const.CIM_INT_RES_NUM_BANKS * const.CIM_INT_RES_BANK_SIZE_NUM_WORD):
        data_format = const.FxFormatIntRes.INT_RES_SW_FX_5_X # Default for all int res
        data_width = const.DataWidth.SINGLE_WIDTH # Default for all int res
        data = random.uniform(-2**(data_format.value-1)+1, 2**(data_format.value-1)-1)
        await utilities.write_one_word_cent(dut, addr=addr, data=data, device="int_res", data_format=data_format, data_width=data_width)

# ----- TEST ----- #
@cocotb.test()
async def inference_tb(dut):
    cocotb.start_soon(Clock(dut.clk, 1/CLK_FREQ_MHZ, 'us').start())
    
    # Reset
    dut.soc_ctrl_rst_n.value = 0
    await RisingEdge(dut.clk)
    dut.soc_ctrl_rst_n.value = 1

    # Fill memory concurrently
    params_task = cocotb.start_soon(fill_params_mem(dut))
    int_res_task = cocotb.start_soon(fill_int_res_mem(dut))

    # Wait for both tasks to complete
    await params_task
    await int_res_task

    # Inference
    dut.soc_ctrl_new_sleep_epoch.value = 1

    while not dut.soc_ctrl_inference_complete.value:
        await RisingEdge(dut.clk)
