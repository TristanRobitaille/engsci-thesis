import cocotb
import random
from cocotb.triggers import RisingEdge
from cocotb.clock import Clock

import Constants

# ----- CONSTANTS ----- #
CLK_FREQ_MHZ = 100

# ----- HELPERS ----- #
async def write_one_word(dut, addr:int, data:int, device:str, width:int):
    if device == "params":
        assert (width == 0), "Params memory only compatible with SINGLE_WIDTH!"
        dut.cim_centralized.param_write.chip_en.value = 1
        dut.cim_centralized.param_write.en.value = 1
        dut.cim_centralized.param_write.data_width.value = width
        dut.cim_centralized.param_write.addr.value = addr
        dut.cim_centralized.param_write.data.value = data
        await RisingEdge(dut.clk)
        dut.cim_centralized.param_write.en.value = 0
    elif device == "int_res":
        dut.cim_centralized.int_res_write.chip_en.value = 1
        dut.cim_centralized.int_res_write.en.value = 1
        dut.cim_centralized.int_res_write.data_width.value = width
        dut.cim_centralized.int_res_write.addr.value = addr
        dut.cim_centralized.int_res_write.data.value = data
        await RisingEdge(dut.clk)
        dut.cim_centralized.int_res_write.en.value = 0

async def fill_params_mem(dut):
    for addr in range(Constants.CIM_PARAMS_NUM_BANKS * Constants.CIM_PARAMS_BANK_SIZE_NUM_WORD):
        data = random.randint(0, 2**Constants.N_STO_PARAMS-1)
        await write_one_word(dut, addr=addr, data=data, device="params", width=Constants.DataWidth.SINGLE_WIDTH.value)

async def fill_int_res_mem(dut):
    for addr in range(Constants.CIM_INT_RES_NUM_BANKS * Constants.CIM_INT_RES_BANK_SIZE_NUM_WORD):
        data = random.randint(0, 2**Constants.N_STO_INT_RES-1)
        await write_one_word(dut, addr=addr, data=data, device="int_res", width=Constants.DataWidth.SINGLE_WIDTH.value)

# ----- TEST ----- #
@cocotb.test()
async def inference_tb(dut):
    cocotb.start_soon(Clock(dut.clk, 1/CLK_FREQ_MHZ, 'us').start())
    
    # Reset
    dut.soc_ctrl_rst_n.value = 0
    await RisingEdge(dut.clk)
    dut.soc_ctrl_rst_n.value = 1

    # Fill memory
    await fill_params_mem(dut)
    await fill_int_res_mem(dut)

    # Inference
    dut.soc_ctrl_new_sleep_epoch.value = 1
