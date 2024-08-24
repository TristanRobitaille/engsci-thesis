import cocotb
import random
from cocotb.triggers import RisingEdge
from cocotb.clock import Clock

import Constants as const

# ----- CONSTANTS ----- #
CLK_FREQ_MHZ = 100
NUM_WRITES = 1000

# ----- HELPERS ----- #
async def reset(dut):
    dut.rst_n.value = 0
    await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)

async def write_one_word(dut, addr:int, data:int, device:str, width:int):
    if device == "params":
        assert (width == 0), "Params memory only compatible with SINGLE_WIDTH!"
        dut.param_chip_en.value = 1
        dut.param_write_en.value = 1
        dut.param_write_data_width.value = width
        dut.param_write_addr.value = addr
        dut.param_write_data.value = data
        await RisingEdge(dut.clk)
        dut.param_write_en.value = 0
    elif device == "int_res":
        dut.int_res_chip_en.value = 1
        dut.int_res_write_en.value = 1
        dut.int_res_write_data_width.value = width
        dut.int_res_write_addr.value = addr
        dut.int_res_write_data.value = data
        await RisingEdge(dut.clk)
        dut.int_res_write_en.value = 0

async def test_params(dut):
    for _ in range(NUM_WRITES):
        addr = random.randint(0, const.CIM_PARAMS_NUM_BANKS * const.CIM_PARAMS_BANK_SIZE_NUM_WORD)
        data = random.randint(0, 2**const.N_STO_PARAMS-1)
        await write_one_word(dut, addr=addr, data=data, device="params", width=const.DataWidth.SINGLE_WIDTH.value)

        # Check if the data was written correctly
        dut.param_read_en.value = 1
        dut.param_read_addr.value = addr
        for _ in range(2): await RisingEdge(dut.clk)
        assert (dut.param_read_data.value == data), f"Expected {data}, got {dut.param_read_data.value}"
        dut.param_read_en.value = 0

async def test_int_res(dut):
    for width in [const.DataWidth.SINGLE_WIDTH, const.DataWidth.DOUBLE_WIDTH]:
        for _ in range(NUM_WRITES):
            addr = random.randint(0, const.CIM_INT_RES_NUM_BANKS * const.CIM_INT_RES_BANK_SIZE_NUM_WORD)
            if (width == const.DataWidth.SINGLE_WIDTH): data = random.randint(0, 2**const.N_STO_INT_RES-1)
            elif (width == const.DataWidth.DOUBLE_WIDTH): data = random.randint(0, 2**(2*const.N_STO_INT_RES)-1)

            await write_one_word(dut, addr=addr, data=data, device="int_res", width=width.value)

            # Check if the data was written correctly
            dut.int_res_read_en.value = 1
            dut.int_res_read_data_width.value = width.value
            dut.int_res_read_addr.value = addr
            for _ in range(2): await RisingEdge(dut.clk)
            assert (dut.int_res_read_data.value == data), f"Expected {data}, got {dut.int_res_read_data.value}"
            dut.int_res_read_en.value = 0

# ----- TEST ----- #
@cocotb.test()
async def mem_tb(dut):
    cocotb.start_soon(Clock(dut.clk, 1/CLK_FREQ_MHZ, 'us').start())

    await reset(dut)
    await test_params(dut)
    await test_int_res(dut)
