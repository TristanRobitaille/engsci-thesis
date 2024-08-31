import cocotb
import random
from FixedPoint import FXfamily, FXnum
from cocotb.triggers import RisingEdge
from cocotb.clock import Clock

import utilities
import Constants as const

# ----- CONSTANTS ----- #
CLK_FREQ_MHZ = 100
NUM_WRITES = 1000

# ----- FUNCTIONS ----- #
async def test_params(dut):
    for _ in range(NUM_WRITES):
        data_format = random.choice(list(const.FxFormatParams))
        addr = random.randint(0, const.CIM_PARAMS_NUM_BANKS * const.CIM_PARAMS_BANK_SIZE_NUM_WORD-1)
        data_in = random.uniform(0, 2**(data_format.value-1)-1) # Currently only positive values are supported in testbench, but negative values are supported in RTL
        await utilities.write_one_word_cent(dut, addr=addr, data=data_in, device="params", data_width=const.DataWidth.SINGLE_WIDTH, data_format=data_format)

        # Check if the data was written correctly
        dut.param_read_en.value = 1
        dut.param_read_addr.value = addr
        dut.param_read_format.value = const.params_fx_rtl_enum[data_format]
        for _ in range(2): await RisingEdge(dut.clk)
        received = (dut.param_read_data.value >> (const.Q_COMP - (const.N_STO_PARAMS-data_format.value))) / (2**(const.N_STO_PARAMS-data_format.value))
        expected = FXnum(data_in, FXfamily(const.N_STO_PARAMS-data_format.value, data_format.value))
        assert (received == expected), f"Expected {expected}, got {received}"
        dut.param_read_en.value = 0

async def test_int_res(dut):
    for width in [const.DataWidth.SINGLE_WIDTH, const.DataWidth.DOUBLE_WIDTH]:
        for _ in range(NUM_WRITES):
            addr = random.randint(0, const.CIM_INT_RES_NUM_BANKS * const.CIM_INT_RES_BANK_SIZE_NUM_WORD-1)
            # Currently only positive values are supported in testbench, but negative values are supported in RTL
            if (width == const.DataWidth.SINGLE_WIDTH):
                data_format = random.choice([const.FxFormatIntRes.INT_RES_SW_FX_1_X, const.FxFormatIntRes.INT_RES_SW_FX_2_X, const.FxFormatIntRes.INT_RES_SW_FX_5_X, const.FxFormatIntRes.INT_RES_SW_FX_6_X])
                data_in = random.uniform(0, 2**(data_format.value-1)-1)
            elif (width == const.DataWidth.DOUBLE_WIDTH):
                data_format = const.FxFormatIntRes.INT_RES_DW_FX
                data_in = random.uniform(0, 2**(data_format.value-1)-1)

            await utilities.write_one_word_cent(dut, addr=addr, data=data_in, device="int_res", data_width=width, data_format=data_format)

            # Check if the data was written correctly
            dut.int_res_read_en.value = 1
            dut.int_res_read_data_width.value = width.value
            dut.int_res_read_addr.value = addr
            dut.int_res_read_format.value = const.int_res_fx_rtl_enum[data_format]
            for _ in range(2): await RisingEdge(dut.clk)
            if (width == const.DataWidth.SINGLE_WIDTH):
                received = (dut.int_res_read_data.value >> (const.Q_COMP - (const.N_STO_INT_RES-data_format.value))) / (2**(const.N_STO_INT_RES-data_format.value))
            elif (width == const.DataWidth.DOUBLE_WIDTH):
                received = (dut.int_res_read_data.value >> (const.Q_COMP - (2*const.N_STO_INT_RES-data_format.value))) / (2**(2*const.N_STO_INT_RES-data_format.value))
            expected = FXnum(data_in, FXfamily(const.N_STO_INT_RES-data_format.value, data_format.value))
            assert (received == expected), f"Expected {expected}, got {received}"
            
            dut.int_res_read_en.value = 0

# ----- TEST ----- #
@cocotb.test()
async def mem_tb(dut):
    cocotb.start_soon(Clock(dut.clk, 1/CLK_FREQ_MHZ, 'us').start())

    await utilities.reset(dut)
    await test_params(dut)
    await test_int_res(dut)
