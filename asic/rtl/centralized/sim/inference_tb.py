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
async def load_eeg(dut):
    dut.soc_ctrl_start_eeg_load.value = 1
    await RisingEdge(dut.clk)
    dut.soc_ctrl_start_eeg_load.value = 0

    for _ in range(utilities.NUM_PATCHES*utilities.PATCH_LEN):
        await RisingEdge(dut.clk)
        dut.soc_ctrl_new_eeg_data.value = 1
        dut.soc_ctrl_eeg.value = random.randint(0, 2**16-1)
        await RisingEdge(dut.clk)
        dut.soc_ctrl_new_eeg_data.value = 0
        for _ in range(3): await RisingEdge(dut.clk)

# ----- TEST ----- #
@cocotb.test()
async def inference_tb(dut):
    cocotb.start_soon(Clock(dut.clk, 1/CLK_FREQ_MHZ, 'us').start())
    
    # Reset
    dut.soc_ctrl_rst_n.value = 0
    await RisingEdge(dut.clk)
    dut.soc_ctrl_rst_n.value = 1

    # Fill memory concurrently
    params_fill = cocotb.start_soon(utilities.fill_params_mem(dut))
    int_res_fill = cocotb.start_soon(utilities.fill_int_res_mem(dut))
    await params_fill
    await int_res_fill

    # Load EEG data as if it was a stream from the ADC
    await load_eeg(dut)

    # Inference
    dut.soc_ctrl_new_sleep_epoch.value = 1

    while not dut.soc_ctrl_inference_complete.value:
        await RisingEdge(dut.clk)
