# Testbench for the centralized compute CiM module
import os
import h5py
import cocotb
import random
import utilities
import Constants as const
from cocotb.triggers import RisingEdge
from cocotb.clock import Clock

# ----- CONSTANTS ----- #
CLK_FREQ_MHZ = 100
CLIP_INDEX = 0
home_dir = os.path.expanduser("~/../tmp")
eeg = h5py.File(f"{home_dir}/asic/fixed_point_accuracy_study/reference_data/eeg.h5", "r")
params_file = h5py.File(f"{home_dir}/asic/fixed_point_accuracy_study/reference_data/model_weights.h5", "r")

# ----- HELPERS ----- #
async def load_eeg(dut):
    dut.soc_ctrl_start_eeg_load.value = 1
    await RisingEdge(dut.clk)
    dut.soc_ctrl_start_eeg_load.value = 0
    await RisingEdge(dut.clk)

    for sample_index in range(utilities.NUM_PATCHES*utilities.PATCH_LEN):
        dut.soc_ctrl_new_eeg_data.value = 1
        dut.soc_ctrl_eeg.value = int(eeg["eeg"][CLIP_INDEX][sample_index])
        await RisingEdge(dut.clk)
        dut.soc_ctrl_new_eeg_data.value = 0
        for _ in range(3): await RisingEdge(dut.clk)

async def load_params(dut):
    # TODO: This has only been very partially tested for correctness
    for metadata in const.ParamMetadataKernels.values():
        weight_matrix = utilities.params(metadata.name, params_file)
        major_axis = metadata.y_len if metadata.index_type == "col-major" else metadata.x_len
        minor_axis = metadata.x_len if metadata.index_type == "col-major" else metadata.y_len

        for minor_axis_index in range(minor_axis):
            for major_axis_index in range(major_axis):
                addr = metadata.addr + major_axis_index + major_axis*minor_axis_index
                row = major_axis_index if metadata.index_type == "col-major" else minor_axis_index
                col = minor_axis_index if metadata.index_type == "col-major" else major_axis_index
                data = weight_matrix[row][col]
                await utilities.write_one_word_cent(dut, addr=addr, data=data, device="params", data_format=metadata.format, data_width=const.DataWidth.SINGLE_WIDTH)

async def print_progress(dut):
    current_step = 0
    while True:
        if const.InferenceStep(dut.current_inf_step.value) != current_step:
            current_step = const.InferenceStep(dut.current_inf_step.value)
            cocotb.log.info(f"Current step: {current_step}")
        for _ in range(5000): await RisingEdge(dut.clk)

# ----- TEST ----- #
@cocotb.test()
async def inference_tb(dut):
    cocotb.start_soon(Clock(dut.clk, 1/CLK_FREQ_MHZ, 'us').start())

    # Reset
    dut.soc_ctrl_rst_n.value = 0
    await RisingEdge(dut.clk)
    dut.soc_ctrl_rst_n.value = 1

    cocotb.start_soon(print_progress(dut))

    # Fill memory concurrently
    params_fill = cocotb.start_soon(load_params(dut))
    int_res_fill = cocotb.start_soon(utilities.fill_int_res_mem(dut))
    await params_fill
    await int_res_fill

    # Load EEG data as if it was a stream from the ADC
    await load_eeg(dut)

    # Inference
    dut.soc_ctrl_new_sleep_epoch.value = 1

    while not dut.soc_ctrl_inference_complete.value:
        await RisingEdge(dut.clk)
