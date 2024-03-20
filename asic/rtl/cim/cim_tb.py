# Simple tests for the fixed-point counter module
import sys
sys.path.append("..")
from utilities import *
from FixedPoint import FXnum

import h5py
import cocotb
import cocotb.triggers
from cocotb.clock import Clock

#----- EEG -----#
eeg = h5py.File("../../func_sim/reference_data/eeg.h5", "r")

#----- TESTS -----#
@cocotb.test()
async def basic_reset(dut):
    cocotb.start_soon(Clock(dut.clk, 1/ASIC_FREQUENCY_MHZ, units="us").start()) # 100MHz clock
    await reset(dut)
    await cocotb.triggers.ClockCycles(dut.clk, 100)

    # Simulate being the master and sending EEG patch data
    print(f"Although we should send a new EEG sample at every {int(1000000*ASIC_FREQUENCY_MHZ/SAMPLING_FREQ_HZ)} clock cycles, we will shorten that to {EEG_SAMPLING_PERIOD_CLOCK_CYCLE} clock cycles.")
    dut.bus_op_read.value = BusOp.PATCH_LOAD_BROADCAST_START_OP.value
    await RisingEdge(dut.clk)

    # Patch load test
    # Fill patch project bias parameters with random data
    patch_proj_kernel = []
    for i in range(PATCH_LEN):
        param = random_input()
        patch_proj_kernel.append(FXnum(param, num_Q_comp))
        dut.mem.params[i].value = BinToDec(param, num_Q_storage)
    
    bias = random_input()
    patch_proj_bias = FXnum(bias, num_Q_comp)
    dut.mem.params[param_addr_map["patch_proj_bias"]].value = BinToDec(bias, num_Q_storage)

    cnt = 0
    mac_result = 0
    eeg_vals = []
    for _ in range(int(CLIP_LENGTH_S*SAMPLING_FREQ_HZ)): # TODO: Remove the /100 once patch load and MAC is fully validated to load all data
        scaled_raw_eeg = send_eeg_from_master(dut, eeg)
        eeg_vals.append(FXnum(scaled_raw_eeg, num_Q_comp))
        await RisingEdge(dut.clk)
        dut.bus_op_read.value = BusOp.NOP.value
        await cocotb.triggers.ClockCycles(dut.clk, 3)
        if (dut.mac_start == 1):
            while (dut.mac_done == 0): await cocotb.triggers.RisingEdge(dut.clk)
            for i in range(PATCH_LEN): mac_result += patch_proj_kernel[i]*eeg_vals[i]
            mac_result += patch_proj_bias
            expected_str = mac_result.toBinaryString(logBase=1).replace(".","")
            assert ((int(expected_str, base=2)-50) <= int(dut.mac_out.value) <= (int(expected_str, base=2)+50)), f"MAC result during patch load doesn't match expected. Expected: {BinToDec(mac_result, num_Q_comp)}, got: {int(dut.mac_out.value)}"
            eeg_vals = []
            mac_result = 0
        
        cnt += 1

    await cocotb.triggers.ClockCycles(dut.clk, 1000)
