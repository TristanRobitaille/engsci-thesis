# Simple tests for the fixed-point counter module
import sys
sys.path.append("..")
from utilities import *

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

    for _ in range(int(CLIP_LENGTH_S*SAMPLING_FREQ_HZ)): # TODO: Remove the /100 once patch load and MAC is fully validated to load all data
        eeg_index = send_eeg_from_master(dut, eeg)
        await RisingEdge(dut.clk)
        dut.bus_op_read.value = BusOp.NOP.value
        # Since the patch MAC takes just under EEG_SAMPLING_PERIOD_CLOCK_CYCLE clock cycles to complete, we need to wait for the next clock cycle to send the next EEG sample
        # but to save simulation time, we will wait for EEG_SAMPLING_PERIOD_CLOCK_CYCLE clock cycles only if we are at the end of a patch
        if (eeg_index%PATCH_LEN == 0): await cocotb.triggers.ClockCycles(dut.clk, EEG_SAMPLING_PERIOD_CLOCK_CYCLE-1)
        else:  await cocotb.triggers.ClockCycles(dut.clk, 5)

    await cocotb.triggers.ClockCycles(dut.clk, 1000)
