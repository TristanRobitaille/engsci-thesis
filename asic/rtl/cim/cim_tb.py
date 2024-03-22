# Simple tests for the fixed-point counter module
import sys
sys.path.append("..")
from utilities import *
from FixedPoint import FXnum

import h5py
import math
import cocotb
import cocotb.triggers
from cocotb.clock import Clock

#----- HELPERS -----#
def write_full_bus(dut, op:Enum, target_or_sender, data:list):
    dut.bus_op_read.value = op.value
    dut.bus_target_or_sender_read.value = target_or_sender
    write_to_data_bus_multi(dut, data)

def write_to_data_bus_multi(dut, data:list):
    # Overwrite all three words
    dut.bus_data_read.value = data[0] | (data[1] << NUM_INT_BITS_STORAGE+NUM_FRACT_BITS) | (data[2] << 2*(NUM_INT_BITS_STORAGE+NUM_FRACT_BITS))

def write_to_data_bus(dut, word_num:int, data):
    # Calling this function more than once per clock cycle will overwrite the previous data. One one byte may be changed per clock cycle.
    d2 = data if word_num == 2 else dut.bus_data_read.value[0:15]
    d1 = data if word_num == 1 else dut.bus_data_read.value[16:31]
    d0 = data if word_num == 0 else dut.bus_data_read.value[32:47]
    dut.bus_data_read.value = d0 | (d1 << NUM_INT_BITS_STORAGE+NUM_FRACT_BITS) | (d2 << 2*(NUM_INT_BITS_STORAGE+NUM_FRACT_BITS))

#----- CONSTANTS -----#
INTERDATA_DELAY = 5

#----- EEG -----#
eeg = h5py.File("../../func_sim/reference_data/eeg.h5", "r")

async def param_load(dut, cim_id:int):
    # Load parameters (only sending for CiM #0 and #1, to see whether CiM #0 ignores it)
    for param_info in param_addr_map.values():
        write_full_bus(dut, op=BusOp.PARAM_STREAM_START_OP, target_or_sender=cim_id, data=[param_info["start_addr"], param_info["len"], 0]) #[start_addr, len, N/A]
        await RisingEdge(dut.clk)
        dut.bus_op_read.value = BusOp.NOP.value
        await cocotb.triggers.ClockCycles(dut.clk, INTERDATA_DELAY-1)

        for i in range(math.ceil(param_info["len"]/3)):
            for j in range(3): # 3 parameters per transaction
                dut.bus_op_read.value = BusOp.PARAM_STREAM_OP.value
                param = random_input(-MAX_VAL, MAX_VAL)
                write_to_data_bus(dut, j, BinToDec(param, num_Q_storage))
                await RisingEdge(dut.clk)
                dut.bus_op_read.value = BusOp.NOP.value
                await cocotb.triggers.ClockCycles(dut.clk, INTERDATA_DELAY-1)

async def patch_load(dut):
    print(f"Although we should send a new EEG sample at every {int(1000000*ASIC_FREQUENCY_MHZ/SAMPLING_FREQ_HZ)} clock cycles, we will shorten that to {EEG_SAMPLING_PERIOD_CLOCK_CYCLE} clock cycles.")
    dut.bus_op_read.value = BusOp.PATCH_LOAD_BROADCAST_START_OP.value
    await RisingEdge(dut.clk)

    # Fill patch project bias parameters with random data
    patch_proj_kernel = []
    for i in range(PATCH_LEN):
        param = random_input(-MAX_VAL, MAX_VAL)
        patch_proj_kernel.append(FXnum(param, num_Q_comp))
        dut.mem.params[i].value = BinToDec(param, num_Q_storage)
    
    bias = random_input(-MAX_VAL, MAX_VAL)
    patch_proj_bias = FXnum(bias, num_Q_comp)
    dut.mem.params[param_addr_map["single_params"]["start_addr"]+0].value = BinToDec(bias, num_Q_storage) # Patch projection bias has zero offset from single_params start address

    cnt = 0
    mac_result = 0
    eeg_vals = []
    for _ in range(int(CLIP_LENGTH_S*SAMPLING_FREQ_HZ)):
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

#----- TESTS -----#
@cocotb.test()
async def basic_reset(dut):
    cocotb.start_soon(Clock(dut.clk, 1/ASIC_FREQUENCY_MHZ, units="us").start()) # 100MHz clock
    await reset(dut)
    await cocotb.triggers.ClockCycles(dut.clk, 100)

    # Random parameters load
    await param_load(dut, 0) # Load parameters on CiM #0
    await param_load(dut, 31) # Load parameters on CiM #31 (ensure CiM #0 ignores the data)
    await cocotb.triggers.ClockCycles(dut.clk, INTERLUDE_CLOCK_CYCLES)

    # Simulate being the master and sending EEG patch data (note that this overwrites the parameters loaded above)
    await patch_load(dut)

    await cocotb.triggers.ClockCycles(dut.clk, INTERLUDE_CLOCK_CYCLES)
