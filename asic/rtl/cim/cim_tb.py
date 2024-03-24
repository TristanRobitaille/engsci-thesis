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

params_loaded = dict()

#----- HELPERS -----#
def write_to_data_bus_multi(dut, data:list):
    # Overwrite all three words
    dut.bus_data_read.value = data[0] | (data[1] << NUM_INT_BITS_STORAGE+NUM_FRACT_BITS) | (data[2] << 2*(NUM_INT_BITS_STORAGE+NUM_FRACT_BITS))

def write_to_data_bus(dut, word_num:int, data):
    # Calling this function more than once per clock cycle will overwrite the previous data. One one byte may be changed per clock cycle.
    d2 = data if word_num == 2 else dut.bus_data_read.value[0:15]
    d1 = data if word_num == 1 else dut.bus_data_read.value[16:31]
    d0 = data if word_num == 0 else dut.bus_data_read.value[32:47]
    dut.bus_data_read.value = d0 | (d1 << NUM_INT_BITS_STORAGE+NUM_FRACT_BITS) | (d2 << 2*(NUM_INT_BITS_STORAGE+NUM_FRACT_BITS))

def write_linear_words_to_int_res(dut, len):
    tx_addr = random.randint(len, TEMP_RES_STORAGE_SIZE_CIM-2-len)
    rx_addr = tx_addr = random.randint(len, TEMP_RES_STORAGE_SIZE_CIM-2-len)
    if (tx_addr - rx_addr < len): rx_addr = tx_addr - len
    elif (rx_addr - tx_addr < len): tx_addr = rx_addr - len
    data_in_mem = []
    for i in range(len):
        dut.mem.int_res[tx_addr+i].value = i+1 # Fill memory with fake data to send
        data_in_mem.append(i+1)

    return tx_addr, rx_addr, data_in_mem

#----- CONSTANTS -----#
INTERDATA_DELAY = 5
INTERTRANSPOSE_TRANSACTIONS_DELAY = 7
INTERSTEP_CLOCK_CYCLES = 10

#----- EEG -----#
eeg = h5py.File("../../func_sim/reference_data/eeg.h5", "r")

#----- TEST FUNCTIONS -----#
async def write_full_bus(dut, op:Enum, target_or_sender, data:list):
    dut.bus_op_read.value = op.value
    dut.bus_target_or_sender_read.value = target_or_sender
    write_to_data_bus_multi(dut, data)
    await cocotb.triggers.RisingEdge(dut.clk)
    dut.bus_op_read.value = 0
    dut.bus_target_or_sender_read.value = 0
    dut.bus_data_read.value = 0

async def full_transpose_broadcast_emulation(dut, tx_addr, rx_addr, data_len, num_cim):
    for i in range(num_cim):
        await write_full_bus(dut, op=BusOp.TRANS_BROADCAST_START_OP, target_or_sender=i, data=[tx_addr, data_len, rx_addr])
        if (i == 0): # CiM #0 is instantiated so it will actually send data so the testbench shouldn't do it
            await cocotb.triggers.RisingEdge(dut.clk)
            while (dut.is_ready == 0): await cocotb.triggers.RisingEdge(dut.clk)
        else:
            for j in range(data_len):
                await cocotb.triggers.ClockCycles(dut.clk, INTERTRANSPOSE_TRANSACTIONS_DELAY)
                word0 = BinToDec(random_input(-MAX_VAL, MAX_VAL), num_Q_storage)
                word1 = BinToDec(random_input(-MAX_VAL, MAX_VAL), num_Q_storage)
                word2 = BinToDec(random_input(-MAX_VAL, MAX_VAL), num_Q_storage)
                await write_full_bus(dut, op=BusOp.TRANS_BROADCAST_DATA_OP, target_or_sender=i, data=[word0, word1, word2])
        await cocotb.triggers.ClockCycles(dut.clk, 50)
            
async def trigger_transpose_broadcast(dut):
    data_len = [6, 1, 60, 61, EMB_DEPTH, 32]
    for len in data_len:
        (tx_addr, rx_addr, data_in_mem) = write_linear_words_to_int_res(dut, len)
        await write_full_bus(dut, op=BusOp.TRANS_BROADCAST_START_OP, target_or_sender=0, data=[tx_addr, len, rx_addr]) # tx_addr, data_len, rx_addr
        await cocotb.triggers.RisingEdge(dut.clk)
        #TODO: Check that data being sent is correct
        #TODO: On a CiM other than #0, check that data saved is correct
        while (dut.is_ready == 0): await cocotb.triggers.RisingEdge(dut.clk)
        await cocotb.triggers.ClockCycles(dut.clk, 1000)

async def trigger_dense_broadcast(dut):
    data_len = [EMB_DEPTH, NUM_HEADS, MLP_DIM, NUM_PATCHES+1]
    for len in data_len:
        (tx_addr, rx_addr, data_in_mem) = write_linear_words_to_int_res(dut, len)
        await write_full_bus(dut, op=BusOp.DENSE_BROADCAST_START_OP, target_or_sender=0, data=[tx_addr, len, rx_addr]) # tx_addr, data_len, rx_addr
        await cocotb.triggers.RisingEdge(dut.clk)
        while (dut.is_ready == 0): await cocotb.triggers.RisingEdge(dut.clk)
        await cocotb.triggers.ClockCycles(dut.clk, 2)
        #TODO: On a CiM other than #0, check that data saved is correct
        # Verify that data was moved correctly (here, we are just checking that the CiM that sends the data also moves its own data correctly)
        for i in range(len): assert (dut.mem.int_res[rx_addr+i].value == data_in_mem[i]), f"Data mismatch following dense broadcast. Expected: {data_in_mem[i]} at addr: {rx_addr+i}, got: {dut.mem.int_res[rx_addr+i].value}"
        await cocotb.triggers.ClockCycles(dut.clk, 1000)

async def param_load(dut, cim_id:int):
    global params_loaded
    for param_info in param_addr_map.values():
        await write_full_bus(dut, op=BusOp.PARAM_STREAM_START_OP, target_or_sender=cim_id, data=[param_info["start_addr"], param_info["len"], 0]) #[start_addr, len, N/A]
        await cocotb.triggers.ClockCycles(dut.clk, INTERDATA_DELAY-1)

        for i in range(math.ceil(param_info["len"]/3)):
            for j in range(3): # 3 parameters per transaction
                param_addr = param_info["start_addr"]+3*i+j
                param = random_input(-MAX_VAL, MAX_VAL)
                if (cim_id == 0): params_loaded[param_addr] = param
                #Send instruction to load parameter
                dut.bus_op_read.value = BusOp.PARAM_STREAM_OP.value
                dut.bus_target_or_sender_read.value = cim_id
                write_to_data_bus(dut, j, BinToDec(param, num_Q_storage))
                await RisingEdge(dut.clk)
                dut.bus_op_read.value = BusOp.NOP.value
                await cocotb.triggers.ClockCycles(dut.clk, 3)
                # Check parameter loaded in memory
                if (param_addr < 528):
                    if (cim_id == 0): assert (dut.mem.params[param_addr].value == BinToDec(param, num_Q_storage)), f"Parameter load failed. Expected: {BinToDec(param, num_Q_storage)}, got: {dut.mem.params[3*i+j].value}"
                    else: assert (dut.mem.params[param_addr].value == BinToDec(params_loaded[param_addr], num_Q_storage)), f"Parameter load failed. Expected: {BinToDec(params_loaded[param_addr], num_Q_storage)}, got: {dut.mem.params[3*i+j].value}" # If we sent params to another CiM, CiM should not update its params memory
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
async def basic_test(dut):
    cocotb.start_soon(Clock(dut.clk, 1/ASIC_FREQUENCY_MHZ, units="us").start()) # 100MHz clock
    await cocotb.start(bus_mirror(dut))
    await reset(dut)
    await cocotb.triggers.ClockCycles(dut.clk, 100)

    # Random parameters load
    await param_load(dut, 0) # Load parameters on CiM #0
    await param_load(dut, 31) # Load parameters on CiM #31 (ensure CiM #0 ignores the data)
    await cocotb.triggers.ClockCycles(dut.clk, INTERLUDE_CLOCK_CYCLES)
    print("Params load done.")

    # Simulate being the master and sending EEG patch data (note that this overwrites the parameters loaded above)
    await patch_load(dut)
    await cocotb.triggers.ClockCycles(dut.clk, INTERLUDE_CLOCK_CYCLES)

    # Test tranpose broadcast and dense broadcast
    # trigger_transpose_broadcast(dut)
    # await cocotb.triggers.ClockCycles(dut.clk, INTERLUDE_CLOCK_CYCLES)
    # trigger_dense_broadcast(dut)

    # Go through inference steps
    await cocotb.triggers.ClockCycles(dut.clk, INTERLUDE_CLOCK_CYCLES)

    await full_transpose_broadcast_emulation(dut, inf_steps[0].tx_addr, inf_steps[0].rx_addr, inf_steps[0].len, inf_steps[0].num_cim) # Pre-LayerNorm #1
    while (dut.is_ready == 0): await cocotb.triggers.RisingEdge(dut.clk)
    await write_full_bus(dut, op=BusOp.PISTOL_START_OP, target_or_sender=0, data=[0, 0, 0])

    await cocotb.triggers.ClockCycles(dut.clk, INTERSTEP_CLOCK_CYCLES)

    await full_transpose_broadcast_emulation(dut, inf_steps[1].tx_addr, inf_steps[1].rx_addr, inf_steps[1].len, inf_steps[1].num_cim) # Intra-LayerNorm #1
    while (dut.is_ready == 0): await cocotb.triggers.RisingEdge(dut.clk)
    await write_full_bus(dut, op=BusOp.PISTOL_START_OP, target_or_sender=0, data=[0, 0, 0])

    await cocotb.triggers.ClockCycles(dut.clk, INTERSTEP_CLOCK_CYCLES)

    await full_transpose_broadcast_emulation(dut, inf_steps[2].tx_addr, inf_steps[2].rx_addr, inf_steps[2].len, inf_steps[2].num_cim) # Post-LayerNorm #1
    await write_full_bus(dut, op=BusOp.PISTOL_START_OP, target_or_sender=0, data=[0, 0, 0])

    await cocotb.triggers.ClockCycles(dut.clk, 100)
