# Simple tests for the fixed-point counter module
import h5py
import math
import sys
sys.path.append("..")
from utilities import *
from FixedPoint import FXnum

import cocotb
import cocotb.triggers
from cocotb.clock import Clock

#----- EXTERNAL MEMORY -----#
params = h5py.File("../../func_sim/reference_data/model_params.h5", "r")
READ_LATENCY = 3 # Number of clock cycles to read from external memory
ext_mem_latency_cnt = READ_LATENCY+1
prev_ext_mem_read_pulse = 0
param_map = { # Starting row of each layer's parameter in external memory. Each row is 64 elements wide.
    "patch_proj_kernel":                0,
    "pos_emb":                          64,
    "enc_Q_dense_kernel":               125,
    "enc_K_dense_kernel":               189,
    "enc_V_dense_kernel":               253,
    "mhsa_combine_head_dense_kernel":   317,
    "mlp_head_dense_1_kernel":          381, # Offset by MLP_DIM
    "mlp_dense_2_kernel":               445,
    "mlp_dense_1_kernel":               381,
    "class_emb":                        477,
    "patch_proj_bias":                  478,
    "enc_Q_dense_bias":                 479,
    "enc_K_dense_bias":                 480,
    "enc_V_dense_bias":                 481,
    "mhsa_combine_head_dense_bias":     482,
    "mlp_head_dense_1_bias":            483,
    "mlp_head_softmax_bias":            483, # Offest by MLP_DIM
    "sqrt_num_heads":                   483, # Offest by MLP_DIM + NUM_SLEEP_STAGES
    "mlp_dense_2_bias":                 484,
    "enc_layernorm_1_beta":             485,
    "enc_layernorm_1_gamma":            486,
    "enc_layernorm_2_beta":             487,
    "enc_layernorm_2_gamma":            488,
    "mlp_head_layernorm_beta":          489,
    "mlp_head_layernorm_gamma":         490,
    "mlp_dense_1_bias":                 491,
    "mlp_head_softmax_kernel":          492
}
data_last_inst = [0, 0, 0]
data_inst_index = 0

# Make set of addresses we need to sample from
addr_requested = set()
addr_needed = set(range(0, 31744))
addr_needed -= set(range(30950, 30976))
for i in range(6): addr_needed -= set(range((491+i)*64+32, (492+i)*64))

def read_param_from_ext_mem(dut):
    # Emulate the external memory that the master reads from. Has programmable read latency.
    # Expects to be called every clock cycle.
    global ext_mem_latency_cnt, prev_ext_mem_read_pulse, data_last_inst, data_inst_index
    ext_mem_latency_cnt += 1
    dut.ext_mem_data_valid.value = 0
    # On a read request pulse
    if ((prev_ext_mem_read_pulse == 0) and (dut.ext_mem_data_read_pulse.value == 1)):
        ext_mem_latency_cnt = 0

    if (ext_mem_latency_cnt == READ_LATENCY):
        row = math.floor(int(dut.ext_mem_addr.value) / 64)
        col = int(dut.ext_mem_addr.value) % 64

        if (param_map["patch_proj_kernel"] <= row < param_map["pos_emb"]):                                  dut.ext_mem_data.value = BinToDec( params["patch_projection_dense"]["vision_transformer"]["patch_projection_dense"]["kernel:0"][row - param_map["patch_proj_kernel"]][col], num_Q_storage ) # Patch projection kernel
        elif (param_map["pos_emb"] <= row < param_map["enc_Q_dense_kernel"]):                               dut.ext_mem_data.value = BinToDec( params["top_level_model_weights"]["pos_emb:0"][0][row - param_map["pos_emb"]][col], num_Q_storage ) # Positional embedding
        elif (param_map["enc_Q_dense_kernel"] <= row < param_map["enc_K_dense_kernel"]):                    dut.ext_mem_data.value = BinToDec( params["Encoder_1"]["mhsa_query_dense"]["kernel:0"][row - param_map["enc_Q_dense_kernel"]][col], num_Q_storage ) # Encoder Q dense kernel
        elif (param_map["enc_K_dense_kernel"] <= row < param_map["enc_V_dense_kernel"]):                    dut.ext_mem_data.value = BinToDec( params["Encoder_1"]["mhsa_key_dense"]["kernel:0"][row - param_map["enc_K_dense_kernel"]][col], num_Q_storage ) # Encoder K dense kernel
        elif (param_map["enc_V_dense_kernel"] <= row < param_map["mhsa_combine_head_dense_kernel"]):        dut.ext_mem_data.value = BinToDec( params["Encoder_1"]["mhsa_value_dense"]["kernel:0"][row - param_map["enc_V_dense_kernel"]][col], num_Q_storage ) # Encoder V dense kernel
        elif (param_map["mhsa_combine_head_dense_kernel"] <= row < param_map["mlp_head_dense_1_kernel"]):   dut.ext_mem_data.value = BinToDec( params["Encoder_1"]["mhsa_combine_head_dense"]["kernel:0"][row - param_map["mhsa_combine_head_dense_kernel"]][col], num_Q_storage ) # MHSA combine head dense kernel
        elif (param_map["mlp_head_dense_1_kernel"] <= row < param_map["mlp_dense_2_kernel"] and (col>=32)): dut.ext_mem_data.value = BinToDec( params["Encoder_1"]["vision_transformer"]["Encoder_1"]["mlp_dense1_encoder"]["kernel:0"][row - param_map["mlp_dense_1_kernel"]][col-32], num_Q_storage ) # Encoder MLP dense 1 kernel
        elif (param_map["mlp_dense_1_kernel"] <= row < param_map["mlp_dense_2_kernel"] and (col<32)):       dut.ext_mem_data.value = BinToDec( params["mlp_head"]["mlp_head_dense1"]["kernel:0"][row - param_map["mlp_head_dense_1_kernel"]][col], num_Q_storage ) # MLP head dense 1 kernel
        elif (param_map["mlp_dense_2_kernel"] <= row < param_map["class_emb"]):                             dut.ext_mem_data.value = BinToDec( params["Encoder_1"]["vision_transformer"]["Encoder_1"]["mlp_dense2_encoder"]["kernel:0"][row - param_map["mlp_dense_2_kernel"]][col], num_Q_storage ) # Encoder MLP dense 2 kernel
        elif (row == param_map["class_emb"]):                                                               dut.ext_mem_data.value = BinToDec( params["top_level_model_weights"]["class_emb:0"][0][0][col], num_Q_storage ) # Class embedding
        elif (row == param_map["patch_proj_bias"]):                                                         dut.ext_mem_data.value = BinToDec( params["patch_projection_dense"]["vision_transformer"]["patch_projection_dense"]["bias:0"][row - param_map["patch_proj_bias"]], num_Q_storage ) # Patch projection bias
        elif (row == param_map["enc_Q_dense_bias"]):                                                        dut.ext_mem_data.value = BinToDec( params["Encoder_1"]["mhsa_query_dense"]["bias:0"][col], num_Q_storage ) # Encoder Q dense bias
        elif (row == param_map["enc_K_dense_bias"]):                                                        dut.ext_mem_data.value = BinToDec( params["Encoder_1"]["mhsa_key_dense"]["bias:0"][col], num_Q_storage ) # Encoder K dense bias
        elif (row == param_map["enc_V_dense_bias"]):                                                        dut.ext_mem_data.value = BinToDec( params["Encoder_1"]["mhsa_value_dense"]["bias:0"][col], num_Q_storage ) # Encoder V dense bias
        elif (row == param_map["mhsa_combine_head_dense_bias"]):                                            dut.ext_mem_data.value = BinToDec( params["Encoder_1"]["mhsa_combine_head_dense"]["bias:0"][col], num_Q_storage ) # MHSA combine head dense bias
        elif (row == param_map["mlp_head_dense_1_bias"] and (col<32)):                                      dut.ext_mem_data.value = BinToDec( params["mlp_head"]["mlp_head_dense1"]["bias:0"][col], num_Q_storage ) # MLP head dense 1 bias
        elif (row == param_map["mlp_head_softmax_bias"] and (32<=col<37)):                                  dut.ext_mem_data.value = BinToDec( params["mlp_head_softmax"]["vision_transformer"]["mlp_head_softmax"]["bias:0"][col - 32], num_Q_storage ) # MLP head softmax bias
        elif (row == param_map["sqrt_num_heads"] and (col==37)):                                            dut.ext_mem_data.value = BinToDec( FXnum(NUM_HEADS, num_Q_storage).sqrt(), num_Q_storage ) # sqrt(# heads)
        elif (row == param_map["mlp_dense_2_bias"]):                                                        dut.ext_mem_data.value = BinToDec( params["Encoder_1"]["vision_transformer"]["Encoder_1"]["mlp_dense2_encoder"]["bias:0"][col], num_Q_storage ) # Encoder MLP dense 2 bias
        elif (row == param_map["enc_layernorm_1_beta"]):                                                    dut.ext_mem_data.value = BinToDec( params["Encoder_1"]["vision_transformer"]["Encoder_1"]["layerNorm1_encoder"]["beta:0"][col], num_Q_storage ) # Encoder LayerNorm 1 beta
        elif (row == param_map["enc_layernorm_1_gamma"]):                                                   dut.ext_mem_data.value = BinToDec( params["Encoder_1"]["vision_transformer"]["Encoder_1"]["layerNorm1_encoder"]["gamma:0"][col], num_Q_storage ) # Encoder LayerNorm 1 gamma
        elif (row == param_map["enc_layernorm_2_beta"]):                                                    dut.ext_mem_data.value = BinToDec( params["Encoder_1"]["vision_transformer"]["Encoder_1"]["layerNorm2_encoder"]["beta:0"][col], num_Q_storage ) # Encoder LayerNorm 2 beta
        elif (row == param_map["enc_layernorm_2_gamma"]):                                                   dut.ext_mem_data.value = BinToDec( params["Encoder_1"]["vision_transformer"]["Encoder_1"]["layerNorm2_encoder"]["gamma:0"][col], num_Q_storage ) # Encoder LayerNorm 2 gamma
        elif (row == param_map["mlp_head_layernorm_beta"]):                                                 dut.ext_mem_data.value = BinToDec( params["Encoder_1"]["vision_transformer"]["Encoder_1"]["layerNorm2_encoder"]["beta:0"][col], num_Q_storage ) # MLP head LayerNorm beta
        elif (row == param_map["mlp_head_layernorm_gamma"]):                                                dut.ext_mem_data.value = BinToDec( params["Encoder_1"]["vision_transformer"]["Encoder_1"]["layerNorm2_encoder"]["gamma:0"][col], num_Q_storage ) # MLP head LayerNorm gamma
        elif (row == param_map["mlp_dense_1_bias"]):                                                        dut.ext_mem_data.value = BinToDec( params["Encoder_1"]["vision_transformer"]["Encoder_1"]["mlp_dense1_encoder"]["bias:0"][col], num_Q_storage ) # Encoder MLP dense 1 bias
        elif (param_map["mlp_head_softmax_kernel"] <= row < param_map["mlp_head_softmax_kernel"]+5):        dut.ext_mem_data.value = BinToDec( params["mlp_head_softmax"]["vision_transformer"]["mlp_head_softmax"]["kernel:0"][col][row - param_map["mlp_head_softmax_kernel"]], num_Q_storage ) # MLP head softmax kernel
        elif (int(dut.ext_mem_addr.value) == 64*(param_map["mlp_head_softmax_kernel"]+5-1)+64):
            print(f"Received the address of 64 + max ({int(dut.ext_mem_addr.value)}), but that's OK since it simplifies the logic")
        else:
            print("Invalid address: ", int(dut.ext_mem_addr.value))
            raise ValueError
            
        dut.ext_mem_data_valid.value = 1
        addr_requested.add(int(dut.ext_mem_addr.value))

    if (dut.ext_mem_data_valid.value == 1):
        data_last_inst[data_inst_index] = int(dut.ext_mem_data.value)
        data_inst_index += 1
        data_inst_index %= 3

    if (dut.bus_op_write.value == 4): # Data stream start
        data_inst_index = 0
        for i in range(3): data_last_inst[i] = 0

    prev_ext_mem_read_pulse = dut.ext_mem_data_read_pulse.value

#----- EEG -----#
eeg = h5py.File("../../func_sim/reference_data/eeg.h5", "r")
eeg_index = 0
async def send_eeg_from_adc(dut):
    global eeg_index
    dut.new_eeg_sample.value = 1
    dut.eeg_sample.value = int(eeg["eeg"][eeg_index])
    await RisingEdge(dut.clk)
    dut.new_eeg_sample.value = 0
    for _ in range(2): await RisingEdge(dut.clk)
    expected_eeg = BinToDec(int(eeg["eeg"][eeg_index])/(2**16), num_Q_storage)
    assert ((expected_eeg-1) <= int(dut.bus_data_write.value[32:47]) <= (expected_eeg+1)), f"EEG data sent on bus ({int(dut.bus_data_write.value[32:47])}) doesn't match expected, normalize, fixed-point EEG data ({expected_eeg})"
    eeg_index += 1

def param_load_assertions(dut):
    # Assertions associated with parameters loading
    global data_last_inst
    if ((dut.bus_op_write.value) == 5):
        d0 = dut.bus_data_write.value[32:47]
        d1 = dut.bus_data_write.value[16:31]
        d2 = dut.bus_data_write.value[0:15]
        assert ((d0 == data_last_inst[0]) and (d1 == data_last_inst[1]) and (d2 == data_last_inst[2])), f"Data sent on internal bus ({d0}, {d1}, {d2}) doesn't match last three elements loaded from external memory ({data_last_inst[0], data_last_inst[1], data_last_inst[2]})"
    pass

#----- TESTS -----#
@cocotb.test()
async def load_params(dut):
    cocotb.start_soon(Clock(dut.clk, 1/ASIC_FREQUENCY_MHZ, units="us").start()) # 100MHz clock
    await reset(dut)

    # Load params and distribute to CiMs
    dut.start_param_load.value = 1
    await RisingEdge(dut.clk)
    dut.start_param_load.value = 0
    while (int(dut.params_curr_layer.value) < 10):
        param_load_assertions(dut)
        read_param_from_ext_mem(dut)
        await RisingEdge(dut.clk)
    
    # Check that all addresses that need to be read have been read
    addr_needed_but_not_requested = addr_needed.difference(addr_requested)
    assert len(addr_needed_but_not_requested) == 0, "Some addresses that are needed have not been requested!"

    # Interlude
    for _ in range(INTERLUDE_CLOCK_CYCLES): await RisingEdge(dut.clk)

    # Start loading EEG data
    print(f"Although we should send a new EEG sample at every {int(1000000*ASIC_FREQUENCY_MHZ/SAMPLING_FREQ_HZ)} clock cycles, we will shorten that to {EEG_SAMPLING_PERIOD_CLOCK_CYCLE} clock cycles.")
    dut.new_sleep_epoch.value = 1
    for _ in range(CLIP_LENGTH_S*SAMPLING_FREQ_HZ):
        await send_eeg_from_adc(dut)
        for _ in range(EEG_SAMPLING_PERIOD_CLOCK_CYCLE-3): await RisingEdge(dut.clk) # Sampling delay of EEG data

    # Interlude
    for _ in range(INTERLUDE_CLOCK_CYCLES): await RisingEdge(dut.clk)
    dut.new_sleep_epoch.value = 0
    
    for inf_step in range(len(inf_steps)):
        if (inf_step == 8): # Skip ENC_MHSA_SOFTMAX_STEP
            dut.all_cims_ready.value = 1
            await RisingEdge(dut.clk)
            dut.all_cims_ready.value = 0
            continue
        for num_run in range(inf_steps[inf_step].num_runs):
            for num_cim in range(inf_steps[inf_step].num_cim):
                # Let master run through inference steps by emulating the CiMs
                cnt = 0
                while (dut.bus_op_write.value != (inf_steps[inf_step].op.value-1)): # Wait for the master to start the broadcast
                    await RisingEdge(dut.clk)
                    cnt += 1
                    if (cnt == 100): raise ValueError(f"Master didn't start the broadcast (bus_op_write != {inf_steps[inf_step].op.value-1}) at inf_step {inf_step}!")

                assert (dut.bus_target_or_sender_write.value == (inf_steps[inf_step].start_cim + num_cim)), f"Master didn't start the broadcast for the correct CiM ({dut.bus_target_or_sender_write.value} != {num_cim}) at inf_step {inf_step}!"

                for _ in range(math.ceil(inf_steps[inf_step].len//3)):
                    await RisingEdge(dut.clk)
                    dut.bus_op_read.value = inf_steps[inf_step].op.value
                # Perform math...
                dut.bus_op_read.value = BusOp.NOP.value
                await cocotb.triggers.ClockCycles(dut.clk, 10)

                # Done with math
                if (num_cim == inf_steps[inf_step].num_cim-1): await cocotb.triggers.ClockCycles(dut.clk, 200)
                dut.all_cims_ready.value = 1
                await RisingEdge(dut.clk)
                dut.all_cims_ready.value = 0

            while (dut.bus_op_write.value != BusOp.PISTOL_START_OP.value): # Wait for pistol start
                await RisingEdge(dut.clk)
                cnt += 1
                if (cnt == 100): raise ValueError(f"Master didn't send pistol start after inf_step {inf_step}!")

        print(f"Done with inference step {inf_step}")

    # Interlude
    for _ in range(INTERLUDE_CLOCK_CYCLES): await RisingEdge(dut.clk)
