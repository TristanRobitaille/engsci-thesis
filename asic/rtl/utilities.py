# Some common functions that are used in the testbenches
import cocotb
import Constants as const
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge
from FixedPoint import FXfamily, FXnum

import h5py
import random
from enum import Enum

#----- CLASSES -----#
class BusOp(Enum):
    NOP                             = 0
    PATCH_LOAD_BROADCAST_START_OP   = 1
    PATCH_LOAD_BROADCAST_OP         = 2
    DENSE_BROADCAST_START_OP        = 3
    DENSE_BROADCAST_DATA_OP         = 4
    PARAM_STREAM_START_OP           = 5
    PARAM_STREAM_OP                 = 6
    TRANS_BROADCAST_START_OP        = 7
    TRANS_BROADCAST_DATA_OP         = 8
    PISTOL_START_OP                 = 9
    INFERENCE_RESULT_OP             = 10
    OP_NUM                          = 11

class MACParamType(Enum):
    MODEL_PARAM = 0
    INTERMEDIATE_RES = 1

class ActivationType(Enum):
    NO_ACTIVATION = 0
    LINEAR_ACTIVATION = 1
    SWISH_ACTIVATION = 2

class LayerNormHalfSelect(Enum):
    FIRST_HALF = 0
    SECOND_HALF = 1

class broadcast_op():
    def __init__(self, op:BusOp, tx_addr:int, rx_addr:int, len:int, num_cim:int, num_runs:int=1, start_cim:int=1):
        self.op = op
        self.tx_addr = tx_addr
        self.rx_addr = rx_addr
        self.len = len
        self.num_cim = num_cim
        self.num_runs = num_runs
        self.start_cim = start_cim

#----- CONSTANTS -----#
NUM_INT_BITS_STORAGE = 6
NUM_INT_BITS_COMP = 12
NUM_FRACT_BITS = 10
MAX_INT_ADD = 2**(NUM_INT_BITS_COMP-1)/2 - 1
MAX_INT_MULT = 2**(NUM_INT_BITS_COMP//2-1) - 1
MAX_VAL = 5
NUM_HEADS = 8
NUM_CIM = 64
NUM_PATCHES = 60
PATCH_LEN = 64
EMB_DEPTH = 64
MLP_DIM = 32
NUM_SLEEP_STAGES = 5
ASIC_FREQUENCY_MHZ = 100
SAMPLING_FREQ_HZ = 128
CLIP_LENGTH_S = 30
EEG_SAMPLING_PERIOD_CLOCK_CYCLE = 500 # Cannot simulate the real sampling period as it would take too long (however, this needs to be > 362)
INTERLUDE_CLOCK_CYCLES = 100
CLIP_INDEX = 1 # Clip index to be used for the test
TEMP_RES_STORAGE_SIZE_CIM = 848 
PARAMS_STORAGE_SIZE_CIM = 528

num_Q_storage = FXfamily(NUM_FRACT_BITS, NUM_INT_BITS_STORAGE)
num_Q_comp = FXfamily(NUM_FRACT_BITS, NUM_INT_BITS_COMP)

inf_steps = [
    broadcast_op(BusOp.TRANS_BROADCAST_DATA_OP, tx_addr=0,                              rx_addr=NUM_PATCHES+1,                  len=NUM_PATCHES+1,  num_cim=NUM_CIM,            num_runs=1, start_cim=0), #0 --> PRE_LAYERNORM_1_TRANS_STEP
    broadcast_op(BusOp.TRANS_BROADCAST_DATA_OP, tx_addr=NUM_PATCHES+1,                  rx_addr=NUM_PATCHES+1+EMB_DEPTH,        len=EMB_DEPTH,      num_cim=NUM_PATCHES+1,      num_runs=1, start_cim=0), #1 --> INTRA_LAYERNORM_1_TRANS_STEP
    broadcast_op(BusOp.TRANS_BROADCAST_DATA_OP, tx_addr=NUM_PATCHES+1+EMB_DEPTH,        rx_addr=2*EMB_DEPTH+3*(NUM_PATCHES+1),  len=NUM_PATCHES+1,  num_cim=NUM_CIM,            num_runs=1, start_cim=0), #2 --> POST_LAYERNORM_1_TRANS_STEP
    broadcast_op(BusOp.DENSE_BROADCAST_DATA_OP, tx_addr=2*EMB_DEPTH+3*(NUM_PATCHES+1),  rx_addr=NUM_PATCHES+1+EMB_DEPTH,        len=EMB_DEPTH,      num_cim=NUM_PATCHES+1,      num_runs=1, start_cim=0), #3 --> ENC_MHSA_DENSE_STEP
    broadcast_op(BusOp.TRANS_BROADCAST_DATA_OP, tx_addr=2*EMB_DEPTH+NUM_PATCHES+1,      rx_addr=NUM_PATCHES+1+EMB_DEPTH,        len=NUM_PATCHES+1,  num_cim=NUM_CIM,            num_runs=1, start_cim=0), #4 --> ENC_MHSA_Q_TRANS_STEP
    broadcast_op(BusOp.TRANS_BROADCAST_DATA_OP, tx_addr=2*(EMB_DEPTH+NUM_PATCHES+1),    rx_addr=2*EMB_DEPTH+NUM_PATCHES+1,      len=NUM_PATCHES+1,  num_cim=NUM_CIM,            num_runs=1, start_cim=0), #5 --> ENC_MHSA_K_TRANS_STEP
    broadcast_op(BusOp.DENSE_BROADCAST_DATA_OP, tx_addr=NUM_PATCHES+1+EMB_DEPTH,        rx_addr=3*EMB_DEPTH+NUM_PATCHES+1,      len=NUM_HEADS,      num_cim=NUM_PATCHES+1,      num_runs=8, start_cim=0), #6 --> ENC_MHSA_QK_T_STEP
    broadcast_op(BusOp.TRANS_BROADCAST_DATA_OP, tx_addr=3*EMB_DEPTH+2*(NUM_PATCHES+1),  rx_addr=2*(EMB_DEPTH+NUM_PATCHES+1),    len=NUM_PATCHES+1,  num_cim=NUM_PATCHES+1,      num_runs=8, start_cim=0), #7 --> ENC_MHSA_PRE_SOFTMAX_TRANS_STEP
    broadcast_op(BusOp.NOP,                     tx_addr=0,                              rx_addr=0,                              len=0,              num_cim=0,                  num_runs=1, start_cim=0), # Dummy
    broadcast_op(BusOp.DENSE_BROADCAST_DATA_OP, tx_addr=2*(EMB_DEPTH+NUM_PATCHES+1),    rx_addr=NUM_PATCHES+1+EMB_DEPTH,        len=NUM_PATCHES+1,  num_cim=NUM_PATCHES+1,      num_runs=8, start_cim=0), #9 --> ENC_MHSA_V_MULT_STEP
    broadcast_op(BusOp.TRANS_BROADCAST_DATA_OP, tx_addr=2*EMB_DEPTH+NUM_PATCHES+1,      rx_addr=NUM_PATCHES+1,                  len=NUM_PATCHES+1,  num_cim=NUM_CIM,            num_runs=1, start_cim=0), #10 --> ENC_MHSA_POST_V_TRANS_STEP
    broadcast_op(BusOp.DENSE_BROADCAST_DATA_OP, tx_addr=NUM_PATCHES+1,                  rx_addr=NUM_PATCHES+1+EMB_DEPTH,        len=EMB_DEPTH,      num_cim=NUM_PATCHES+1,      num_runs=1, start_cim=0), #11 --> ENC_MHSA_POST_V_DENSE_STEP
    broadcast_op(BusOp.TRANS_BROADCAST_DATA_OP, tx_addr=2*EMB_DEPTH+NUM_PATCHES+1,      rx_addr=NUM_PATCHES+1,                  len=NUM_PATCHES+1,  num_cim=NUM_CIM,            num_runs=1, start_cim=0), #12 --> PRE_LAYERNORM_2_TRANS_STEP
    broadcast_op(BusOp.TRANS_BROADCAST_DATA_OP, tx_addr=NUM_PATCHES+1,                  rx_addr=NUM_PATCHES+1+EMB_DEPTH,        len=EMB_DEPTH,      num_cim=NUM_PATCHES+1,      num_runs=1, start_cim=0), #13 --> INTRA_LAYERNORM_2_TRANS_STEP
    broadcast_op(BusOp.TRANS_BROADCAST_DATA_OP, tx_addr=NUM_PATCHES+1+EMB_DEPTH,        rx_addr=4*EMB_DEPTH+NUM_PATCHES+1,      len=NUM_PATCHES+1,  num_cim=NUM_CIM,            num_runs=1, start_cim=0), #14 --> ENC_PRE_MLP_TRANSPOSE_STEP
    broadcast_op(BusOp.DENSE_BROADCAST_DATA_OP, tx_addr=4*EMB_DEPTH+NUM_PATCHES+1,      rx_addr=NUM_PATCHES+1,                  len=EMB_DEPTH,      num_cim=NUM_CIM,            num_runs=1, start_cim=0), #15 --> ENC_MLP_DENSE_1_STEP
    broadcast_op(BusOp.TRANS_BROADCAST_DATA_OP, tx_addr=NUM_PATCHES+1+EMB_DEPTH,        rx_addr=NUM_PATCHES+1,                  len=1,              num_cim=MLP_DIM,            num_runs=1, start_cim=0), #16 --> ENC_MLP_DENSE_2_TRANSPOSE_STEP
    broadcast_op(BusOp.DENSE_BROADCAST_DATA_OP, tx_addr=NUM_PATCHES+1,                  rx_addr=NUM_PATCHES+1,                  len=MLP_DIM,        num_cim=1,                  num_runs=1, start_cim=0), #17 --> ENC_MLP_DENSE_2_AND_SUM_STEP
    broadcast_op(BusOp.TRANS_BROADCAST_DATA_OP, tx_addr=3*EMB_DEPTH+NUM_PATCHES+2,      rx_addr=0,                              len=1,              num_cim=NUM_CIM,            num_runs=1, start_cim=0), #18 --> PRE_LAYERNORM_3_TRANS_STEP
    broadcast_op(BusOp.TRANS_BROADCAST_DATA_OP, tx_addr=0,                              rx_addr=EMB_DEPTH,                      len=EMB_DEPTH,      num_cim=1,                  num_runs=1, start_cim=0), #19 --> INTRA_LAYERNORM_3_TRANS_STEP
    broadcast_op(BusOp.TRANS_BROADCAST_DATA_OP, tx_addr=EMB_DEPTH,                      rx_addr=0,                              len=1,              num_cim=EMB_DEPTH,          num_runs=1, start_cim=0), #20 --> PRE_MLP_HEAD_DENSE_TRANS_STEP
    broadcast_op(BusOp.DENSE_BROADCAST_DATA_OP, tx_addr=0,                              rx_addr=EMB_DEPTH,                      len=EMB_DEPTH,      num_cim=1,                  num_runs=1, start_cim=0), #21 --> MLP_HEAD_DENSE_1_STEP
    broadcast_op(BusOp.TRANS_BROADCAST_DATA_OP, tx_addr=2*EMB_DEPTH,                    rx_addr=0,                              len=1,              num_cim=MLP_DIM,            num_runs=1, start_cim=MLP_DIM), #22 --> PRE_MLP_HEAD_DENSE_2_TRANS_STEP
    broadcast_op(BusOp.DENSE_BROADCAST_DATA_OP, tx_addr=MLP_DIM,                        rx_addr=EMB_DEPTH,                      len=MLP_DIM,        num_cim=1,                  num_runs=1, start_cim=0), #23 --> MLP_HEAD_DENSE_2_STEP
    broadcast_op(BusOp.TRANS_BROADCAST_DATA_OP, tx_addr=2*EMB_DEPTH,                    rx_addr=MLP_DIM,                        len=1,              num_cim=NUM_SLEEP_STAGES,   num_runs=1, start_cim=0)  #24 --> MLP_HEAD_SOFTMAX_TRANS_STEP
]

param_addr_map = {
    "patch_proj_kernel":                            {"start_addr": 0,       "len": PATCH_LEN},
    "pos_emb":                                      {"start_addr": 1*64,    "len": NUM_PATCHES+1},
    "enc_Q_dense_kernel":                           {"start_addr": 2*64,    "len": EMB_DEPTH},
    "enc_K_dense_kernel":                           {"start_addr": 3*64,    "len": EMB_DEPTH},
    "enc_V_dense_kernel":                           {"start_addr": 4*64,    "len": EMB_DEPTH},
    "enc_comb_head_kernel":                         {"start_addr": 5*64,    "len": EMB_DEPTH},
    "enc_mlp_dense_1_or_mlp_head_dense_1_kernel":   {"start_addr": 6*64,    "len": EMB_DEPTH},
    "enc_mlp_dense_2_kernel":                       {"start_addr": 7*64,    "len": MLP_DIM},
    "enc_enc_comb_2_kernel":                        {"start_addr": 7*64+32, "len": MLP_DIM},
    "single_params":                                {"start_addr": 8*64,    "len": 16}
}

#----- HELPERS -----#
async def reset(dut):
    dut.rst_n.value = 0
    await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)

async def start_pulse(dut):
    dut.start.value = 1
    await RisingEdge(dut.clk)
    dut.start.value = 0
    await RisingEdge(dut.clk)

async def start_routine_basic_arithmetic(dut):
    cocotb.start_soon(Clock(dut.clk, 10, units="ns").start()) # 100MHz clock
    await RisingEdge(dut.clk)
    await reset(dut)
    
def BinToDec(dec:float, num_type:FXfamily):
    z2 = num_type(dec)
    z2_str = z2.toBinaryString(logBase=1, twosComp=True).replace(".","")
    return int(z2_str, base=2)

def random_input(min, max):
    val = random.normalvariate(mu = 0.0, sigma = (max/5)) # Random number from a normal distribution (max. value at 5 std. dev.)
    if val > max: val = max
    elif val < min: val = min
    return val

def params(param_name:str, params_file:h5py.File):
    if (param_name == "patch_proj_kernel"):                 return params_file["patch_projection_dense"]["vision_transformer"]["patch_projection_dense"]["kernel:0"] # Patch projection kernel
    elif (param_name == "pos_emb"):                         return params_file["top_level_model_weights"]["pos_emb:0"][0] # Positional embedding
    elif (param_name == "enc_Q_dense_kernel"):              return params_file["Encoder_1"]["mhsa_query_dense"]["kernel:0"] # Encoder Q dense kernel
    elif (param_name == "enc_K_dense_kernel"):              return params_file["Encoder_1"]["mhsa_key_dense"]["kernel:0"] # Encoder K dense kernel
    elif (param_name == "enc_V_dense_kernel"):              return params_file["Encoder_1"]["mhsa_value_dense"]["kernel:0"] # Encoder V dense kernel
    elif (param_name == "mhsa_combine_head_dense_kernel"):  return params_file["Encoder_1"]["mhsa_combine_head_dense"]["kernel:0"] # MHSA combine head dense kernel
    elif (param_name == "mlp_head_dense_1_kernel"):         return params_file["Encoder_1"]["vision_transformer"]["Encoder_1"]["mlp_dense1_encoder"]["kernel:0"] # Encoder MLP dense 1 kernel
    elif (param_name == "mlp_dense_1_kernel"):              return params_file["mlp_head"]["mlp_head_dense1"]["kernel:0"] # MLP head dense 1 kernel
    elif (param_name == "mlp_dense_2_kernel"):              return params_file["Encoder_1"]["vision_transformer"]["Encoder_1"]["mlp_dense2_encoder"]["kernel:0"] # Encoder MLP dense 2 kernel
    elif (param_name == "class_emb"):                       return params_file["top_level_model_weights"]["class_emb:0"][0] # Class embedding
    elif (param_name == "patch_proj_bias"):                 return params_file["patch_projection_dense"]["vision_transformer"]["patch_projection_dense"]["bias:0"] # Patch projection bias
    elif (param_name == "enc_Q_dense_bias"):                return params_file["Encoder_1"]["mhsa_query_dense"]["bias:0"] # Encoder Q dense bias
    elif (param_name == "enc_K_dense_bias"):                return params_file["Encoder_1"]["mhsa_key_dense"]["bias:0"] # Encoder K dense bias
    elif (param_name == "enc_V_dense_bias"):                return params_file["Encoder_1"]["mhsa_value_dense"]["bias:0"] # Encoder V dense bias
    elif (param_name == "mhsa_combine_head_dense_bias"):    return params_file["Encoder_1"]["mhsa_combine_head_dense"]["bias:0"] # MHSA combine head dense bias
    elif (param_name == "mlp_head_dense_1_bias"):           return params_file["mlp_head"]["mlp_head_dense1"]["bias:0"] # MLP head dense 1 bias
    elif (param_name == "mlp_head_softmax_bias"):           return params_file["mlp_head_softmax"]["vision_transformer"]["mlp_head_softmax"]["bias:0"] # MLP head softmax bias
    elif (param_name == "inv_sqrt_num_heads"):              return FXnum(1/NUM_HEADS, num_Q_storage).sqrt() # 1/sqrt(# heads)
    elif (param_name == "mlp_dense_2_bias"):                return params_file["Encoder_1"]["vision_transformer"]["Encoder_1"]["mlp_dense2_encoder"]["bias:0"] # Encoder MLP dense 2 bias
    elif (param_name == "enc_layernorm_1_beta"):            return params_file["Encoder_1"]["vision_transformer"]["Encoder_1"]["layerNorm1_encoder"]["beta:0"] # Encoder LayerNorm 1 beta
    elif (param_name == "enc_layernorm_1_gamma"):           return params_file["Encoder_1"]["vision_transformer"]["Encoder_1"]["layerNorm1_encoder"]["gamma:0"] # Encoder LayerNorm 1 gamma
    elif (param_name == "enc_layernorm_2_beta"):            return params_file["Encoder_1"]["vision_transformer"]["Encoder_1"]["layerNorm2_encoder"]["beta:0"] # Encoder LayerNorm 2 beta
    elif (param_name == "enc_layernorm_2_gamma"):           return params_file["Encoder_1"]["vision_transformer"]["Encoder_1"]["layerNorm2_encoder"]["gamma:0"] # Encoder LayerNorm 2 gamma
    elif (param_name == "mlp_head_layernorm_beta"):         return params_file["Encoder_1"]["vision_transformer"]["Encoder_1"]["layerNorm2_encoder"]["beta:0"] # MLP head LayerNorm beta
    elif (param_name == "mlp_head_layernorm_gamma"):        return params_file["Encoder_1"]["vision_transformer"]["Encoder_1"]["layerNorm2_encoder"]["gamma:0"] # MLP head LayerNorm gamma
    elif (param_name == "mlp_dense_1_bias"):                return params_file["Encoder_1"]["vision_transformer"]["Encoder_1"]["mlp_dense1_encoder"]["bias:0"] # Encoder MLP dense 1 bias
    elif (param_name == "mlp_head_softmax_kernel"):         return params_file["mlp_head_softmax"]["vision_transformer"]["mlp_head_softmax"]["kernel:0"]# MLP head softmax kernel
    else: raise ValueError(f"Unknown parameter name: {param_name}")

def twos_complement_to_float(input_data, bit_width:int=const.N_COMP):
    input_data = str(input_data)
    integer_value = int(input_data, 2) # Convert binary string to an integer
    if input_data[0] == '1': integer_value -= (1 << bit_width)
    return float(integer_value / 2**const.Q_COMP)

async def write_one_word_cent(dut, addr:int, data:int, device:str, data_format, data_width:const.DataWidth) -> None:
    if device == "params":
        try:
            data_fx = FXnum(data, FXfamily(const.N_STO_PARAMS-data_format.value, data_format.value))
        except:
            if data < 0: data_fx = FXnum(-(2**(data_format.value-1)) + 1/2**(const.N_STO_PARAMS-data_format.value), FXfamily(const.N_STO_PARAMS-data_format.value, data_format.value))
            elif data > 0: data_fx = FXnum(2**(data_format.value-1) - 1/2**(const.N_STO_PARAMS-data_format.value), FXfamily(const.N_STO_PARAMS-data_format.value, data_format.value))
    elif device == "int_res":
        if data_width == const.DataWidth.SINGLE_WIDTH: data_fx = FXnum(data, FXfamily(const.N_STO_INT_RES-data_format.value, data_format.value))
        elif data_width == const.DataWidth.DOUBLE_WIDTH: data_fx = FXnum(data, FXfamily(2*const.N_STO_INT_RES-data_format.value, data_format.value))
    
    data_fx = FXnum(data_fx, const.num_Q_comp)
    data_fx = BinToDec(data_fx, const.num_Q_comp)

    if device == "params":
        assert (data_width.value == const.DataWidth.SINGLE_WIDTH.value), "Params memory only compatible with SINGLE_WIDTH!"
        dut.param_chip_en.value = 1
        dut.param_write_en.value = 1
        dut.param_write_addr.value = addr
        dut.param_write_data.value = data_fx
        dut.param_write_format.value = const.params_fx_rtl_enum[data_format]
        await RisingEdge(dut.clk)
        dut.param_write_en.value = 0
    elif device == "int_res":
        dut.int_res_chip_en.value = 1
        dut.int_res_write_en.value = 1
        dut.int_res_write_data_width.value = data_width.value
        dut.int_res_write_addr.value = addr
        dut.int_res_write_data.value = data_fx
        dut.int_res_write_format.value = const.int_res_fx_rtl_enum[data_format]
        await RisingEdge(dut.clk)
        dut.int_res_write_en.value = 0

async def read_one_word_cent(dut, addr:int, device:str, data_format, data_width:const.DataWidth) -> float:
    await RisingEdge(dut.clk)
    if device == "params":
        assert (data_width.value == const.DataWidth.SINGLE_WIDTH.value), "Params memory only compatible with SINGLE_WIDTH!"
        dut.param_chip_en.value = 1
        dut.param_read_en.value = 1
        dut.param_read_addr.value = addr
        dut.param_read_format.value = const.params_fx_rtl_enum[data_format]
        await RisingEdge(dut.clk)
        dut.param_read_en.value = 0
        await RisingEdge(dut.clk)
        return twos_complement_to_float(dut.param_read_data.value)
    elif device == "int_res":
        dut.int_res_chip_en.value = 1
        dut.int_res_read_en.value = 1
        dut.int_res_read_data_width.value = data_width.value
        dut.int_res_read_addr.value = addr
        dut.int_res_read_format.value = const.int_res_fx_rtl_enum[data_format]
        await RisingEdge(dut.clk)
        dut.int_res_read_en.value = 0
        await RisingEdge(dut.clk)
        return twos_complement_to_float(dut.int_res_read_data.value)

async def fill_params_mem(dut, data_format:const.FxFormatParams=const.FxFormatParams.PARAMS_FX_4_X):
    for addr in range(const.CIM_PARAMS_NUM_BANKS * const.CIM_PARAMS_BANK_SIZE_NUM_WORD):
        data = random.uniform(-2**(data_format.value-1)+1, 2**(data_format.value-1)-1)
        await write_one_word_cent(dut, addr=addr, data=data, device="params", data_format=data_format, data_width=const.DataWidth.SINGLE_WIDTH)

async def fill_int_res_mem(dut, data_format:const.FxFormatIntRes=const.FxFormatIntRes.INT_RES_SW_FX_5_X, data_width:const.DataWidth=const.DataWidth.SINGLE_WIDTH):
    for addr in range(const.CIM_INT_RES_NUM_BANKS * const.CIM_INT_RES_BANK_SIZE_NUM_WORD):
        data = random.uniform(-2**(data_format.value-1)+1, 2**(data_format.value-1)-1)
        await write_one_word_cent(dut, addr=addr, data=data, device="int_res", data_format=data_format, data_width=data_width)

#----- EEG -----#
eeg_index = 0
async def send_eeg_from_adc(dut, eeg_file):
    global eeg_index
    dut.new_eeg_sample.value = 1
    dut.eeg_sample.value = int(eeg_file["eeg"][CLIP_INDEX][eeg_index])
    await RisingEdge(dut.clk)
    dut.new_eeg_sample.value = 0
    for _ in range(2): await RisingEdge(dut.clk)
    expected_eeg = BinToDec(int(eeg_file["eeg"][CLIP_INDEX][eeg_index])/(2**16), num_Q_storage)
    assert ((expected_eeg-1) <= int(dut.bus_data_write.value[32:47]) <= (expected_eeg+1)), f"EEG data sent on bus ({int(dut.bus_data_write.value[32:47])}) doesn't match expected, normalize, fixed-point EEG data ({expected_eeg})"
    eeg_index += 1

def send_eeg_from_master(dut, eeg_file):
    global eeg_index
    scaled_raw_eeg = int(eeg_file["eeg"][CLIP_INDEX][eeg_index])/(2**16)
    val = BinToDec(scaled_raw_eeg, num_Q_storage)
    dut.bus_op_read.value = BusOp.PATCH_LOAD_BROADCAST_OP.value
    dut.bus_data_read.value = val
    eeg_index += 1
    return scaled_raw_eeg

#----- COROUTINES -----#
@cocotb.coroutine
def bus_mirror(dut):
    # Writes on bus_x_read was is on bus_x, as would be done in the ASIC. This is needed because the CiM sometimes reacts to what it itself sends.
    while True:
        # Wait for a change on the bus
        yield [ cocotb.triggers.Edge(dut.bus_op),
                cocotb.triggers.Edge(dut.bus_target_or_sender),
                cocotb.triggers.Edge(dut.bus_data)]
        dut.bus_op_read.value = dut.bus_op.value
        dut.bus_target_or_sender_read.value = dut.bus_target_or_sender.value
        dut.bus_data_read.value = dut.bus_data.value