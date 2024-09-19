from enum import Enum
from FixedPoint import FXfamily, FXnum

# ----- CONSTANTS ----- #
CIM_PARAMS_BANK_SIZE_NUM_WORD   = 15872
CIM_INT_RES_BANK_SIZE_NUM_WORD  = 14336
CIM_PARAMS_NUM_BANKS            = 2
CIM_INT_RES_NUM_BANKS           = 4
N_STO_INT_RES                   = 15
N_STO_PARAMS                    = 15
N_COMP                          = 39
Q_COMP                          = 21

NUM_PATCHES = 60
PATCH_LEN = 64
EMB_DEPTH = 64
MLP_DIM = 32
NUM_SLEEP_STAGES = 5

# ----- TEST CONSTANTS ----- #
MAX_INT_ADD = 2**(N_COMP-Q_COMP-1)/2 - 1
MAX_INT_MULT = 2**(N_COMP-Q_COMP-1)/2 - 1

# ----- ENUM ----- #
class DataWidth(Enum):
    SINGLE_WIDTH = 0
    DOUBLE_WIDTH = 1

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

class MACDirection(Enum):
    HORIZONTAL = 0
    VERTICAL = 1

class FxFormatIntRes(Enum): # Value is the number of integer bits
    INT_RES_SW_FX_1_X = 1
    INT_RES_SW_FX_2_X = 2
    INT_RES_SW_FX_4_X = 4
    INT_RES_SW_FX_5_X = 5
    INT_RES_SW_FX_6_X = 6
    INT_RES_DW_FX = 10

class FxFormatParams(Enum): # Value is the number of integer bits
    PARAMS_FX_2_X = 2
    PARAMS_FX_3_X = 3
    PARAMS_FX_4_X = 4
    PARAMS_FX_5_X = 5

class InferenceStep(Enum):
    PATCH_PROJ_STEP = 0
    CLASS_TOKEN_CONCAT_STEP = 1
    POS_EMB_STEP = 2
    ENC_LAYERNORM_1_1ST_HALF_STEP = 3
    ENC_LAYERNORM_1_2ND_HALF_STEP = 4
    POS_EMB_COMPRESSION_STEP = 5
    ENC_MHSA_Q_STEP = 6
    ENC_MHSA_K_STEP = 7
    ENC_MHSA_V_STEP = 8
    ENC_MHSA_QK_T_STEP = 9
    ENC_MHSA_SOFTMAX_STEP = 10
    ENC_MHSA_MULT_V_STEP = 11
    ENC_POST_MHSA_DENSE_AND_INPUT_SUM_STEP = 12
    ENC_LAYERNORM_2_1ST_HALF_STEP = 13
    ENC_LAYERNORM_2_2ND_HALF_STEP = 14
    MLP_DENSE_1_STEP = 15
    MLP_DENSE_2_AND_SUM_STEP = 16
    ENC_LAYERNORM_3_1ST_HALF_STEP = 17
    ENC_LAYERNORM_3_2ND_HALF_STEP = 18
    MLP_HEAD_DENSE_1_STEP = 19
    MLP_HEAD_DENSE_2_STEP = 20
    MLP_HEAD_SOFTMAX_STEP = 21
    SOFTMAX_DIVIDE_STEP = 22
    SOFTMAX_AVERAGING_STEP = 23
    SOFTMAX_AVERAGE_ARGMAX_STEP = 24
    SOFTMAX_RETIRE_STEP = 25
    INFERENCE_COMPLETE = 26
    INVALID_STEP = 27

class ParamKernelsAddr(Enum):
    PATCH_PROJ_KERNEL_PARAMS = 0
    POS_EMB_PARAMS = 1
    ENC_Q_DENSE_PARAMS = 2
    ENC_K_DENSE_PARAMS = 3
    ENC_V_DENSE_PARAMS = 4
    ENC_COMB_HEAD_PARAMS = 5
    ENC_MLP_DENSE_1_PARAMS = 6
    ENC_MLP_DENSE_2_PARAMS = 7
    MLP_HEAD_DENSE_1_PARAMS = 8
    MLP_HEAD_DENSE_2_PARAMS = 9
    SINGLE_PARAMS = 10

class ParamBiasOffset(Enum):
    PATCH_PROJ_BIAS = 0
    CLASS_TOKEN = 1
    ENC_LAYERNORM_1_GAMMA = 2
    ENC_LAYERNORM_1_BETA = 3
    ENC_Q_DENSE_BIAS = 4
    ENC_K_DENSE_BIAS = 5
    ENC_V_DENSE_BIAS = 6
    ENC_INV_SQRT_NUM_HEADS = 7
    ENC_COMB_HEAD_BIAS = 8
    ENC_LAYERNORM_2_GAMMA = 9
    ENC_LAYERNORM_2_BETA = 10
    ENC_MLP_DENSE_1_BIAS = 11
    ENC_MLP_DENSE_2_BIAS = 12
    ENC_LAYERNORM_3_GAMMA = 13
    ENC_LAYERNORM_3_BETA = 14
    MLP_HEAD_DENSE_1_BIAS = 15
    MLP_HEAD_DENSE_2_BIAS = 16

class Param():
    def __init__(self, addr:int, name:str, y_len:int, x_len:int, format:FxFormatParams, index_type:str):
        self.addr = addr
        self.y_len = y_len
        self.x_len = x_len
        self.name = name
        self.format = format
        self.index_type = index_type

# ----- ADDRESSES ----- #
ParamMetadataKernels = {
    ParamKernelsAddr.PATCH_PROJ_KERNEL_PARAMS: Param(0, "patch_proj_kernel", PATCH_LEN, EMB_DEPTH, FxFormatParams.PARAMS_FX_2_X, "col-major"),
    ParamKernelsAddr.POS_EMB_PARAMS: Param(4096, "pos_emb", (NUM_PATCHES+1), EMB_DEPTH, FxFormatParams.PARAMS_FX_2_X, "row-major"),
    ParamKernelsAddr.ENC_Q_DENSE_PARAMS: Param(8000, "enc_Q_dense_kernel", EMB_DEPTH, EMB_DEPTH, FxFormatParams.PARAMS_FX_2_X, "col-major"),
    ParamKernelsAddr.ENC_K_DENSE_PARAMS: Param(12096, "enc_K_dense_kernel", EMB_DEPTH, EMB_DEPTH, FxFormatParams.PARAMS_FX_2_X, "col-major"),
    ParamKernelsAddr.ENC_V_DENSE_PARAMS: Param(16192, "enc_V_dense_kernel", EMB_DEPTH, EMB_DEPTH, FxFormatParams.PARAMS_FX_2_X, "col-major"),
    ParamKernelsAddr.ENC_COMB_HEAD_PARAMS: Param(20288, "mhsa_combine_head_dense_kernel", EMB_DEPTH, EMB_DEPTH, FxFormatParams.PARAMS_FX_2_X, "col-major"),
    ParamKernelsAddr.ENC_MLP_DENSE_1_PARAMS: Param(24384, "mlp_dense_1_kernel", EMB_DEPTH, MLP_DIM, FxFormatParams.PARAMS_FX_2_X, "col-major"),
    ParamKernelsAddr.ENC_MLP_DENSE_2_PARAMS: Param(26432, "mlp_dense_2_kernel", MLP_DIM, EMB_DEPTH, FxFormatParams.PARAMS_FX_2_X, "col-major"),
    ParamKernelsAddr.MLP_HEAD_DENSE_1_PARAMS: Param(28480, "mlp_head_dense_1_kernel", EMB_DEPTH, MLP_DIM, FxFormatParams.PARAMS_FX_2_X, "row-major"),
    ParamKernelsAddr.MLP_HEAD_DENSE_2_PARAMS: Param(30528, "mlp_head_softmax_kernel", MLP_DIM, NUM_SLEEP_STAGES, FxFormatParams.PARAMS_FX_5_X, "col-major"),
}

ParamMetadataBias = {
    ParamBiasOffset.PATCH_PROJ_BIAS: Param(30688+0, "patch_proj_bias", 1, EMB_DEPTH, FxFormatParams.PARAMS_FX_2_X, "row-major"),
    ParamBiasOffset.CLASS_TOKEN: Param(30688+64, "class_emb", 1, EMB_DEPTH, FxFormatParams.PARAMS_FX_2_X, "row-major"),
    ParamBiasOffset.ENC_LAYERNORM_1_GAMMA: Param(30688+128, "enc_layernorm_1_gamma", 1, EMB_DEPTH, FxFormatParams.PARAMS_FX_3_X, "row-major"),
    ParamBiasOffset.ENC_LAYERNORM_1_BETA: Param(30688+192, "enc_layernorm_1_beta", 1, EMB_DEPTH, FxFormatParams.PARAMS_FX_3_X, "row-major"),
    ParamBiasOffset.ENC_Q_DENSE_BIAS: Param(30688+256, "enc_Q_dense_bias", 1, EMB_DEPTH, FxFormatParams.PARAMS_FX_2_X, "row-major"),
    ParamBiasOffset.ENC_K_DENSE_BIAS: Param(30688+320, "enc_K_dense_bias", 1, EMB_DEPTH, FxFormatParams.PARAMS_FX_2_X, "row-major"),
    ParamBiasOffset.ENC_V_DENSE_BIAS: Param(30688+384, "enc_V_dense_bias", 1, EMB_DEPTH, FxFormatParams.PARAMS_FX_2_X, "row-major"),
    ParamBiasOffset.ENC_INV_SQRT_NUM_HEADS: Param(30688+448, "inv_sqrt_num_heads", 1, 1, FxFormatParams.PARAMS_FX_4_X, "n/a"),
    ParamBiasOffset.ENC_COMB_HEAD_BIAS: Param(30688+449, "mhsa_combine_head_dense_bias", 1, EMB_DEPTH, FxFormatParams.PARAMS_FX_2_X, "row-major"),
    ParamBiasOffset.ENC_LAYERNORM_2_GAMMA: Param(30688+513, "enc_layernorm_2_gamma", 1, EMB_DEPTH, FxFormatParams.PARAMS_FX_3_X, "row-major"),
    ParamBiasOffset.ENC_LAYERNORM_2_BETA: Param(30688+577, "enc_layernorm_2_beta", 1, EMB_DEPTH, FxFormatParams.PARAMS_FX_3_X, "row-major"),
    ParamBiasOffset.ENC_MLP_DENSE_1_BIAS: Param(30688+641, "mlp_dense_1_bias", 1, MLP_DIM, FxFormatParams.PARAMS_FX_2_X, "row-major"),
    ParamBiasOffset.ENC_MLP_DENSE_2_BIAS: Param(30688+673, "mlp_dense_2_bias", 1, EMB_DEPTH, FxFormatParams.PARAMS_FX_2_X, "row-major"),
    ParamBiasOffset.ENC_LAYERNORM_3_GAMMA: Param(30688+737, "enc_layernorm_3_gamma", 1, EMB_DEPTH, FxFormatParams.PARAMS_FX_3_X, "row-major"),
    ParamBiasOffset.ENC_LAYERNORM_3_BETA: Param(30688+801, "enc_layernorm_3_beta", 1, EMB_DEPTH, FxFormatParams.PARAMS_FX_3_X, "row-major"),
    ParamBiasOffset.MLP_HEAD_DENSE_1_BIAS: Param(30688+865, "mlp_dense_1_bias", 1, MLP_DIM, FxFormatParams.PARAMS_FX_2_X, "row-major"),
    ParamBiasOffset.MLP_HEAD_DENSE_2_BIAS: Param(30688+897, "mlp_head_softmax_bias", 1, EMB_DEPTH, FxFormatParams.PARAMS_FX_2_X, "row-major")
}

# ----- DICTIONARIES ----- #
int_res_fx_rtl_enum = {
    FxFormatIntRes.INT_RES_SW_FX_1_X: 0,
    FxFormatIntRes.INT_RES_SW_FX_2_X: 1,
    FxFormatIntRes.INT_RES_SW_FX_4_X: 2,
    FxFormatIntRes.INT_RES_SW_FX_5_X: 3,
    FxFormatIntRes.INT_RES_SW_FX_6_X: 4,
    FxFormatIntRes.INT_RES_DW_FX: 5
}

params_fx_rtl_enum = {
    FxFormatParams.PARAMS_FX_2_X: 0,
    FxFormatParams.PARAMS_FX_3_X: 1,
    FxFormatParams.PARAMS_FX_4_X: 2,
    FxFormatParams.PARAMS_FX_5_X: 3
}

# ----- CLASSES ----- #
num_Q_comp          = FXfamily(Q_COMP, N_COMP-Q_COMP)
num_Q_comp_overflow = FXfamily(Q_COMP, N_COMP-Q_COMP+10)
