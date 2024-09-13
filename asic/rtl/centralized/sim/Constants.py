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
