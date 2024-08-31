from enum import Enum
from FixedPoint import FXfamily, FXnum

# ----- CONSTANTS ----- #
CIM_PARAMS_BANK_SIZE_NUM_WORD   = 15872
CIM_INT_RES_BANK_SIZE_NUM_WORD  = 14336
CIM_PARAMS_NUM_BANKS            = 2
CIM_INT_RES_NUM_BANKS           = 4
N_STO_INT_RES                   = 8
N_STO_PARAMS                    = 9
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

class FxFormatIntRes(Enum): # Value is the number of integer bits
    INT_RES_SW_FX_1_X = 1
    INT_RES_SW_FX_2_X = 2
    INT_RES_SW_FX_5_X = 5
    INT_RES_SW_FX_6_X = 6
    INT_RES_DW_FX = 10

class FxFormatParams(Enum): # Value is the number of integer bits
    PARAMS_FX_2_X = 2
    PARAMS_FX_3_X = 3
    PARAMS_FX_4_X = 4
    PARAMS_FX_5_X = 5

# ----- DICTIONARIES ----- #
int_res_fx_rtl_enum = {
    FxFormatIntRes.INT_RES_SW_FX_1_X: 0,
    FxFormatIntRes.INT_RES_SW_FX_2_X: 1,
    FxFormatIntRes.INT_RES_SW_FX_5_X: 2,
    FxFormatIntRes.INT_RES_SW_FX_6_X: 3,
    FxFormatIntRes.INT_RES_DW_FX: 4
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
