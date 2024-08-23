from enum import Enum

# ----- CONSTANTS ----- #
CIM_PARAMS_BANK_SIZE_NUM_WORD   = 15872
CIM_INT_RES_BANK_SIZE_NUM_WORD  = 14336
CIM_PARAMS_NUM_BANKS            = 2
CIM_INT_RES_NUM_BANKS           = 4
N_STO_INT_RES                   = 9
N_STO_PARAMS                    = 9

# ----- ENUM ----- #
class DataWidth(Enum):
    SINGLE_WIDTH = 0
    DOUBLE_WIDTH = 0
