from enum import Enum
from FixedPoint import FXfamily, FXnum

# ----- CONSTANTS ----- #
CIM_PARAMS_BANK_SIZE_NUM_WORD   = 15872
CIM_INT_RES_BANK_SIZE_NUM_WORD  = 14336
CIM_PARAMS_NUM_BANKS            = 2
CIM_INT_RES_NUM_BANKS           = 4
N_STO_INT_RES                   = 9
N_STO_PARAMS                    = 8
N_COMP                          = 38
Q_COMP                          = 21

# ----- TEST CONSTANTS ----- #
Q_STO_INT_RES   = 4
MAX_INT_ADD = 2**(N_COMP-Q_COMP-1)/2 - 1
MAX_INT_MULT = 2**(N_COMP-Q_COMP-1)/2 - 1

# ----- ENUM ----- #
class DataWidth(Enum):
    SINGLE_WIDTH = 0
    DOUBLE_WIDTH = 0

# ----- CLASSES ----- #
num_Q_sto           = FXfamily(Q_STO_INT_RES, N_STO_INT_RES-Q_STO_INT_RES)
num_Q_comp          = FXfamily(Q_COMP, N_COMP-Q_COMP)
num_Q_comp_overflow = FXfamily(Q_COMP, N_COMP-Q_COMP+10)
