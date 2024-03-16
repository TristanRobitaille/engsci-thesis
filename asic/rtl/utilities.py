# Some common functions that are used in the testbenches
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge
from FixedPoint import FXfamily
from enum import Enum

#----- CLASSES -----#
class BusOp(Enum):
    PATCH_LOAD_BROADCAST_START_OP   = 0
    PATCH_LOAD_BROADCAST_OP         = 1
    DENSE_BROADCAST_START_OP        = 2
    DENSE_BROADCAST_DATA_OP         = 3
    DATA_STREAM_START_OP            = 4
    DATA_STREAM_OP                  = 5
    TRANS_BROADCAST_START_OP        = 6
    TRANS_BROADCAST_DATA_OP         = 7
    PISTOL_START_OP                 = 8
    INFERENCE_RESULT_OP             = 9
    NOP                             = 10
    OP_NUM                          = 11

class broadcast_op():
    def __init__(self, op:BusOp, len:int, num_cim:int, num_runs:int=1, start_cim:int=1):
        self.op = op
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
NUM_HEADS = 8
NUM_CIM = 64
NUM_PATCHES = 60
EMB_DEPTH = 64
MLP_DIM = 32
NUM_SLEEP_STAGES = 5
ASIC_FREQUENCY_MHZ = 100
SAMPLING_FREQ_HZ = 128
CLIP_LENGTH_S = 30
EEG_SAMPLING_PERIOD_CLOCK_CYCLE = 10 # Cannot simulate the real sampling period as it would take too long
INTERLUDE_CLOCK_CYCLES = 10000

num_Q_storage = FXfamily(NUM_FRACT_BITS, NUM_INT_BITS_STORAGE)
num_Q_comp = FXfamily(NUM_FRACT_BITS, NUM_INT_BITS_COMP)

inf_steps = [
    broadcast_op(BusOp.TRANS_BROADCAST_DATA_OP, len=NUM_PATCHES+1,  num_cim=NUM_CIM,            num_runs=1, start_cim=0),
    broadcast_op(BusOp.TRANS_BROADCAST_DATA_OP, len=EMB_DEPTH,      num_cim=NUM_PATCHES+1,      num_runs=1, start_cim=0),
    broadcast_op(BusOp.TRANS_BROADCAST_DATA_OP, len=NUM_PATCHES+1,  num_cim=NUM_CIM,            num_runs=1, start_cim=0),
    broadcast_op(BusOp.DENSE_BROADCAST_DATA_OP, len=EMB_DEPTH,      num_cim=NUM_PATCHES+1,      num_runs=1, start_cim=0),
    broadcast_op(BusOp.TRANS_BROADCAST_DATA_OP, len=NUM_PATCHES+1,  num_cim=NUM_CIM,            num_runs=1, start_cim=0),
    broadcast_op(BusOp.TRANS_BROADCAST_DATA_OP, len=NUM_PATCHES+1,  num_cim=NUM_CIM,            num_runs=1, start_cim=0),
    broadcast_op(BusOp.DENSE_BROADCAST_DATA_OP, len=NUM_HEADS,      num_cim=NUM_PATCHES+1,      num_runs=8, start_cim=0),
    broadcast_op(BusOp.TRANS_BROADCAST_DATA_OP, len=NUM_PATCHES+1,  num_cim=NUM_PATCHES+1,      num_runs=8, start_cim=0),
    broadcast_op(BusOp.NOP,                     len=0,              num_cim=0,                  num_runs=1, start_cim=0), # Dummy
    broadcast_op(BusOp.DENSE_BROADCAST_DATA_OP, len=NUM_PATCHES+1,  num_cim=NUM_PATCHES+1,      num_runs=8, start_cim=0),
    broadcast_op(BusOp.TRANS_BROADCAST_DATA_OP, len=NUM_PATCHES+1,  num_cim=NUM_CIM,            num_runs=1, start_cim=0),
    broadcast_op(BusOp.DENSE_BROADCAST_DATA_OP, len=EMB_DEPTH,      num_cim=NUM_PATCHES+1,      num_runs=1, start_cim=0),
    broadcast_op(BusOp.TRANS_BROADCAST_DATA_OP, len=NUM_PATCHES+1,  num_cim=NUM_CIM,            num_runs=1, start_cim=0),
    broadcast_op(BusOp.TRANS_BROADCAST_DATA_OP, len=EMB_DEPTH,      num_cim=NUM_PATCHES+1,      num_runs=1, start_cim=0),
    broadcast_op(BusOp.TRANS_BROADCAST_DATA_OP, len=NUM_PATCHES+1,  num_cim=NUM_CIM,            num_runs=1, start_cim=0),
    broadcast_op(BusOp.DENSE_BROADCAST_DATA_OP, len=EMB_DEPTH,      num_cim=NUM_CIM,            num_runs=1, start_cim=0),
    broadcast_op(BusOp.TRANS_BROADCAST_DATA_OP, len=1,              num_cim=MLP_DIM,            num_runs=1, start_cim=0),
    broadcast_op(BusOp.DENSE_BROADCAST_DATA_OP, len=MLP_DIM,        num_cim=1,                  num_runs=1, start_cim=0),
    broadcast_op(BusOp.TRANS_BROADCAST_DATA_OP, len=1,              num_cim=NUM_CIM,            num_runs=1, start_cim=0),
    broadcast_op(BusOp.TRANS_BROADCAST_DATA_OP, len=EMB_DEPTH,      num_cim=1,                  num_runs=1, start_cim=0),
    broadcast_op(BusOp.TRANS_BROADCAST_DATA_OP, len=1,              num_cim=EMB_DEPTH,          num_runs=1, start_cim=0),
    broadcast_op(BusOp.DENSE_BROADCAST_DATA_OP, len=EMB_DEPTH,      num_cim=1,                  num_runs=1, start_cim=0),
    broadcast_op(BusOp.TRANS_BROADCAST_DATA_OP, len=1,              num_cim=MLP_DIM,            num_runs=1, start_cim=MLP_DIM),
    broadcast_op(BusOp.DENSE_BROADCAST_DATA_OP, len=MLP_DIM,        num_cim=1,                  num_runs=1, start_cim=0),
    broadcast_op(BusOp.TRANS_BROADCAST_DATA_OP, len=1,              num_cim=NUM_SLEEP_STAGES,   num_runs=1, start_cim=0)
]

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

def BinToDec(dec:float, num_type:FXfamily):
    z2 = num_type(dec)
    z2_str = z2.toBinaryString(logBase=1, twosComp=True).replace(".","")
    return int(z2_str, base=2)

async def start_routine_basic_arithmetic(dut):
    cocotb.start_soon(Clock(dut.clk, 1, units="ns").start())
    await RisingEdge(dut.clk)
    await reset(dut)
    dut.refresh.value = 1