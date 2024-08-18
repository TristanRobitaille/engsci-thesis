import cocotb
from cocotb.triggers import RisingEdge
from cocotb.clock import Clock

# ----- CONSTANTS ----- #
CLK_FREQ_MHZ = 100

# ----- TEST ----- #
@cocotb.test()
async def inference_tb(dut):
    cocotb.start_soon(Clock(dut.clk, 1/CLK_FREQ_MHZ, 'us').start())

    for _ in range(1000):
        await RisingEdge(dut.clk)
