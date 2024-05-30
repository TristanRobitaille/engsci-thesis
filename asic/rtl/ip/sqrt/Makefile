SIM ?= verilator
WAVES = 1
TOPLEVEL_LANG ?= verilog
EXTRA_ARGS += --trace --trace-structs --Wall

VERILOG_SOURCES = ../../parameters.svh ../../types.svh sqrt.sv
TOPLEVEL = sqrt
MODULE = sqrt_tb

include $(shell cocotb-config --makefiles)/Makefile.sim