SIM ?= verilator
WAVES = 1
TOPLEVEL_LANG ?= verilog
EXTRA_ARGS += --trace --trace-structs --Wall

VERILOG_SOURCES = adder.sv ../../types.svh
TOPLEVEL = adder
MODULE = adder_tb

include $(shell cocotb-config --makefiles)/Makefile.sim