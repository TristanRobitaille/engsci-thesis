SIM ?= verilator
WAVES = 1
TOPLEVEL_LANG ?= verilog
EXTRA_ARGS += --trace --trace-structs --Wall

VERILOG_SOURCES = divider.sv ../../types.svh
TOPLEVEL = divider
MODULE = divider_tb

include $(shell cocotb-config --makefiles)/Makefile.sim