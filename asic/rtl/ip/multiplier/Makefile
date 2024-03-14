SIM ?= verilator
WAVES = 1
TOPLEVEL_LANG ?= verilog
EXTRA_ARGS += --trace --trace-structs --Wall

VERILOG_SOURCES = multiplier.sv
TOPLEVEL = multiplier
MODULE = multiplier_tb

include $(shell cocotb-config --makefiles)/Makefile.sim