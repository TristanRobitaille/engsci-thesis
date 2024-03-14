SIM ?= verilator
WAVES = 1
TOPLEVEL_LANG ?= verilog
EXTRA_ARGS += --trace --trace-structs --Wall

VERILOG_SOURCES = counter.sv
TOPLEVEL = counter
MODULE = counter_tb

include $(shell cocotb-config --makefiles)/Makefile.sim