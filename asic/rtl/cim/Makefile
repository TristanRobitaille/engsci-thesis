SIM ?= verilator
WAVES = 1
TOPLEVEL_LANG ?= verilog
EXTRA_ARGS += --trace --trace-structs --Wall --structs-packed --cc --j 0 --threads 2

VERILOG_SOURCES = cim.sv
TOPLEVEL = cim
MODULE = cim_tb

include $(shell cocotb-config --makefiles)/Makefile.sim