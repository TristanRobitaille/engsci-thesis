SIM ?= verilator
WAVES = 1
TOPLEVEL_LANG ?= verilog
EXTRA_ARGS += --trace --trace-structs --Wall

VERILOG_SOURCES = mac.sv top.sv
TOPLEVEL = top
MODULE = mac_tb

include $(shell cocotb-config --makefiles)/Makefile.sim