SIM ?= verilator
WAVES = 1
TOPLEVEL_LANG ?= verilog
EXTRA_ARGS += --trace --trace-structs --Wall
COMPILE_ARGS += +incdir+../../
VERILOG_SOURCES = divider.sv ../../types.svh ../../parameters.svh
TOPLEVEL = divider
MODULE = divider_tb

include $(shell cocotb-config --makefiles)/Makefile.sim