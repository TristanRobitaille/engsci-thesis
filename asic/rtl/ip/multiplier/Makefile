SIM ?= verilator
WAVES = 1
TOPLEVEL_LANG ?= verilog
EXTRA_ARGS += --trace --trace-structs --Wall
COMPILE_ARGS += +incdir+../../
VERILOG_SOURCES = multiplier.sv ../../types.svh ../../parameters.svh
TOPLEVEL = multiplier
MODULE = multiplier_tb

include $(shell cocotb-config --makefiles)/Makefile.sim