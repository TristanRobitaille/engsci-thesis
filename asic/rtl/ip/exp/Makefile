SIM ?= verilator
WAVES = 1
TOPLEVEL_LANG ?= verilog
EXTRA_ARGS += --trace --trace-structs --Wall
COMPILE_ARGS += +incdir+../../
VERILOG_SOURCES = exp_wrapper_tb.sv ../../types.svh ../../parameters.svh ../adder/adder.sv ../multiplier/multiplier.sv
TOPLEVEL = exp_wrapper_tb
MODULE = exp_tb

include $(shell cocotb-config --makefiles)/Makefile.sim