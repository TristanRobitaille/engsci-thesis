SIM ?= verilator
WAVES = 1
TOPLEVEL_LANG ?= verilog
EXTRA_ARGS += --trace --trace-structs --Wall
COMPILE_ARGS += +incdir+../ +incdir+../adder/ +incdir+../../ +incdir+../multiplier/ +incdir+../sqrt/ +incdir+../divider/ +incdir+../../cim
VERILOG_SOURCES = layernorm.sv ../../cim/cim.svh ../../types.svh
VERILOG_SOURCES = layernorm.sv top_layernorm.sv
TOPLEVEL = top_layernorm
MODULE = layernorm_tb

include $(shell cocotb-config --makefiles)/Makefile.sim