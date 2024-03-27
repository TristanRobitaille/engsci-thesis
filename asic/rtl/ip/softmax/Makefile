SIM ?= verilator
WAVES = 1
TOPLEVEL_LANG ?= verilog
EXTRA_ARGS += --trace --trace-structs --Wall
COMPILE_ARGS += +incdir+../ +incdir+../adder/ +incdir+../../ +incdir+../multiplier/ +incdir+../divider/ +incdir+../exp/
VERILOG_SOURCES = softmax.sv top_softmax.sv
TOPLEVEL = top_softmax
MODULE = softmax_tb

include $(shell cocotb-config --makefiles)/Makefile.sim