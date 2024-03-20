SIM ?= verilator
WAVES = 1
TOPLEVEL_LANG ?= verilog
EXTRA_ARGS += --trace --trace-structs --Wall
COMPILE_ARGS += +incdir+../ +incdir+../adder/ +incdir+../../ +incdir+../multiplier/
VERILOG_SOURCES = cim.sv cim.svh ../../types.svh
VERILOG_SOURCES = mac.sv top_mac.sv
TOPLEVEL = top_mac
MODULE = mac_tb

include $(shell cocotb-config --makefiles)/Makefile.sim