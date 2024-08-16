SIM ?= verilator
WAVES = 1
TOPLEVEL_LANG ?= verilog
EXTRA_ARGS += --trace --trace-structs --Wall --structs-packed --cc --j 0 --threads 2
COMPILE_ARGS += -GSTANDALONE_TB=1 # We cannot easily drive inout signals with CocoTB, so we need to directly drive bus_x_read signals and avoid assigning to them if STANDALONE_TB is set to 1
COMPILE_ARGS += +incdir+../
VERILOG_SOURCES = master.sv master.svh ../types.svh ../ip/counter/counter.sv
TOPLEVEL = master
MODULE = master_tb

include $(shell cocotb-config --makefiles)/Makefile.sim