SIM ?= verilator
WAVES = 1
TOPLEVEL_LANG ?= verilog
EXTRA_ARGS += --trace --trace-structs --Wall --structs-packed --cc # --j 0 --threads 2
COMPILE_ARGS += --assert -GSTANDALONE_TB=1 # We cannot easily drive inout signals with CocoTB, so we need to directly drive bus_x_read signals and avoid assigning to them if STANDALONE_TB is set to 1
COMPILE_ARGS += +incdir+../
VERILOG_SOURCES = cim.sv cim.svh ../types.svh ../ip/adder/adder.sv ../ip/counter/counter.sv ../ip/divider/divider.sv ../ip/multiplier/multiplier.sv ../ip/mac/mac.sv ../ip/exp/exp.sv ../ip/sqrt/sqrt.sv
TOPLEVEL = cim
MODULE = cim_tb

include $(shell cocotb-config --makefiles)/Makefile.sim