#!/usr/bin/env tclsh

# Define the base memory name and path
set BASE_PATH "/autofs/fs1.ece/fs1.eecg.xliugrp/robita46/engsci-thesis/asic/rtl/centralized/mem"

# Create a list of memory names
set MEM_NAMES {
    "params_15872x15"
    "int_res_14336x15"
}

# Loop through each memory name
foreach MEM_NAME $MEM_NAMES {
    # Define the full path for the current memory name
    set CURRENT_BASE_PATH "${BASE_PATH}/${MEM_NAME}"

    # Read and write libraries for different variations
    read_lib [file join $CURRENT_BASE_PATH "${MEM_NAME}_nldm_ff_1p10v_1p10v_0c_syn.lib"]
    write_lib "${MEM_NAME}_nldm_ff_1p10v_1p10v_0c" -format db -output [file join $CURRENT_BASE_PATH "${MEM_NAME}_nldm_ff_1p10v_1p10v_0c.db"]

    read_lib [file join $CURRENT_BASE_PATH "${MEM_NAME}_nldm_ff_1p10v_1p10v_125c_syn.lib"]
    write_lib "${MEM_NAME}_nldm_ff_1p10v_1p10v_125c" -format db -output [file join $CURRENT_BASE_PATH "${MEM_NAME}_nldm_ff_1p10v_1p10v_125c.db"]

    read_lib [file join $CURRENT_BASE_PATH "${MEM_NAME}_nldm_ff_1p10v_1p10v_m40c_syn.lib"]
    write_lib "${MEM_NAME}_nldm_ff_1p10v_1p10v_m40c" -format db -output [file join $CURRENT_BASE_PATH "${MEM_NAME}_nldm_ff_1p10v_1p10v_m40c.db"]

    read_lib [file join $CURRENT_BASE_PATH "${MEM_NAME}_nldm_ss_0p90v_0p90v_125c_syn.lib"]
    write_lib "${MEM_NAME}_nldm_ss_0p90v_0p90v_125c" -format db -output [file join $CURRENT_BASE_PATH "${MEM_NAME}_nldm_ss_0p90v_0p90v_125c.db"]

    read_lib [file join $CURRENT_BASE_PATH "${MEM_NAME}_nldm_ss_0p90v_0p90v_m40c_syn.lib"]
    write_lib "${MEM_NAME}_nldm_ss_0p90v_0p90v_m40c" -format db -output [file join $CURRENT_BASE_PATH "${MEM_NAME}_nldm_ss_0p90v_0p90v_m40c.db"]

    read_lib [file join $CURRENT_BASE_PATH "${MEM_NAME}_nldm_tt_1p00v_1p00v_25c_syn.lib"]
    write_lib "${MEM_NAME}_nldm_tt_1p00v_1p00v_25c" -format db -output [file join $CURRENT_BASE_PATH "${MEM_NAME}_nldm_tt_1p00v_1p00v_25c.db"]
}

exit