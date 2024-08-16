`ifndef _parameters_vh_
`define _parameters_vh_

/*----- PARAMETERS -----*/
package parameters;
    // verilator lint_off UNUSEDPARAM
    // Fixed-point parameters
    parameter   N_STORAGE = 16, // 16b total (for storage)
                N_COMP = 22,    // 22b total (for temporary results of computation)
                Q = 10;         // 10b fractional

    // Bus parameters
    parameter   BUS_OP_WIDTH = 4;

    // Other
    parameter   NUM_CIMS                    = 64,
                NUM_PATCHES                 = 60,
                PATCH_LEN                   = 64,
                EMB_DEPTH                   = 64, // Note: Must be a power of two as we are bit-shifting instead of dividing in the LayerNorm module based on this
                MLP_DIM                     = 32,
                NUM_SLEEP_STAGES            = 5,
                NUM_HEADS                   = 8,
                MAC_MAX_LEN                 = 64,
                SOFTMAX_MAX_LEN             = 64,
                NUM_SAMPLES_OUT_AVG         = 3,
                INV_NUM_SAMPLES_OUT_AVG     = 341; // 1/NUM_SAMPLES_OUT_AVG in fixed-point

    parameter   NUM_PARAMS  = 31589;
    parameter   PARAMS_STORAGE_SIZE_CIM = 528,
                TEMP_RES_STORAGE_SIZE_CIM = 848;
 
    parameter   EEG_SAMPLE_DEPTH = 16;
    // verilator lint_on UNUSEDPARAM
endpackage

`endif
