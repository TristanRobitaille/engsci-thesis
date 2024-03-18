`ifndef _shared_parameters_vh_
`define _shared_parameters_vh_

/*----- PARAMETERS -----*/
    // Fixed-point parameters
    // verilator lint_off UNUSEDPARAM
    parameter   N_STORAGE = 16, // 16b total (for storage)
                N_COMP = 22, // 22b total (for temporary results of computation)
                Q = 10; // 10b fractional

    // Bus parameters
    parameter   BUS_OP_WIDTH = 4;

    // Other
    parameter   NUM_CIMS            = 64,
                NUM_PATCHES         = 60,
                PATCH_LEN           = 64,
                EMB_DEPTH           = 64,
                MLP_DIM             = 32,
                NUM_SLEEP_STAGES    = 5,
                NUM_HEADS           = 8;

    parameter   NUM_PARAMS  = 31589;
    parameter   PARAMS_STORAGE_SIZE_CIM = 528,
                TEMP_RES_STORAGE_SIZE_CIM = 848;
 
    parameter   EEG_SAMPLE_DEPTH = 16;
    // verilator lint_on UNUSEDPARAM

/*----- TYPES -----*/
typedef enum reg [BUS_OP_WIDTH-1:0] {
    PATCH_LOAD_BROADCAST_START_OP,
    PATCH_LOAD_BROADCAST_OP,
    DENSE_BROADCAST_START_OP,
    DENSE_BROADCAST_DATA_OP,
    DATA_STREAM_START_OP,
    DATA_STREAM_OP,
    TRANS_BROADCAST_START_OP,
    TRANS_BROADCAST_DATA_OP,
    PISTOL_START_OP,
    INFERENCE_RESULT_OP,
    NOP,
    OP_NUM // Number of ops
} bus_op_t;

/*----- ENUM -----*/
typedef enum logic {
    RST = 1'b0,
    RUN = 1'b1
} RESET_STATE_T;

`endif
