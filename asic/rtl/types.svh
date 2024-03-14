`ifndef _shared_parameters_vh_
`define _shared_parameters_vh_

/*----- PARAMETERS -----*/
    // Fixed-point parameters
    parameter   N_STORAGE = 16, // 16b total (for storage)
                // verilator lint_off UNUSEDPARAM
                N_COMP = 22, // 22b total (for temporary results of computation)
                Q = 10; // 10b fractional
                // verilator lint_on UNUSEDPARAM

    // Bus parameters
    parameter   BUS_OP_WIDTH = 4; // Have 11 ops

    // Other
    parameter   NUM_CIMS            = 64,
                NUM_PATCHES         = 60,
                PATCH_LEN           = 64,
                EMB_DEPTH           = 64,
                MLP_DIM             = 32,
                NUM_SLEEP_STAGES    = 5;

    parameter   NUM_PARAMS  = 31589;
    parameter   PARAMS_STORAGE_SIZE_CIM = 528,
                // verilator lint_off UNUSEDPARAM
                TEMP_RES_STORAGE_SIZE_CIM = 848;
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
    NOP
} bus_op_t;

typedef struct packed {
    reg [BUS_OP_WIDTH-1:0] op;
    reg signed [N_STORAGE-1:0] data_0;
    reg signed [N_STORAGE-1:0] data_1;
    reg signed [N_STORAGE-1:0] data_2;
    reg [$clog2(NUM_CIMS)-1:0] target_or_sender;
} bus_t;

/*----- ENUM -----*/
typedef enum logic {
    RST = 1'b0,
    RUN = 1'b1
} RESET_STATE_T;

`endif
