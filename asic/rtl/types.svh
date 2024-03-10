`ifndef _shared_parameters_vh_
`define _shared_parameters_vh_

/*----- PARAMETERS -----*/
    // Fixed-point parameters
    parameter   N = 22, // 22b total
                Q = 10; // 10b fractional

    // Bus parameters
    parameter   BUS_OP_WIDTH = 4, // Have 11 ops
                BUS_DATA_WIDTH = 3 * N; // Fit 3 data words

    // Other
    parameter NUM_CIMS = 64;

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
    reg signed [BUS_DATA_WIDTH-1:0] data;
    reg [$clog2(NUM_CIMS)-1:0] target_or_sender;
} bus_t;

typedef reg signed [N-1:0] fix_pt_t;

`endif
