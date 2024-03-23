`ifndef _cim_svh_
`define _cim_svh_

`include "../types.svh"

/*----- ENUM -----*/
typedef enum logic [2:0] {
    CIM_IDLE,
    CIM_RESET,
    CIM_PATCH_LOAD,
    CIM_INFERENCE_RUNNING,
    CIM_INVALID,
    CIM_NUM_STATES
} CIM_STATE_T;

typedef enum logic [5:0] {
    CLASS_TOKEN_CONCAT_STEP,
    POS_EMB_STEP,
    ENC_LAYERNORM_1_1ST_HALF_STEP,
    ENC_LAYERNORM_1_2ND_HALF_STEP,
    ENC_LAYERNORM_2_1ST_HALF_STEP,
    ENC_LAYERNORM_2_2ND_HALF_STEP,
    ENC_LAYERNORM_3_1ST_HALF_STEP,
    ENC_LAYERNORM_3_2ND_HALF_STEP,
    POST_LAYERNORM_TRANSPOSE_STEP,
    ENC_MHSA_DENSE_STEP,
    ENC_MHSA_Q_TRANSPOSE_STEP,
    ENC_MHSA_K_TRANSPOSE_STEP,
    ENC_MHSA_QK_T_STEP,
    ENC_MHSA_PRE_SOFTMAX_TRANSPOSE_STEP,
    ENC_MHSA_SOFTMAX_STEP,
    ENC_MHSA_MULT_V_STEP,
    ENC_POST_MHSA_TRANSPOSE_STEP,
    ENC_POST_MHSA_DENSE_AND_INPUT_SUM_STEP,
    ENC_PRE_MLP_TRANSPOSE_STEP,
    MLP_DENSE_1_STEP,
    ENC_POST_DENSE_1_TRANSPOSE_STEP,
    MLP_DENSE_2_AND_SUM_STEP,
    MLP_HEAD_PRE_DENSE_1_TRANSPOSE_STEP,
    MLP_HEAD_DENSE_1_STEP,
    MLP_HEAD_PRE_DENSE_2_TRANSPOSE_STEP,
    MLP_HEAD_DENSE_2_STEP,
    MLP_HEAD_PRE_SOFTMAX_TRANSPOSE_STEP,
    MLP_HEAD_SOFTMAX_STEP,
    POST_SOFTMAX_DIVIDE_STEP,
    POST_SOFTMAX_AVERAGING_STEP,
    RETIRE_SOFTMAX_STEP,
    INFERENCE_COMPLETE,
    INVALID_INF_STEP
} INFERENCE_STEP_T;

typedef enum logic [4:0] {
    PATCH_MEM,
    CLASS_TOKEN_MEM,
    POS_EMB_MEM,
    ENC_LN1_1ST_HALF_MEM,
    ENC_LN1_2ND_HALF_MEM,
    ENC_QVK_IN_MEM,
    ENC_Q_MEM,
    ENC_K_MEM,
    ENC_V_MEM,
    ENC_K_T_MEM,
    ENC_QK_T_IN_MEM,
    ENC_QK_T_MEM,
    ENC_PRE_SOFTMAX_MEM,
    ENC_V_MULT_IN_MEM,
    ENC_V_MULT_MEM,
    ENC_DENSE_IN_MEM,
    ENC_MHSA_OUT_MEM,
    ENC_LN2_1ST_HALF_MEM,
    ENC_LN2_2ND_HALF_MEM,
    ENC_MLP_IN_MEM,
    ENC_MLP_DENSE1_MEM,
    ENC_MLP_DENSE2_IN_MEM,
    ENC_MLP_OUT_MEM,
    MLP_HEAD_LN_1ST_HALF_MEM,
    MLP_HEAD_LN_2ND_HALF_MEM,
    MLP_HEAD_DENSE_1_IN_MEM,
    MLP_HEAD_DENSE_1_OUT_MEM,
    MLP_HEAD_DENSE_2_IN_MEM,
    MLP_HEAD_DENSE_2_OUT_MEM,
    MLP_HEAD_SOFTMAX_IN_MEM,
    SOFTMAX_AVG_SUM_MEM,
    PREV_SOFTMAX_OUTPUT_MEM
} TEMP_DATA_NAME_T;

/*----- LUT -----*/
logic [$clog2(TEMP_RES_STORAGE_SIZE_CIM)-1:0] mem_map [PREV_SOFTMAX_OUTPUT_MEM+'d1];
ParamInfo_t param_addr_map[PARAMS_NUM-1];

/*----- INTERFACE -----*/
typedef enum logic [2:0] {
    BUS_FSM,
    LOGIC_FSM,
    DATA_FILL_FSM,
    MAC,
    LAYERNORM,
    SOFTMAX,
    MEM_ACCESS_SRC_NUM
} MEM_ACCESS_SRC_T;

typedef enum logic {
    FIRST_HALF = 1'b0,
    SECOND_HALF = 1'b1
} LAYERNORM_HALF_SELECT_T;

/* verilator lint_off DECLFILENAME */
interface MemAccessSignals;
    wire [MEM_ACCESS_SRC_NUM-1:0] read_req_src, write_req_src; // One-hot signal indicating the source of the read or write request
    wire [$clog2(TEMP_RES_STORAGE_SIZE_CIM)-1:0] addr_table [MEM_ACCESS_SRC_NUM]; // Contains the addresses that different part of the CiM want to read/write
    wire [N_STORAGE-1:0] write_data [MEM_ACCESS_SRC_NUM]; // Data to be written to the memory
endinterface
/* verilator lint_on DECLFILENAME */
`endif
