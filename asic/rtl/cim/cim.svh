`ifndef _cim_svh_
`define _cim_svh_

`include "types.svh"

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
    CLASS_TOKEN_CONCAT_STEP_CIM                 = 'd0,
    POS_EMB_STEP_CIM                            = 'd1,
    ENC_LAYERNORM_1_1ST_HALF_STEP_CIM           = 'd2,
    ENC_LAYERNORM_1_2ND_HALF_STEP_CIM           = 'd3,
    ENC_LAYERNORM_2_1ST_HALF_STEP_CIM           = 'd4,
    ENC_LAYERNORM_2_2ND_HALF_STEP_CIM           = 'd5,
    ENC_LAYERNORM_3_1ST_HALF_STEP_CIM           = 'd6,
    ENC_LAYERNORM_3_2ND_HALF_STEP_CIM           = 'd7,
    POST_LAYERNORM_TRANSPOSE_STEP_CIM           = 'd8,
    ENC_MHSA_DENSE_STEP_CIM                     = 'd9,
    ENC_MHSA_Q_TRANSPOSE_STEP_CIM               = 'd10,
    ENC_MHSA_K_TRANSPOSE_STEP_CIM               = 'd11,
    ENC_MHSA_QK_T_STEP_CIM                      = 'd12,
    ENC_MHSA_PRE_SOFTMAX_TRANSPOSE_STEP_CIM     = 'd13,
    ENC_MHSA_SOFTMAX_STEP_CIM                   = 'd14,
    ENC_MHSA_MULT_V_STEP_CIM                    = 'd15,
    ENC_POST_MHSA_TRANSPOSE_STEP_CIM            = 'd16,
    ENC_POST_MHSA_DENSE_AND_INPUT_SUM_STEP_CIM  = 'd17,
    ENC_PRE_MLP_TRANSPOSE_STEP_CIM              = 'd18,
    MLP_DENSE_1_STEP_CIM                        = 'd19,
    ENC_POST_DENSE_1_TRANSPOSE_STEP_CIM         = 'd20,
    MLP_DENSE_2_AND_SUM_STEP_CIM                = 'd21,
    MLP_HEAD_PRE_DENSE_1_TRANSPOSE_STEP_CIM     = 'd22,
    MLP_HEAD_DENSE_1_STEP_CIM                   = 'd23,
    MLP_HEAD_PRE_DENSE_2_TRANSPOSE_STEP_CIM     = 'd24,
    MLP_HEAD_DENSE_2_STEP_CIM                   = 'd25,
    MLP_HEAD_PRE_SOFTMAX_TRANSPOSE_STEP_CIM     = 'd26,
    MLP_HEAD_SOFTMAX_STEP_CIM                   = 'd27,
    POST_SOFTMAX_DIVIDE_STEP_CIM                = 'd28,
    POST_SOFTMAX_AVERAGING_STEP_CIM             = 'd29,
    POST_SOFTMAX_ARGMAX_STEP_CIM                = 'd30,
    RETIRE_SOFTMAX_STEP_CIM                     = 'd31,
    INFERENCE_COMPLETE_CIM                      = 'd32,
    INVALID_INF_STEP_CIM
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

/*----- INTERFACE -----*/
typedef enum logic [2:0] {
    BUS_FSM                     = 'd0,
    LOGIC_FSM                   = 'd1,
    DATA_FILL_FSM               = 'd2,
    DENSE_BROADCAST_SAVE_FSM    = 'd3,
    MAC                         = 'd4,
    LAYERNORM                   = 'd5,
    SOFTMAX                     = 'd6,
    MEM_ACCESS_SRC_NUM          = 'd7
} MEM_ACCESS_SRC_T;

typedef enum logic {
    FIRST_HALF = 1'b0,
    SECOND_HALF = 1'b1
} LAYERNORM_HALF_SELECT_T;

/* verilator lint_off DECLFILENAME */
interface MemAccessSignals;
    logic [MEM_ACCESS_SRC_NUM-1:0] read_req_src, write_req_src; // One-hot signal indicating the source of the read or write request
    TEMP_RES_ADDR_T addr_table [MEM_ACCESS_SRC_NUM]; // Contains the addresses that different part of the CiM want to read/write
    STORAGE_WORD_T write_data [MEM_ACCESS_SRC_NUM]; // Data to be written to the memory
endinterface
/* verilator lint_on DECLFILENAME */
`endif
