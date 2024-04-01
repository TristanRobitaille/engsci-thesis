`ifndef _types_vh_
`define _types_vh_

`include "parameters.svh"
/* verilator lint_off IMPORTSTAR */
import parameters::*;
/* verilator lint_on IMPORTSTAR */

/*----- TYPES -----*/
typedef logic [$clog2(TEMP_RES_STORAGE_SIZE_CIM)-1:0] TEMP_RES_ADDR_T;
typedef logic [$clog2(PARAMS_STORAGE_SIZE_CIM)-1:0] PARAMS_ADDR_T;
typedef logic [$clog2(NUM_SLEEP_STAGES)-1:0] SLEEP_STAGE_T;
typedef logic signed [N_COMP-1:0] COMP_WORD_T;
typedef logic signed [N_STORAGE-1:0] STORAGE_WORD_T;

/*----- STRUCT -----*/
typedef struct packed {
    PARAMS_ADDR_T addr;
    logic [$clog2(NUM_CIMS+1)-1:0] len;
    logic [$clog2(NUM_CIMS+1)-1:0] num_rec;
} ParamInfo_t;

/*----- ENUMS -----*/
typedef enum reg [BUS_OP_WIDTH-1:0] {
    NOP,
    PATCH_LOAD_BROADCAST_START_OP,
    PATCH_LOAD_BROADCAST_OP,
    DENSE_BROADCAST_START_OP,
    DENSE_BROADCAST_DATA_OP,
    PARAM_STREAM_START_OP,
    PARAM_STREAM_OP,
    TRANS_BROADCAST_START_OP,
    TRANS_BROADCAST_DATA_OP,
    PISTOL_START_OP,
    INFERENCE_RESULT_OP,
    OP_NUM // Number of ops
} BUS_OP_T;

typedef enum logic {
    RST = 1'b0,
    RUN = 1'b1
} RESET_STATE_T;

typedef enum logic {
    MODEL_PARAM = 'd0,
    INTERMEDIATE_RES = 'd1
} PARAM_TYPE_T;

typedef enum logic [1:0] {
    NO_ACTIVATION = 2'd0,
    LINEAR_ACTIVATION = 2'd1,
    SWISH_ACTIVATION = 2'd2
} ACTIVATION_TYPE_T;

typedef enum logic [3:0] {
    POS_EMB_PARAMS,
    PATCH_PROJ_KERNEL_PARAMS,
    ENC_Q_DENSE_KERNEL_PARAMS,
    ENC_K_DENSE_KERNEL_PARAMS,
    ENC_V_DENSE_KERNEL_PARAMS,
    ENC_COMB_HEAD_KERNEL_PARAMS,
    ENC_MLP_DENSE_1_OR_MLP_HEAD_DENSE_1_KERNEL_PARAMS,
    ENC_MLP_DENSE_2_KERNEL_PARAMS,
    MLP_HEAD_DENSE_2_KERNEL_PARAMS,
    SINGLE_PARAMS,
    PARAM_LOAD_FINISHED,
    PARAMS_NUM
} PARAM_NAME_T;

typedef enum PARAMS_ADDR_T {
    PATCH_PROJ_BIAS_OFF,
    CLASS_TOKEN_OFF,
    ENC_LAYERNORM_1_GAMMA_OFF,
    ENC_LAYERNORM_1_BETA_OFF,
    ENC_Q_DENSE_BIAS_0FF,
    ENC_K_DENSE_BIAS_0FF,
    ENC_V_DENSE_BIAS_0FF,
    ENC_INV_SQRT_NUM_HEADS_OFF,
    ENC_COMB_HEAD_BIAS_OFF,
    ENC_LAYERNORM_2_GAMMA_OFF,
    ENC_LAYERNORM_2_BETA_OFF,
    ENC_MLP_DENSE_1_MLP_HEAD_DENSE_1_BIAS_OFF, // This is the same offset for both encoder's MLP's dense 1 bias and MLP head's dense 1 bias
    ENC_MLP_DENSE_2_BIAS_OFF,
    ENC_LAYERNORM_3_GAMMA_OFF,
    ENC_LAYERNORM_3_BETA_OFF,
    MLP_HEAD_DENSE_2_BIAS_OFF
} SINGLE_PARAM_OFFSET_T;

`endif
