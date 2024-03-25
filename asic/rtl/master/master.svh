`ifndef _master_vh_
`define _master_vh_

`include "../types.svh"

/*----- ENUM -----*/
    typedef enum logic [2:0] {
        MASTER_STATE_IDLE,
        MASTER_STATE_PARAM_LOAD,
        MASTER_STATE_SIGNAL_LOAD,
        MASTER_STATE_INFERENCE_RUNNING,
        MASTER_STATE_BROADCAST_MANAGEMENT,
        MASTER_STATE_DONE_INFERENCE
    } MASTER_STATE_T;

    typedef enum logic [4:0] {
        PRE_LAYERNORM_1_TRANS_STEP,
        INTRA_LAYERNORM_1_TRANS_STEP,
        POST_LAYERNORM_1_TRANS_STEP,
        ENC_MHSA_DENSE_STEP,
        ENC_MHSA_Q_TRANS_STEP,
        ENC_MHSA_K_TRANS_STEP,
        ENC_MHSA_QK_T_STEP,
        ENC_MHSA_PRE_SOFTMAX_TRANS_STEP,
        ENC_MHSA_SOFTMAX_STEP,
        ENC_MHSA_V_MULT_STEP,
        ENC_MHSA_POST_V_TRANS_STEP,
        ENC_MHSA_POST_V_DENSE_STEP,
        PRE_LAYERNORM_2_TRANS_STEP,
        INTRA_LAYERNORM_2_TRANS_STEP,
        ENC_PRE_MLP_TRANSPOSE_STEP,
        ENC_MLP_DENSE_1_STEP,
        ENC_MLP_DENSE_2_TRANSPOSE_STEP,
        ENC_MLP_DENSE_2_AND_SUM_STEP,
        PRE_LAYERNORM_3_TRANS_STEP,
        INTRA_LAYERNORM_3_TRANS_STEP,
        PRE_MLP_HEAD_DENSE_TRANS_STEP,
        MLP_HEAD_DENSE_1_STEP,
        PRE_MLP_HEAD_DENSE_2_TRANS_STEP,
        MLP_HEAD_DENSE_2_STEP,
        MLP_HEAD_SOFTMAX_TRANS_STEP,
        SOFTMAX_AVERAGING,
        INFERENCE_FINISHED,
        HIGH_LEVEL_INFERENCE_STEP_NUM
    } HIGH_LEVEL_INFERENCE_STEP_T;

    typedef enum logic [4:0] {
        PATCH_PROJ_KERNEL_EXT_MEM,
        PATCH_PROJ_BIAS_EXT_MEM,
        POS_EMB_EXT_MEM,
        ENC_Q_DENSE_KERNEL_EXT_MEM,
        ENC_Q_DENSE_BIAS_EXT_MEM,
        ENC_K_DENSE_KERNEL_EXT_MEM,
        ENC_K_DENSE_BIAS_EXT_MEM,
        ENC_V_DENSE_KERNEL_EXT_MEM,
        ENC_V_DENSE_BIAS_EXT_MEM,
        ENC_COMB_HEAD_KERNEL_EXT_MEM,
        ENC_COMB_HEAD_BIAS_EXT_MEM,
        MLP_HEAD_DENSE_1_KERNEL_EXT_MEM,
        MLP_HEAD_DENSE_1_BIAS_EXT_MEM,
        MLP_DENSE_1_KERNEL_EXT_MEM,
        MLP_DENSE_1_BIAS_EXT_MEM,
        MLP_DENSE_2_KERNEL_EXT_MEM,
        MLP_DENSE_2_BIAS_EXT_MEM,
        CLASS_EMB_EXT_MEM,
        ENC_LAYERNORM_1_BETA_EXT_MEM,
        ENC_LAYERNORM_1_GAMMA_EXT_MEM,
        ENC_LAYERNORM_2_BETA_EXT_MEM,
        ENC_LAYERNORM_2_GAMMA_EXT_MEM,
        MLP_HEAD_LAYERNORM_BETA_EXT_MEM,
        MLP_HEAD_LAYERNORM_GAMMA_EXT_MEM,
        MLP_HEAD_DENSE_SOFTMAX_KERNEL_EXT_MEM,
        MLP_HEAD_DENSE_SOFTMAX_BIAS_EXT_MEM,
        SQRT_NUM_HEAD_EXT_MEM,
        EXT_MEM_PARAM_NUM
    } EXT_MEM_PARAM_NAME_T;

/*----- STRUCT -----*/
    typedef struct packed {
        logic [$clog2(OP_NUM)-1:0] op;
        TEMP_RES_ADDR_T tx_addr;
        logic [$clog2(NUM_CIMS+1)-1:0] len;
        TEMP_RES_ADDR_T rx_addr;
        logic [$clog2(NUM_CIMS+1)-1:0] num_cim;
    } BroadcastOpInfo_t;

/*----- LUT -----*/
    ParamInfo_t param_addr_map[PARAMS_NUM-1];
    BroadcastOpInfo_t broadcast_ops[HIGH_LEVEL_INFERENCE_STEP_NUM];
    logic [$clog2(NUM_PARAMS/64)-1:0] ext_mem_param_addr_map[EXT_MEM_PARAM_NUM];

`endif
