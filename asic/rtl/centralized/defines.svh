`ifndef _defines_vh_
`define _defines_vh_

package Defines;
    /* ----- Constants ----- */
    localparam int CIM_PARAMS_BANK_SIZE_NUM_WORD    = 15872; // Need 2 banks
    localparam int CIM_INT_RES_BANK_SIZE_NUM_WORD   = 14336; // Need 4 banks
    localparam int CIM_PARAMS_NUM_BANKS             = 2;
    localparam int CIM_INT_RES_NUM_BANKS            = 4;
    localparam int N_STO_INT_RES                    = 9;
    localparam int N_STO_PARAMS                     = 9;

    localparam int EMB_DEPTH = 64;
    localparam int NUM_PATCHES = 60;
    localparam int PATCH_LEN = 64;
    localparam int MLP_DIM = 32;
    localparam int NUM_HEADS = 8;
    localparam int NUM_SLEEP_STAGES = 5;
    localparam int NUM_SAMPLES_OUT_AVG = 3;

    /* ----- Types ----- */
    typedef logic [$clog2(CIM_INT_RES_NUM_BANKS*CIM_PARAMS_BANK_SIZE_NUM_WORD)-1:0] IntResAddr_t;
    typedef logic [$clog2(CIM_PARAMS_NUM_BANKS*CIM_INT_RES_BANK_SIZE_NUM_WORD)-1:0] ParamAddr_t;
    typedef logic [$clog2(CIM_INT_RES_BANK_SIZE_NUM_WORD)-1:0]                      IntResBankAddr_t;
    typedef logic [$clog2(CIM_PARAMS_BANK_SIZE_NUM_WORD)-1:0]                       ParamBankAddr_t;
    typedef logic [N_STO_PARAMS-1:0]                                                Param_t;
    typedef logic [N_STO_INT_RES-1:0]                                               IntResSingle_t;
    typedef logic [2*N_STO_INT_RES-1:0]                                             IntResDouble_t;

    /* ----- Enum ----- */
    typedef enum logic {
        SINGLE_WIDTH,
        DOUBLE_WIDTH
    } DataWidth_t;

    typedef enum logic [1:0] {
        IDLE_CIM,
        INFERENCE_RUNNING,
        INVALID_CIM
    } State_t;

    typedef enum logic [4:0] {
        PATCH_PROJ_STEP,
        CLASS_TOKEN_CONCAT_STEP,
        POS_EMB_STEP,
        ENC_LAYERNORM_1_1ST_HALF_STEP,
        ENC_LAYERNORM_1_2ND_HALF_STEP,
        POS_EMB_COMPRESSION_STEP,
        ENC_MHSA_Q_STEP,
        ENC_MHSA_K_STEP,
        ENC_MHSA_V_STEP,
        ENC_MHSA_QK_T_STEP,
        ENC_MHSA_SOFTMAX_STEP,
        ENC_MHSA_MULT_V_STEP,
        ENC_POST_MHSA_DENSE_AND_INPUT_SUM_STEP,
        ENC_LAYERNORM_2_1ST_HALF_STEP,
        ENC_LAYERNORM_2_2ND_HALF_STEP,
        MLP_DENSE_1_STEP,
        MLP_DENSE_2_AND_SUM_STEP,
        ENC_LAYERNORM_3_1ST_HALF_STEP,
        ENC_LAYERNORM_3_2ND_HALF_STEP,
        MLP_HEAD_DENSE_1_STEP,
        MLP_HEAD_DENSE_2_STEP,
        MLP_HEAD_SOFTMAX_STEP,
        SOFTMAX_DIVIDE_STEP,
        SOFTMAX_AVERAGING_STEP,
        SOFTMAX_AVERAGE_ARGMAX_STEP,
        SOFTMAX_RETIRE_STEP,
        INFERENCE_COMPLETE,
        INVALID_STEP
    } InferenceStep_t;

    typedef enum int {
        EEG_INPUT_MEM,
        PATCH_MEM,
        CLASS_TOKEN_MEM,
        POS_EMB_MEM,
        ENC_LN1_MEM,
        ENC_Q_MEM,
        ENC_K_MEM,
        ENC_V_MEM,
        ENC_QK_T_MEM,
        ENC_V_MULT_MEM,
        ENC_MHSA_OUT_MEM,
        ENC_LN2_MEM,
        ENC_MLP_DENSE1_MEM,
        ENC_MLP_OUT_MEM,
        ENC_LN3_MEM,
        MLP_HEAD_DENSE_1_OUT_MEM,
        MLP_HEAD_DENSE_2_OUT_MEM,
        SOFTMAX_AVG_SUM_MEM,
        PREV_SOFTMAX_OUTPUT_MEM,
        NUM_DATA_ADDR
    } DataAddr_t;

    /* ----- ADDRESSES ----- */
    localparam int mem_map [NUM_DATA_ADDR] = '{
        0, // EEG_INPUT_MEM
        SINGLE_WIDTH*NUM_PATCHES*PATCH_LEN + PATCH_LEN, // PATCH_MEM
        SINGLE_WIDTH*NUM_PATCHES*PATCH_LEN, // CLASS_TOKEN_MEM
        0, // POS_EMB_MEM
        20000, // ENC_LN1_MEM
        2*(NUM_PATCHES+1)*EMB_DEPTH, // ENC_Q_MEM
        2*(NUM_PATCHES+1)*EMB_DEPTH+(NUM_PATCHES+1)*EMB_DEPTH, // ENC_K_MEM
        (NUM_PATCHES+1)*EMB_DEPTH, // ENC_V_MEM
        4*(NUM_PATCHES+1)*EMB_DEPTH, // ENC_QK_T_MEM
        4*(NUM_PATCHES+1)*EMB_DEPTH + NUM_HEADS*(NUM_PATCHES+1)*(NUM_PATCHES+1), // ENC_V_MULT_MEM
        (NUM_PATCHES+1)*EMB_DEPTH, // ENC_MHSA_OUT_MEM
        0, // ENC_LN2_MEM
        EMB_DEPTH*EMB_DEPTH, // ENC_MLP_DENSE1_MEM
        0, // ENC_MLP_OUT_MEM
        EMB_DEPTH, // ENC_LN3_MEM
        0, // MLP_HEAD_DENSE_1_OUT_MEM
        MLP_DIM, // MLP_HEAD_DENSE_2_OUT_MEM
        0, // SOFTMAX_AVG_SUM_MEM
        CIM_INT_RES_NUM_BANKS*CIM_INT_RES_BANK_SIZE_NUM_WORD - NUM_SLEEP_STAGES*(NUM_SAMPLES_OUT_AVG-1) // PREV_SOFTMAX_OUTPUT_MEM
    };

endpackage

`endif // _defines_vh_