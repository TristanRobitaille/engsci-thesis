`ifndef _defines_svh_
`define _defines_svh_

package Defines;
    /* ----- Constants ----- */
    localparam int CIM_PARAMS_BANK_SIZE_NUM_WORD    = 15872; // Need 2 banks
    localparam int CIM_INT_RES_BANK_SIZE_NUM_WORD   = 14336; // Need 4 banks
    localparam int CIM_PARAMS_NUM_BANKS             = 2;
    localparam int CIM_INT_RES_NUM_BANKS            = 4;
    localparam int N_STO_INT_RES                    = 15;
    localparam int N_STO_PARAMS                     = 15;
    localparam int Q_STO_INT_RES_DOUBLE             = 2*N_STO_INT_RES - 10;
    localparam int N_COMP                           = 39;
    localparam int Q_COMP                           = 21;
    localparam int ADC_BITWIDTH                     = 16; // Assume that the ADC feeding EEG data is 16b unsigned integer

    localparam int EMB_DEPTH = 64;
    localparam int NUM_PATCHES = 60;
    localparam int PATCH_LEN = 64;
    localparam int MLP_DIM = 32;
    localparam int NUM_HEADS = 8;
    localparam int NUM_SLEEP_STAGES = 5;
    localparam int NUM_SAMPLES_OUT_AVG = 3;

    localparam int VECTOR_MAX_LEN = 64;

    /* ----- Types ----- */
    typedef logic [$clog2(CIM_INT_RES_NUM_BANKS*CIM_PARAMS_BANK_SIZE_NUM_WORD)-1:0] IntResAddr_t;
    typedef logic [$clog2(CIM_PARAMS_NUM_BANKS*CIM_INT_RES_BANK_SIZE_NUM_WORD)-1:0] ParamAddr_t;
    typedef logic [$clog2(CIM_INT_RES_BANK_SIZE_NUM_WORD)-1:0]                      IntResBankAddr_t;
    typedef logic [$clog2(CIM_PARAMS_BANK_SIZE_NUM_WORD)-1:0]                       ParamBankAddr_t;
    typedef logic [$clog2(VECTOR_MAX_LEN+1)-1:0]                                    VectorLen_t;
    typedef logic signed [N_STO_PARAMS-1:0]                                         Param_t;
    typedef logic signed [N_STO_INT_RES-1:0]                                        IntResSingle_t;
    typedef logic signed [2*N_STO_INT_RES-1:0]                                      IntResDouble_t;
    typedef logic signed [N_COMP-1:0]                                               CompFx_t;
    typedef logic signed                                                            FxFormat_Unused_t; // Needed for MemoryInterface to instantiate correctly for banks
    typedef logic [ADC_BITWIDTH-1:0]                                                AdcData_t;

    /* ----- Enum ----- */
    typedef enum logic {
        POSEDGE_TRIGGERED,
        LEVEL_TRIGGERED
    } CounterMode_t;

    typedef enum logic {
        SINGLE_WIDTH,
        DOUBLE_WIDTH
    } DataWidth_t;

    typedef enum logic {
        FIRST_HALF,
        SECOND_HALF
    } HalfSelect_t;

    typedef enum logic [2:0] {
        INT_RES_SW_FX_1_X,
        INT_RES_SW_FX_2_X,
        INT_RES_SW_FX_5_X,
        INT_RES_SW_FX_6_X,
        INT_RES_DW_FX
    } FxFormatIntRes_t;

    typedef enum logic [1:0] {
        PARAMS_FX_2_X,
        PARAMS_FX_3_X,
        PARAMS_FX_4_X,
        PARAMS_FX_5_X
    } FxFormatParams_t;

    typedef enum logic [1:0] {
        IDLE_CIM,
        EEG_LOAD,
        INFERENCE_RUNNING,
        INVALID_CIM
    } State_t;

    typedef enum logic [1:0] {
        MODEL_PARAM,
        INTERMEDIATE_RES,
        IMMEDIATE_VAL,
        ADC_INPUT
    } ParamType_t;

    typedef enum logic [1:0] {
        NO_ACTIVATION,
        LINEAR_ACTIVATION,
        SWISH_ACTIVATION
    } Activation_t;

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
        NUM_DATA_STEP
    } DataStep_t;

    typedef enum int {
        PATCH_PROJ_KERNEL_PARAMS,
        POS_EMB_PARAMS,
        ENC_Q_DENSE_PARAMS,
        ENC_K_DENSE_PARAMS,
        ENC_V_DENSE_PARAMS,
        ENC_COMB_HEAD_PARAMS,
        ENC_MLP_DENSE_1_PARAMS,
        ENC_MLP_DENSE_2_PARAMS,
        MLP_HEAD_DENSE_1_PARAMS,
        MLP_HEAD_DENSE_2_PARAMS,
        SINGLE_PARAMS,
        NUM_PARAM_STEP
    } ParamStep_t;

    typedef enum int {
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
        ENC_MLP_DENSE_1_BIAS_OFF,
        ENC_MLP_DENSE_2_BIAS_OFF,
        ENC_LAYERNORM_3_GAMMA_OFF,
        ENC_LAYERNORM_3_BETA_OFF,
        MLP_HEAD_DENSE_1_BIAS_OFF,
        MLP_HEAD_DENSE_2_BIAS_OFF,
        NUM_PARAM_BIAS_STEP
    } ParamBiasStep_t;

    /* ----- ADDRESSES ----- */
    const IntResAddr_t mem_map [NUM_DATA_STEP] = '{
        IntResAddr_t'(0),                                                                                               // EEG_INPUT_MEM
        IntResAddr_t'(NUM_PATCHES*PATCH_LEN + PATCH_LEN),                                                               // PATCH_MEM
        IntResAddr_t'(NUM_PATCHES*PATCH_LEN),                                                                           // CLASS_TOKEN_MEM
        IntResAddr_t'(0),                                                                                               // POS_EMB_MEM
        IntResAddr_t'(20000),                                                                                           // ENC_LN1_MEM
        IntResAddr_t'(2*(NUM_PATCHES+1)*EMB_DEPTH),                                                                     // ENC_Q_MEM
        IntResAddr_t'(2*(NUM_PATCHES+1)*EMB_DEPTH+(NUM_PATCHES+1)*EMB_DEPTH),                                           // ENC_K_MEM
        IntResAddr_t'((NUM_PATCHES+1)*EMB_DEPTH),                                                                       // ENC_V_MEM
        IntResAddr_t'(4*(NUM_PATCHES+1)*EMB_DEPTH),                                                                     // ENC_QK_T_MEM
        IntResAddr_t'(4*(NUM_PATCHES+1)*EMB_DEPTH + NUM_HEADS*(NUM_PATCHES+1)*(NUM_PATCHES+1)),                         // ENC_V_MULT_MEM
        IntResAddr_t'((NUM_PATCHES+1)*EMB_DEPTH),                                                                       // ENC_MHSA_OUT_MEM
        IntResAddr_t'(0),                                                                                               // ENC_LN2_MEM
        IntResAddr_t'(EMB_DEPTH*EMB_DEPTH),                                                                             // ENC_MLP_DENSE1_MEM
        IntResAddr_t'(0),                                                                                               // ENC_MLP_OUT_MEM
        IntResAddr_t'(EMB_DEPTH),                                                                                       // ENC_LN3_MEM
        IntResAddr_t'(0),                                                                                               // MLP_HEAD_DENSE_1_OUT_MEM
        IntResAddr_t'(MLP_DIM),                                                                                         // MLP_HEAD_DENSE_2_OUT_MEM
        IntResAddr_t'(0),                                                                                               // SOFTMAX_AVG_SUM_MEM
        IntResAddr_t'(CIM_INT_RES_NUM_BANKS*CIM_INT_RES_BANK_SIZE_NUM_WORD - NUM_SLEEP_STAGES*(NUM_SAMPLES_OUT_AVG-1))  // PREV_SOFTMAX_OUTPUT_MEM
    };

    const ParamAddr_t param_addr_map [NUM_PARAM_STEP] = '{
        0,      // PATCH_PROJ_KERNEL_PARAMS
        4096,   // POS_EMB_PARAMS
        8000,   // ENC_Q_DENSE_PARAMS
        12096,  // ENC_K_DENSE_PARAMS
        16192,  // ENC_V_DENSE_PARAMS
        20288,  // ENC_COMB_HEAD_PARAMS
        24384,  // ENC_MLP_DENSE_1_PARAMS
        26432,  // ENC_MLP_DENSE_2_PARAMS
        28480,  // MLP_HEAD_DENSE_1_PARAMS
        30528,  // MLP_HEAD_DENSE_2_PARAMS
        30688   // SINGLE_PARAMS
    };

    const ParamAddr_t param_addr_map_bias [NUM_PARAM_BIAS_STEP] = '{
        param_addr_map[NUM_PARAM_STEP-1],       // PATCH_PROJ_BIAS_OFF
        param_addr_map[NUM_PARAM_STEP-1] + 64,  // CLASS_TOKEN_OFF
        param_addr_map[NUM_PARAM_STEP-1] + 128, // ENC_LAYERNORM_1_GAMMA_OFF
        param_addr_map[NUM_PARAM_STEP-1] + 192, // ENC_LAYERNORM_1_BETA_OFF
        param_addr_map[NUM_PARAM_STEP-1] + 256, // ENC_Q_DENSE_BIAS_0FF
        param_addr_map[NUM_PARAM_STEP-1] + 320, // ENC_K_DENSE_BIAS_0FF
        param_addr_map[NUM_PARAM_STEP-1] + 384, // ENC_V_DENSE_BIAS_0FF
        param_addr_map[NUM_PARAM_STEP-1] + 448, // ENC_INV_SQRT_NUM_HEADS_OFF
        param_addr_map[NUM_PARAM_STEP-1] + 449, // ENC_COMB_HEAD_BIAS_OFF
        param_addr_map[NUM_PARAM_STEP-1] + 513, // ENC_LAYERNORM_2_GAMMA_OFF
        param_addr_map[NUM_PARAM_STEP-1] + 577, // ENC_LAYERNORM_2_BETA_OFF
        param_addr_map[NUM_PARAM_STEP-1] + 641, // ENC_MLP_DENSE_1_BIAS_OFF
        param_addr_map[NUM_PARAM_STEP-1] + 673, // ENC_MLP_DENSE_2_BIAS_OFF
        param_addr_map[NUM_PARAM_STEP-1] + 737, // ENC_LAYERNORM_3_GAMMA_OFF
        param_addr_map[NUM_PARAM_STEP-1] + 801, // ENC_LAYERNORM_3_BETA_OFF
        param_addr_map[NUM_PARAM_STEP-1] + 865, // MLP_HEAD_DENSE_1_BIAS_OFF
        param_addr_map[NUM_PARAM_STEP-1] + 897  // MLP_HEAD_DENSE_2_BIAS_OFF
    };

    /* ----- CASTS ----- */
    typedef enum int {
        EEG_WIDTH,
        PATCH_PROJ_OUTPUT_WIDTH,
        CLASS_EMB_TOKEN_WIDTH,
        NUM_INT_RES_WIDTHS
    } IntResWidth_t;

    typedef enum int {
        EEG_FORMAT,
        PATCH_PROJ_OUTPUT_FORMAT,
        CLASS_EMB_TOKEN_FORMAT,
        NUM_INT_RES_FORMATS
    } IntResFormat_t;

    typedef enum int {
        PATCH_PROJ_PARAM_FORMAT,
        CLASS_EMB_TOKEN_PARAM_FORMAT,
        NUM_PARAMS_FORMATS
    } ParamWidth_t;

    const DataWidth_t int_res_width [NUM_INT_RES_WIDTHS] = '{
        DOUBLE_WIDTH, // EEG_WIDTH
        DOUBLE_WIDTH, // PATCH_PROJ_OUTPUT_WIDTH
        DOUBLE_WIDTH  // CLASS_TOKEN_WIDTH
    };

    const FxFormatIntRes_t int_res_format [NUM_INT_RES_FORMATS] = '{
        INT_RES_DW_FX, // EEG_FORMAT
        INT_RES_DW_FX, // PATCH_PROJ_OUTPUT_FORMAT
        INT_RES_DW_FX  // CLASS_EMB_TOKEN_FORMAT
    };

    const FxFormatParams_t params_format [NUM_PARAMS_FORMATS] = '{
        PARAMS_FX_2_X, // PATCH_PROJ_PARAM_FORMAT
        PARAMS_FX_2_X  // CLASS_EMB_TOKEN_PARAM_FORMAT
    };

endpackage
`endif // _defines_svh_
