#ifndef PARAM_LAYER_MAPPING_H
#define PARAM_LAYER_MAPPING_H

#include <map>

/*----- ENUM -----*/
enum PARAM_NAME { // Master goes through these layers sequentially, loading weights into the appropriate CiM
    PATCH_PROJ_KERNEL_PARAMS,
    POS_EMB_PARAMS,
    ENC_Q_DENSE_PARAMS,
    ENC_K_DENSE_PARAMS,
    ENC_V_DENSE_PARAMS,
    ENC_COMB_HEAD_PARAMS,
    ENC_MLP_DENSE_1_OR_MLP_HEAD_DENSE_1_PARAMS, // Because these two share the same space in different CiMs, we will use the same enum for both
    ENC_MLP_DENSE_2_PARAMS,
    MLP_HEAD_DENSE_2_PARAMS,    
    SINGLE_PARAMS, // The 1-element parameters stored at the bottom of the storage
    PARAM_LOAD_FINISHED,
#if CENTRALIZED_ARCH
    MLP_HEAD_DENSE_1_PARAMS,
    ENC_MLP_DENSE_1_PARAMS,
#endif
};

enum SINGLE_PARAM_OFFSET {
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
    MLP_HEAD_DENSE_2_BIAS_OFF,
#if CENTRALIZED_ARCH
    MLP_HEAD_DENSE_1_BIAS_OFF,
    ENC_MLP_DENSE_1_BIAS_OFF,
#endif
};

enum DATA {
    EEG_INPUT_MEM,
    PATCH_MEM,
    CLASS_TOKEN_MEM,
    POS_EMB_MEM,
#if DISTRIBUTED_ARCH
    ENC_LN1_1ST_HALF_MEM,
    ENC_LN1_2ND_HALF_MEM,
    ENC_LN2_1ST_HALF_MEM,
    ENC_LN2_2ND_HALF_MEM,
    MLP_HEAD_LN_1ST_HALF_MEM,
    MLP_HEAD_LN_2ND_HALF_MEM,
#elif CENTRALIZED_ARCH
    ENC_LN1_MEM,
    ENC_LN2_MEM,
    ENC_LN3_MEM,
#endif
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
    ENC_MLP_IN_MEM,
    ENC_MLP_DENSE1_MEM,
    ENC_MLP_DENSE2_IN_MEM,
    ENC_MLP_OUT_MEM,
    MLP_HEAD_DENSE_1_IN_MEM,
    MLP_HEAD_DENSE_1_OUT_MEM,
    MLP_HEAD_DENSE_2_IN_MEM,
    MLP_HEAD_DENSE_2_OUT_MEM,
    MLP_HEAD_SOFTMAX_IN_MEM,
    SOFTMAX_AVG_SUM_MEM,
    PREV_SOFTMAX_OUTPUT_MEM
};

/*----- TYPEDEF -----*/
typedef std::array<float, EMB_DEPTH> EmbDepthVect_t;
typedef std::array<float, MLP_DIM> MlpDimVect_t;
typedef std::array<std::array<float, EMB_DEPTH>, PATCH_LEN> PatchProjKernel_t;
typedef std::array<std::array<float, EMB_DEPTH>, NUM_PATCHES+1> PosEmb_t;
typedef std::array<std::array<float, EMB_DEPTH>, EMB_DEPTH> EncEmbDepthMat_t;
typedef std::array<std::array<float, EMB_DEPTH>, 3> EncEmbDepthVect3_t;
typedef std::array<std::array<float, MLP_DIM>, EMB_DEPTH> EmbDepthxMlpDimMat_t;
typedef std::array<std::array<float, EMB_DEPTH>, MLP_DIM> EncMlpDimxEmbDepthMat_t;
typedef std::array<std::array<float, NUM_SLEEP_STAGES>, MLP_DIM> NumSleepStagesxMlpDimMat_t;
typedef std::array<float, NUM_SLEEP_STAGES> NumSleepStagesVect_t;

#if DISTRIBUTED_ARCH
/*----- MAP -----*/
const std::map<DATA, uint32_t> mem_map = {
    {EEG_INPUT_MEM,             0},
    {PATCH_MEM,                 PATCH_LEN+DOUBLE_WIDTH},
    {CLASS_TOKEN_MEM,           PATCH_LEN},
    {POS_EMB_MEM,               0},
    {ENC_LN1_1ST_HALF_MEM,      DOUBLE_WIDTH*(NUM_PATCHES+1)},
    {ENC_LN1_2ND_HALF_MEM,      DOUBLE_WIDTH*(NUM_PATCHES+1+EMB_DEPTH)},
    {ENC_QVK_IN_MEM,            DOUBLE_WIDTH*(2*(NUM_PATCHES+1)+2*EMB_DEPTH)},
    {ENC_Q_MEM,                 DOUBLE_WIDTH*(2*(NUM_PATCHES+1)+3*EMB_DEPTH)},
    {ENC_K_MEM,                 DOUBLE_WIDTH*(3*(NUM_PATCHES+1)+3*EMB_DEPTH)},
    {ENC_V_MEM,                 NUM_PATCHES+1},
    {ENC_K_T_MEM,               DOUBLE_WIDTH*(EMB_DEPTH)+EMB_DEPTH+NUM_PATCHES+1},
    {ENC_QK_T_IN_MEM,           DOUBLE_WIDTH*(EMB_DEPTH)+2*EMB_DEPTH+NUM_PATCHES+1},
    {ENC_QK_T_MEM,              DOUBLE_WIDTH*(EMB_DEPTH)+2*EMB_DEPTH+2*(NUM_PATCHES+1)}, // This address + NUM_HEADS*(NUM_PATCHES+1) is the determining factor in the total memory requirement
    {ENC_PRE_SOFTMAX_MEM,       DOUBLE_WIDTH*(EMB_DEPTH)+NUM_PATCHES+1},
    {ENC_V_MULT_IN_MEM,         DOUBLE_WIDTH*(EMB_DEPTH)+8*EMB_DEPTH+NUM_PATCHES+1},
    {ENC_V_MULT_MEM,            DOUBLE_WIDTH*(EMB_DEPTH)+9*EMB_DEPTH+NUM_PATCHES+1},
    {ENC_DENSE_IN_MEM,          DOUBLE_WIDTH*(NUM_PATCHES+1+EMB_DEPTH)},
    {ENC_MHSA_OUT_MEM,          DOUBLE_WIDTH*(2*EMB_DEPTH+NUM_PATCHES+1)},
    {ENC_LN2_1ST_HALF_MEM,      DOUBLE_WIDTH*(NUM_PATCHES+1)},
    {ENC_LN2_2ND_HALF_MEM,      DOUBLE_WIDTH*(NUM_PATCHES+1+EMB_DEPTH)},
    {ENC_MLP_IN_MEM,            DOUBLE_WIDTH*(NUM_PATCHES+1)},
    {ENC_MLP_DENSE1_MEM,        DOUBLE_WIDTH*(NUM_PATCHES+1+EMB_DEPTH)},
    {ENC_MLP_DENSE2_IN_MEM,     DOUBLE_WIDTH*(NUM_PATCHES+1)},
    {ENC_MLP_OUT_MEM,           DOUBLE_WIDTH*(3*EMB_DEPTH+NUM_PATCHES+2)},
    {MLP_HEAD_LN_1ST_HALF_MEM,  DOUBLE_WIDTH*(0)},
    {MLP_HEAD_LN_2ND_HALF_MEM,  DOUBLE_WIDTH*(EMB_DEPTH)},
    {MLP_HEAD_DENSE_1_IN_MEM,   DOUBLE_WIDTH*(EMB_DEPTH)},
    {MLP_HEAD_DENSE_1_OUT_MEM,  DOUBLE_WIDTH*(2*EMB_DEPTH)},
    {MLP_HEAD_DENSE_2_IN_MEM,   DOUBLE_WIDTH*(EMB_DEPTH)},
    {MLP_HEAD_DENSE_2_OUT_MEM,  DOUBLE_WIDTH*(2*EMB_DEPTH)},
    {MLP_HEAD_SOFTMAX_IN_MEM,   DOUBLE_WIDTH*(MLP_DIM)},
    {PREV_SOFTMAX_OUTPUT_MEM,   866}, // Only relevant for CiM #0
    {SOFTMAX_AVG_SUM_MEM,       DOUBLE_WIDTH*(2*EMB_DEPTH)}
};

/*----- STRUCT -----*/
struct ParamInfo {
    uint32_t addr;
    uint16_t len; // Length in # of elements
    uint16_t num_rec = 1; // Number of CiMs that will receive the data (some parameters only have to be sent to a subset of CiMs)
};

/*----- STATIC -----*/
static std::map<PARAM_NAME, ParamInfo> param_addr_map = {
    {PATCH_PROJ_KERNEL_PARAMS,                      {/*addr*/ 0,     /*len*/ PATCH_LEN,          /*num rec*/ NUM_CIM}},
    {SINGLE_PARAMS,                                 {/*addr*/ 64*8,  /*len*/ 16,                 /*num rec*/ NUM_CIM}},
    {POS_EMB_PARAMS,                                {/*addr*/ 64*1,  /*len*/ NUM_PATCHES+1,      /*num rec*/ NUM_CIM}},
    {ENC_Q_DENSE_PARAMS,                            {/*addr*/ 128,   /*len*/ EMB_DEPTH,          /*num rec*/ NUM_CIM}},
    {ENC_K_DENSE_PARAMS,                            {/*addr*/ 192,   /*len*/ EMB_DEPTH,          /*num rec*/ NUM_CIM}},
    {ENC_V_DENSE_PARAMS,                            {/*addr*/ 256,   /*len*/ EMB_DEPTH,          /*num rec*/ NUM_CIM}},
    {ENC_COMB_HEAD_PARAMS,                          {/*addr*/ 320,   /*len*/ EMB_DEPTH,          /*num rec*/ NUM_CIM}},
    {ENC_MLP_DENSE_1_OR_MLP_HEAD_DENSE_1_PARAMS,    {/*addr*/ 384,   /*len*/ EMB_DEPTH,          /*num rec*/ NUM_CIM}}, // CiMs 0-31 have enc_mlp_dense_1, CiMs 32-63 have enc_head_mlp_dense_1
    {ENC_MLP_DENSE_2_PARAMS,                        {/*addr*/ 448,   /*len*/ MLP_DIM,            /*num rec*/ EMB_DEPTH}},
    {MLP_HEAD_DENSE_2_PARAMS,                       {/*addr*/ 480,   /*len*/ MLP_DIM,            /*num rec*/ NUM_SLEEP_STAGES}},
};
#elif CENTRALIZED_ARCH
/*----- MAP -----*/
const std::map<DATA, uint32_t> mem_map = {
    {EEG_INPUT_MEM,             0},
    {PATCH_MEM,                 SINGLE_WIDTH*NUM_PATCHES*PATCH_LEN + DOUBLE_WIDTH*PATCH_LEN},
    {CLASS_TOKEN_MEM,           SINGLE_WIDTH*NUM_PATCHES*PATCH_LEN},
    {POS_EMB_MEM,               0},
    {ENC_LN1_MEM,               DOUBLE_WIDTH*(NUM_PATCHES+1)*EMB_DEPTH},
    {ENC_Q_MEM,                 2*DOUBLE_WIDTH*(NUM_PATCHES+1)*EMB_DEPTH},
    {ENC_K_MEM,                 2*DOUBLE_WIDTH*(NUM_PATCHES+1)*EMB_DEPTH+(NUM_PATCHES+1)*EMB_DEPTH},
    {ENC_V_MEM,                 (NUM_PATCHES+1)*EMB_DEPTH}, // Need to protect positional embedding as we'll use it below
    {ENC_QK_T_MEM,              2*DOUBLE_WIDTH*(NUM_PATCHES+1)*EMB_DEPTH + 2*(NUM_PATCHES+1)*EMB_DEPTH},
    {ENC_V_MULT_MEM,            2*DOUBLE_WIDTH*(NUM_PATCHES+1)*EMB_DEPTH + 2*(NUM_PATCHES+1)*EMB_DEPTH + NUM_HEADS*(NUM_PATCHES+1)*(NUM_PATCHES+1)},
    {ENC_MHSA_OUT_MEM,          (NUM_PATCHES+1)*EMB_DEPTH},
    {ENC_LN2_MEM,               DOUBLE_WIDTH*2*(NUM_PATCHES+1)*EMB_DEPTH},
    {ENC_MLP_DENSE1_MEM,        DOUBLE_WIDTH*3*(NUM_PATCHES+1)*EMB_DEPTH},
    {ENC_MLP_OUT_MEM,           DOUBLE_WIDTH*4*(NUM_PATCHES+1)*EMB_DEPTH},
    {ENC_LN3_MEM,               DOUBLE_WIDTH*4*(NUM_PATCHES+2)*EMB_DEPTH},
    {MLP_HEAD_DENSE_1_OUT_MEM,  DOUBLE_WIDTH*4*(NUM_PATCHES+3)*EMB_DEPTH},
    {MLP_HEAD_DENSE_2_OUT_MEM,  DOUBLE_WIDTH*4*(NUM_PATCHES+3)*EMB_DEPTH+DOUBLE_WIDTH*MLP_DIM},
    {SOFTMAX_AVG_SUM_MEM,       0},
    {PREV_SOFTMAX_OUTPUT_MEM,   CIM_INT_RES_SIZE_NUM_ELEM - DOUBLE_WIDTH*NUM_SLEEP_STAGES*(NUM_SAMPLES_OUT_AVG-1)}
};

/*----- STRUCT -----*/
struct ParamInfo {
    uint16_t addr;
    uint16_t len; // Length in # of elements
};

/*----- STATIC -----*/
static std::map<PARAM_NAME, ParamInfo> param_addr_map = {
    {PATCH_PROJ_KERNEL_PARAMS,  {/*addr*/ 0,        /*len*/ PATCH_LEN*EMB_DEPTH}},
    {POS_EMB_PARAMS,            {/*addr*/ 4096,     /*len*/ (NUM_PATCHES+1)*EMB_DEPTH}},
    {ENC_Q_DENSE_PARAMS,        {/*addr*/ 8000,     /*len*/ PATCH_LEN*EMB_DEPTH}},
    {ENC_K_DENSE_PARAMS,        {/*addr*/ 12096,    /*len*/ PATCH_LEN*EMB_DEPTH}},
    {ENC_V_DENSE_PARAMS,        {/*addr*/ 16192,    /*len*/ PATCH_LEN*EMB_DEPTH}},
    {ENC_COMB_HEAD_PARAMS,      {/*addr*/ 20288,    /*len*/ PATCH_LEN*EMB_DEPTH}},
    {ENC_MLP_DENSE_1_PARAMS,    {/*addr*/ 24384,    /*len*/ EMB_DEPTH*MLP_DIM}},
    {ENC_MLP_DENSE_2_PARAMS,    {/*addr*/ 26432,    /*len*/ MLP_DIM*EMB_DEPTH}},
    {MLP_HEAD_DENSE_1_PARAMS,   {/*addr*/ 28480,    /*len*/ MLP_DIM*EMB_DEPTH}},
    {MLP_HEAD_DENSE_2_PARAMS,   {/*addr*/ 30528,    /*len*/ NUM_SLEEP_STAGES*MLP_DIM}},
    {SINGLE_PARAMS,             {/*addr*/ 30688,    /*len*/ 961}}
};

static std::map<SINGLE_PARAM_OFFSET, ParamInfo> param_addr_map_bias = {
    {PATCH_PROJ_BIAS_OFF,           {/*addr*/ param_addr_map[SINGLE_PARAMS].addr,       /*len*/ EMB_DEPTH}},
    {CLASS_TOKEN_OFF,               {/*addr*/ param_addr_map[SINGLE_PARAMS].addr+64,    /*len*/ EMB_DEPTH}},
    {ENC_LAYERNORM_1_GAMMA_OFF,     {/*addr*/ param_addr_map[SINGLE_PARAMS].addr+128,   /*len*/ EMB_DEPTH}},
    {ENC_LAYERNORM_1_BETA_OFF,      {/*addr*/ param_addr_map[SINGLE_PARAMS].addr+192,   /*len*/ EMB_DEPTH}},
    {ENC_Q_DENSE_BIAS_0FF,          {/*addr*/ param_addr_map[SINGLE_PARAMS].addr+256,   /*len*/ EMB_DEPTH}},
    {ENC_K_DENSE_BIAS_0FF,          {/*addr*/ param_addr_map[SINGLE_PARAMS].addr+320,   /*len*/ EMB_DEPTH}},
    {ENC_V_DENSE_BIAS_0FF,          {/*addr*/ param_addr_map[SINGLE_PARAMS].addr+384,   /*len*/ EMB_DEPTH}},
    {ENC_INV_SQRT_NUM_HEADS_OFF,    {/*addr*/ param_addr_map[SINGLE_PARAMS].addr+448,   /*len*/ 1}},
    {ENC_COMB_HEAD_BIAS_OFF,        {/*addr*/ param_addr_map[SINGLE_PARAMS].addr+449,   /*len*/ EMB_DEPTH}},
    {ENC_LAYERNORM_2_GAMMA_OFF,     {/*addr*/ param_addr_map[SINGLE_PARAMS].addr+513,   /*len*/ EMB_DEPTH}},
    {ENC_LAYERNORM_2_BETA_OFF,      {/*addr*/ param_addr_map[SINGLE_PARAMS].addr+577,   /*len*/ EMB_DEPTH}},
    {ENC_MLP_DENSE_1_BIAS_OFF,      {/*addr*/ param_addr_map[SINGLE_PARAMS].addr+641,   /*len*/ MLP_DIM}},
    {ENC_MLP_DENSE_2_BIAS_OFF,      {/*addr*/ param_addr_map[SINGLE_PARAMS].addr+673,   /*len*/ EMB_DEPTH}},
    {ENC_LAYERNORM_3_GAMMA_OFF,     {/*addr*/ param_addr_map[SINGLE_PARAMS].addr+737,   /*len*/ EMB_DEPTH}},
    {ENC_LAYERNORM_3_BETA_OFF,      {/*addr*/ param_addr_map[SINGLE_PARAMS].addr+801,   /*len*/ EMB_DEPTH}},
    {MLP_HEAD_DENSE_1_BIAS_OFF,     {/*addr*/ param_addr_map[SINGLE_PARAMS].addr+865,   /*len*/ MLP_DIM}},
    {MLP_HEAD_DENSE_2_BIAS_OFF,     {/*addr*/ param_addr_map[SINGLE_PARAMS].addr+897,   /*len*/ EMB_DEPTH}}
};

#endif
#endif // PARAM_LAYER_MAPPING_H
