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
};

enum SINGLE_PARAM_OFFSET {
    PATCH_PROJ_BIAS_OFF,
    CLASS_EMB_OFF,
    ENC_LAYERNORM_1_GAMMA_OFF,
    ENC_LAYERNORM_1_BETA_OFF,
    ENC_Q_DENSE_BIAS_0FF,
    ENC_K_DENSE_BIAS_0FF,
    ENC_V_DENSE_BIAS_0FF,
    ENC_SQRT_NUM_HEADS_OFF,
    ENC_COMB_HEAD_BIAS_OFF,
    ENC_LAYERNORM_2_GAMMA_OFF,
    ENC_LAYERNORM_2_BETA_OFF,
    ENC_MLP_DENSE_1_MLP_HEAD_DENSE_1_BIAS_OFF, // This is the same offset for both encoder's MLP's dense 1 bias and MLP head's dense 1 bias
    ENC_MLP_DENSE_2_BIAS_OFF,
    ENC_LAYERNORM_3_GAMMA_OFF,
    ENC_LAYERNORM_3_BETA_OFF,
    MLP_HEAD_DENSE_2_BIAS_OFF
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

/*----- STRUCT -----*/
struct ParamInfo {
    uint16_t addr;
    uint16_t len; // Length in # of elements
    uint16_t num_rec = 1; // Number of CiMs that will receive the data (some parameters only have to be sent to a subset of CiMs)
};

/*----- STATIC -----*/
static std::map<PARAM_NAME, ParamInfo> param_addr_map = {
    {PATCH_PROJ_KERNEL_PARAMS,                      {0,     /*len*/ PATCH_LEN,          /*num rec*/ NUM_CIM}},
    {SINGLE_PARAMS,                                 {64*8,  /*len*/ 16,                 /*num rec*/ NUM_CIM}},
    {POS_EMB_PARAMS,                                {64*1,  /*len*/ NUM_PATCHES+1,      /*num rec*/ NUM_CIM}},
    {ENC_Q_DENSE_PARAMS,                            {128,   /*len*/ EMB_DEPTH,          /*num rec*/ NUM_CIM}},
    {ENC_K_DENSE_PARAMS,                            {192,   /*len*/ EMB_DEPTH,          /*num rec*/ NUM_CIM}},
    {ENC_V_DENSE_PARAMS,                            {256,   /*len*/ EMB_DEPTH,          /*num rec*/ NUM_CIM}},
    {ENC_COMB_HEAD_PARAMS,                          {320,   /*len*/ EMB_DEPTH,          /*num rec*/ NUM_CIM}},
    {ENC_MLP_DENSE_1_OR_MLP_HEAD_DENSE_1_PARAMS,    {384,   /*len*/ EMB_DEPTH,          /*num rec*/ NUM_CIM}}, // CiMs 0-31 have enc_mlp_dense_1, CiMs 32-63 have enc_head_mlp_dense_1
    {ENC_MLP_DENSE_2_PARAMS,                        {448,   /*len*/ MLP_DIM,            /*num rec*/ EMB_DEPTH}},
    {MLP_HEAD_DENSE_2_PARAMS,                       {480,   /*len*/ MLP_DIM,            /*num rec*/ NUM_SLEEP_STAGES}},
};

#endif // PARAM_LAYER_MAPPING_H