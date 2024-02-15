#ifndef PARAM_LAYER_MAPPING_H
#define PARAM_LAYER_MAPPING_H

#include <map>

/*----- ENUM -----*/
enum PARAM_NAME { // Master goes through these layers sequentially, loading weights into the appropriate CiM
    PATCH_PROJ_KERNEL_PARAMS,
    SINGLE_PARAMS, // The 1-element parameters stored at the bottom of the storage
    POS_EMB_PARAMS,
    ENC_Q_DENSE_PARAMS,
    ENC_K_DENSE_PARAMS,
    ENC_V_DENSE_PARAMS,
    ENC_COMB_HEAD_PARAMS,
    MLP_DENSE_1_PARAMS,
    MLP_DENSE_2_PARAMS,
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
    ENC_MLP_DENSE_1_BIAS_OFF,
    ENC_MLP_DENSE_2_BIAS_OFF
};

/*----- TYPEDEF -----*/
typedef std::array<float, EMB_DEPTH> EmbDepthVect_t;
typedef std::array<float, MLP_DIM> MlpDimVect_t;
typedef std::array<std::array<float, EMB_DEPTH>, PATCH_LEN> PatchProjKernel_t;
typedef std::array<std::array<float, EMB_DEPTH>, NUM_PATCHES+1> PosEmb_t;
typedef std::array<std::array<float, EMB_DEPTH>, EMB_DEPTH> EncEmbDepthMat_t;
typedef std::array<std::array<float, EMB_DEPTH>, 2> EncEmbDepthVect2_t;
typedef std::array<std::array<float, MLP_DIM>, EMB_DEPTH> EncEmbDepthxMlpDimMat_t;
typedef std::array<std::array<float, EMB_DEPTH>, MLP_DIM> EncMlpDimxEmbDepthMat_t;
typedef std::variant<EmbDepthVect_t, PatchProjKernel_t, PosEmb_t, EncEmbDepthMat_t, EncEmbDepthVect2_t, EncEmbDepthxMlpDimMat_t, MlpDimVect_t, EncMlpDimxEmbDepthMat_t> ParamType;

/*----- STRUCT -----*/
struct ParamInfo {
    uint16_t addr;
    uint16_t len; // Length in # of elements
    uint16_t num_rec = 1; // Number of CiMs that will receive the data (some parameters only have to be sent to a subset of CiMs)
};

/*----- STATIC -----*/
static std::map<PARAM_NAME, ParamInfo> param_addr_map = {
    {PATCH_PROJ_KERNEL_PARAMS,  {0,     /*len*/ PATCH_LEN,      /*num rec*/ NUM_CIM}},
    {SINGLE_PARAMS,             {64*11, /*len*/ 13,             /*num rec*/ NUM_CIM}},
    {POS_EMB_PARAMS,            {64*1,  /*len*/ NUM_PATCHES+1,  /*num rec*/ NUM_CIM}},
    {ENC_Q_DENSE_PARAMS,        {128,   /*len*/ EMB_DEPTH,      /*num rec*/ NUM_CIM}},
    {ENC_K_DENSE_PARAMS,        {192,   /*len*/ EMB_DEPTH,      /*num rec*/ NUM_CIM}},
    {ENC_V_DENSE_PARAMS,        {256,   /*len*/ EMB_DEPTH,      /*num rec*/ NUM_CIM}},
    {ENC_COMB_HEAD_PARAMS,      {320,   /*len*/ EMB_DEPTH,      /*num rec*/ NUM_CIM}},
    {MLP_DENSE_1_PARAMS,        {384,   /*len*/ EMB_DEPTH,      /*num rec*/ MLP_DIM}},
    {MLP_DENSE_2_PARAMS,        {448,   /*len*/ MLP_DIM,        /*num rec*/ EMB_DEPTH}}
};

#endif // PARAM_LAYER_MAPPING_H