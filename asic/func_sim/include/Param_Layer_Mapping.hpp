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
    ENC_LAYERNORM_2_BETA_OFF
};

/*----- TYPEDEF -----*/
typedef std::array<float, EMB_DEPTH> EmbDepthVect_t;
typedef std::array<std::array<float, EMB_DEPTH>, PATCH_LENGTH_NUM_SAMPLES> PatchProjKernel_t;
typedef std::array<std::array<float, EMB_DEPTH>, NUM_PATCHES+1> PosEmb_t;
typedef std::array<float, EMB_DEPTH> EncEmbDepthVect_t;
typedef std::array<std::array<float, EMB_DEPTH>, EMB_DEPTH> EncEmbDepthMat_t;
typedef std::array<std::array<float, EMB_DEPTH>, 2> EncEmbDepthVect2_t;
typedef std::array<std::array<std::array<float, EMB_DEPTH>, EMB_DEPTH>, 2> EncEmbDepthMat2_t;
typedef std::variant<EmbDepthVect_t, PatchProjKernel_t, PosEmb_t, EncEmbDepthVect_t, EncEmbDepthMat_t, EncEmbDepthVect2_t, EncEmbDepthMat2_t> ParamType;

/*----- STRUCT -----*/
struct ParamInfo {
    uint16_t addr;
    uint16_t len; // Length in # of elements
};

/*----- STATIC -----*/
static std::map<PARAM_NAME, ParamInfo> param_addr_map = {
    {PATCH_PROJ_KERNEL_PARAMS,  {0, PATCH_LENGTH_NUM_SAMPLES}},
    {SINGLE_PARAMS,             {64*11, /*len*/ 11}}, // TODO: 11 so far, but will add
    {POS_EMB_PARAMS,            {64*1,  /*len*/ NUM_PATCHES+1}},
    {ENC_Q_DENSE_PARAMS,        {128,   /*len*/ EMB_DEPTH}},
    {ENC_K_DENSE_PARAMS,        {192,   /*len*/ EMB_DEPTH}},
    {ENC_V_DENSE_PARAMS,        {256,   /*len*/ EMB_DEPTH}},
    {ENC_COMB_HEAD_PARAMS,      {320,   /*len*/ EMB_DEPTH}}
};

#endif // PARAM_LAYER_MAPPING_H