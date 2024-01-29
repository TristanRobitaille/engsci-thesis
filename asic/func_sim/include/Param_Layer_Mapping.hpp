#ifndef PARAM_LAYER_MAPPING_H
#define PARAM_LAYER_MAPPING_H

#include <map>

enum PARAM_NAMES { // Master goes through these layers sequentially, loading weights into the appropriate CiM
    PATCH_PROJ_KERNEL_PARAMS,
    SINGLE_PARAMS, // The 1-element parameters stored at the bottom of the storage
    POS_EMB_PARAMS,
    PARAM_LOAD_FINISHED
};

enum SINGLE_PARAM_OFFSET {
    PATCH_PROJ_BIAS_OFF,
    CLASS_EMB_OFF,
    ENC1_LAYERNORM1_GAMMA,
    ENC1_LAYERNORM1_BETA,
};

enum FIELD_NAME {
    ADDR,
    LEN
};

static std::map<PARAM_NAMES, std::array<uint16_t, 2> /*{start_addr, length (# elem)}*/> param_addr_map = {
    {PATCH_PROJ_KERNEL_PARAMS,  {0, PATCH_LENGTH_NUM_SAMPLES}},
    {SINGLE_PARAMS,             {64*11, 10}}, // TODO: 10 so far, but will add
    {POS_EMB_PARAMS,            {64*1, NUM_PATCHES+1}}
};

#endif // PARAM_LAYER_MAPPING_H