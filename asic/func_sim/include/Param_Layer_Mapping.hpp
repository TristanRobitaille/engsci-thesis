#ifndef PARAM_LAYER_MAPPING_H
#define PARAM_LAYER_MAPPING_H

#include <map>

enum PARAM_NAMES { // Master goes through these layers sequentially, loading weights into the appropriate CiM
    PATCH_PROJ_DENSE_KERNEL,
    PATCH_PROJ_DENSE_BIAS,
    CLASS_EMB,
    POS_EMB,
    PARAM_LOAD_FINISHED
};

static std::map<int, std::array<uint16_t, 2> /*{start_addr, length (# elem)}*/> param_address_mapping = {
    {PATCH_PROJ_DENSE_KERNEL,   {0, 64}},
    {PATCH_PROJ_DENSE_BIAS,     {64*11, 1}},
    {CLASS_EMB,                 {64*11+1, 1}},
    {POS_EMB,                   {64*1, 61}}
};

#endif // PARAM_LAYER_MAPPING_H