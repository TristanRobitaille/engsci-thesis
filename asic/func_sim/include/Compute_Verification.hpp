#include <map>
#include <../rapidcsv/src/rapidcsv.h>
#include <CiM.hpp>

#ifndef COMPUTE_VERIFICATION_H
#define COMPUTE_VERIFICATION_H

/*----- DEFINE -----*/
#define ENABLE_COMPUTATION_VERIFICATION true
#define REL_TOLERANCE 0.001f //0.1% tolerance
#define ABS_TOLERANCE 0.0001f

/*----- ENUM -----*/
enum COMPUTE_VERIFICATION_STEP {
    PATCH_PROJECTION_VERIF,
    CLASS_TOKEN_VERIF,
    POS_EMB_VERIF,
    ENC_LAYERNORM1_VERIF
};

/*----- STRUCT -----*/
struct StepVerifInfo {
    std::string csv_fp;
};

/*----- STATIC -----*/
static std::map<COMPUTE_VERIFICATION_STEP, StepVerifInfo> step_verif_info = {
    {PATCH_PROJECTION_VERIF,    {"reference_data/patch_projection.csv"}},
    {CLASS_TOKEN_VERIF,         {"reference_data/clip_with_class_embedding.csv"}},
    {POS_EMB_VERIF,             {"reference_data/clip_with_class_embedding_and_pos_emb.csv"}},
    {ENC_LAYERNORM1_VERIF,      {"reference_data/enc_layernorm1.csv"}}
};

/*----- FUNCTION -----*/
inline bool are_equal(float a, float b, uint16_t index, uint8_t id) {
    float diff = abs(a - b);
    if ((diff > REL_TOLERANCE*abs(b)) && (diff > ABS_TOLERANCE)) {
        std::cout << "Mismatch for CiM #" << (uint16_t) id << " at index " << index << ": Expected: " << a << ", got " << b << std::endl;
        return false;
    }
    return true;
}

inline void verify_computation(COMPUTE_VERIFICATION_STEP cim_state, uint8_t id, float* data, uint16_t starting_addr) {
    if (ENABLE_COMPUTATION_VERIFICATION == false) { return; }

    rapidcsv::Document csv(step_verif_info[cim_state].csv_fp, rapidcsv::LabelParams(-1, -1));
    std::vector<float> col = csv.GetColumn<float>(id);
    for (int i = 0; i < col.size(); i++) { are_equal(col[i], data[i+starting_addr], i, id); }
}

#endif //COMPUTE_VERIFICATION_H
