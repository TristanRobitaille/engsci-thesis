#ifndef COMPUTE_VERIFICATION_H
#define COMPUTE_VERIFICATION_H

#include <map>
#include <armadillo>
#include <../rapidcsv/src/rapidcsv.h>
#include <CiM.hpp>

/*----- DEFINE -----*/
#define ENABLE_COMPUTATION_VERIFICATION true
#define REL_TOLERANCE 0.001f //0.1% tolerance
#define ABS_TOLERANCE 0.0001f

/*----- ENUM -----*/
enum COMPUTE_VERIFICATION_STEP {
    PATCH_PROJECTION_VERIF,
    CLASS_TOKEN_VERIF,
    POS_EMB_VERIF,
    ENC_LAYERNORM1_VERIF,
    ENC_MHSA_DENSE_Q_VERIF,
    ENC_MHSA_DENSE_K_VERIF,
    ENC_MHSA_DENSE_V_VERIF,
    ENC_MHSA_DENSE_QK_T_VERIF,
    ENC_SOFTMAX_VERIF,
    ENC_MULT_V_VERIF,
    ENC_RES_SUM_1_VERIF,
    ENC_LAYERNORM2_VERIF,
    ENC_MLP_DENSE1_VERIF,
    ENC_OUT_VERIF,
    MLP_HEAD_LAYERNORM_VERIF,
    MLP_HEAD_DENSE_1_VERIF,
    MLP_HEAD_SOFTMAX_VERIF,
    MLP_HEAD_SOFTMAX_DIV_VERIF,
    POST_SOFTMAX_AVG_VERIF
};

enum RESULT_TYPE {
    MEAN,
    VARIANCE
};

/*----- STRUCT -----*/
struct StepVerifInfo {
    std::string csv_fp;
};

/*----- STATIC -----*/
static std::map<COMPUTE_VERIFICATION_STEP, StepVerifInfo> step_verif_info = {
    {PATCH_PROJECTION_VERIF,    {"reference_data/patch_proj.csv"}},
    {CLASS_TOKEN_VERIF,         {"reference_data/class_emb.csv"}},
    {POS_EMB_VERIF,             {"reference_data/pos_emb.csv"}},
    {ENC_LAYERNORM1_VERIF,      {"reference_data/enc_layernorm1.csv"}},
    {ENC_MHSA_DENSE_Q_VERIF,    {"reference_data/enc_Q_dense.csv"}},
    {ENC_MHSA_DENSE_K_VERIF,    {"reference_data/enc_K_dense.csv"}},
    {ENC_MHSA_DENSE_V_VERIF,    {"reference_data/enc_V_dense.csv"}},
    {ENC_MHSA_DENSE_QK_T_VERIF, {"reference_data/enc_scaled_score_"}},
    {ENC_SOFTMAX_VERIF,         {"reference_data/enc_softmax_"}},
    {ENC_MULT_V_VERIF,          {"reference_data/enc_softmax_mult_V.csv"}},
    {ENC_RES_SUM_1_VERIF,       {"reference_data/enc_res_sum_1.csv"}},
    {ENC_LAYERNORM2_VERIF,      {"reference_data/enc_layernorm2.csv"}},
    {ENC_MLP_DENSE1_VERIF,      {"reference_data/enc_mlp_dense1.csv"}},
    {ENC_OUT_VERIF,             {"reference_data/enc_output.csv"}},
    {MLP_HEAD_LAYERNORM_VERIF,  {"reference_data/mlp_head_layernorm.csv"}},
    {MLP_HEAD_DENSE_1_VERIF,    {"reference_data/mlp_head_out.csv"}},
    {MLP_HEAD_SOFTMAX_VERIF,    {"reference_data/mlp_head_softmax.csv"}},
    {MLP_HEAD_SOFTMAX_DIV_VERIF,{"reference_data/mlp_head_softmax.csv"}},
    {POST_SOFTMAX_AVG_VERIF,    {"reference_data/dummy_softmax_"}}
};

/*----- FUNCTION -----*/
bool are_equal(float a, float b, uint16_t index, uint8_t id);
void verify_computation(COMPUTE_VERIFICATION_STEP cim_state, uint8_t id, float* data, uint16_t starting_addr);
void verify_result(RESULT_TYPE type, float result, float* input_data, uint16_t starting_addr, uint16_t len, uint8_t id);
void verify_softmax_storage(float* intermediate_res, float* prev_softmax_storage);
void print_intermediate_value_stats();

#endif //COMPUTE_VERIFICATION_H
