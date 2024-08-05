
#if CENTRALIZED_ARCH
#include "CiM_Centralized.hpp"

/*----- NAMESPACE -----*/
using namespace std;

/*----- DEFINITION -----*/
CiM_Centralized::CiM_Centralized(const string params_filepath) : gen_cnt_7b(7), gen_cnt_7b_2(7) {
    reset();
    load_params_from_h5(params_filepath);
}

int CiM_Centralized::reset(){
    fill(begin(int_res), end(int_res), 0);
    gen_cnt_7b.reset();
    gen_cnt_7b_2.reset();
    cim_state = IDLE_CIM;
    current_inf_step = INVALID_STEP;
    system_state = IDLE;
    return 0;
}

void CiM_Centralized::load_params_from_h5(const string params_filepath) {
    // Load parameters from H5 file to memory (done by RISC-V). Data is load in row-major order.

    HighFive::File file(params_filepath, HighFive::File::ReadOnly);

    // Determine name of the transformer. Different runs in a training will have different names for the transformer.
    string transformer_name;
    if (file.getGroup("patch_projection_dense").exist("vision_transformer") == true) { transformer_name = "vision_transformer"; }
    else if (file.getGroup("patch_projection_dense").exist("vision_transformer_1") == true) { transformer_name = "vision_transformer_1"; }
    else if (file.getGroup("patch_projection_dense").exist("vision_transformer_2") == true) { transformer_name = "vision_transformer_2"; }
    else if (file.getGroup("patch_projection_dense").exist("vision_transformer_3") == true) { transformer_name = "vision_transformer_3"; }
    else if (file.getGroup("patch_projection_dense").exist("vision_transformer_4") == true) { transformer_name = "vision_transformer_4"; }

    // Patch projection
    PatchProjKernel_t patch_proj_kernel = file.getGroup("patch_projection_dense").getGroup(transformer_name).getGroup("patch_projection_dense").getDataSet("kernel:0").read<PatchProjKernel_t>();
    for (int col=0; col<PATCH_LEN; col++) {
        for (int row=0; row<EMB_DEPTH; row++) {
            params[param_addr_map[PATCH_PROJ_KERNEL_PARAMS].addr + row + col*EMB_DEPTH] = patch_proj_kernel[row][col];
        }
    }
    EmbDepthVect_t patch_proj_bias = file.getGroup("patch_projection_dense").getGroup(transformer_name).getGroup("patch_projection_dense").getDataSet("bias:0").read<EmbDepthVect_t>();
    for (int col=0; col<param_addr_map_bias[PATCH_PROJ_BIAS_OFF].len; col++) {
        params[param_addr_map_bias[PATCH_PROJ_BIAS_OFF].addr + col] = patch_proj_bias[col];
    }

    // Class token and positional embedding
    EmbDepthVect_t class_emb = file.getGroup("top_level_model_weights").getDataSet("class_emb:0").read<EmbDepthVect_t>();
    for (int col=0; col<param_addr_map_bias[CLASS_TOKEN_OFF].len; col++) {
        params[param_addr_map_bias[CLASS_TOKEN_OFF].addr + col] = class_emb[col];
    }
    PosEmb_t pos_emb_kernel = file.getGroup("top_level_model_weights").getDataSet("pos_emb:0").read<array<PosEmb_t, 1>>()[0];
    for (int row=0; row<(NUM_PATCHES+1); row++) {
        for (int col=0; col<EMB_DEPTH; col++) {
            params[param_addr_map[POS_EMB_PARAMS].addr + row*EMB_DEPTH + col] = pos_emb_kernel[row][col];
        }
    }

    // Encoder
    HighFive::Group enc = file.getGroup("Encoder_1");
    // LayerNorm
    EmbDepthVect_t layernorm_beta = enc.getGroup(transformer_name).getGroup("Encoder_1").getGroup("layerNorm1_encoder").getDataSet("beta:0").read<EmbDepthVect_t>(); // Encoder's LayerNorm 1
    EmbDepthVect_t layernorm_gamma = enc.getGroup(transformer_name).getGroup("Encoder_1").getGroup("layerNorm1_encoder").getDataSet("gamma:0").read<EmbDepthVect_t>(); // Encoder's LayerNorm 1
    for (int col=0; col<param_addr_map_bias[ENC_LAYERNORM_1_BETA_OFF].len; col++) {
        params[param_addr_map_bias[ENC_LAYERNORM_1_BETA_OFF].addr + col] = layernorm_beta[col];
    }
    for (int col=0; col<param_addr_map_bias[ENC_LAYERNORM_1_GAMMA_OFF].len; col++) {
        params[param_addr_map_bias[ENC_LAYERNORM_1_GAMMA_OFF].addr + col] = layernorm_gamma[col];
    }
    
    layernorm_beta = enc.getGroup(transformer_name).getGroup("Encoder_1").getGroup("layerNorm2_encoder").getDataSet("beta:0").read<EmbDepthVect_t>(); // Encoder's LayerNorm 2
    layernorm_gamma = enc.getGroup(transformer_name).getGroup("Encoder_1").getGroup("layerNorm2_encoder").getDataSet("gamma:0").read<EmbDepthVect_t>(); // Encoder's LayerNorm 2
    for (int col=0; col<param_addr_map_bias[ENC_LAYERNORM_2_BETA_OFF].len; col++) {
        params[param_addr_map_bias[ENC_LAYERNORM_2_BETA_OFF].addr + col] = layernorm_beta[col];
    }
    for (int col=0; col<param_addr_map_bias[ENC_LAYERNORM_2_GAMMA_OFF].len; col++) {
        params[param_addr_map_bias[ENC_LAYERNORM_2_GAMMA_OFF].addr + col] = layernorm_gamma[col];
    }

    // MHSA
    EncEmbDepthMat_t enc_mhsa_Q_kernel = enc.getGroup("mhsa_query_dense").getDataSet("kernel:0").read<EncEmbDepthMat_t>();
    EncEmbDepthMat_t enc_mhsa_K_kernel = enc.getGroup("mhsa_key_dense").getDataSet("kernel:0").read<EncEmbDepthMat_t>();
    EncEmbDepthMat_t enc_mhsa_V_kernel = enc.getGroup("mhsa_value_dense").getDataSet("kernel:0").read<EncEmbDepthMat_t>();
    for (int row=0; row<EMB_DEPTH; row++) {
        for (int col=0; col<EMB_DEPTH; col++) {
            params[param_addr_map[ENC_Q_DENSE_PARAMS].addr + row*EMB_DEPTH + col] = enc_mhsa_Q_kernel[row][col];
            params[param_addr_map[ENC_K_DENSE_PARAMS].addr + row*EMB_DEPTH + col] = enc_mhsa_K_kernel[row][col];
            params[param_addr_map[ENC_V_DENSE_PARAMS].addr + row*EMB_DEPTH + col] = enc_mhsa_V_kernel[row][col];
        }
    }
    EmbDepthVect_t enc_mhsa_Q_bias = enc.getGroup("mhsa_query_dense").getDataSet("bias:0").read<EmbDepthVect_t>();
    EmbDepthVect_t enc_mhsa_K_bias = enc.getGroup("mhsa_key_dense").getDataSet("bias:0").read<EmbDepthVect_t>();
    EmbDepthVect_t enc_mhsa_V_bias = enc.getGroup("mhsa_value_dense").getDataSet("bias:0").read<EmbDepthVect_t>();
    for (int col=0; col<param_addr_map_bias[ENC_Q_DENSE_BIAS_0FF].len; col++) {
        params[param_addr_map_bias[ENC_Q_DENSE_BIAS_0FF].addr + col] = enc_mhsa_Q_bias[col];
        params[param_addr_map_bias[ENC_K_DENSE_BIAS_0FF].addr + col] = enc_mhsa_K_bias[col];
        params[param_addr_map_bias[ENC_V_DENSE_BIAS_0FF].addr + col] = enc_mhsa_V_bias[col];
    }
    
    EncEmbDepthMat_t enc_mhsa_combine_kernel = enc.getGroup("mhsa_combine_head_dense").getDataSet("kernel:0").read<EncEmbDepthMat_t>();
    EmbDepthVect_t enc_mhsa_combine_bias = enc.getGroup("mhsa_combine_head_dense").getDataSet("bias:0").read<EmbDepthVect_t>();
    for (int row=0; row<EMB_DEPTH; row++) {
        for (int col=0; col<EMB_DEPTH; col++) {
            params[param_addr_map[ENC_COMB_HEAD_PARAMS].addr + row*EMB_DEPTH + col] = enc_mhsa_combine_kernel[row][col];
        }
    }
    for (int col=0; col<param_addr_map_bias[ENC_COMB_HEAD_BIAS_OFF].len; col++) {
        params[param_addr_map_bias[ENC_COMB_HEAD_BIAS_OFF].addr + col] = enc_mhsa_combine_bias[col];
    }
    float enc_mhsa_inv_sqrt_num_heads = static_cast<float>(1 / sqrt(NUM_HEADS)); // To save compute, we store the reciprocal of the square root of the number of heads such that we can multiply instead of divide
    params[param_addr_map_bias[ENC_INV_SQRT_NUM_HEADS_OFF].addr] = enc_mhsa_inv_sqrt_num_heads;

    // MLP
    EmbDepthxMlpDimMat_t enc_mlp_dense_1_kernel = enc.getGroup(transformer_name).getGroup("Encoder_1").getGroup("mlp_dense1_encoder").getDataSet("kernel:0").read<EmbDepthxMlpDimMat_t>();
    MlpDimVect_t enc_mlp_dense_1_bias = enc.getGroup(transformer_name).getGroup("Encoder_1").getGroup("mlp_dense1_encoder").getDataSet("bias:0").read<MlpDimVect_t>();
    EncMlpDimxEmbDepthMat_t enc_mlp_dense_2_kernel = enc.getGroup(transformer_name).getGroup("Encoder_1").getGroup("mlp_dense2_encoder").getDataSet("kernel:0").read<EncMlpDimxEmbDepthMat_t>();
    EmbDepthVect_t enc_mlp_dense_2_bias = enc.getGroup(transformer_name).getGroup("Encoder_1").getGroup("mlp_dense2_encoder").getDataSet("bias:0").read<EmbDepthVect_t>();
    for (int row=0; row<EMB_DEPTH; row++) {
        for (int col=0; col<MLP_DIM; col++) {
            params[param_addr_map[ENC_MLP_DENSE_1_PARAMS].addr + row*MLP_DIM + col] = enc_mlp_dense_1_kernel[row][col];
        }
    }
    for (int col=0; col<param_addr_map_bias[ENC_MLP_DENSE_1_BIAS_OFF].len; col++) {
        params[param_addr_map_bias[ENC_MLP_DENSE_1_BIAS_OFF].addr + col] = enc_mlp_dense_1_bias[col];
    }
    for (int row=0; row<MLP_DIM; row++) {
        for (int col=0; col<EMB_DEPTH; col++) {
            params[param_addr_map[ENC_MLP_DENSE_2_PARAMS].addr + row*EMB_DEPTH + col] = enc_mlp_dense_2_kernel[row][col];
        }
    }
    for (int col=0; col<param_addr_map_bias[ENC_MLP_DENSE_2_BIAS_OFF].len; col++) {
        params[param_addr_map_bias[ENC_MLP_DENSE_2_BIAS_OFF].addr + col] = enc_mlp_dense_2_bias[col];
    }

    // MLP head
    layernorm_beta = file.getGroup("mlp_head_layerNorm").getGroup(transformer_name).getGroup("mlp_head_layerNorm").getDataSet("beta:0").read<EmbDepthVect_t>(); // Encoder's LayerNorm 2
    layernorm_gamma = file.getGroup("mlp_head_layerNorm").getGroup(transformer_name).getGroup("mlp_head_layerNorm").getDataSet("gamma:0").read<EmbDepthVect_t>(); // Encoder's LayerNorm 2
    for (int col=0; col<param_addr_map_bias[ENC_LAYERNORM_3_BETA_OFF].len; col++) {
        params[param_addr_map_bias[ENC_LAYERNORM_3_BETA_OFF].addr + col] = layernorm_beta[col];
    }
    for (int col=0; col<param_addr_map_bias[ENC_LAYERNORM_3_GAMMA_OFF].len; col++) {
        params[param_addr_map_bias[ENC_LAYERNORM_3_GAMMA_OFF].addr + col] = layernorm_gamma[col];
    }

    EmbDepthxMlpDimMat_t mlp_head_dense_1_kernel = file.getGroup("mlp_head").getGroup("mlp_head_dense1").getDataSet("kernel:0").read<EmbDepthxMlpDimMat_t>();
    MlpDimVect_t mlp_head_dense_1_bias = file.getGroup("mlp_head").getGroup("mlp_head_dense1").getDataSet("bias:0").read<MlpDimVect_t>();
    NumSleepStagesxMlpDimMat_t mlp_head_dense_2_kernel = file.getGroup("mlp_head_softmax").getGroup(transformer_name).getGroup("mlp_head_softmax").getDataSet("kernel:0").read<NumSleepStagesxMlpDimMat_t>();
    NumSleepStagesVect_t mlp_head_dense_2_bias = file.getGroup("mlp_head_softmax").getGroup(transformer_name).getGroup("mlp_head_softmax").getDataSet("bias:0").read<NumSleepStagesVect_t>();
    for (int row=0; row<EMB_DEPTH; row++) {
        for (int col=0; col<MLP_DIM; col++) {
            params[param_addr_map[MLP_HEAD_DENSE_1_PARAMS].addr + row*MLP_DIM + col] = mlp_head_dense_1_kernel[row][col];
        }
    }
    for (int col=0; col<param_addr_map_bias[MLP_HEAD_DENSE_1_BIAS_OFF].len; col++) {
        params[param_addr_map_bias[MLP_HEAD_DENSE_1_BIAS_OFF].addr + col] = mlp_head_dense_1_bias[col];
    }
    for (int row=0; row<MLP_DIM; row++) {
        for (int col=0; col<NUM_SLEEP_STAGES; col++) {
            params[param_addr_map[MLP_HEAD_DENSE_2_PARAMS].addr + row*NUM_SLEEP_STAGES + col] = mlp_head_dense_2_kernel[row][col];
        }
    }
    for (int col=0; col<param_addr_map_bias[MLP_HEAD_DENSE_2_BIAS_OFF].len; col++) {
        params[param_addr_map_bias[MLP_HEAD_DENSE_2_BIAS_OFF].addr + col] = mlp_head_dense_2_bias[col];
    }
}

void CiM_Centralized::load_eeg_from_h5(const string eeg_filepath, uint16_t clip_index) {
    // Load parameters from H5 file to memory (done by RISC-V)

    // EEG data file
    HighFive::File eeg_file(eeg_filepath, HighFive::File::ReadOnly);
    vector<vector<int64_t>> eeg_ds = eeg_file.getDataSet("eeg").read<vector<vector<int64_t>>>();

    for (int i=0; i<NUM_PATCHES*PATCH_LEN; i++) {
        int_res_write(static_cast<float>(eeg_ds[clip_index][i])/EEG_SCALE_FACTOR, mem_map.at(EEG_INPUT_MEM) + i*SINGLE_WIDTH, SINGLE_WIDTH);
    }
}

void CiM_Centralized::load_previous_softmax(const string prev_softmax_base_filepath) {
    for (int i = 0; i < (NUM_SAMPLES_OUT_AVG-1); i++) { // Softmax for previous dummy epochs
        string filename = prev_softmax_base_filepath + to_string(i) + ".csv";
        rapidcsv::Document csv(filename, rapidcsv::LabelParams(-1, -1));
        vector<float> dummy_softmax = csv.GetRow<float>(0);
        for (int j = 0; j < NUM_SLEEP_STAGES; j++) {
            int_res_write((dummy_softmax[j] / NUM_SAMPLES_OUT_AVG), mem_map.at(PREV_SOFTMAX_OUTPUT_MEM) + i*NUM_SLEEP_STAGES*DOUBLE_WIDTH + DOUBLE_WIDTH*j, DOUBLE_WIDTH);
        }
    }
}

SYSTEM_STATE CiM_Centralized::run(struct ext_signals* ext_sigs, string softmax_base_filepath, string eeg_filepath, uint16_t clip_index) {
    /* Run the CiM FSM */
    if (ext_sigs->master_nrst == false) {
        reset();
        load_previous_softmax(softmax_base_filepath);
        load_eeg_from_h5(eeg_filepath, clip_index);
    }

    // Update compute process counter
    update_compute_process_cnt();

    switch (cim_state){
        case IDLE_CIM:
            if (ext_sigs->new_sleep_epoch) {
                cim_state = INFERENCE_RUNNING;
                current_inf_step = PATCH_PROJ_STEP;
            }
            break;

        case INFERENCE_RUNNING:
            if (current_inf_step == POS_EMB_STEP) { system_state = EVERYTHING_FINISHED; }
            break;

        case INVALID_CIM:
            break;
    }

    switch (current_inf_step){
        case PATCH_PROJ_STEP:
            /* gen_cnt_7b holds current parameters row 
               gen_cnt_7b_2 holds current patch */

            if (compute_done || (gen_cnt_7b.get_cnt() == 0 && !compute_in_progress)) { // Start a new MAC
                uint32_t patch_addr = mem_map.at(EEG_INPUT_MEM) + SINGLE_WIDTH*EMB_DEPTH*gen_cnt_7b_2.get_cnt();
                uint32_t param_addr = param_addr_map[PATCH_PROJ_KERNEL_PARAMS].addr + EMB_DEPTH*gen_cnt_7b.get_cnt();
                uint32_t param_bias_addr = param_addr_map_bias[PATCH_PROJ_BIAS_OFF].addr + gen_cnt_7b.get_cnt();
                MAC<dw_fx_x_t,params_fx_2_x_t>(patch_addr, param_addr, PATCH_LEN, param_bias_addr, MODEL_PARAM, LINEAR_ACTIVATION, SINGLE_WIDTH);
            }

            if (compute_done) {
                // Save data
                uint32_t addr = mem_map.at(PATCH_MEM) + DOUBLE_WIDTH*(gen_cnt_7b.get_cnt() + gen_cnt_7b_2.get_cnt()*PATCH_LEN);
                int_res_write(computation_result, addr, DOUBLE_WIDTH);

                // Update index control
                if (gen_cnt_7b.get_cnt() == EMB_DEPTH-1) { // Done going through all parameters rows with given patch
                    gen_cnt_7b.reset();
                    if (gen_cnt_7b_2.get_cnt() == NUM_PATCHES-1) { // Done going through all patches
                        gen_cnt_7b_2.reset();
                        verify_layer_out(PATCH_PROJECTION_VERIF, int_res, mem_map.at(PATCH_MEM), EMB_DEPTH, DOUBLE_WIDTH);
                        current_inf_step = CLASS_TOKEN_CONCAT_STEP;
                        if (PRINT_INF_PROGRESS) { cout << "Patch projection done" << endl; }
                    }
                    gen_cnt_7b_2.inc(); // New patch
                } else {
                    gen_cnt_7b.inc();
                }
            }
            break;

        case CLASS_TOKEN_CONCAT_STEP:
            if (gen_cnt_7b.get_cnt() < EMB_DEPTH) {
                uint32_t class_token_addr = param_addr_map_bias[CLASS_TOKEN_OFF].addr + gen_cnt_7b.get_cnt();
                int_res_write(params[class_token_addr], mem_map.at(CLASS_TOKEN_MEM) + DOUBLE_WIDTH*gen_cnt_7b.get_cnt(), DOUBLE_WIDTH);
                gen_cnt_7b.inc();
            } else {
                gen_cnt_7b.reset();
                verify_layer_out(CLASS_TOKEN_VERIF, int_res, mem_map.at(CLASS_TOKEN_MEM), EMB_DEPTH, DOUBLE_WIDTH);
                current_inf_step = POS_EMB_STEP;
                if (PRINT_INF_PROGRESS) { cout << "Classification token concatenation done" << endl; }
            }
            break;

        case POS_EMB_STEP:
            break;

        case INVALID_STEP:
            break;
    }

    // Reset compute done (stays on for only one cycle)
    reset_compute_done();

    return system_state;
}

#endif