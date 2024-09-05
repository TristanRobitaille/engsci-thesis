
#if CENTRALIZED_ARCH
#include "CiM_Centralized.hpp"

/*----- NAMESPACE -----*/
using namespace std;

/*----- DEFINITION -----*/
float CiM_Compute::int_res_0[CIM_INT_RES_BANK_SIZE_NUM_WORD];
float CiM_Compute::int_res_1[CIM_INT_RES_BANK_SIZE_NUM_WORD];
float CiM_Compute::int_res_2[CIM_INT_RES_BANK_SIZE_NUM_WORD];
float CiM_Compute::int_res_3[CIM_INT_RES_BANK_SIZE_NUM_WORD];

CiM_Centralized::CiM_Centralized(const string params_filepath) : gen_cnt_7b(7), gen_cnt_9b(9), gen_cnt_4b(4) {
    reset();
    load_params_from_h5(params_filepath);
}

int CiM_Centralized::reset(){
    fill(begin(int_res_0), end(int_res_0), 0);
    fill(begin(int_res_1), end(int_res_1), 0);
    fill(begin(int_res_2), end(int_res_2), 0);
    fill(begin(int_res_3), end(int_res_3), 0);
    gen_cnt_7b.reset();
    gen_cnt_9b.reset();
    cim_state = IDLE_CIM;
    current_inf_step = INVALID_STEP;
    mac_or_div = MAC_OP;
    mac_or_add = ADD_OP;
    generic_done = false;
    _softmax_max_index = 0;
    _inferred_sleep_stage = -1;
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
            param_write(patch_proj_kernel[row][col], param_addr_map[PATCH_PROJ_KERNEL_PARAMS].addr + row + col*EMB_DEPTH);
        }
    }
    EmbDepthVect_t patch_proj_bias = file.getGroup("patch_projection_dense").getGroup(transformer_name).getGroup("patch_projection_dense").getDataSet("bias:0").read<EmbDepthVect_t>();
    for (int col=0; col<param_addr_map_bias[PATCH_PROJ_BIAS_OFF].len; col++) {
        param_write(patch_proj_bias[col], param_addr_map_bias[PATCH_PROJ_BIAS_OFF].addr + col);
    }

    // Class token and positional embedding
    EmbDepthVect_t class_emb = file.getGroup("top_level_model_weights").getDataSet("class_emb:0").read<EmbDepthVect_t>();
    for (int col=0; col<param_addr_map_bias[CLASS_TOKEN_OFF].len; col++) {
        param_write(class_emb[col], param_addr_map_bias[CLASS_TOKEN_OFF].addr + col);
    }
    PosEmb_t pos_emb_kernel = file.getGroup("top_level_model_weights").getDataSet("pos_emb:0").read<array<PosEmb_t, 1>>()[0];
    for (int row=0; row<(NUM_PATCHES+1); row++) {
        for (int col=0; col<EMB_DEPTH; col++) {
            param_write(pos_emb_kernel[row][col], param_addr_map[POS_EMB_PARAMS].addr + row*EMB_DEPTH + col);
        }
    }

    // Encoder
    HighFive::Group enc = file.getGroup("Encoder_1");
    // LayerNorm
    EmbDepthVect_t layernorm_beta = enc.getGroup(transformer_name).getGroup("Encoder_1").getGroup("layerNorm1_encoder").getDataSet("beta:0").read<EmbDepthVect_t>(); // Encoder's LayerNorm 1
    EmbDepthVect_t layernorm_gamma = enc.getGroup(transformer_name).getGroup("Encoder_1").getGroup("layerNorm1_encoder").getDataSet("gamma:0").read<EmbDepthVect_t>(); // Encoder's LayerNorm 1
    for (int col=0; col<param_addr_map_bias[ENC_LAYERNORM_1_BETA_OFF].len; col++) {
        param_write(layernorm_beta[col], param_addr_map_bias[ENC_LAYERNORM_1_BETA_OFF].addr + col);
    }
    for (int col=0; col<param_addr_map_bias[ENC_LAYERNORM_1_GAMMA_OFF].len; col++) {
        param_write(layernorm_gamma[col], param_addr_map_bias[ENC_LAYERNORM_1_GAMMA_OFF].addr + col);
    }

    layernorm_beta = enc.getGroup(transformer_name).getGroup("Encoder_1").getGroup("layerNorm2_encoder").getDataSet("beta:0").read<EmbDepthVect_t>(); // Encoder's LayerNorm 2
    layernorm_gamma = enc.getGroup(transformer_name).getGroup("Encoder_1").getGroup("layerNorm2_encoder").getDataSet("gamma:0").read<EmbDepthVect_t>(); // Encoder's LayerNorm 2
    for (int col=0; col<param_addr_map_bias[ENC_LAYERNORM_2_BETA_OFF].len; col++) {
        param_write(layernorm_beta[col], param_addr_map_bias[ENC_LAYERNORM_2_BETA_OFF].addr + col);
    }
    for (int col=0; col<param_addr_map_bias[ENC_LAYERNORM_2_GAMMA_OFF].len; col++) {
        param_write(layernorm_gamma[col], param_addr_map_bias[ENC_LAYERNORM_2_GAMMA_OFF].addr + col);
    }

    // MHSA
    // Kernels stored in column-major order!
    EncEmbDepthMat_t enc_mhsa_Q_kernel = enc.getGroup("mhsa_query_dense").getDataSet("kernel:0").read<EncEmbDepthMat_t>();
    EncEmbDepthMat_t enc_mhsa_K_kernel = enc.getGroup("mhsa_key_dense").getDataSet("kernel:0").read<EncEmbDepthMat_t>();
    EncEmbDepthMat_t enc_mhsa_V_kernel = enc.getGroup("mhsa_value_dense").getDataSet("kernel:0").read<EncEmbDepthMat_t>();
    for (int row=0; row<EMB_DEPTH; row++) {
        for (int col=0; col<EMB_DEPTH; col++) {
            param_write(enc_mhsa_Q_kernel[row][col], param_addr_map[ENC_Q_DENSE_PARAMS].addr + row + EMB_DEPTH*col);
            param_write(enc_mhsa_K_kernel[row][col], param_addr_map[ENC_K_DENSE_PARAMS].addr + row + EMB_DEPTH*col);
            param_write(enc_mhsa_V_kernel[row][col], param_addr_map[ENC_V_DENSE_PARAMS].addr + row + EMB_DEPTH*col);
        }
    }
    EmbDepthVect_t enc_mhsa_Q_bias = enc.getGroup("mhsa_query_dense").getDataSet("bias:0").read<EmbDepthVect_t>();
    EmbDepthVect_t enc_mhsa_K_bias = enc.getGroup("mhsa_key_dense").getDataSet("bias:0").read<EmbDepthVect_t>();
    EmbDepthVect_t enc_mhsa_V_bias = enc.getGroup("mhsa_value_dense").getDataSet("bias:0").read<EmbDepthVect_t>();
    for (int col=0; col<param_addr_map_bias[ENC_Q_DENSE_BIAS_0FF].len; col++) {
        param_write(enc_mhsa_Q_bias[col], param_addr_map_bias[ENC_Q_DENSE_BIAS_0FF].addr + col);
        param_write(enc_mhsa_K_bias[col], param_addr_map_bias[ENC_K_DENSE_BIAS_0FF].addr + col);
        param_write(enc_mhsa_V_bias[col], param_addr_map_bias[ENC_V_DENSE_BIAS_0FF].addr + col);
    }

    // Kernels stored in column-major order!
    EncEmbDepthMat_t enc_mhsa_combine_kernel = enc.getGroup("mhsa_combine_head_dense").getDataSet("kernel:0").read<EncEmbDepthMat_t>();
    EmbDepthVect_t enc_mhsa_combine_bias = enc.getGroup("mhsa_combine_head_dense").getDataSet("bias:0").read<EmbDepthVect_t>();
    for (int row=0; row<EMB_DEPTH; row++) {
        for (int col=0; col<EMB_DEPTH; col++) {
            param_write(enc_mhsa_combine_kernel[row][col], param_addr_map[ENC_COMB_HEAD_PARAMS].addr + row + EMB_DEPTH*col);
        }
    }
    for (int col=0; col<param_addr_map_bias[ENC_COMB_HEAD_BIAS_OFF].len; col++) {
        param_write(enc_mhsa_combine_bias[col], param_addr_map_bias[ENC_COMB_HEAD_BIAS_OFF].addr + col);
    }
    float enc_mhsa_inv_sqrt_num_heads = static_cast<float>(1 / sqrt(NUM_HEADS)); // To save compute, we store the reciprocal of the square root of the number of heads such that we can multiply instead of divide
    param_write(enc_mhsa_inv_sqrt_num_heads, param_addr_map_bias[ENC_INV_SQRT_NUM_HEADS_OFF].addr);

    // MLP
    // Kernels stored in column-major order!
    EmbDepthxMlpDimMat_t enc_mlp_dense_1_kernel = enc.getGroup(transformer_name).getGroup("Encoder_1").getGroup("mlp_dense1_encoder").getDataSet("kernel:0").read<EmbDepthxMlpDimMat_t>();
    MlpDimVect_t enc_mlp_dense_1_bias = enc.getGroup(transformer_name).getGroup("Encoder_1").getGroup("mlp_dense1_encoder").getDataSet("bias:0").read<MlpDimVect_t>();
    EncMlpDimxEmbDepthMat_t enc_mlp_dense_2_kernel = enc.getGroup(transformer_name).getGroup("Encoder_1").getGroup("mlp_dense2_encoder").getDataSet("kernel:0").read<EncMlpDimxEmbDepthMat_t>();
    EmbDepthVect_t enc_mlp_dense_2_bias = enc.getGroup(transformer_name).getGroup("Encoder_1").getGroup("mlp_dense2_encoder").getDataSet("bias:0").read<EmbDepthVect_t>();
    for (int row=0; row<EMB_DEPTH; row++) {
        for (int col=0; col<MLP_DIM; col++) {
            param_write(enc_mlp_dense_1_kernel[row][col], param_addr_map[ENC_MLP_DENSE_1_PARAMS].addr + row + col*EMB_DEPTH);
        }
    }
    for (int col=0; col<param_addr_map_bias[ENC_MLP_DENSE_1_BIAS_OFF].len; col++) {
        param_write(enc_mlp_dense_1_bias[col], param_addr_map_bias[ENC_MLP_DENSE_1_BIAS_OFF].addr + col);
    }
    for (int row=0; row<MLP_DIM; row++) {
        for (int col=0; col<EMB_DEPTH; col++) {
            param_write(enc_mlp_dense_2_kernel[row][col], param_addr_map[ENC_MLP_DENSE_2_PARAMS].addr + row + col*MLP_DIM);
        }
    }
    for (int col=0; col<param_addr_map_bias[ENC_MLP_DENSE_2_BIAS_OFF].len; col++) {
        param_write(enc_mlp_dense_2_bias[col], param_addr_map_bias[ENC_MLP_DENSE_2_BIAS_OFF].addr + col);
    }

    // MLP head
    layernorm_beta = file.getGroup("mlp_head_layerNorm").getGroup(transformer_name).getGroup("mlp_head_layerNorm").getDataSet("beta:0").read<EmbDepthVect_t>(); // Encoder's LayerNorm 2
    layernorm_gamma = file.getGroup("mlp_head_layerNorm").getGroup(transformer_name).getGroup("mlp_head_layerNorm").getDataSet("gamma:0").read<EmbDepthVect_t>(); // Encoder's LayerNorm 2
    for (int col=0; col<param_addr_map_bias[ENC_LAYERNORM_3_BETA_OFF].len; col++) {
        param_write(layernorm_beta[col], param_addr_map_bias[ENC_LAYERNORM_3_BETA_OFF].addr + col);   
    }
    for (int col=0; col<param_addr_map_bias[ENC_LAYERNORM_3_GAMMA_OFF].len; col++) {
        param_write(layernorm_gamma[col], param_addr_map_bias[ENC_LAYERNORM_3_GAMMA_OFF].addr + col);
    }

    // Kernels stored in column-major order!
    EmbDepthxMlpDimMat_t mlp_head_dense_1_kernel = file.getGroup("mlp_head").getGroup("mlp_head_dense1").getDataSet("kernel:0").read<EmbDepthxMlpDimMat_t>();
    MlpDimVect_t mlp_head_dense_1_bias = file.getGroup("mlp_head").getGroup("mlp_head_dense1").getDataSet("bias:0").read<MlpDimVect_t>();
    NumSleepStagesxMlpDimMat_t mlp_head_dense_2_kernel = file.getGroup("mlp_head_softmax").getGroup(transformer_name).getGroup("mlp_head_softmax").getDataSet("kernel:0").read<NumSleepStagesxMlpDimMat_t>();
    NumSleepStagesVect_t mlp_head_dense_2_bias = file.getGroup("mlp_head_softmax").getGroup(transformer_name).getGroup("mlp_head_softmax").getDataSet("bias:0").read<NumSleepStagesVect_t>();
    for (int row=0; row<EMB_DEPTH; row++) {
        for (int col=0; col<MLP_DIM; col++) {
            param_write(mlp_head_dense_1_kernel[row][col], param_addr_map[MLP_HEAD_DENSE_1_PARAMS].addr + row + col*EMB_DEPTH);
        }
    }
    for (int col=0; col<param_addr_map_bias[MLP_HEAD_DENSE_1_BIAS_OFF].len; col++) {
        param_write(mlp_head_dense_1_bias[col], param_addr_map_bias[MLP_HEAD_DENSE_1_BIAS_OFF].addr + col);
    }

    for (int row=0; row<MLP_DIM; row++) {
        for (int col=0; col<NUM_SLEEP_STAGES; col++) {
            param_write(mlp_head_dense_2_kernel[row][col], param_addr_map[MLP_HEAD_DENSE_2_PARAMS].addr + row + col*MLP_DIM);
        }
    }
    for (int col=0; col<param_addr_map_bias[MLP_HEAD_DENSE_2_BIAS_OFF].len; col++) {
        param_write(mlp_head_dense_2_bias[col], param_addr_map_bias[MLP_HEAD_DENSE_2_BIAS_OFF].addr + col);
    }
}

void CiM_Centralized::load_eeg_from_h5(const string eeg_filepath, uint16_t clip_index) {
    // Load parameters from H5 file to memory (done by RISC-V)

    // EEG data file
    HighFive::File eeg_file(eeg_filepath, HighFive::File::ReadOnly);
    vector<vector<int64_t>> eeg_ds = eeg_file.getDataSet("eeg").read<vector<vector<int64_t>>>();

    for (int i=0; i<NUM_PATCHES*PATCH_LEN; i++) {
        int_res_write(static_cast<float>(eeg_ds[clip_index][i])/EEG_SCALE_FACTOR, mem_map.at(EEG_INPUT_MEM) + i, SINGLE_WIDTH);
    }
}

void CiM_Centralized::load_previous_softmax(const string prev_softmax_base_filepath) {
    for (int i = 0; i < (NUM_SAMPLES_OUT_AVG-1); i++) { // Softmax for previous dummy epochs
        string filename = prev_softmax_base_filepath + to_string(i) + ".csv";
        rapidcsv::Document csv(filename, rapidcsv::LabelParams(-1, -1));
        vector<float> dummy_softmax = csv.GetRow<float>(0);
        for (int j = 0; j < NUM_SLEEP_STAGES; j++) {
            int_res_write((dummy_softmax[j] / NUM_SAMPLES_OUT_AVG), mem_map.at(PREV_SOFTMAX_OUTPUT_MEM) + i*NUM_SLEEP_STAGES + j, SINGLE_WIDTH);
        }
    }
}

uint16_t CiM_Centralized::get_softmax_max_index() {
    return _softmax_max_index;
}

bool CiM_Centralized::run(struct ext_signals* ext_sigs, string softmax_base_filepath, string eeg_filepath, uint16_t clip_index) {
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
            if (current_inf_step == INFERENCE_COMPLETE) {
                cim_state = IDLE_CIM;
            }
            break;

        case INVALID_CIM:
            break;
    }

    switch (current_inf_step){
        case PATCH_PROJ_STEP:
            /* gen_cnt_7b holds current parameters row
               gen_cnt_9b holds current patch */

            if (compute_done || (gen_cnt_7b.get_cnt() == 0 && !compute_in_progress)) { // Start a new MAC
                uint32_t patch_addr         = mem_map.at(EEG_INPUT_MEM) + EMB_DEPTH*gen_cnt_9b.get_cnt();
                uint32_t param_addr         = param_addr_map[PATCH_PROJ_KERNEL_PARAMS].addr + EMB_DEPTH*gen_cnt_7b.get_cnt();
                uint32_t param_bias_addr    = param_addr_map_bias[PATCH_PROJ_BIAS_OFF].addr + gen_cnt_7b.get_cnt();
                MAC<dw_fx_x_t,params_fx_2_x_t>(patch_addr, param_addr, PATCH_LEN, param_bias_addr, MODEL_PARAM, LINEAR_ACTIVATION);
            }

            if (compute_done) {
                // Save data
                uint32_t addr = mem_map.at(PATCH_MEM) + gen_cnt_7b.get_cnt() + gen_cnt_9b.get_cnt()*PATCH_LEN;
                int_res_write(computation_result, addr, DOUBLE_WIDTH);

                // Update index control
                if (gen_cnt_7b.get_cnt() == EMB_DEPTH-1) { // Done going through all parameters rows with given patch
                    gen_cnt_7b.reset();
                    if (gen_cnt_9b.get_cnt() == NUM_PATCHES-1) { // Done going through all patches
                        gen_cnt_9b.reset();
                        if (PRINT_INF_PROGRESS) { cout << "Patch projection done" << endl; }
                        verify_layer_out(PATCH_PROJECTION_VERIF, int_res_read, mem_map.at(PATCH_MEM), EMB_DEPTH);
                        current_inf_step = CLASS_TOKEN_CONCAT_STEP;
                    } else { gen_cnt_9b.inc(); } // New patch
                } else { gen_cnt_7b.inc(); }
            }
            break;

        case CLASS_TOKEN_CONCAT_STEP:
            if (gen_cnt_7b.get_cnt() < EMB_DEPTH) {
                uint32_t class_token_addr = param_addr_map_bias[CLASS_TOKEN_OFF].addr + gen_cnt_7b.get_cnt();
                float data = static_cast<float> ( params_fx_2_x_t { param_read(class_token_addr) } );
                int_res_write(data, mem_map.at(CLASS_TOKEN_MEM) + gen_cnt_7b.get_cnt(), DOUBLE_WIDTH);
                gen_cnt_7b.inc();
            } else {
                gen_cnt_7b.reset();
                if (PRINT_INF_PROGRESS) { cout << "Classification token concatenation done" << endl; }
                verify_layer_out(CLASS_TOKEN_VERIF, int_res_read, mem_map.at(CLASS_TOKEN_MEM), EMB_DEPTH);
                current_inf_step = POS_EMB_STEP;
            }
            break;

        case POS_EMB_STEP:
            /* gen_cnt_7b holds the column
               gen_cnt_9b holds the row */

            if (compute_done || (gen_cnt_7b.get_cnt() == 0 && gen_cnt_9b.get_cnt() == 0 && !compute_in_progress)) { // Start a new ADD
                uint32_t int_res_addr   = mem_map.at(CLASS_TOKEN_MEM)           + gen_cnt_7b.get_cnt() + EMB_DEPTH*gen_cnt_9b.get_cnt();
                uint32_t params_addr    = param_addr_map[POS_EMB_PARAMS].addr   + gen_cnt_7b.get_cnt() + EMB_DEPTH*gen_cnt_9b.get_cnt();
                ADD<dw_fx_x_t,params_fx_2_x_t>(int_res_addr, params_addr, MODEL_PARAM);
            }

            if (compute_done) {
                // Save data
                uint32_t addr = mem_map.at(POS_EMB_MEM) + gen_cnt_7b.get_cnt() + EMB_DEPTH*gen_cnt_9b.get_cnt( );
                int_res_write(computation_result, addr, DOUBLE_WIDTH);

                // Update index control
                if (gen_cnt_7b.get_cnt() == EMB_DEPTH-1) { // Done going through all columns of a given row
                    gen_cnt_7b.reset();
                    if (gen_cnt_9b.get_cnt() == NUM_PATCHES) { // Done going through all rows
                        gen_cnt_9b.reset();
                        if (PRINT_INF_PROGRESS) { cout << "Positional embedding done" << endl; }
                        verify_layer_out(POS_EMB_VERIF, int_res_read, mem_map.at(POS_EMB_MEM), EMB_DEPTH);
                        current_inf_step = ENC_LAYERNORM_1_1ST_HALF_STEP;
                    } else { gen_cnt_9b.inc(); }// New row
                } else { gen_cnt_7b.inc(); }
            }
            break;

        case ENC_LAYERNORM_1_1ST_HALF_STEP:
        case ENC_LAYERNORM_2_1ST_HALF_STEP:
        case ENC_LAYERNORM_3_1ST_HALF_STEP: {
            /* gen_cnt_7b holds the current row to which we apply normalization */

            uint16_t num_rows;

            if (current_inf_step == ENC_LAYERNORM_3_1ST_HALF_STEP) { num_rows = 1; }
            else { num_rows = NUM_PATCHES+1; }

            if (compute_done || (gen_cnt_7b.get_cnt() == 0 && !compute_in_progress)) { // Start a new LAYERNORM
                uint32_t input_starting_addr, output_starting_addr;
                if (current_inf_step == ENC_LAYERNORM_1_1ST_HALF_STEP) {
                    input_starting_addr = mem_map.at(POS_EMB_MEM) + EMB_DEPTH*gen_cnt_7b.get_cnt();
                    output_starting_addr = mem_map.at(ENC_LN1_MEM) + EMB_DEPTH*gen_cnt_7b.get_cnt();
                } else if (current_inf_step == ENC_LAYERNORM_2_1ST_HALF_STEP) {
                    input_starting_addr = mem_map.at(ENC_MHSA_OUT_MEM) + EMB_DEPTH*gen_cnt_7b.get_cnt();
                    output_starting_addr = mem_map.at(ENC_LN2_MEM) + EMB_DEPTH*gen_cnt_7b.get_cnt();
                } else if (current_inf_step == ENC_LAYERNORM_3_1ST_HALF_STEP) {
                    input_starting_addr = mem_map.at(ENC_MLP_OUT_MEM);
                    output_starting_addr = mem_map.at(ENC_LN3_MEM);
                }
                LAYERNORM_1ST_HALF<dw_fx_x_t>(input_starting_addr, output_starting_addr);
            }

            if (compute_done) {
                if (gen_cnt_7b.get_cnt() == num_rows-1) { // Done going through all rows
                    gen_cnt_7b.reset();
                    current_inf_step = static_cast<INFERENCE_STEP> (static_cast<int> (current_inf_step) + 1);
                } else { gen_cnt_7b.inc(); }
            }
            break;
        }

        case ENC_LAYERNORM_1_2ND_HALF_STEP:
        case ENC_LAYERNORM_2_2ND_HALF_STEP:
        case ENC_LAYERNORM_3_2ND_HALF_STEP:
            /* gen_cnt_7b holds the current column to which we apply centering and scaling */

            uint16_t num_col;

            if (compute_done || (gen_cnt_7b.get_cnt() == 0 && !compute_in_progress)) { // Start a new LAYERNORM
            uint32_t input_starting_addr, gamma_addr, beta_addr, num_rows;
                if (current_inf_step == ENC_LAYERNORM_1_2ND_HALF_STEP) {
                    input_starting_addr = mem_map.at(ENC_LN1_MEM) + gen_cnt_7b.get_cnt();
                    gamma_addr = param_addr_map_bias[ENC_LAYERNORM_1_GAMMA_OFF].addr + gen_cnt_7b.get_cnt();
                    beta_addr = param_addr_map_bias[ENC_LAYERNORM_1_BETA_OFF].addr + gen_cnt_7b.get_cnt();
                    num_rows = NUM_PATCHES + 1;
                } else if (current_inf_step == ENC_LAYERNORM_2_2ND_HALF_STEP) {
                    input_starting_addr = mem_map.at(ENC_LN2_MEM) + gen_cnt_7b.get_cnt();
                    gamma_addr = param_addr_map_bias[ENC_LAYERNORM_2_GAMMA_OFF].addr + gen_cnt_7b.get_cnt();
                    beta_addr = param_addr_map_bias[ENC_LAYERNORM_2_BETA_OFF].addr + gen_cnt_7b.get_cnt();
                    num_rows = NUM_PATCHES + 1;
                } else if (current_inf_step == ENC_LAYERNORM_3_2ND_HALF_STEP) {
                    input_starting_addr = mem_map.at(ENC_LN3_MEM) + gen_cnt_7b.get_cnt();
                    gamma_addr = param_addr_map_bias[ENC_LAYERNORM_3_GAMMA_OFF].addr + gen_cnt_7b.get_cnt();
                    beta_addr = param_addr_map_bias[ENC_LAYERNORM_3_BETA_OFF].addr + gen_cnt_7b.get_cnt();
                    num_rows = 1;
                }
                LAYERNORM_2ND_HALF<dw_fx_x_t, params_fx_3_x_t>(input_starting_addr, gamma_addr, beta_addr, num_rows);
            }

            if (compute_done) {
                if (gen_cnt_7b.get_cnt() == EMB_DEPTH-1) { // Done going through all columns
                    gen_cnt_7b.reset();
                    if (PRINT_INF_PROGRESS) { cout << "Finished LayerNorm" << endl; }
                    if (current_inf_step == ENC_LAYERNORM_1_2ND_HALF_STEP) {
                        verify_layer_out(ENC_LAYERNORM1_VERIF, int_res_read, mem_map.at(ENC_LN1_MEM), EMB_DEPTH);
                    } else if (current_inf_step == ENC_LAYERNORM_2_2ND_HALF_STEP) {
                        verify_layer_out(ENC_LAYERNORM2_VERIF, int_res_read, mem_map.at(ENC_LN2_MEM), EMB_DEPTH);
                    } else if (current_inf_step == ENC_LAYERNORM_3_2ND_HALF_STEP) {
                        verify_layer_out(MLP_HEAD_LAYERNORM_VERIF, int_res_read, mem_map.at(ENC_LN3_MEM), 1);
                    }
                    current_inf_step = static_cast<INFERENCE_STEP> (static_cast<int> (current_inf_step) + 1);
                } else { gen_cnt_7b.inc(); }
            }
            break;

        case POS_EMB_COMPRESSION_STEP: {
            /* Compress the positional embedding to single width to save on storage
            -gen_cnt_7b holds the current column
            -gen_cnt_9b holds the current row*/
            uint32_t in_addr = mem_map.at(POS_EMB_MEM) + gen_cnt_7b.get_cnt() + EMB_DEPTH*gen_cnt_9b.get_cnt();
            uint32_t out_addr = mem_map.at(POS_EMB_MEM) + gen_cnt_7b.get_cnt() + EMB_DEPTH*gen_cnt_9b.get_cnt();
            int_res_write(int_res_read(in_addr), out_addr, SINGLE_WIDTH);

            if (gen_cnt_7b.get_cnt() == EMB_DEPTH-1) {
                gen_cnt_7b.reset();
                gen_cnt_9b.inc();
            } else { gen_cnt_7b.inc(); }

            if (gen_cnt_9b.get_cnt() == NUM_PATCHES+1) {
                gen_cnt_7b.reset();
                gen_cnt_9b.reset();
                current_inf_step = ENC_MHSA_Q_STEP;
                if (PRINT_INF_PROGRESS) { cout << "Done compressing positional embedding from DOUBLE_WIDTH to SINGLE_WIDTH" << endl; }
                verify_layer_out(POS_EMB_VERIF, int_res_read, mem_map.at(POS_EMB_MEM), EMB_DEPTH);
            }
            break;
        }
        case ENC_MHSA_Q_STEP:
        case ENC_MHSA_K_STEP:
        case ENC_MHSA_V_STEP:
        case MLP_DENSE_1_STEP:
        case MLP_HEAD_DENSE_1_STEP:
        case MLP_HEAD_DENSE_2_STEP: {
            /* gen_cnt_7b holds current data row
               gen_cnt_9b holds current parameter column (assumes kernel is stored in column-major order)*/

            uint16_t input_height, kernel_width;
            DATA_WIDTH output_data_width;

            if (current_inf_step == MLP_DENSE_1_STEP) {
                kernel_width = MLP_DIM;
                input_height = NUM_PATCHES + 1;
                output_data_width = DOUBLE_WIDTH;
            } else if (current_inf_step == MLP_HEAD_DENSE_1_STEP) {
                kernel_width = MLP_DIM;
                input_height = 1;
                output_data_width = DOUBLE_WIDTH;
            } else if (current_inf_step == MLP_HEAD_DENSE_2_STEP) {
                kernel_width = NUM_SLEEP_STAGES;
                input_height = 1;
                output_data_width = DOUBLE_WIDTH;
            } else {
                kernel_width = EMB_DEPTH;
                input_height = NUM_PATCHES + 1;
                output_data_width = SINGLE_WIDTH;
            }

            if (compute_done || (gen_cnt_7b.get_cnt() == 0 && gen_cnt_9b.get_cnt() == 0 && !compute_in_progress)) { // Start a new MAC
                uint32_t kernel_col_addr, bias_addr, data_row_addr;
                if (current_inf_step == ENC_MHSA_Q_STEP) {
                    kernel_col_addr = param_addr_map[ENC_Q_DENSE_PARAMS].addr + EMB_DEPTH*gen_cnt_9b.get_cnt();
                    bias_addr = param_addr_map_bias[ENC_Q_DENSE_BIAS_0FF].addr + gen_cnt_9b.get_cnt();
                    data_row_addr = mem_map.at(ENC_LN1_MEM) + EMB_DEPTH*gen_cnt_7b.get_cnt();
                } else if (current_inf_step == ENC_MHSA_K_STEP) {
                    kernel_col_addr = param_addr_map[ENC_K_DENSE_PARAMS].addr + EMB_DEPTH*gen_cnt_9b.get_cnt();
                    bias_addr = param_addr_map_bias[ENC_K_DENSE_BIAS_0FF].addr + gen_cnt_9b.get_cnt();
                    data_row_addr = mem_map.at(ENC_LN1_MEM) + EMB_DEPTH*gen_cnt_7b.get_cnt();
                } else if (current_inf_step == ENC_MHSA_V_STEP) {
                    kernel_col_addr = param_addr_map[ENC_V_DENSE_PARAMS].addr + EMB_DEPTH*gen_cnt_9b.get_cnt();
                    bias_addr = param_addr_map_bias[ENC_V_DENSE_BIAS_0FF].addr + gen_cnt_9b.get_cnt();
                    data_row_addr = mem_map.at(ENC_LN1_MEM) + EMB_DEPTH*gen_cnt_7b.get_cnt();
                } else if (current_inf_step == MLP_DENSE_1_STEP) {
                    kernel_col_addr = param_addr_map[ENC_MLP_DENSE_1_PARAMS].addr + EMB_DEPTH*gen_cnt_9b.get_cnt();
                    bias_addr = param_addr_map_bias[ENC_MLP_DENSE_1_BIAS_OFF].addr + gen_cnt_9b.get_cnt();
                    data_row_addr = mem_map.at(ENC_LN2_MEM) + EMB_DEPTH*gen_cnt_7b.get_cnt();
                } else if (current_inf_step == MLP_HEAD_DENSE_1_STEP) {
                    kernel_col_addr = param_addr_map[MLP_HEAD_DENSE_1_PARAMS].addr + EMB_DEPTH*gen_cnt_9b.get_cnt();
                    bias_addr = param_addr_map_bias[MLP_HEAD_DENSE_1_BIAS_OFF].addr + gen_cnt_9b.get_cnt();
                    data_row_addr = mem_map.at(ENC_LN3_MEM);
                } else if (current_inf_step == MLP_HEAD_DENSE_2_STEP) {
                    kernel_col_addr = param_addr_map[MLP_HEAD_DENSE_2_PARAMS].addr + MLP_DIM*gen_cnt_9b.get_cnt();
                    bias_addr = param_addr_map_bias[MLP_HEAD_DENSE_2_BIAS_OFF].addr + gen_cnt_9b.get_cnt();
                    data_row_addr = mem_map.at(MLP_HEAD_DENSE_1_OUT_MEM);
                }

                if (current_inf_step == MLP_DENSE_1_STEP || current_inf_step == MLP_HEAD_DENSE_1_STEP) { MAC<dw_fx_x_t,params_fx_2_x_t>(data_row_addr, kernel_col_addr, EMB_DEPTH, bias_addr, MODEL_PARAM, SWISH_ACTIVATION); }
                else if (current_inf_step == MLP_HEAD_DENSE_2_STEP) { MAC<dw_fx_x_t,params_fx_5_x_t>(data_row_addr, kernel_col_addr, MLP_DIM, bias_addr, MODEL_PARAM, LINEAR_ACTIVATION); }
                else { MAC<dw_fx_x_t,params_fx_2_x_t>(data_row_addr, kernel_col_addr, EMB_DEPTH, bias_addr, MODEL_PARAM, LINEAR_ACTIVATION); }
            }

            if (compute_done) {
                // Save data
                uint32_t addr;
                if (current_inf_step == ENC_MHSA_Q_STEP) { addr = mem_map.at(ENC_Q_MEM) + EMB_DEPTH*gen_cnt_7b.get_cnt() + gen_cnt_9b.get_cnt(); }
                else if (current_inf_step == ENC_MHSA_V_STEP) { addr = mem_map.at(ENC_V_MEM) + EMB_DEPTH*gen_cnt_7b.get_cnt() + gen_cnt_9b.get_cnt(); }
                else if (current_inf_step == ENC_MHSA_K_STEP) { addr = mem_map.at(ENC_K_MEM) + EMB_DEPTH*gen_cnt_7b.get_cnt() + gen_cnt_9b.get_cnt(); }
                else if (current_inf_step == MLP_DENSE_1_STEP) { addr = mem_map.at(ENC_MLP_DENSE1_MEM) + MLP_DIM*gen_cnt_7b.get_cnt() + gen_cnt_9b.get_cnt(); }
                else if (current_inf_step == MLP_HEAD_DENSE_1_STEP) { addr = mem_map.at(MLP_HEAD_DENSE_1_OUT_MEM) + gen_cnt_9b.get_cnt(); }
                else if (current_inf_step == MLP_HEAD_DENSE_2_STEP) { addr = mem_map.at(MLP_HEAD_DENSE_2_OUT_MEM) + gen_cnt_9b.get_cnt(); }

                int_res_write(computation_result, addr, output_data_width);

                // Update index control
                if (gen_cnt_7b.get_cnt() == input_height-1) { // Done going through all data rows with given parameter column
                    gen_cnt_7b.reset();
                    if (gen_cnt_9b.get_cnt() == kernel_width-1) { // Done going through all kernel columns
                        gen_cnt_9b.reset();
                        if (current_inf_step == ENC_MHSA_Q_STEP) {
                            if (PRINT_INF_PROGRESS) { cout << "Encoder's Q matrix done" << endl; }
                            verify_layer_out(ENC_MHSA_DENSE_Q_VERIF, int_res_read, mem_map.at(ENC_Q_MEM), EMB_DEPTH);
                            current_inf_step = ENC_MHSA_K_STEP;
                        } else if (current_inf_step == ENC_MHSA_K_STEP) {
                            if (PRINT_INF_PROGRESS) { cout << "Encoder's K matrix done" << endl; }
                            verify_layer_out(ENC_MHSA_DENSE_K_VERIF, int_res_read, mem_map.at(ENC_K_MEM), EMB_DEPTH);
                            current_inf_step = ENC_MHSA_V_STEP;
                        } else if (current_inf_step == ENC_MHSA_V_STEP) {
                            if (PRINT_INF_PROGRESS) { cout << "Encoder's V matrix done" << endl; }
                            verify_layer_out(ENC_MHSA_DENSE_V_VERIF, int_res_read, mem_map.at(ENC_V_MEM), EMB_DEPTH);
                            current_inf_step = ENC_MHSA_QK_T_STEP;
                        } else if (current_inf_step == MLP_DENSE_1_STEP) {
                            if (PRINT_INF_PROGRESS) { cout << "MLP dense 1 done" << endl; }
                            verify_layer_out(ENC_MLP_DENSE1_VERIF, int_res_read, mem_map.at(ENC_MLP_DENSE1_MEM), MLP_DIM);
                            current_inf_step = MLP_DENSE_2_AND_SUM_STEP;
                        } else if (current_inf_step == MLP_HEAD_DENSE_1_STEP) {
                            if (PRINT_INF_PROGRESS) { cout << "MLP head dense 1 done" << endl; }
                            verify_layer_out(MLP_HEAD_DENSE_1_VERIF, int_res_read, mem_map.at(MLP_HEAD_DENSE_1_OUT_MEM), 1);
                            current_inf_step = MLP_HEAD_DENSE_2_STEP;
                        } else if (current_inf_step == MLP_HEAD_DENSE_2_STEP) {
                            if (PRINT_INF_PROGRESS) { cout << "MLP head dense 2 done" << endl; }
                            current_inf_step = MLP_HEAD_SOFTMAX_STEP;
                        }
                    } else { gen_cnt_9b.inc(); } // New kernel column
                } else { gen_cnt_7b.inc(); } // New input row
            }
            break;
        }

        case ENC_MHSA_QK_T_STEP:
            /* gen_cnt_7b holds x
               gen_cnt_9b holds y
               gen_cnt_4b holds z

            for z in 0...(NUM_HEADS-1):
                for y in 0...(NUM_PATCHES):
                    for x 0...(NUM_PATCHES):
            */

            if (compute_done || (gen_cnt_7b.get_cnt() == 0 && gen_cnt_9b.get_cnt() == 0 && gen_cnt_4b.get_cnt() == 0 && !compute_in_progress)) { // Start a new MAC
                uint32_t Q_addr     = mem_map.at(ENC_Q_MEM) + /*x*/ 0*gen_cnt_7b.get_cnt()          + /*y*/ EMB_DEPTH*gen_cnt_9b.get_cnt()  + /*z*/ NUM_HEADS*gen_cnt_4b.get_cnt();
                uint32_t K_T_addr   = mem_map.at(ENC_K_MEM) + /*x*/ EMB_DEPTH*gen_cnt_7b.get_cnt()  + /*y*/ 0*gen_cnt_9b.get_cnt()          + /*z*/ NUM_HEADS*gen_cnt_4b.get_cnt();
                MAC<sw_fx_5_x_t,sw_fx_5_x_t>(Q_addr, K_T_addr, NUM_HEADS, 0, INTERMEDIATE_RES, NO_ACTIVATION);
                mac_or_div = DIV_OP; // Next we do the division by sqrt(NUM_HEADS)
            }

            if (compute_done || generic_done) {
                // Save data
                if (mac_or_div == DIV_OP) {
                    float inv_sqrt_num_heads = param_read(param_addr_map_bias[ENC_INV_SQRT_NUM_HEADS_OFF].addr);
                    computation_result = static_cast<float>(static_cast<comp_fx_t>(computation_result ) * static_cast<params_fx_4_x_t>(inv_sqrt_num_heads)); // Divide by sqrt(NUM_HEADS). Done in ASIC before writing to mem, so can be left cast as comp_fx_t
                    generic_done = true;
                    mac_or_div = MAC_OP;
                } else {
                    generic_done = false;
                    uint32_t output_addr = mem_map.at(ENC_QK_T_MEM) + gen_cnt_7b.get_cnt() + (NUM_PATCHES+1)*gen_cnt_9b.get_cnt() + (NUM_PATCHES+1)*(NUM_PATCHES+1)*gen_cnt_4b.get_cnt();
                    int_res_write(computation_result, output_addr, SINGLE_WIDTH);

                    // Update counters
                    if (gen_cnt_7b.get_cnt() == NUM_PATCHES) {
                        gen_cnt_7b.reset();
                        if (gen_cnt_9b.get_cnt() == NUM_PATCHES) {
                            gen_cnt_9b.reset();
                            if (gen_cnt_4b.get_cnt() == (NUM_HEADS-1)) {
                                gen_cnt_4b.reset();
                                if (PRINT_INF_PROGRESS) { cout << "Finished Encoder MHSA's QK_T." << endl; }
                                current_inf_step = ENC_MHSA_SOFTMAX_STEP;
                                verify_layer_out(ENC_MHSA_DENSE_QK_T_VERIF, int_res_read, mem_map.at(ENC_QK_T_MEM), NUM_PATCHES+1);
                            } else { gen_cnt_4b.inc(); } // z++
                        } else { gen_cnt_9b.inc(); } // y++
                    } else { gen_cnt_7b.inc(); } // x++
                }
            }
            break;

        case ENC_MHSA_SOFTMAX_STEP:
        case MLP_HEAD_SOFTMAX_STEP: {
            /* gen_cnt_9b holds row over which to do softmax */
            uint16_t num_rows;
            if (current_inf_step == ENC_MHSA_SOFTMAX_STEP) { num_rows = NUM_HEADS*(NUM_PATCHES+1); }
            else { num_rows = 1; }

            if (compute_done || (gen_cnt_9b.get_cnt() == 0 && !compute_in_progress)) { // Start a new MAC
                uint32_t MAC_storage_addr, len;
                if (current_inf_step == ENC_MHSA_SOFTMAX_STEP) {
                    MAC_storage_addr = mem_map.at(ENC_QK_T_MEM) + (NUM_PATCHES+1)*gen_cnt_9b.get_cnt();
                    SOFTMAX<sw_fx_6_x_t>(MAC_storage_addr, NUM_PATCHES+1, SINGLE_WIDTH);
                } else { SOFTMAX<dw_fx_x_t>(mem_map.at(MLP_HEAD_DENSE_2_OUT_MEM), NUM_SLEEP_STAGES, DOUBLE_WIDTH); }
            }

            if (compute_done) {
                if (gen_cnt_9b.get_cnt() == num_rows-1) {
                    gen_cnt_9b.reset();
                    if (current_inf_step == ENC_MHSA_SOFTMAX_STEP) {
                        current_inf_step = ENC_MHSA_MULT_V_STEP;
                        if (PRINT_INF_PROGRESS) { cout << "Finished encoder's MHSA softmax" << endl; }
                        verify_layer_out(ENC_SOFTMAX_VERIF, int_res_read, mem_map.at(ENC_QK_T_MEM), NUM_PATCHES+1);
                    } else {
                        current_inf_step = SOFTMAX_DIVIDE_STEP;
                        if (PRINT_INF_PROGRESS) { cout << "Finished MLP head softmax" << endl; }
                        verify_layer_out(MLP_HEAD_SOFTMAX_VERIF, int_res_read, mem_map.at(MLP_HEAD_DENSE_2_OUT_MEM), 1);
                        _softmax_max_index = find_softmax_argmax(mem_map.at(MLP_HEAD_DENSE_2_OUT_MEM));
                        print_softmax_error(int_res_0, mem_map.at(MLP_HEAD_DENSE_2_OUT_MEM));
                    }
                } else { gen_cnt_9b.inc(); }
            }
            break;
        }

        case ENC_MHSA_MULT_V_STEP:
            /* gen_cnt_7b holds x
               gen_cnt_9b holds y
               gen_cnt_4b holds z

            for z in 0...(NUM_HEADS-1):
                for y in 0...(NUM_PATCHES):
                    for x 0...(EMB_DEPTH/NUM_HEADS-1):
            */

            if (compute_done || (gen_cnt_7b.get_cnt() == 0 && gen_cnt_9b.get_cnt() == 0 && gen_cnt_4b.get_cnt() == 0 && !compute_in_progress)) { // Start a new MAC
                uint32_t QK_T_addr  = mem_map.at(ENC_QK_T_MEM)  + /*x*/ 0*gen_cnt_7b.get_cnt()  + /*y*/ (NUM_PATCHES+1)*gen_cnt_9b.get_cnt()    + /*z*/ (NUM_PATCHES+1)*(NUM_PATCHES+1)*gen_cnt_4b.get_cnt();
                uint32_t V_addr     = mem_map.at(ENC_V_MEM)     + /*x*/ gen_cnt_7b.get_cnt()    + /*y*/ 0*gen_cnt_9b.get_cnt()                  + /*z*/ NUM_HEADS*gen_cnt_4b.get_cnt();
                MAC<sw_fx_2_x_t,sw_fx_5_x_t>(QK_T_addr, V_addr, NUM_PATCHES+1, 0, INTERMEDIATE_RES, NO_ACTIVATION, VERTICAL, EMB_DEPTH);
            }

            if (compute_done) {
                // Save data
                uint32_t output_addr = mem_map.at(ENC_V_MULT_MEM) + gen_cnt_7b.get_cnt() + EMB_DEPTH*gen_cnt_9b.get_cnt() + (EMB_DEPTH/NUM_HEADS)*gen_cnt_4b.get_cnt();
                int_res_write(computation_result, output_addr, SINGLE_WIDTH);

                // Update counters
                if (gen_cnt_7b.get_cnt() == (EMB_DEPTH/NUM_HEADS-1)) {
                    gen_cnt_7b.reset();
                    if (gen_cnt_9b.get_cnt() == NUM_PATCHES) {
                        gen_cnt_9b.reset();
                        if (gen_cnt_4b.get_cnt() == (NUM_HEADS-1)) {
                            gen_cnt_4b.reset();
                            current_inf_step = ENC_POST_MHSA_DENSE_AND_INPUT_SUM_STEP;
                            if (PRINT_INF_PROGRESS) { cout << "Finished encoder's MHSA mult V" << endl; }
                            verify_layer_out(ENC_MULT_V_VERIF, int_res_read, mem_map.at(ENC_V_MULT_MEM), EMB_DEPTH);
                        } else { gen_cnt_4b.inc(); }// z++
                    } else { gen_cnt_9b.inc(); } // y++
                } else { gen_cnt_7b.inc(); } // x++
            }
            break;

        case ENC_POST_MHSA_DENSE_AND_INPUT_SUM_STEP:
        case MLP_DENSE_2_AND_SUM_STEP: {
            /* gen_cnt_7b holds x
               gen_cnt_9b holds y

            for y in 0...(NUM_PATCHES):
                for x 0...(EMB_DEPTH-1):
            */

            uint16_t input_height;

            if (current_inf_step == ENC_POST_MHSA_DENSE_AND_INPUT_SUM_STEP) {
                input_height = NUM_PATCHES+1;
            } else if (current_inf_step == MLP_DENSE_2_AND_SUM_STEP) {
                input_height = 1; // We only compute the first row of the output
            }

            if (compute_done || (gen_cnt_7b.get_cnt() == 0 && gen_cnt_9b.get_cnt() == 0 && !compute_in_progress)) { // Start a new MAC
                if (current_inf_step == ENC_POST_MHSA_DENSE_AND_INPUT_SUM_STEP) {
                    uint32_t input_addr = mem_map.at(ENC_V_MULT_MEM) + EMB_DEPTH*gen_cnt_9b.get_cnt();
                    uint32_t kernel_addr = param_addr_map[ENC_COMB_HEAD_PARAMS].addr + EMB_DEPTH*gen_cnt_7b.get_cnt();
                    uint32_t bias_addr = param_addr_map_bias[ENC_COMB_HEAD_BIAS_OFF].addr + gen_cnt_7b.get_cnt();
                    MAC<dw_fx_x_t,params_fx_2_x_t>(input_addr, kernel_addr, EMB_DEPTH, bias_addr, MODEL_PARAM, LINEAR_ACTIVATION);
                } else if (current_inf_step == MLP_DENSE_2_AND_SUM_STEP) {
                    uint32_t input_addr = mem_map.at(ENC_MLP_DENSE1_MEM); // Only one row
                    uint32_t kernel_addr = param_addr_map[ENC_MLP_DENSE_2_PARAMS].addr + MLP_DIM*gen_cnt_7b.get_cnt();
                    uint32_t bias_addr = param_addr_map_bias[ENC_MLP_DENSE_2_BIAS_OFF].addr + gen_cnt_7b.get_cnt();
                    MAC<dw_fx_x_t,params_fx_2_x_t>(input_addr, kernel_addr, MLP_DIM, bias_addr, MODEL_PARAM, LINEAR_ACTIVATION);
                }
                mac_or_add = ADD_OP;
            }

            if (compute_done || generic_done) {
                if (mac_or_add == ADD_OP) {
                    uint32_t add_addr;
                    if (current_inf_step == ENC_POST_MHSA_DENSE_AND_INPUT_SUM_STEP) { add_addr = mem_map.at(POS_EMB_MEM) + gen_cnt_7b.get_cnt() + EMB_DEPTH*gen_cnt_9b.get_cnt(); }
                    else { add_addr = mem_map.at(ENC_MHSA_OUT_MEM) + gen_cnt_7b.get_cnt(); }
                    computation_result += int_res_read(add_addr);
                    mac_or_add = MAC_OP;
                    generic_done = true;
                } else {
                    uint32_t output_addr;
                    if (current_inf_step == ENC_POST_MHSA_DENSE_AND_INPUT_SUM_STEP) { output_addr = mem_map.at(ENC_MHSA_OUT_MEM) + gen_cnt_7b.get_cnt() + EMB_DEPTH*gen_cnt_9b.get_cnt(); }
                    else if (current_inf_step == MLP_DENSE_2_AND_SUM_STEP) { output_addr = mem_map.at(ENC_MLP_OUT_MEM) + gen_cnt_7b.get_cnt(); }
                    int_res_write(computation_result, output_addr, DOUBLE_WIDTH);
                    generic_done = false;

                    if (gen_cnt_7b.get_cnt() == EMB_DEPTH-1) {
                        gen_cnt_7b.reset();
                        if (gen_cnt_9b.get_cnt() == input_height-1) {
                            gen_cnt_9b.reset();

                            if (current_inf_step == ENC_POST_MHSA_DENSE_AND_INPUT_SUM_STEP) {
                                current_inf_step = ENC_LAYERNORM_2_1ST_HALF_STEP;
                                if (PRINT_INF_PROGRESS) { cout << "Finished encoder's MHSA dense and input step" << endl; }
                                verify_layer_out(ENC_RES_SUM_1_VERIF, int_res_read, mem_map.at(ENC_MHSA_OUT_MEM), EMB_DEPTH);
                            } else if (current_inf_step == MLP_DENSE_2_AND_SUM_STEP) {
                                current_inf_step = ENC_LAYERNORM_3_1ST_HALF_STEP;
                                if (PRINT_INF_PROGRESS) { cout << "Finished MLP's dense 2 and input sum step" << endl; }
                                verify_layer_out(ENC_OUT_VERIF, int_res_read, mem_map.at(ENC_MLP_OUT_MEM), EMB_DEPTH);
                            }
                        } else { gen_cnt_9b.inc(); }
                    } else { gen_cnt_7b.inc(); }
                }
            }
            break;
        }

        case SOFTMAX_DIVIDE_STEP:
            /* gen_cnt_4b holds internal step
               gen_cnt_7b holds the sleep stage
            */

            if (gen_cnt_4b.get_cnt() == 0) { // Grab from memory
                computation_result = comp_fx_t { int_res_read(mem_map.at(MLP_HEAD_DENSE_2_OUT_MEM) + gen_cnt_7b.get_cnt()) };
            } else if (gen_cnt_4b.get_cnt() == 1) {
                computation_result *= static_cast<float> ( comp_fx_t { 1.0f / NUM_SAMPLES_OUT_AVG } ); // Multiply by 1/NUM_SAMPLES_OUT_AVG saves cycles on the ASIC vs dividing by NUM_SAMPLES_OUT_AVG
            } else if (gen_cnt_4b.get_cnt() == 2) {
                int_res_write(computation_result, mem_map.at(MLP_HEAD_DENSE_2_OUT_MEM) + gen_cnt_7b.get_cnt(), SINGLE_WIDTH);
            } else if (gen_cnt_4b.get_cnt() == 3) {
                int_res_write(computation_result, mem_map.at(SOFTMAX_AVG_SUM_MEM) + gen_cnt_7b.get_cnt(), SINGLE_WIDTH);
            }

            if (gen_cnt_4b.get_cnt() == 3) {
                gen_cnt_4b.reset();
                if (gen_cnt_7b.get_cnt() == NUM_SLEEP_STAGES-1) {
                    gen_cnt_7b.reset();
                    if (PRINT_INF_PROGRESS) { cout << "Finished MLP head's Softmax averaging divide" << endl; }
                    verify_layer_out(MLP_HEAD_SOFTMAX_DIV_VERIF, int_res_read, mem_map.at(SOFTMAX_AVG_SUM_MEM), 1);
                    current_inf_step = SOFTMAX_AVERAGING_STEP;
                } else { gen_cnt_7b.inc(); } // Next sleep stage
            } else { gen_cnt_4b.inc(); }
            break;

        case SOFTMAX_AVERAGING_STEP: {
            /* gen_cnt_7b holds the current sleep stage within an epoch's softmax
               gen_cnt_9b holds the epoch
               gen_cnt_4b holds internal step
            */

            uint32_t addr_prev_softmax = mem_map.at(PREV_SOFTMAX_OUTPUT_MEM) + gen_cnt_7b.get_cnt() + gen_cnt_9b.get_cnt()*NUM_SLEEP_STAGES;
            uint32_t addr_softmax_divide_sum = mem_map.at(SOFTMAX_AVG_SUM_MEM) + gen_cnt_7b.get_cnt();

            if (gen_cnt_4b.get_cnt() == 0) {
                computation_result = int_res_read(addr_prev_softmax); // Prev softmax
            } else if (gen_cnt_4b.get_cnt() == 1) {
                computation_result += int_res_read(addr_softmax_divide_sum); // Current accumulator
            } else if (gen_cnt_4b.get_cnt() == 2) {
                int_res_write(computation_result, addr_softmax_divide_sum, SINGLE_WIDTH); // Update accumulator
            }

            if (gen_cnt_4b.get_cnt() == 2) {
                gen_cnt_4b.reset();
                if (gen_cnt_7b.get_cnt() == NUM_SLEEP_STAGES-1) {
                    gen_cnt_7b.reset();
                    if (gen_cnt_9b.get_cnt() == NUM_SAMPLES_OUT_AVG-2) {
                        gen_cnt_9b.reset();
                        current_inf_step = SOFTMAX_AVERAGE_ARGMAX_STEP;
                    } else { gen_cnt_9b.inc(); }
                } else { gen_cnt_7b.inc(); }
            } else { gen_cnt_4b.inc(); }
            break;
        }

        case SOFTMAX_AVERAGE_ARGMAX_STEP:
            if (compute_done) {
                current_inf_step = SOFTMAX_RETIRE_STEP;
                _inferred_sleep_stage = static_cast<int16_t> (computation_result);
                if (PRINT_INF_PROGRESS) { cout << "Finished softmax argmax." << endl; }
                verify_layer_out(POST_SOFTMAX_AVG_VERIF, int_res_read, mem_map.at(SOFTMAX_AVG_SUM_MEM), 1);
            } else if (!compute_in_progress) { ARGMAX(mem_map.at(SOFTMAX_AVG_SUM_MEM), NUM_SLEEP_STAGES); } // Start a ARGMAX in the background. Result will be in computation_result.
            break;

        case SOFTMAX_RETIRE_STEP:
            /* Move dummy #0 into dummy #1's position and current softmax into dummy #0
               gen_cnt_7b holds the current sleep stage within an epoch's softmax
               gen_cnt_4b holds internal step
            */

            if (gen_cnt_4b.get_cnt() == 0) { // Grab dummy #0
                uint32_t addr = mem_map.at(PREV_SOFTMAX_OUTPUT_MEM) + gen_cnt_7b.get_cnt();
                computation_result = int_res_read(addr);
            } else if (gen_cnt_4b.get_cnt() == 1) { // Write to dummy #1
                uint32_t addr = mem_map.at(PREV_SOFTMAX_OUTPUT_MEM) + (gen_cnt_7b.get_cnt() + NUM_SLEEP_STAGES);
                int_res_write(computation_result, addr, SINGLE_WIDTH);
            } else if (gen_cnt_4b.get_cnt() == 2) { // Grab current epoch
                uint32_t addr = mem_map.at(MLP_HEAD_DENSE_2_OUT_MEM) + gen_cnt_7b.get_cnt();
                computation_result = int_res_read(addr);
            } else if (gen_cnt_4b.get_cnt() == 3) { // Write to dummy #0
                uint32_t addr = mem_map.at(PREV_SOFTMAX_OUTPUT_MEM) + gen_cnt_7b.get_cnt();
                int_res_write(computation_result, addr, SINGLE_WIDTH);
            }

            if (gen_cnt_4b.get_cnt() == 3) {
                gen_cnt_4b.reset();
                if (gen_cnt_7b.get_cnt() == NUM_SLEEP_STAGES-1) {
                    gen_cnt_7b.reset();
                    current_inf_step = INFERENCE_COMPLETE;
                    verify_softmax_storage(int_res_3, mem_map.at(PREV_SOFTMAX_OUTPUT_MEM) - 3*CIM_INT_RES_BANK_SIZE_NUM_WORD);
                    cout << ">----- STATS -----<" << endl;
                    cout << "Inference complete. Inferred sleep stage: " << _inferred_sleep_stage  << endl;
                    cout << "Number of exponent operations with negative argument = " << _neg_exp_cnt << "/" << _total_exp_cnt << " (" << 100*_neg_exp_cnt/_total_exp_cnt  << "%)" << endl;
                    cout << "Min./Max. inputs to exponential = " << static_cast<float>(_min_exp_input_arg) << " and " << static_cast<float>(_max_exp_input_arg) << endl;
                } else { gen_cnt_7b.inc(); }
            } else { gen_cnt_4b.inc(); }
            break;

            break;

        case INFERENCE_COMPLETE:
            break;

        case INVALID_STEP:
            break;
    }

    // Reset compute done (stays on for only one cycle)
    reset_compute_done();

    return (current_inf_step == INFERENCE_COMPLETE) & (cim_state == IDLE_CIM);
}

#endif
