#include <Master_Ctrl.hpp>

/*----- NAMESPACE -----*/
using namespace std;

/*----- DECLARATION -----*/
Master_ctrl::Master_ctrl(const string eeg_filepath, const string params_filepath) : gen_cnt_8b(8), gen_cnt_10b(10) {
    state = RESET;

    // EEG data file
    HighFive::File eeg_file(eeg_filepath, HighFive::File::ReadOnly);
    eeg_ds = eeg_file.getDataSet("eeg").read<vector<float>>();
    eeg = eeg_ds.begin();

    // Parameters data file
    load_params_from_h5(params_filepath);
}

int Master_ctrl::reset(){
    gen_cnt_8b.reset();
    gen_cnt_10b.reset();
    gen_reg_16b = 0;
    gen_reg_16b_2 = 0;
    gen_reg_16b_3 = 0;
    high_level_inf_step = PRE_LAYERNORM_1_TRANS_STEP;
    state = IDLE;
    return 0;
}

SYSTEM_STATE Master_ctrl::run(struct ext_signals* ext_sigs, Bus* bus, std::vector<CiM> cims){
    /* Run the master controller FSM */
    SYSTEM_STATE sys_state = RUNNING;

    // Act on priority external signals
    if  (ext_sigs->master_nrst == false) { state = RESET; }

    // Check if CiMs are idle
    all_cims_ready = true;
    for (int i=0; i<num_necessary_idles.at(high_level_inf_step); i++){ 
        if (high_level_inf_step == MLP_HEAD_DENSE_1_STEP) { all_cims_ready &= cims[i+MLP_DIM].get_is_ready(); }
        else { all_cims_ready &= cims[i].get_is_ready(); }
    }

    switch (state) {
    case IDLE:
        if (ext_sigs->start_param_load == true) {
            assert(ext_sigs->new_sleep_epoch == false && "ERROR: Both 'start_param_load' and 'new_sleep_epoch' are set simultaneously!");
            gen_reg_16b = false; // Need to make sure this is false before starting the parameter load
            state = PARAM_LOAD;
            cout << "Starting parameters load" << endl;
        } else if (ext_sigs->new_sleep_epoch == true) { start_signal_load(bus); }
        break;

    case PARAM_LOAD:
        if (all_cims_ready) { bus->push_inst(param_to_send()); }
        break;

    case SIGNAL_LOAD:
        /* Sequentially parses the input EEG file and broadcasts it to all CiMs to emulate the ADC feed in the ASIC */
        if ((eeg != eeg_ds.end()) && (all_cims_ready == true)) {
            float rescaled_data = (*eeg)*EEG_SCALE_FACTOR;
            struct instruction inst = {/*op*/ PATCH_LOAD_BROADCAST_OP, /*target_or_sender*/ 0, /*data*/ {rescaled_data,0}, /*extra_field*/ 0};
            bus->push_inst(inst); // Broadcast on bus
            ++eeg;
        } else if ((eeg == eeg_ds.end()) && (all_cims_ready == true)) {
            state = INFERENCE_RUNNING;
            cout << "Reached end of signal file" << endl;
        }
        break;

    case INFERENCE_RUNNING:
        switch (high_level_inf_step){
        case PRE_LAYERNORM_1_TRANS_STEP:
        case INTRA_LAYERNORM_1_TRANS_STEP:
        case POST_LAYERNORM_1_TRANS_STEP:
        case ENC_MHSA_DENSE_STEP:
        case ENC_MHSA_Q_TRANS_STEP:
        case ENC_MHSA_K_TRANS_STEP:
        case ENC_MHSA_QK_T_STEP:
        case ENC_MHSA_PRE_SOFTMAX_TRANS_STEP:
        case ENC_MHSA_V_MULT_STEP:
        case ENC_MHSA_POST_V_TRANS_STEP:
        case ENC_MHSA_POST_V_DENSE_STEP:
        case PRE_LAYERNORM_2_TRANS_STEP:
        case INTRA_LAYERNORM_2_TRANS_STEP:
        case ENC_PRE_MLP_TRANSPOSE_STEP:
        case ENC_MLP_DENSE_1_STEP:
        case ENC_MLP_DENSE_2_TRANSPOSE_STEP:
        case ENC_MLP_DENSE_2_AND_SUM_STEP:
        case PRE_LAYERNORM_3_TRANS_STEP:
        case INTRA_LAYERNORM_3_TRANS_STEP:
        case PRE_MLP_HEAD_DENSE_TRANS_STEP:
        case MLP_HEAD_DENSE_1_STEP:
        case PRE_MLP_HEAD_DENSE_2_TRANS_STEP:
        case MLP_HEAD_DENSE_2_STEP:
        case MLP_HEAD_SOFTMAX_TRANS_STEP:
        case SOFTMAX_AVERAGING:
            if (bus->get_inst().op == INFERENCE_RESULT_OP && all_cims_ready == true) {
                sys_state = EVERYTHING_FINISHED;
                break;
            } else if (high_level_inf_step != SOFTMAX_AVERAGING) {
                if (high_level_inf_step == ENC_MHSA_QK_T_STEP) { cout << "Master: Performing encoder's MHSA QK_T. Starting matrix #" << gen_reg_16b_3 << " in the Z-stack (out of " << NUM_HEADS << ")" << endl; }
                else if (high_level_inf_step == ENC_MHSA_PRE_SOFTMAX_TRANS_STEP) { cout << "Master: Performing encoder's MHSA pre-softmax tranpose. Starting matrix #" << gen_reg_16b_3 << " in the Z-stack (out of " << NUM_HEADS << ")" << endl; }
                else if (high_level_inf_step == ENC_MHSA_V_MULT_STEP) { cout << "Master: Performing encoder's MHSA V_MULT. Starting matrix #" << gen_reg_16b_3 << " in the Z-stack (out of " << NUM_HEADS << ")" << endl; }
                else { cout << "Master: Starting high-level step #" << high_level_inf_step << endl; }
                prepare_for_broadcast(broadcast_ops.at(high_level_inf_step), bus);
            }
            break;

        case ENC_MHSA_SOFTMAX_STEP:
            high_level_inf_step = (all_cims_ready == true) ? (static_cast<HIGH_LEVEL_INFERENCE_STEP> (high_level_inf_step+1)) : (high_level_inf_step);
            break;

        default:
            sys_state = EVERYTHING_FINISHED;
            break;
        }
        break;

    case BROADCAST_MANAGEMENT:
        if ((gen_cnt_10b.get_cnt() == 0) && (bus->get_inst().op == NOP) && (all_cims_ready == true)) { // All transactions sent and bus free, start a new CiM
            gen_cnt_8b.inc(); // CiM counter

            if (gen_cnt_8b.get_cnt() == gen_reg_16b_2) { // All CiMs sent all data and finished using it, can go back to running inference
                if ((high_level_inf_step != ENC_MHSA_QK_T_STEP && high_level_inf_step != ENC_MHSA_V_MULT_STEP && high_level_inf_step != ENC_MHSA_PRE_SOFTMAX_TRANS_STEP) || (gen_reg_16b_3 == NUM_HEADS)) { // These two steps require going through the Z-stack of the Q and K matrices so only increment the step counter when we're done with the Z-stack
                    struct instruction inst = {/*op*/ PISTOL_START_OP, /*target_or_sender*/ 0, /*data*/ {0,0}, /*extra_fields*/ 0}; // Tell CiMs that they can go to the next step
                    bus->push_inst(inst);
                    high_level_inf_step = static_cast<HIGH_LEVEL_INFERENCE_STEP> (high_level_inf_step+1);
                    gen_reg_16b_3 = 0;
                }
                state = INFERENCE_RUNNING;
            } else { // Trigger new CiM to send data
                struct broadcast_op_info op_info = broadcast_ops.at(high_level_inf_step);
                struct instruction inst;
                if (high_level_inf_step == PRE_MLP_HEAD_DENSE_2_TRANS_STEP) {
                    inst = {op_info.op, /*target_or_sender*/ gen_cnt_8b.get_cnt()+MLP_DIM, {op_info.tx_addr, op_info.len}, op_info.rx_addr}; // CiMs 32-63 have the data that we want to broadcast
                } else if (high_level_inf_step == ENC_MHSA_QK_T_STEP) {
                    inst = {op_info.op, /*target_or_sender*/ gen_cnt_8b.get_cnt(), {op_info.tx_addr + (gen_reg_16b_3-1)*NUM_HEADS, op_info.len}, op_info.rx_addr};
                } else if (high_level_inf_step == ENC_MHSA_PRE_SOFTMAX_TRANS_STEP) {
                    inst = {op_info.op, /*target_or_sender*/ gen_cnt_8b.get_cnt(), {op_info.tx_addr + (gen_reg_16b_3-1)*(NUM_PATCHES+1), op_info.len}, op_info.rx_addr + (gen_reg_16b_3-1)*(NUM_PATCHES+1)};
                } else if (high_level_inf_step == ENC_MHSA_V_MULT_STEP) {
                    inst = {op_info.op, /*target_or_sender*/ gen_cnt_8b.get_cnt(), {op_info.tx_addr + (gen_reg_16b_3-1)*(NUM_PATCHES+1), op_info.len}, op_info.rx_addr};
                } else {
                    inst = {op_info.op, /*target_or_sender*/ gen_cnt_8b.get_cnt(), {op_info.tx_addr, op_info.len}, op_info.rx_addr};
                }
                bus->push_inst(inst);
                gen_cnt_10b.set_val(NUM_TRANS(op_info.len)); // Transaction counter
            }
        }
        if (bus->get_inst().op == TRANS_BROADCAST_DATA_OP || bus->get_inst().op == DENSE_BROADCAST_DATA_OP){
            gen_cnt_10b.dec();
        }
        break;

    case RESET:
        if (ext_sigs->master_nrst == true) { state = IDLE; } // No longer under reset
        reset();
        break;

    case INVALID:
    default:
        throw invalid_argument("Master controller in an invalid state!");
        break;
    }

    return sys_state;
}

int Master_ctrl::start_signal_load(Bus* bus){
    state = SIGNAL_LOAD;
    gen_cnt_8b.reset();
    gen_cnt_10b.reset();
    cout << "Starting signal load" << endl;
    instruction inst = {PATCH_LOAD_BROADCAST_START_OP, /*target_or_sender*/ 0, /*data*/ {0,0}, /*extra_fields*/ 0};
    bus->push_inst(inst);
    return 0;
}

struct instruction Master_ctrl::param_to_send(){
    /* Parses the parameter file and returns the instruction that master needs to send to CiM */
    struct instruction inst;

    /* 
    * Note: gen_reg_16b holds whether we've sent the first CiM of a layer
    *       gen_cnt_8b() holds number of bytes sent to the current CiM for the current param
    *       gen_cnt_10b() holds the current CiM we are sending data to for the current param
    */

    // New params layer
    switch (params_curr_layer) {
    case PATCH_PROJ_KERNEL_PARAMS:
    case POS_EMB_PARAMS:
    case ENC_Q_DENSE_PARAMS:
    case ENC_K_DENSE_PARAMS:
    case ENC_V_DENSE_PARAMS:
    case ENC_COMB_HEAD_PARAMS:
    case ENC_MLP_DENSE_2_PARAMS:
    case ENC_MLP_DENSE_1_OR_MLP_HEAD_DENSE_1_PARAMS:
    case MLP_HEAD_DENSE_2_PARAMS:
        if (((gen_reg_16b == false) || (gen_cnt_8b.get_cnt() >= param_addr_map[params_curr_layer].len)) && (gen_cnt_10b.get_cnt() < param_addr_map[params_curr_layer].num_rec)) { // Starting a new CiM (if not done all layers)
            uint16_t addr = param_addr_map[params_curr_layer].addr;
            uint16_t length = param_addr_map[params_curr_layer].len;
            array<float, 2> data = {static_cast<float>(addr), static_cast<float>(length)};

            inst = {DATA_STREAM_START_OP, /*target_or_sender*/ gen_cnt_10b.get_cnt(), /*data={start_addr, length_elem}*/ data, /*extra_fields*/ 0};
            gen_reg_16b = true;
            gen_cnt_8b.reset();
        } else if (gen_cnt_8b.get_cnt() < param_addr_map[params_curr_layer].len) { // Sending data to a CiM
            inst.op = DATA_STREAM_OP;
            inst.target_or_sender = gen_cnt_10b.get_cnt();
            update_inst_with_params(params_curr_layer, &inst);
            gen_cnt_8b.inc(3); // Increment by 3 since we send 3 bytes per transaction
            if (gen_cnt_8b.get_cnt() >= param_addr_map[params_curr_layer].len) { gen_cnt_10b.inc(); } // Increment number of CiM we've sent data to as we've now sent all the data to the current CiM
        } else if ((gen_cnt_10b.get_cnt() == param_addr_map[params_curr_layer].num_rec) && (gen_cnt_8b.get_cnt() >= param_addr_map[params_curr_layer].len)) { // Need to start a new param layer
            gen_cnt_10b.reset(); // Holds number of CiMs we've sent data to for the current param
            gen_cnt_8b.reset(); // Holds number of bytes we've sent to the current CiM for the current param
            params_curr_layer = static_cast<PARAM_NAME> (params_curr_layer+1);
            gen_reg_16b = false;
        } else { // Shouldn't get here
            cout << "ERROR: Got into invalid state while sending parameters! Exiting." << endl;
            exit(-1);
        }
        break;
    case SINGLE_PARAMS:
        if ((gen_cnt_10b.get_cnt() < param_addr_map[params_curr_layer].num_rec) && (gen_cnt_8b.get_cnt() == 0)) {
            uint16_t addr = param_addr_map[params_curr_layer].addr;
            uint16_t length = param_addr_map[params_curr_layer].len;
            std::array<float, 2> data = {static_cast<float>(addr), static_cast<float>(length)};
            inst = {DATA_STREAM_START_OP, /*target_or_sender*/ gen_cnt_10b.get_cnt(), /*data={start_addr, length_elem}*/ data, /*extra_fields*/ 0};
            gen_cnt_8b.inc(); // Use as indication that next time this runs, we go to the else if () below

        } else if (gen_cnt_10b.get_cnt() < param_addr_map[params_curr_layer].num_rec){
            if (gen_cnt_8b.get_cnt() == 1) {
                inst = {DATA_STREAM_OP, /*target_or_sender*/ gen_cnt_10b.get_cnt(), /*data*/ {params.patch_proj_bias[gen_cnt_10b.get_cnt()], params.class_emb[gen_cnt_10b.get_cnt()]}, /*extra_fields*/ params.layernorm_gamma[0][gen_cnt_10b.get_cnt()]};
                gen_cnt_8b.inc();
            } else if (gen_cnt_8b.get_cnt() == 2) {
                inst = {DATA_STREAM_OP, /*target_or_sender*/ gen_cnt_10b.get_cnt(), /*data*/ {params.layernorm_beta[0][gen_cnt_10b.get_cnt()], params.enc_mhsa_Q_bias[gen_cnt_10b.get_cnt()]}, /*extra_fields*/ params.enc_mhsa_K_bias[gen_cnt_10b.get_cnt()]};
                gen_cnt_8b.inc();
            } else if (gen_cnt_8b.get_cnt() == 3) {
                inst = {DATA_STREAM_OP, /*target_or_sender*/ gen_cnt_10b.get_cnt(), /*data*/ {params.enc_mhsa_V_bias[gen_cnt_10b.get_cnt()], params.enc_mhsa_sqrt_num_heads}, /*extra_fields*/ params.enc_mhsa_combine_bias[gen_cnt_10b.get_cnt()]};
                gen_cnt_8b.inc();
            } else if (gen_cnt_8b.get_cnt() == 4) {
                // Note: CiM's 0-31 receive bias for the encoder's MLP and CiM's 32-63 receive bias for the MLP head
                if (gen_cnt_10b.get_cnt() < MLP_DIM) { inst = {DATA_STREAM_OP, /*target_or_sender*/ gen_cnt_10b.get_cnt(), /*data*/ {params.layernorm_gamma[1][gen_cnt_10b.get_cnt()], params.layernorm_beta[1][gen_cnt_10b.get_cnt()]}, /*extra_fields*/ params.enc_mlp_dense_1_bias[gen_cnt_10b.get_cnt()]}; }
                else {                                 inst = {DATA_STREAM_OP, /*target_or_sender*/ gen_cnt_10b.get_cnt(), /*data*/ {params.layernorm_gamma[1][gen_cnt_10b.get_cnt()], params.layernorm_beta[1][gen_cnt_10b.get_cnt()]}, /*extra_fields*/ params.mlp_head_dense_1_bias[gen_cnt_10b.get_cnt()-MLP_DIM]}; }
                gen_cnt_8b.inc();
            } else if (gen_cnt_8b.get_cnt() == 5) {
                inst = {DATA_STREAM_OP, /*target_or_sender*/ gen_cnt_10b.get_cnt(), /*data*/ {params.enc_mlp_dense_2_bias[gen_cnt_10b.get_cnt()], params.layernorm_gamma[2][gen_cnt_10b.get_cnt()]}, /*extra_fields*/ params.layernorm_beta[2][gen_cnt_10b.get_cnt()]};
                gen_cnt_8b.inc();
            } else if (gen_cnt_8b.get_cnt() == 6) {
                inst = {DATA_STREAM_OP, /*target_or_sender*/ gen_cnt_10b.get_cnt(), /*data*/ {params.mlp_head_dense_2_bias[gen_cnt_10b.get_cnt()], 0}, /*extra_fields*/ 0};
                gen_cnt_10b.inc();
                gen_cnt_8b.reset();
            }
        } else {
            gen_cnt_10b.reset();
            gen_cnt_8b.reset();
            params_curr_layer = static_cast<PARAM_NAME> (params_curr_layer+1);
        }
        break;
    case PARAM_LOAD_FINISHED:
    default:
        state = IDLE;
        break;
    }

    return inst;
}

int Master_ctrl::load_params_from_h5(const std::string params_filepath) {
    // Load parameters from .h5 file and store into struct for easier and faster access

    HighFive::File file(params_filepath, HighFive::File::ReadOnly);

    // Patch projection
    params.patch_proj_kernel = file.getGroup("patch_projection_dense").getGroup("vision_transformer").getGroup("patch_projection_dense").getDataSet("kernel:0").read<PatchProjKernel_t>();
    params.patch_proj_bias = file.getGroup("patch_projection_dense").getGroup("vision_transformer").getGroup("patch_projection_dense").getDataSet("bias:0").read<EmbDepthVect_t>();

    params.class_emb = file.getGroup("top_level_model_weights").getDataSet("class_emb:0").read<EmbDepthVect_t>();
    params.pos_emb = file.getGroup("top_level_model_weights").getDataSet("pos_emb:0").read<array<array<array<float, EMB_DEPTH>, NUM_PATCHES+1>, 1>>()[0];

    // Encoders
    HighFive::Group enc = file.getGroup("Encoder_1");
    // LayerNorm
    params.layernorm_beta[0] = enc.getGroup("vision_transformer").getGroup("Encoder_1").getGroup("layerNorm1_encoder").getDataSet("beta:0").read<EmbDepthVect_t>(); // Encoder's LayerNorm 1
    params.layernorm_gamma[0] = enc.getGroup("vision_transformer").getGroup("Encoder_1").getGroup("layerNorm1_encoder").getDataSet("gamma:0").read<EmbDepthVect_t>(); // Encoder's LayerNorm 1
    params.layernorm_beta[1] = enc.getGroup("vision_transformer").getGroup("Encoder_1").getGroup("layerNorm2_encoder").getDataSet("beta:0").read<EmbDepthVect_t>(); // Encoder's LayerNorm 2
    params.layernorm_gamma[1] = enc.getGroup("vision_transformer").getGroup("Encoder_1").getGroup("layerNorm2_encoder").getDataSet("gamma:0").read<EmbDepthVect_t>(); // Encoder's LayerNorm 2

    // MHSA
    params.enc_mhsa_Q_kernel = enc.getGroup("mhsa_query_dense").getDataSet("kernel:0").read<EncEmbDepthMat_t>();
    params.enc_mhsa_K_kernel = enc.getGroup("mhsa_key_dense").getDataSet("kernel:0").read<EncEmbDepthMat_t>();
    params.enc_mhsa_V_kernel = enc.getGroup("mhsa_value_dense").getDataSet("kernel:0").read<EncEmbDepthMat_t>();
    params.enc_mhsa_Q_bias = enc.getGroup("mhsa_query_dense").getDataSet("bias:0").read<EmbDepthVect_t>();
    params.enc_mhsa_K_bias = enc.getGroup("mhsa_key_dense").getDataSet("bias:0").read<EmbDepthVect_t>();
    params.enc_mhsa_V_bias = enc.getGroup("mhsa_value_dense").getDataSet("bias:0").read<EmbDepthVect_t>();
    params.enc_mhsa_combine_kernel = enc.getGroup("mhsa_combine_head_dense").getDataSet("kernel:0").read<EncEmbDepthMat_t>();
    params.enc_mhsa_combine_bias = enc.getGroup("mhsa_combine_head_dense").getDataSet("bias:0").read<EmbDepthVect_t>();
    params.enc_mhsa_sqrt_num_heads = static_cast<float>(sqrt(NUM_HEADS));

    // MLP
    params.enc_mlp_dense_1_kernel = enc.getGroup("vision_transformer").getGroup("Encoder_1").getGroup("mlp_dense1_encoder").getDataSet("kernel:0").read<EmbDepthxMlpDimMat_t>();
    params.enc_mlp_dense_1_bias = enc.getGroup("vision_transformer").getGroup("Encoder_1").getGroup("mlp_dense1_encoder").getDataSet("bias:0").read<MlpDimVect_t>();
    params.enc_mlp_dense_2_kernel = enc.getGroup("vision_transformer").getGroup("Encoder_1").getGroup("mlp_dense2_encoder").getDataSet("kernel:0").read<EncMlpDimxEmbDepthMat_t>();
    params.enc_mlp_dense_2_bias = enc.getGroup("vision_transformer").getGroup("Encoder_1").getGroup("mlp_dense2_encoder").getDataSet("bias:0").read<EmbDepthVect_t>();

    // MLP head
    params.layernorm_beta[2] = file.getGroup("mlp_head_layerNorm").getGroup("vision_transformer").getGroup("mlp_head_layerNorm").getDataSet("beta:0").read<EmbDepthVect_t>(); // MLP head's LayerNorm 1
    params.layernorm_gamma[2] = file.getGroup("mlp_head_layerNorm").getGroup("vision_transformer").getGroup("mlp_head_layerNorm").getDataSet("gamma:0").read<EmbDepthVect_t>(); // MLP head's LayerNorm 1
    params.mlp_head_dense_1_kernel = file.getGroup("mlp_head").getGroup("mlp_head_dense1").getDataSet("kernel:0").read<EmbDepthxMlpDimMat_t>();
    params.mlp_head_dense_1_bias = file.getGroup("mlp_head").getGroup("mlp_head_dense1").getDataSet("bias:0").read<MlpDimVect_t>();
    params.mlp_head_dense_2_kernel = file.getGroup("mlp_head_softmax").getGroup("vision_transformer").getGroup("mlp_head_softmax").getDataSet("kernel:0").read<NumSleepStagesxMlpDimMat_t>();
    params.mlp_head_dense_2_bias = file.getGroup("mlp_head_softmax").getGroup("vision_transformer").getGroup("mlp_head_softmax").getDataSet("bias:0").read<NumSleepStagesVect_t>();

    return 0;
}

void Master_ctrl::update_inst_with_params(PARAM_NAME param_name, struct instruction* inst) {
    /* Returns the parameters loaded from the .h5 file corresponding to the passed name 
    *  Note: gen_cnt_8b() holds number of bytes sent to the current CiM for the current param
    *        gen_cnt_10b() holds the current CiM we are sending data to for the current param
    */

    int num_left = param_addr_map[param_name].len-gen_cnt_8b.get_cnt();

    switch (param_name) {
    case PATCH_PROJ_KERNEL_PARAMS:
        inst->data = {params.patch_proj_kernel[gen_cnt_8b.get_cnt()][gen_cnt_10b.get_cnt()], (num_left == 1) ? (0) : (params.patch_proj_kernel[gen_cnt_8b.get_cnt()+1][gen_cnt_10b.get_cnt()])};
        inst->extra_fields = (num_left <= 2) ? (0) : params.patch_proj_kernel[gen_cnt_8b.get_cnt()+2][gen_cnt_10b.get_cnt()];
        break;
    case POS_EMB_PARAMS:
        inst->data = {params.pos_emb[gen_cnt_8b.get_cnt()][gen_cnt_10b.get_cnt()], (num_left == 1) ? (0) : (params.pos_emb[gen_cnt_8b.get_cnt()+1][gen_cnt_10b.get_cnt()])};
        inst->extra_fields = (num_left <= 2) ? (0) : params.pos_emb[gen_cnt_8b.get_cnt()+2][gen_cnt_10b.get_cnt()];
        break;
    case ENC_Q_DENSE_PARAMS:
        inst->data = {params.enc_mhsa_Q_kernel[gen_cnt_8b.get_cnt()][gen_cnt_10b.get_cnt()], (num_left == 1) ? (0) : (params.enc_mhsa_Q_kernel[gen_cnt_8b.get_cnt()+1][gen_cnt_10b.get_cnt()])};
        inst->extra_fields = (num_left <= 2) ? (0) : params.enc_mhsa_Q_kernel[gen_cnt_8b.get_cnt()+2][gen_cnt_10b.get_cnt()];
        break;
    case ENC_K_DENSE_PARAMS:
        inst->data = {params.enc_mhsa_K_kernel[gen_cnt_8b.get_cnt()][gen_cnt_10b.get_cnt()], (num_left == 1) ? (0) : (params.enc_mhsa_K_kernel[gen_cnt_8b.get_cnt()+1][gen_cnt_10b.get_cnt()])};
        inst->extra_fields = (num_left <= 2) ? (0) : params.enc_mhsa_K_kernel[gen_cnt_8b.get_cnt()+2][gen_cnt_10b.get_cnt()];
        break;
    case ENC_V_DENSE_PARAMS:
        inst->data = {params.enc_mhsa_V_kernel[gen_cnt_8b.get_cnt()][gen_cnt_10b.get_cnt()], (num_left == 1) ? (0) : (params.enc_mhsa_V_kernel[gen_cnt_8b.get_cnt()+1][gen_cnt_10b.get_cnt()])};
        inst->extra_fields = (num_left <= 2) ? (0) : params.enc_mhsa_V_kernel[gen_cnt_8b.get_cnt()+2][gen_cnt_10b.get_cnt()];
        break;
    case ENC_COMB_HEAD_PARAMS:
        inst->data = {params.enc_mhsa_combine_kernel[gen_cnt_8b.get_cnt()][gen_cnt_10b.get_cnt()], (num_left == 1) ? (0) : (params.enc_mhsa_combine_kernel[gen_cnt_8b.get_cnt()+1][gen_cnt_10b.get_cnt()])};
        inst->extra_fields = (num_left <= 2) ? (0) : params.enc_mhsa_combine_kernel[gen_cnt_8b.get_cnt()+2][gen_cnt_10b.get_cnt()];
        break;
    case ENC_MLP_DENSE_1_OR_MLP_HEAD_DENSE_1_PARAMS:
        if (gen_cnt_10b.get_cnt() < MLP_DIM) { // Encoder's MLP (onyl CiMs 0-31 receive bias for the encoder's MLP and CiM's 32-63 receive bias for the MLP head)
            inst->data = {params.enc_mlp_dense_1_kernel[gen_cnt_8b.get_cnt()][gen_cnt_10b.get_cnt()], (num_left == 1) ? (0) : (params.enc_mlp_dense_1_kernel[gen_cnt_8b.get_cnt()+1][gen_cnt_10b.get_cnt()])};
            inst->extra_fields = (num_left <= 2) ? (0) : params.enc_mlp_dense_1_kernel[gen_cnt_8b.get_cnt()+2][gen_cnt_10b.get_cnt()];
        } else { // MLP head (only 1 layer)
            inst->data = {params.mlp_head_dense_1_kernel[gen_cnt_8b.get_cnt()][gen_cnt_10b.get_cnt()-MLP_DIM], (num_left == 1) ? (0) : (params.mlp_head_dense_1_kernel[gen_cnt_8b.get_cnt()+1][gen_cnt_10b.get_cnt()-MLP_DIM])};
            inst->extra_fields = (num_left <= 2) ? (0) : params.mlp_head_dense_1_kernel[gen_cnt_8b.get_cnt()+2][gen_cnt_10b.get_cnt()-MLP_DIM];
        }
        break;
    case ENC_MLP_DENSE_2_PARAMS:
        inst->data = {params.enc_mlp_dense_2_kernel[gen_cnt_8b.get_cnt()][gen_cnt_10b.get_cnt()], (num_left == 1) ? (0) : (params.enc_mlp_dense_2_kernel[gen_cnt_8b.get_cnt()+1][gen_cnt_10b.get_cnt()])};
        inst->extra_fields = (num_left <= 2) ? (0) : params.enc_mlp_dense_2_kernel[gen_cnt_8b.get_cnt()+2][gen_cnt_10b.get_cnt()];
        break;
    case MLP_HEAD_DENSE_2_PARAMS:
        inst->data = {params.mlp_head_dense_2_kernel[gen_cnt_8b.get_cnt()][gen_cnt_10b.get_cnt()], (num_left == 1) ? (0) : (params.mlp_head_dense_2_kernel[gen_cnt_8b.get_cnt()+1][gen_cnt_10b.get_cnt()])};
        inst->extra_fields = (num_left <= 2) ? (0) : params.mlp_head_dense_2_kernel[gen_cnt_8b.get_cnt()+2][gen_cnt_10b.get_cnt()];
        break;
    default:
        throw invalid_argument("Invalid parameter name");
    }
}

int Master_ctrl::prepare_for_broadcast(broadcast_op_info op_info, Bus* bus) {
    /* Prepares the master controller for a broadcast operation */
    struct instruction inst = {op_info.op, /*target_or_sender*/ 0, {op_info.tx_addr, op_info.len}, op_info.rx_addr};
    state = BROADCAST_MANAGEMENT;
    gen_reg_16b = NUM_TRANS(op_info.len); // Holds the number of transactions each CiM will send (3 elements per transaction)
    gen_reg_16b_2 = op_info.num_cims; // Number of CiMs that will need to send data
    gen_cnt_8b.reset(); // Count CiMs
    gen_cnt_10b.set_val(gen_reg_16b); // Count transactions (add 3 such that we will wait 1 cycle after CiM sent its last transaction to avoid shorting the bus)

    if (high_level_inf_step == ENC_MHSA_QK_T_STEP) { // Need to modify instruction based on where we are in the Z-stack of the Q and K matrices
        inst.data[0] = op_info.tx_addr + gen_reg_16b_3*NUM_HEADS; // tx addr
        gen_reg_16b_3++;
    } else if (high_level_inf_step == ENC_MHSA_V_MULT_STEP) {
        inst.data[0] = op_info.tx_addr + gen_reg_16b_3*(NUM_PATCHES+1); // tx addr
        gen_reg_16b_3++;
    } else if (high_level_inf_step == ENC_MHSA_PRE_SOFTMAX_TRANS_STEP) {
        inst.data[0] = op_info.tx_addr + gen_reg_16b_3*(NUM_PATCHES+1); // tx addr
        inst.extra_fields = op_info.rx_addr + gen_reg_16b_3*(NUM_PATCHES+1); // rx addr
        gen_reg_16b_3++;
    } else if (high_level_inf_step == PRE_MLP_HEAD_DENSE_2_TRANS_STEP) {
        inst.target_or_sender = MLP_DIM; // tx addr (CiMs 32-63 have the data that we want to broadcast)
    }

    bus->push_inst(inst);
    return 0;
}