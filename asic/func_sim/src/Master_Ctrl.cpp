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
    fill(begin(storage), end(storage), 0); // Reset remote storage
    gen_cnt_8b.reset();
    gen_cnt_10b.reset();
    gen_reg_16b = 0;
    gen_reg_16b_2 = 0;
    return 0;
}

SYSTEM_STATE Master_ctrl::run(struct ext_signals* ext_sigs, Bus* bus, CiM cims[]){
    /* Run the master controller FSM */

    bool all_cims_idle = true;
    SYSTEM_STATE sys_state = RUNNING;

    // Act on priority external signals
    if  (ext_sigs->master_nrst == false) { state = RESET; }

    switch (state) {
    case IDLE:
        if (ext_sigs->start_param_load == true) {
            assert(ext_sigs->new_sleep_epoch == false && "ERROR: Both 'start_param_load' and 'new_sleep_epoch' are set simultaneously!");
            state = PARAM_LOAD;
        } else if (ext_sigs->new_sleep_epoch == true) { start_signal_load(); }
        break;

    case PARAM_LOAD:
        bus->push_inst(param_to_send());
        break;

    case SIGNAL_LOAD:
        /* Sequentially parses the input EEG file and broadcasts it to all CiMs to emulate the ADC feed in the ASIC*/
        if (eeg != eeg_ds.end()) {
            struct instruction inst = {
                /*op*/ PATCH_LOAD_BROADCAST_OP,
                /*target_or_sender*/ 0,
                /*data*/ {*eeg,0}, // In ASIC, we would split the 16b into 2x 8b
                /*extra_field*/ 0
            };

            bus->push_inst(inst); // Broadcast on bus
            ++eeg;
        } else {
            state = INFERENCE_RUNNING;
            cout << "Reached end of signal file" << endl;
        }
        break;

    case INFERENCE_RUNNING:
        // Wait for CiM to be done
        for (int i=0; i<NUM_CIM; i++){ all_cims_idle &= cims[i].get_is_idle(); }

        switch (high_level_inf_step){
        case PRE_LAYERNORM_TRANSPOSE_STEP:
            if (all_cims_idle == true) {
                struct instruction inst = {/*op*/ TRANSPOSE_BROADCAST_START_OP, /*target_or_sender*/ 0, /*{addr data to send, length}*/ {0, NUM_PATCHES+1}, /*addr where to store*/ NUM_PATCHES+1};
                state = BROADCAST_MANAGEMENT;
                bus->push_inst(inst);
                gen_reg_16b = NUM_TRANS(NUM_PATCHES+1); // Holds the number of transactions each CiM will send (3 elements per transaction)
                gen_reg_16b_2 = NUM_CIM; // Number of CiMs that will need to send data
                gen_cnt_8b.reset(); // Count CiMs
                gen_cnt_10b.set_val(gen_reg_16b); // Count transactions
                cout << "Starting PRE_LAYERNORM_TRANSPOSE_STEP" << endl;
            }
            break;

        case INTRA_LAYERNORM_TRANSPOSE_STEP:
            if (all_cims_idle == true) {
                struct instruction inst = {/*op*/ TRANSPOSE_BROADCAST_START_OP, /*target_or_sender*/ 0, /*{addr data to send, length}*/ {NUM_PATCHES+1, EMBEDDING_DEPTH}, /*addr where to store*/ NUM_PATCHES+1+EMBEDDING_DEPTH};
                state = BROADCAST_MANAGEMENT;
                bus->push_inst(inst);
                gen_reg_16b = NUM_TRANS(EMBEDDING_DEPTH+1); // Holds the number of transactions each CiM will send (3 elements per transaction)
                gen_reg_16b_2 = NUM_PATCHES+1; // Number of CiMs that will need to send data
                gen_cnt_8b.reset(); // Count CiMs. TODO: Could set_val and decrement instead in order to save a reg but logic would become slightly less intuitive
                gen_cnt_10b.set_val(gen_reg_16b); // Count transactions
                cout << "Starting INTRA_LAYERNORM_TRANSPOSE_STEP" << endl;
            }
            break;

        case POST_LAYERNORM_TRANSPOSE_STEP:
            if (all_cims_idle == true) {
                struct instruction inst = {/*op*/ TRANSPOSE_BROADCAST_START_OP, /*target_or_sender*/ 0, /*{addr data to send, length}*/ {NUM_PATCHES+1+EMBEDDING_DEPTH, NUM_PATCHES+1}, /*addr where to store*/ NUM_PATCHES+1};
                state = BROADCAST_MANAGEMENT;
                bus->push_inst(inst);
                gen_reg_16b = NUM_TRANS(NUM_PATCHES+1); // Holds the number of transactions each CiM will send (3 elements per transaction)
                gen_reg_16b_2 = NUM_CIM; // Number of CiMs that will need to send data
                gen_cnt_8b.reset(); // Count CiMs
                gen_cnt_10b.set_val(gen_reg_16b); // Count transactions
                cout << "Starting POST_LAYERNORM_TRANSPOSE_STEP" << endl;
            }
            break;

        case ENC_MHSA_DENSE_STEP: {
            struct instruction inst = {/*op*/ DENSE_BROADCAST_START_OP, /*target_or_sender*/ 0, /*{addr data to send, length}*/ {NUM_PATCHES+1, EMBEDDING_DEPTH}, /*addr where to store*/ NUM_PATCHES+1+EMBEDDING_DEPTH};
            state = BROADCAST_MANAGEMENT;
            bus->push_inst(inst);
            gen_reg_16b = NUM_TRANS(EMBEDDING_DEPTH); // Holds the number of transactions each CiM will send (3 elements per transaction)
            gen_reg_16b_2 = NUM_PATCHES+1; // Number of CiMs that will need to send data
            gen_cnt_8b.reset(); // Count CiMs
            gen_cnt_10b.set_val(gen_reg_16b); // Count transactions
            cout << "Starting ENC_MHSA_DENSE_STEP" << endl;
            break;
        }

        default:
            sys_state = INFERENCE_FINISHED; // TODO: For now
            break;
        }
        break;

    case BROADCAST_MANAGEMENT:
        if (bus->get_inst().op == TRANSPOSE_BROADCAST_DATA_OP || bus->get_inst().op == DENSE_BROADCAST_DATA_OP){
            gen_cnt_10b.dec();
        }
        if (gen_cnt_10b.get_cnt() == 0) { // All transactions sent, start a new CiM
            gen_cnt_8b.inc(); // CiM counter

            if (gen_cnt_8b.get_cnt() == gen_reg_16b_2) { // All CiMs sent all data and finished using it, can go back to running inference
                state = INFERENCE_RUNNING;
                high_level_inf_step = static_cast<HIGH_LEVEL_INFERENCE_STEP> (high_level_inf_step+1);
                struct instruction inst = {/*op*/ PISTOL_START_OP, /*target_or_sender*/ 0, /*data*/ {0,0}, /*extra_fields*/ 0}; // Tell CiMs that they can go to the next step
                bus->push_inst(inst);
            } else if (all_cims_idle == true) {
                struct instruction inst;
                if (high_level_inf_step == PRE_LAYERNORM_TRANSPOSE_STEP) { // TODO: Consider saving the parameters in these instructions in registers when entering the step instead of hardcoding them here
                    inst = {/*op*/ TRANSPOSE_BROADCAST_START_OP, /*target_or_sender*/ gen_cnt_8b.get_cnt(), /*{addr data to send, length}*/ {0, NUM_PATCHES+1}, /*addr where to store*/ NUM_PATCHES+1};
                } else if (high_level_inf_step == INTRA_LAYERNORM_TRANSPOSE_STEP) {
                    inst = {/*op*/ TRANSPOSE_BROADCAST_START_OP, /*target_or_sender*/ gen_cnt_8b.get_cnt(), /*{addr data to send, length}*/ {NUM_PATCHES+1, EMBEDDING_DEPTH}, /*addr where to store*/ NUM_PATCHES+1+EMBEDDING_DEPTH};
                } else if (high_level_inf_step == POST_LAYERNORM_TRANSPOSE_STEP) {
                    inst = {/*op*/ TRANSPOSE_BROADCAST_START_OP, /*target_or_sender*/ gen_cnt_8b.get_cnt(), /*{addr data to send, length}*/ {NUM_PATCHES+1+EMBEDDING_DEPTH, NUM_PATCHES+1}, /*addr where to store*/ NUM_PATCHES+1};
                } else if (high_level_inf_step == ENC_MHSA_DENSE_STEP) {
                    inst = {/*op*/ DENSE_BROADCAST_START_OP, /*target_or_sender*/ gen_cnt_8b.get_cnt(), /*{addr data to send, length}*/ {NUM_PATCHES+1, EMBEDDING_DEPTH}, /*addr where to store*/ NUM_PATCHES+1+EMBEDDING_DEPTH};
                }
                bus->push_inst(inst);
                gen_cnt_10b.set_val(gen_reg_16b); // Transaction counter
            }
        }
        break;

    case RESET:
        if (ext_sigs->master_nrst == true) { state = IDLE; } // No longer under reset
        reset();
        break;

    case INVALID:
    default:
        cout << "Master controller in an invalid state!\n" << endl;
        break;
    }

    return sys_state;
}

int Master_ctrl::start_signal_load(){
    state = SIGNAL_LOAD;
    gen_cnt_8b.reset();
    gen_cnt_10b.reset();
    return 0;
}

struct instruction Master_ctrl::param_to_send(){
    /* Parses the parameter file and returns the instruction that master needs to send to CiM */
    struct instruction inst;

    // New params layer
    switch (params_curr_layer) {
    case PATCH_PROJ_KERNEL_PARAMS:
    case POS_EMB_PARAMS:
    case ENC_Q_DENSE_PARAMS:
    case ENC_K_DENSE_PARAMS:
    case ENC_V_DENSE_PARAMS:
        if ((params_cim_cnt == NUM_CIM-1) && (params_data_cnt >= param_addr_map[params_curr_layer].len)) { // Start new layer
            params_cim_cnt = -1;
            params_data_cnt = -1;
            gen_cnt_8b.reset();
            params_curr_layer = static_cast<PARAM_NAME> (params_curr_layer+1);

        } else if (params_data_cnt >= param_addr_map[params_curr_layer].len || params_data_cnt == -1) { // Sending to a new CiM
            uint16_t addr = param_addr_map[params_curr_layer].addr;
            uint16_t length = param_addr_map[params_curr_layer].len;
            array<float, 2> data = {static_cast<float>(addr), static_cast<float>(length)};

            params_cim_cnt++;
            inst = {DATA_STREAM_START_OP, /*target_or_sender*/ params_cim_cnt, /*data={start_addr, length_elem}*/ data, /*extra_fields*/ 0};
            params_data_cnt = 0;

        } else { // Stream data to current CiM
            inst.op = DATA_STREAM_OP;
            inst.target_or_sender = params_cim_cnt;
            update_inst_with_params(params_curr_layer, &inst);
            params_data_cnt = params_data_cnt + 3; // Increment by 3 since we send 3 bytes per transaction
        }
        break;

    case SINGLE_PARAMS:
        if ((params_cim_cnt < NUM_CIM) && (gen_cnt_8b.get_cnt() == 0)) {
            uint16_t addr = param_addr_map[params_curr_layer].addr;
            uint16_t length = param_addr_map[params_curr_layer].len;
            std::array<float, 2> data = {static_cast<float>(addr), static_cast<float>(length)};
            inst = {DATA_STREAM_START_OP, /*target_or_sender*/ params_cim_cnt, /*data={start_addr, length_elem}*/ data, /*extra_fields*/ 0};
            gen_cnt_8b.inc(); // Use as indication that next time this runs, we go to the else if () below

        } else if (params_cim_cnt < NUM_CIM){
            // TODO: Incomplete
            if (gen_cnt_8b.get_cnt() == 1) {
                inst = {DATA_STREAM_OP, /*target_or_sender*/ params_cim_cnt, /*data*/ {params.patch_proj_bias[params_cim_cnt], params.class_emb[params_cim_cnt]}, /*extra_fields*/ params.enc_layernorm_gamma[0][params_cim_cnt]};
                gen_cnt_8b.inc();
            } else if (gen_cnt_8b.get_cnt() == 2) { // TODO: Incomplete
                inst = {DATA_STREAM_OP, /*target_or_sender*/ params_cim_cnt, /*data*/ {params.enc_layernorm_beta[0][params_cim_cnt], params.enc_mhsa_Q_bias[params_cim_cnt]}, /*extra_fields*/ params.enc_mhsa_K_bias[params_cim_cnt]};
                gen_cnt_8b.inc();
            } else if (gen_cnt_8b.get_cnt() == 3) {
                inst = {DATA_STREAM_OP, /*target_or_sender*/ params_cim_cnt, /*data*/ {params.enc_mhsa_V_bias[params_cim_cnt], 0}, /*extra_fields*/ 0};
                gen_cnt_8b.reset();
                params_cim_cnt++;
            }
        } else {
            params_cim_cnt = -1;
            params_data_cnt = -1;
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
    params.pos_emb = file.getGroup("top_level_model_weights").getDataSet("pos_emb:0").read<array<array<array<float, EMBEDDING_DEPTH>, NUM_PATCHES+1>, 1>>()[0];

    // Encoders
    HighFive::Group enc = file.getGroup("encoder");
    // LayerNorm
    params.enc_layernorm_beta[0] = enc.getGroup("vision_transformer").getGroup("encoder").getGroup("layerNorm1_encoder").getDataSet("beta:0").read<EmbDepthVect_t>();
    params.enc_layernorm_gamma[0] = enc.getGroup("vision_transformer").getGroup("encoder").getGroup("layerNorm1_encoder").getDataSet("gamma:0").read<EmbDepthVect_t>();
    params.enc_layernorm_beta[1] = enc.getGroup("vision_transformer").getGroup("encoder").getGroup("layerNorm2_encoder").getDataSet("beta:0").read<EmbDepthVect_t>();
    params.enc_layernorm_gamma[1] = enc.getGroup("vision_transformer").getGroup("encoder").getGroup("layerNorm2_encoder").getDataSet("gamma:0").read<EmbDepthVect_t>();

    // MHSA
    params.enc_mhsa_Q_kernel = enc.getGroup("mhsa_query_dense").getDataSet("kernel:0").read<EncEmbDepthMat_t>();
    params.enc_mhsa_K_kernel = enc.getGroup("mhsa_key_dense").getDataSet("kernel:0").read<EncEmbDepthMat_t>();
    params.enc_mhsa_V_kernel = enc.getGroup("mhsa_value_dense").getDataSet("kernel:0").read<EncEmbDepthMat_t>();
    params.enc_mhsa_Q_bias = enc.getGroup("mhsa_query_dense").getDataSet("bias:0").read<EmbDepthVect_t>();
    params.enc_mhsa_K_bias = enc.getGroup("mhsa_key_dense").getDataSet("bias:0").read<EmbDepthVect_t>();
    params.enc_mhsa_V_bias = enc.getGroup("mhsa_value_dense").getDataSet("bias:0").read<EmbDepthVect_t>();
    params.enc_mhsa_combine_kernel = enc.getGroup("mhsa_combine_head_dense").getDataSet("kernel:0").read<EncEmbDepthMat_t>();
    params.enc_mhsa_combine_bias = enc.getGroup("mhsa_combine_head_dense").getDataSet("bias:0").read<EmbDepthVect_t>();

    // MLP
    params.enc_mlp_dense_kernel[0] = enc.getGroup("mlp_dense1_encoder").getDataSet("kernel:0").read<EncEmbDepthMat_t>();
    params.enc_mlp_dense_bias[0] = enc.getGroup("mlp_dense1_encoder").getDataSet("bias:0").read<EmbDepthVect_t>();
    params.enc_mlp_dense_kernel[1] = enc.getGroup("mlp_dense2_encoder").getDataSet("kernel:0").read<EncEmbDepthMat_t>();
    params.enc_mlp_dense_bias[1] = enc.getGroup("mlp_dense2_encoder").getDataSet("bias:0").read<EmbDepthVect_t>();

    return 0;
}

void Master_ctrl::update_inst_with_params(PARAM_NAME param_name, struct instruction* inst) {
    /* Returns the parameters loaded from the .h5 file corresponding to the passed name */

    int num_left = (param_addr_map[param_name].len-params_data_cnt-1);

    switch (param_name) {
    case PATCH_PROJ_KERNEL_PARAMS:
        inst->data = {params.patch_proj_kernel[params_data_cnt][params_cim_cnt], (num_left <= 2) ? (0) : (params.patch_proj_kernel[params_data_cnt+1][params_cim_cnt])};
        inst->extra_fields = (num_left <= 1) ? (0) : params.patch_proj_kernel[params_data_cnt+2][params_cim_cnt];
        break;
    case POS_EMB_PARAMS:
        inst->data = {params.pos_emb[params_data_cnt][params_cim_cnt], (num_left <= 2) ? (0) : (params.pos_emb[params_data_cnt+1][params_cim_cnt])};
        inst->extra_fields = (num_left <= 1) ? (0) : params.pos_emb[params_data_cnt+2][params_cim_cnt];
        break;
    case ENC_Q_DENSE_PARAMS:
        inst->data = {params.enc_mhsa_Q_kernel[params_data_cnt][params_cim_cnt], (num_left <= 2) ? (0) : (params.enc_mhsa_Q_kernel[params_data_cnt+1][params_cim_cnt])};
        inst->extra_fields = (num_left <= 1) ? (0) : params.enc_mhsa_Q_kernel[params_data_cnt+2][params_cim_cnt];
        break;
    case ENC_K_DENSE_PARAMS:
        inst->data = {params.enc_mhsa_K_kernel[params_data_cnt][params_cim_cnt], (num_left <= 2) ? (0) : (params.enc_mhsa_K_kernel[params_data_cnt+1][params_cim_cnt])};
        inst->extra_fields = (num_left <= 1) ? (0) : params.enc_mhsa_K_kernel[params_data_cnt+2][params_cim_cnt];
        break;
    case ENC_V_DENSE_PARAMS:
        inst->data = {params.enc_mhsa_V_kernel[params_data_cnt][params_cim_cnt], (num_left <= 2) ? (0) : (params.enc_mhsa_V_kernel[params_data_cnt+1][params_cim_cnt])};
        inst->extra_fields = (num_left <= 1) ? (0) : params.enc_mhsa_V_kernel[params_data_cnt+2][params_cim_cnt];
        break;
    default:
        throw invalid_argument("Invalid parameter name");
    }
}