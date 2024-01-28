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
                /*target_cim*/ 0,
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
        for (int i=0; i<NUM_CIM; i++){
            all_cims_idle &= cims[i].get_is_idle();
        }
        sys_state = (all_cims_idle) ? (INFERENCE_FINISHED) : (sys_state);
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
        if ((params_cim_cnt == NUM_CIM-1) && (params_data_cnt >= param_addr_map[params_curr_layer][LEN])) { // Start new layer
            params_cim_cnt = 0;
            params_data_cnt = 0;
            gen_cnt_8b.reset();
            params_curr_layer = static_cast<PARAM_NAMES> (params_curr_layer+1);

        } else if (params_data_cnt >= param_addr_map[params_curr_layer][LEN] || params_data_cnt == -1) { // Sending to a new CiM
            uint16_t addr = param_addr_map[params_curr_layer][ADDR];
            uint16_t length = param_addr_map[params_curr_layer][LEN];
            std::array<float, 2> data = {static_cast<float>(addr), static_cast<float>(length)};

            params_cim_cnt++;
            inst = {DATA_STREAM_START_OP, /*target_cim*/ params_cim_cnt, /*data={start_addr, length_elem}*/ data, /*extra_fields*/ 0};
            params_data_cnt = 0;

        } else { // Stream data to current CiM
            std::array<float, 2> data = {0.0f, 0.0f};
            float extra_field = 0.0f;
            int num_left = (param_addr_map[params_curr_layer][LEN]-params_data_cnt-1);

            if (params_curr_layer == PATCH_PROJ_KERNEL_PARAMS) {
                data = {params.patch_proj_kernel[params_data_cnt][params_cim_cnt], (num_left <= 2) ? (0) : (params.patch_proj_kernel[params_data_cnt+1][params_cim_cnt])};
                extra_field = (num_left <= 1) ? (0) : params.patch_proj_kernel[params_data_cnt+2][params_cim_cnt];
            } else if (params_curr_layer == POS_EMB_PARAMS) {
                data = {params.pos_emb[params_data_cnt][params_cim_cnt], (num_left <= 2) ? (0) : (params.pos_emb[params_data_cnt+1][params_cim_cnt])};
                extra_field = (num_left <= 1) ? (0) : params.pos_emb[params_data_cnt+2][params_cim_cnt];
            }

            inst = {DATA_STREAM, /*target_cim*/ params_cim_cnt, /*data*/ data, /*extra_fields*/ extra_field};
            params_data_cnt = params_data_cnt + 3; // Increment by 3 since we send 3 bytes per transaction
        }
        break;

    case SINGLE_PARAMS:
        if ((params_cim_cnt < NUM_CIM) && (gen_cnt_8b.get_cnt() == 0)) {
            uint16_t addr = param_addr_map[params_curr_layer][ADDR];
            uint16_t length = param_addr_map[params_curr_layer][LEN];
            std::array<float, 2> data = {static_cast<float>(addr), static_cast<float>(length)};
            inst = {DATA_STREAM_START_OP, /*target_cim*/ params_cim_cnt, /*data={start_addr, length_elem}*/ data, /*extra_fields*/ 0};
            gen_cnt_8b.inc(); // Use as indication that next time this runs, we go to the else if () below

        } else if (params_cim_cnt < NUM_CIM){
            // TODO: Incomplete
            if (gen_cnt_8b.get_cnt() == 1) {
                inst = {DATA_STREAM, /*target_cim*/ params_cim_cnt, /*data*/ {params.patch_proj_bias[params_cim_cnt], params.class_emb[params_cim_cnt]}, /*extra_fields*/ params.enc_layer_norm_gamma[0][0][params_cim_cnt]};
                gen_cnt_8b.inc();
            } else if (gen_cnt_8b.get_cnt() == 2) { // TODO: Incomplete
                inst = {DATA_STREAM, /*target_cim*/ params_cim_cnt, /*data*/ {params.enc_layer_norm_beta[0][0][params_cim_cnt], 0}, /*extra_fields*/ 0};
                gen_cnt_8b.reset();
                params_cim_cnt++;
            }
        } else {
            params_cim_cnt = -1;
            params_data_cnt = -1;
            gen_cnt_8b.reset();
            params_curr_layer = static_cast<PARAM_NAMES> (params_curr_layer+1);
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
    params.patch_proj_kernel = file.getGroup("patch_projection_dense").getGroup("vision_transformer_1").getGroup("patch_projection_dense").getDataSet("kernel:0").read<array<array<float, EMBEDDING_DEPTH>, PATCH_LENGTH_NUM_SAMPLES>>();
    params.patch_proj_bias = file.getGroup("patch_projection_dense").getGroup("vision_transformer_1").getGroup("patch_projection_dense").getDataSet("bias:0").read<array<float, EMBEDDING_DEPTH>>();

    params.class_emb = file.getGroup("top_level_model_weights").getDataSet("class_emb:0").read<array<float, EMBEDDING_DEPTH>>();
    params.pos_emb = file.getGroup("top_level_model_weights").getDataSet("pos_emb:0").read<array<array<array<float, EMBEDDING_DEPTH>, NUM_PATCHES+1>, 1>>()[0];

    for (int i=0; i<NUM_ENCODERS; i++) {
        HighFive::Group enc = file.getGroup(fmt::format("encoder_{}", 2+i));
        // LayerNorm
        params.enc_layer_norm_beta[i][0] = enc.getGroup("vision_transformer_1").getGroup(fmt::format("encoder_{}", 2+i)).getGroup("layerNorm1_encoder").getDataSet("beta:0").read<array<float, EMBEDDING_DEPTH>>();
        params.enc_layer_norm_gamma[i][0] = enc.getGroup("vision_transformer_1").getGroup(fmt::format("encoder_{}", 2+i)).getGroup("layerNorm1_encoder").getDataSet("gamma:0").read<array<float, EMBEDDING_DEPTH>>();
        params.enc_layer_norm_beta[i][1] = enc.getGroup("vision_transformer_1").getGroup(fmt::format("encoder_{}", 2+i)).getGroup("layerNorm2_encoder").getDataSet("beta:0").read<array<float, EMBEDDING_DEPTH>>();
        params.enc_layer_norm_gamma[i][1] = enc.getGroup("vision_transformer_1").getGroup(fmt::format("encoder_{}", 2+i)).getGroup("layerNorm2_encoder").getDataSet("gamma:0").read<array<float, EMBEDDING_DEPTH>>();
    }

    return 0;
}
