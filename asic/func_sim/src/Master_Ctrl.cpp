#include <Master_Ctrl.hpp>

/*----- NAMESPACE -----*/
using namespace std;

/*----- DECLARATION -----*/
Master_ctrl::Master_ctrl(const string eeg_filepath, const string params_filepath) : gen_cnt_8b(8), gen_cnt_10b(10), params_file(params_filepath, HighFive::File::ReadOnly) {
    state = RESET;

    // EEG data file
    HighFive::File eeg_file(eeg_filepath, HighFive::File::ReadOnly);
    eeg_ds = eeg_file.getDataSet("eeg").read<vector<float>>();
    eeg = eeg_ds.begin();
}

int Master_ctrl::reset(){
    fill(begin(storage), end(storage), 0); // Reset remote storage
    gen_cnt_8b.reset();
    gen_cnt_10b.reset();
    return 0;
}

system_state Master_ctrl::run(struct ext_signals* ext_sigs, Bus* bus){
    /* Run the master controller FSM */

    system_state sys_state = RUNNING;

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
        cout << "Inference running, but exiting for now." << endl;
        sys_state = INFERENCE_FINISHED;
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
    case PATCH_PROJ_DENSE_KERNEL:
        if ((params_cim_cnt == NUM_CIM-1) && (params_data_cnt >= param_address_mapping[params_curr_layer][1])) { // Start new layer
            params_cim_cnt = 0;
            params_data_cnt = 0;
            gen_cnt_8b.reset();
            params_curr_layer++;
            
        } else if (params_data_cnt >= param_address_mapping[params_curr_layer][1]) { // Sending to a new CiM
            uint16_t addr = param_address_mapping[params_curr_layer][0];
            uint16_t length = param_address_mapping[params_curr_layer][1];            
            std::array<float, 2> data = {static_cast<float>(addr), static_cast<float>(length)};

            params_cim_cnt++;
            inst = {DATA_STREAM_START_OP, /*target_cim*/ params_cim_cnt, /*data={start_addr, length_elem}*/ data, /*extra_fields*/ 0};
            params_data_cnt = 0;

        } else {
            auto data = params_file.getGroup("patch_projection_dense").getGroup("vision_transformer_1").getGroup("patch_projection_dense").getDataSet("kernel:0").read<array<array<float, EMBEDDING_DEPTH>, PATCH_LENGTH_NUM_SAMPLES>>(); // TODO: Should not load dataset every time...
            if (params_data_cnt < 63) {
                inst = {DATA_STREAM, /*target_cim*/ params_cim_cnt, /*data={start_addr, length_elem}*/ {data[params_data_cnt][params_cim_cnt], data[params_data_cnt+1][params_cim_cnt]}, /*extra_fields*/ data[params_data_cnt+2][params_cim_cnt]};
            } else {
                inst = {DATA_STREAM, /*target_cim*/ params_cim_cnt, /*data={start_addr, length_elem}*/ {data[params_data_cnt][params_cim_cnt], 0}, /*extra_fields*/ 0};
            }
            params_data_cnt = params_data_cnt + 3;

        }
        break;

    case PATCH_PROJ_DENSE_BIAS:
        if ((params_cim_cnt < NUM_CIM) && (gen_cnt_8b.get_cnt() == 0)) {
            uint16_t addr = param_address_mapping[params_curr_layer][0];
            uint16_t length = param_address_mapping[params_curr_layer][1];            
            std::array<float, 2> data = {static_cast<float>(addr), static_cast<float>(length)};
            inst = {DATA_STREAM_START_OP, /*target_cim*/ params_cim_cnt, /*data={start_addr, length_elem}*/ data, /*extra_fields*/ 0};
            gen_cnt_8b.inc(); // Use as indication that next time this runs, we go to the else if () below
        } else if (params_cim_cnt < NUM_CIM){
            auto data = params_file.getGroup("patch_projection_dense").getGroup("vision_transformer_1").getGroup("patch_projection_dense").getDataSet("bias:0").read<array<float, EMBEDDING_DEPTH>>(); // TODO: Should not load dataset every time...
            inst = {DATA_STREAM, /*target_cim*/ params_cim_cnt, /*data={start_addr, length_elem}*/ {data[params_cim_cnt], 0}, /*extra_fields*/ 0};
            gen_cnt_8b.reset();
            params_cim_cnt++;
        } else {
            params_cim_cnt = 0;
            gen_cnt_8b.reset();
            params_curr_layer++;    
        }
        break;

    case CLASS_EMB:
    case POS_EMB:
    case PARAM_LOAD_FINISHED:
    default:
        state = IDLE;
        break;
    }

    return inst;
}
int Master_ctrl::broadcast_inst(struct instruction){return 0;}
