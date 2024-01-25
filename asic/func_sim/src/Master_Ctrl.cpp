#include <Master_Ctrl.hpp>

/*----- NAMESPACE -----*/
using namespace std;

/*----- DECLARATION -----*/
Master_ctrl::Master_ctrl(const string eeg_filepath, const string params_filepath) : gen_cnt_8b(8), gen_cnt_10b(10) {
    eeg_fp = eeg_filepath;
    params_fp = params_filepath;
    state = RESET;
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
        if (ext_sigs->new_sleep_epoch == true) { start_signal_load(); }
        break;

    case SIGNAL_LOAD:
        /* Sequentially parses the input EEG file and broadcasts it to all CiMs to emulate the ADC feed in the ASIC*/
        if (eeg != eeg_ds.end()) {
            struct instruction inst = {
                /*bus_direction */ MASTER_TO_CIM,
                /*op*/ PATCH_LOAD_BROADCAST_OP,
                /*target_cim*/ 0,
                /*data*/ *eeg
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
        sys_state = DONE;
        break;

    case RESET:
        if (ext_sigs->master_nrst == true) { state = IDLE; }
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

    HighFive::File file(eeg_fp, HighFive::File::ReadOnly);
    eeg_ds = file.getDataSet("eeg").read<vector<uint16_t>>();
    eeg = eeg_ds.begin();

    return 0;
}

int Master_ctrl::start_weights_load(){
    // auto dataset = file.getDataSet("grp/data");
    // auto data = dataset.read<vector<int>>();

    return 0;
}
int Master_ctrl::broadcast_inst(struct instruction){return 0;}
