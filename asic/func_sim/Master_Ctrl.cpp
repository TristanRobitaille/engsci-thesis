#include "Master_Ctrl.hpp"
#include "csv-parser/single_include/csv.hpp"

/*----- NAMESPACE -----*/
using namespace std;

/*----- DECLARATION -----*/
Master_ctrl::Master_ctrl(const std::string eeg_fp) : eeg(eeg_fp), gen_cnt_8b(8), gen_cnt_10b(10) {
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
    csv::CSVRow row;

    // Act on priority external signals
    if  (ext_sigs->master_nrst == false) { state = RESET; }

    switch (state) {
    case IDLE:
        if (ext_sigs->new_sleep_epoch == true) { start_signal_load(); }
        break;

    case SIGNAL_LOAD:
        if (eeg.read_row(row)) {
            struct instruction inst = {
                /*bus_direction */ MASTER_TO_CIM,
                /*op*/ PATCH_LOAD,
                /*target_cim*/ static_cast<uint16_t>(gen_cnt_8b.get_cnt()),
                /*data*/ row["EEG"].get<uint16_t>()
            };
            
            bus->push_inst(inst); // Broadcast on bus
            gen_cnt_10b.inc();
            if (gen_cnt_10b.get_cnt() % PATCH_LENGTH_NUM_SAMPLES == 0) { 
                gen_cnt_10b.reset(); //Avoid overflow
                gen_cnt_8b.inc(); // Increment target CiM counter
            }
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
    /* Sequentially parses the input EEG file and forwards it to the correct CiM */
    /* The CSV stream emulates the ADC feed in the ASIC*/

    state = SIGNAL_LOAD;
    gen_cnt_8b.reset();
    gen_cnt_10b.reset();

    return 0;
}
int Master_ctrl::start_weights_load(){return 0;}
int Master_ctrl::broadcast_inst(struct instruction){return 0;}
