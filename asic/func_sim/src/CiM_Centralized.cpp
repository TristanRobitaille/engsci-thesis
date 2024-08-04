
#if CENTRALIZED_ARCH
#include "Cim_Centralized.hpp"

/*----- NAMESPACE -----*/
using namespace std;

/*----- DEFINITION -----*/
CiM_Centralized::CiM_Centralized() {
    reset();
    load_previous_softmax(); // Load in dummy softmax data of previous epochs for verification
}

void CiM_Centralized::load_previous_softmax() {
    // TODO: Implement this function
}

int CiM_Centralized::reset(){
    fill(begin(int_res), end(int_res), 0); // Reset local int_res
    cim_state = SLEEP_CIM;
    current_inf_step = PATCH_DENSE_STEP;
    system_state = IDLE;

    return 0;
}

SYSTEM_STATE CiM_Centralized::run(struct ext_signals* ext_sigs){
    /* Run the CiM FSM */
    if (ext_sigs->master_nrst == false) { reset(); }

    // Update compute process counter
    update_compute_process_cnt();

    switch (cim_state){
        case SLEEP_CIM:
            if (ext_sigs->start_param_load) {
                cim_state = IDLE_CIM;
            }
            break;

        case IDLE_CIM:
            system_state = EVERYTHING_FINISHED;
            break;

        case INVALID_CIM:
            break;
    }

    return system_state;
}

#endif