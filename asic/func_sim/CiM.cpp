#include "CiM.hpp"

/*----- NAMESPACE -----*/
using namespace std;

/*----- DECLARATION -----*/
CiM::CiM(const int16_t cim_id) : id(cim_id), gen_cnt_10b(10) {
    state = RESET;
}

int CiM::reset(){
    fill(begin(data_and_weights), end(data_and_weights), 0); // Reset local data_and_weights
    fill(begin(storage), end(storage), 0); // Reset local storage
    return 0;
}

int CiM::run(struct ext_signals* ext_sigs, Bus* bus){
    /* Run the CiM FSM */

    // Read bus if new instruction
    struct instruction inst = bus->get_inst();
    if (inst.target_cim == id && bus->is_trans_new()) {
        switch (inst.op){
        case PATCH_LOAD:
            if (prev_bus_op != PATCH_LOAD) {
                cout << inst.data << endl;
                gen_cnt_10b.reset();
            };
            data_and_weights[gen_cnt_10b.get_cnt()] = inst.data;
            break;

        case INVALID:
        default:
            break;
        }

        prev_bus_op = inst.op;
    }

    // Run FSM
    switch (state){
    case IDLE:
        break;

    case RESET:
        if (ext_sigs->master_nrst == true) { state = IDLE; }
        reset();
        break;

    case INVALID:
    default:
        cout << "CiM " << id << " controller in an invalid state!\n" << endl;
        break;
    }
    return 0;
}
