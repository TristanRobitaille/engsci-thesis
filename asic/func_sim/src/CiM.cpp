#include <CiM.hpp>

/*----- NAMESPACE -----*/
using namespace std;

/*----- DECLARATION -----*/
CiM::CiM(const int16_t cim_id) : id(cim_id), gen_cnt_10b(10) {
    state = RESET;
}

int CiM::reset(){
    fill(begin(data_and_param), end(data_and_param), 0); // Reset local data_and_param
    fill(begin(storage), end(storage), 0); // Reset local storage
    return 0;
}

int CiM::run(struct ext_signals* ext_sigs, Bus* bus){
    /* Run the CiM FSM */

    // Read bus if new instruction
    struct instruction inst = bus->get_inst();
    if (inst.target_cim == id && bus->is_trans_new()) {
        switch (inst.op){
        case PATCH_LOAD_BROADCAST_OP:
            if (prev_bus_op != PATCH_LOAD_BROADCAST_OP) { gen_cnt_10b.reset(); };
            data_and_param[gen_cnt_10b.get_cnt()] = inst.data;
            gen_cnt_10b.inc();
            if (gen_cnt_10b.get_cnt() == PATCH_LENGTH_NUM_SAMPLES) { // Received a complete patch, perform part of Dense layer
                /* TODO: Trigger Dense vector-matrix mult */
                gen_cnt_10b.reset();
            }
            break;

        case INVALID_OP:
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
