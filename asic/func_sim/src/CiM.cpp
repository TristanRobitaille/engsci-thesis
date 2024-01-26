#include <CiM.hpp>

/*----- NAMESPACE -----*/
using namespace std;

/*----- DECLARATION -----*/
CiM::CiM(const int16_t cim_id) : id(cim_id), gen_cnt_10b(10), gen_cnt_10b_2(10) {
    state = RESET;
}

int CiM::reset(){
    fill(begin(params), end(params), 0); // Reset local params
    fill(begin(intermediate_res), end(intermediate_res), 0); // Reset local intermediate_res
    return 0;
}

int CiM::run(struct ext_signals* ext_sigs, Bus* bus){
    /* Run the CiM FSM */

    // Read bus if new instruction
    struct instruction inst = bus->get_inst();
    if (inst.op != NOP) {
        switch (inst.op){
        case PATCH_LOAD_BROADCAST_OP:
            if (prev_bus_op != PATCH_LOAD_BROADCAST_OP) { gen_cnt_10b.reset(); };
            intermediate_res[gen_cnt_10b.get_cnt()] = inst.data[0];
            gen_cnt_10b.inc();
            if (gen_cnt_10b.get_cnt() == PATCH_LENGTH_NUM_SAMPLES) { // Received a complete patch, perform part of Dense layer
                float result = MAC(0 /* patch starting address */, 0 /* weight starting address */, PATCH_LENGTH_NUM_SAMPLES);
                intermediate_res[gen_cnt_10b_2.get_cnt() + PATCH_LENGTH_NUM_SAMPLES] = result + intermediate_res[param_address_mapping[PATCH_PROJ_DENSE_BIAS][0]];
                gen_cnt_10b_2.inc(); // Increment number of patches received
                gen_cnt_10b.reset();
            }
            break;

        case DATA_STREAM_START_OP:
            if (inst.target_cim == id) {
                gen_reg_16b = inst.data[0]; // Address
                gen_cnt_10b.reset();
            }
            break;

        case DATA_STREAM:
            if (inst.target_cim == id) {
                params[gen_cnt_10b.get_cnt() + gen_reg_16b] = inst.data[0];
                params[gen_cnt_10b.get_cnt()+1 + gen_reg_16b] = inst.data[1]; // Note: If the length is less than 3, we will be loading junk. That's OK since it will get overwritten
                params[gen_cnt_10b.get_cnt()+2 + gen_reg_16b] = inst.extra_fields;
                gen_cnt_10b.inc(3);
            }
            break;

        case NOP:
        default:
            break;
        }
    }
    prev_bus_op = inst.op;

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
        cout << "CiM " << id << " controller in an invalid state (" << state << ")!\n" << endl;
        break;
    }
    return 0;
}

float CiM::MAC(uint16_t input_start_addr, uint16_t params_start_addr, uint16_t len) {
    /* Dot-product between two vectors. The first vector is in the intermediate storage location, and the second is in the params section storage. */

    float result = 0.0f;
    for (uint16_t i = 0; i < len; ++i) {
        result += intermediate_res[input_start_addr+i] * params[params_start_addr+i];
    }
    return result;
}