#include <CiM.hpp>

/*----- NAMESPACE -----*/
using namespace std;

/*----- DECLARATION -----*/
CiM::CiM(const int16_t cim_id) : id(cim_id), gen_cnt_10b(10), gen_cnt_10b_2(10) {
    state = RESET_CIM;
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
                intermediate_res[gen_cnt_10b_2.get_cnt() + PATCH_LENGTH_NUM_SAMPLES] = result + intermediate_res[param_addr_map[SINGLE_PARAMS][ADDR]+PATCH_PROJ_BIAS_OFF];
                gen_cnt_10b_2.inc(); // Increment number of patches received
                gen_cnt_10b.reset();
                if (gen_cnt_10b_2.get_cnt() == NUM_PATCHES) { state = INFERENCE_RUNNING_CIM; } // Received all patches, automatically start inference
            }
            break;

        case DATA_STREAM_START_OP:
            if (inst.target_or_sender == id) {
                gen_reg_16b = inst.data[0]; // Address
                gen_cnt_10b.reset();
            }
            break;

        case DATA_STREAM_OP:
            if (inst.target_or_sender == id) {
                params[gen_cnt_10b.get_cnt() + gen_reg_16b] = inst.data[0];
                params[gen_cnt_10b.get_cnt()+1 + gen_reg_16b] = inst.data[1]; // Note: If the length is less than 3, we will be loading junk. That's OK since it will get overwritten
                params[gen_cnt_10b.get_cnt()+2 + gen_reg_16b] = inst.extra_fields;
                gen_cnt_10b.inc(3);
            }
            break;

        case TRANSPOSE_BROADCAST_START_OP:
            if (inst.target_or_sender == id) { // Master controller tells me to start broadcasting some data
                int start_addr = static_cast<int> (inst.data[0]);
                struct instruction new_inst = {/*op*/ TRANSPOSE_BROADCAST_DATA_OP, /*target_or_sender*/id, /*data*/{intermediate_res[start_addr], intermediate_res[start_addr+1]}, /*extra_fields*/intermediate_res[start_addr+2]};
                gen_reg_16b = start_addr; // Save address of data to send
                gen_reg_16b_2 = inst.data[1]; // Save length of data to send
                bus->push_inst(new_inst);
            } else {
                gen_reg_16b = static_cast<int> (inst.extra_fields); // Save address where to store data
            }
            gen_cnt_10b.set_val(3); // 3 elements already sent
            break;

        case TRANSPOSE_BROADCAST_DATA_OP:
            if (gen_cnt_10b.get_cnt() < gen_reg_16b_2) { // If still more data to send/receive
                if (inst.target_or_sender == id) { // If last time was me and I still have data to send, continue sending data
                    struct instruction new_inst = {/*op*/ TRANSPOSE_BROADCAST_DATA_OP, /*target_or_sender*/id, /*data*/{0, 0}, /*extra_fields*/0};
                    if ((gen_reg_16b_2 - gen_cnt_10b.get_cnt() == 2)) { // Only two bytes left
                        new_inst.data = {intermediate_res[gen_reg_16b+gen_cnt_10b.get_cnt()], intermediate_res[gen_reg_16b+gen_cnt_10b.get_cnt()+1]};
                    } else if ((gen_reg_16b_2 - gen_cnt_10b.get_cnt() == 1)) { // Only one byte left
                        new_inst.data = {intermediate_res[gen_reg_16b+gen_cnt_10b.get_cnt()], 0};
                    } else {
                        new_inst.data = {intermediate_res[gen_reg_16b+gen_cnt_10b.get_cnt()], intermediate_res[gen_reg_16b+gen_cnt_10b.get_cnt()+1]};
                        new_inst.extra_fields = intermediate_res[gen_reg_16b+gen_cnt_10b.get_cnt()+2];
                    }

                    bus->push_inst(new_inst);

                } else { // I'm not broadcasting, but listening
                    if ((gen_cnt_10b.get_cnt() - id) == 1) {
                        intermediate_res[gen_reg_16b+inst.target_or_sender] = inst.extra_fields;
                    } else if ((gen_cnt_10b.get_cnt() - id) == 2) {
                        intermediate_res[gen_reg_16b+inst.target_or_sender] = inst.data[1];
                    } else if ((gen_cnt_10b.get_cnt() - id) == 3) {
                        intermediate_res[gen_reg_16b+inst.target_or_sender] = inst.data[0];
                    }
                }
                gen_cnt_10b.inc(3); // Increment data sent
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
    case IDLE_CIM:
        break;

    case RESET_CIM:
        if (ext_sigs->master_nrst == true) { state = IDLE_CIM; }
        reset();
        break;

    case INFERENCE_RUNNING_CIM:
        is_idle = false;

        switch (current_inf_step) {
        case CLASS_TOKEN_CONCAT:
            intermediate_res[PATCH_LENGTH_NUM_SAMPLES+NUM_PATCHES] = params[param_addr_map[SINGLE_PARAMS][ADDR]+CLASS_EMB_OFF]; // Move classification token from parameters memory to intermediate storage
            current_inf_step = POS_EMB;
            gen_cnt_10b.reset();
            break;

        case POS_EMB:
            if (gen_cnt_10b.get_cnt() < NUM_PATCHES+1) {
                intermediate_res[gen_cnt_10b.get_cnt()] = ADD(PATCH_LENGTH_NUM_SAMPLES+gen_cnt_10b.get_cnt(), param_addr_map[POS_EMB_PARAMS][ADDR]+gen_cnt_10b.get_cnt());
                gen_cnt_10b.inc();
            } else {
                gen_cnt_10b.reset();
                current_inf_step = ENC_LAYERNORM;
                is_idle = true; // Indicate to master controller that positional embedding computation is done
            }
            break;

        case ENC_LAYERNORM:
            if ((inst.op == PISTOL_START_OP) && (id < (NUM_PATCHES+1))) { // Wait for master's start signal to perform LayerNorm (only CiM # < NUM_PATCHES+1 have a row to LayerNorm)
                if (compute_in_progress == false) {
                    float gamma = params[param_addr_map[SINGLE_PARAMS][0]+ENC1_LAYERNORM1_GAMMA];
                    float beta = params[param_addr_map[SINGLE_PARAMS][0]+ENC1_LAYERNORM1_BETA];
                    gen_reg_16b = 1; // Just a signal to avoid coming here every time FSM runs
                    is_idle = false;
                    LAYERNORM(0, gamma, beta);
                } else if (compute_in_progress == false && gen_reg_16b == 1) { // Done with LayerNorm
                    current_inf_step = ENC_MHSA_DENSE;
                }
            }
            break;

        case ENC_MHSA_DENSE:
        case INVALID_INF_STEP:
        default:
            break;
        }
        break;

    case INVALID_CIM:
    default:
        cout << "CiM " << id << " controller in an invalid state (" << state << ")!\n" << endl;
        break;
    }
    return 0;
}

float CiM::MAC(uint16_t input_start_addr, uint16_t params_start_addr, uint16_t len) {
    /* Dot-product between two vectors. The first vector is in the intermediate storage location, and the second is in params storage. */

    float result = 0.0f;
    for (uint16_t i = 0; i < len; ++i) {
        result += intermediate_res[input_start_addr+i] * params[params_start_addr+i];
    }
    return result;
}

float CiM::ADD(uint16_t input_addr, uint16_t params_addr) {
    /* Return sum of two inputs. First input is in intermediate storage location, and second is in the params storage. */
    return intermediate_res[input_addr] + params[params_addr];
}

void CiM::LAYERNORM(uint16_t input_addr, float gamma, float beta) {
    /* Layer normalization of input. Input is in intermediate storage location. Note: All LayerNorms are done over a row of EMBEDDING_DEPTH length*/
    compute_in_progress = true; // Using this compute_in_progress signal as it will be used in the CiM (in ASIC, this compute is multi-cycle, so leaving this here for visibility)

    float result = 0.0f;
    float result2 = 0.0f;
    
    // Summation along feature axis
    for (uint16_t i = 0; i < EMBEDDING_DEPTH; ++i) {
        result += intermediate_res[input_addr+i];
    }

    result /= EMBEDDING_DEPTH; // Mean

    // Subtract and square mean
    for (uint16_t i = 0; i < EMBEDDING_DEPTH; ++i) {
        intermediate_res[input_addr+i] -= result; // Subtract
        intermediate_res[input_addr+i] *= intermediate_res[input_addr+i]; // Square
        result2 += intermediate_res[input_addr+i]; // Sum
    }

    result2 /= EMBEDDING_DEPTH; // Mean
    result2 += LAYERNORM_EPSILON; // Add epsilon
    result2 = sqrt(result2); // <-- Standard deviation

    // Normalize
    for (uint16_t i = 0; i < EMBEDDING_DEPTH; ++i) {
        intermediate_res[input_addr+i] = gamma * (intermediate_res[input_addr+i] / result2) + beta;
    }

    compute_in_progress = false;
}

bool CiM::get_is_idle() {
    return is_idle;
}