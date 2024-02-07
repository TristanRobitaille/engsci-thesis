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
                float result = MAC(0 /* patch starting address */, 0 /* weight starting address */, PATCH_LENGTH_NUM_SAMPLES, MODEL_PARAM);
                intermediate_res[gen_cnt_10b_2.get_cnt() + PATCH_LENGTH_NUM_SAMPLES] = result + intermediate_res[param_addr_map[SINGLE_PARAMS].addr+PATCH_PROJ_BIAS_OFF];
                gen_cnt_10b_2.inc(); // Increment number of patches received
                gen_cnt_10b.reset();
                if (gen_cnt_10b_2.get_cnt() == NUM_PATCHES) { // Received all patches, automatically start inference
                    state = INFERENCE_RUNNING_CIM;
                    compute_done = false;
                    gen_cnt_10b_2.reset();
                }
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

        case DENSE_BROADCAST_START_OP:
        case TRANSPOSE_BROADCAST_START_OP:
            if (inst.target_or_sender == id) { // Master controller tells me to start broadcasting some data
                int start_addr = static_cast<int> (inst.data[0]);
                struct instruction new_inst;

                if (inst.op == DENSE_BROADCAST_START_OP) {
                    new_inst = {/*op*/ DENSE_BROADCAST_DATA_OP, /*target_or_sender*/id, /*data*/{intermediate_res[start_addr], intermediate_res[start_addr+1]}, /*extra_fields*/intermediate_res[start_addr+2]};
                } else if (inst.op == TRANSPOSE_BROADCAST_START_OP) {
                    new_inst = {/*op*/ TRANSPOSE_BROADCAST_DATA_OP, /*target_or_sender*/id, /*data*/{intermediate_res[start_addr], intermediate_res[start_addr+1]}, /*extra_fields*/intermediate_res[start_addr+2]};
                }

                gen_reg_16b = start_addr; // Save address of data to send
                gen_reg_16b_2 = inst.data[1]; // Save length of data to send
                bus->push_inst(new_inst);

                if ((inst.op == TRANSPOSE_BROADCAST_DATA_OP) && (current_inf_step == ENC_MHSA_QK_T)) { // Since I'm here, move my own data to the correct location in intermediate_res (do I need to do that in the previous op (QVK dense) as well?)
                    intermediate_res[static_cast<int>(inst.extra_fields)+id] = intermediate_res[gen_reg_16b+id];
                } else if ((inst.op == DENSE_BROADCAST_START_OP) && (current_inf_step == ENC_MHSA_QK_T) && (inst.target_or_sender == 0)) { // Keep track of matrix in the Z-stack by incrementing upon receiving a DENSE_BROADCAST_START_OP for the first CiM
                    gen_cnt_10b_2.inc();
                }
            } else {
                gen_reg_16b = static_cast<int> (inst.extra_fields); // Save address where to store data
            }
            gen_cnt_10b.set_val(3); // 3 elements already sent
            break;

        case DENSE_BROADCAST_DATA_OP:
        case TRANSPOSE_BROADCAST_DATA_OP:
            if (gen_cnt_10b.get_cnt() < gen_reg_16b_2) { // If still more data to send/receive
                if (inst.target_or_sender == id) { // If last time was me and I still have data to send, continue sending data
                    struct instruction new_inst = {/*op*/ inst.op, /*target_or_sender*/id, /*data*/{0, 0}, /*extra_fields*/0};
                    if ((gen_reg_16b_2 - gen_cnt_10b.get_cnt() == 2)) { // Only two bytes left
                        new_inst.data = {intermediate_res[gen_reg_16b+gen_cnt_10b.get_cnt()], intermediate_res[gen_reg_16b+gen_cnt_10b.get_cnt()+1]};
                    } else if ((gen_reg_16b_2 - gen_cnt_10b.get_cnt() == 1)) { // Only one byte left
                        new_inst.data = {intermediate_res[gen_reg_16b+gen_cnt_10b.get_cnt()], 0};
                    } else {
                        new_inst.data = {intermediate_res[gen_reg_16b+gen_cnt_10b.get_cnt()], intermediate_res[gen_reg_16b+gen_cnt_10b.get_cnt()+1]};
                        new_inst.extra_fields = intermediate_res[gen_reg_16b+gen_cnt_10b.get_cnt()+2];
                    }

                    bus->push_inst(new_inst);
                } else if (inst.op == TRANSPOSE_BROADCAST_DATA_OP) { // Not broadcasting and, for a transpose, grab only some data
                    intermediate_res[gen_reg_16b+inst.target_or_sender] = ((gen_cnt_10b.get_cnt() - id) == 1) ? (inst.extra_fields) : (intermediate_res[gen_reg_16b+inst.target_or_sender]);
                    intermediate_res[gen_reg_16b+inst.target_or_sender] = ((gen_cnt_10b.get_cnt() - id) == 2) ? (inst.data[1]) : (intermediate_res[gen_reg_16b+inst.target_or_sender]);
                    intermediate_res[gen_reg_16b+inst.target_or_sender] = ((gen_cnt_10b.get_cnt() - id) == 3) ? (inst.data[0]) : (intermediate_res[gen_reg_16b+inst.target_or_sender]);
                }
                
                // Always move data (even my own) to the correct location to perform operations later (TODO: move up with other lines above?)
                if (inst.op == DENSE_BROADCAST_DATA_OP) {
                    intermediate_res[gen_reg_16b+gen_cnt_10b.get_cnt()-3] = inst.data[0];
                    intermediate_res[gen_reg_16b+gen_cnt_10b.get_cnt()-2] = inst.data[1];
                    intermediate_res[gen_reg_16b+gen_cnt_10b.get_cnt()-1] = inst.extra_fields;
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
            intermediate_res[PATCH_LENGTH_NUM_SAMPLES+NUM_PATCHES] = params[param_addr_map[SINGLE_PARAMS].addr+CLASS_EMB_OFF]; // Move classification token from parameters memory to intermediate storage
            current_inf_step = POS_EMB;
            gen_cnt_10b.reset();
            break;

        case POS_EMB:
            if (gen_cnt_10b.get_cnt() < NUM_PATCHES+1) {
                intermediate_res[gen_cnt_10b.get_cnt()] = ADD(PATCH_LENGTH_NUM_SAMPLES+gen_cnt_10b.get_cnt(), param_addr_map[POS_EMB_PARAMS].addr+gen_cnt_10b.get_cnt());
                gen_cnt_10b.inc();
            } else {
                gen_cnt_10b.reset();
                current_inf_step = ENC_LAYERNORM_1ST_HALF;
                is_idle = true; // Indicate to master controller that positional embedding computation is done
            }
            break;

        case ENC_LAYERNORM_1ST_HALF:
            if (inst.op == PISTOL_START_OP){// Wait for master's start signal to perform LayerNorm. Note: Only CiM # < NUM_PATCHES+1 have a row to LayerNorm, so the others will just compute garbage
                if (compute_in_progress == false) {
                    gen_reg_16b = 1; // Just a signal to avoid coming here every time FSM runs
                    is_idle = false;
                    LAYERNORM_1ST_HALF(0);
                }
            } else if (compute_in_progress == false && gen_reg_16b == 1) { // Done with LayerNorm
                current_inf_step = ENC_LAYERNORM_2ND_HALF;
                is_idle = true;
            }
            break;

        case ENC_LAYERNORM_2ND_HALF:
            if (inst.op == PISTOL_START_OP) { // Wait for master's start signal to perform LayerNorm
                if (compute_in_progress == false) {
                    float gamma = params[param_addr_map[SINGLE_PARAMS].addr+ENC_LAYERNORM1_GAMMA_OFF];
                    float beta = params[param_addr_map[SINGLE_PARAMS].addr+ENC_LAYERNORM1_BETA_OFF];
                    gen_reg_16b = 1; // Just a signal to avoid coming here every time FSM runs
                    is_idle = false;
                    LAYERNORM_2ND_HALF(0, gamma, beta);
                }
            } else if (compute_in_progress == false && gen_reg_16b == 1) { // Done with LayerNorm TODO: Use compute_done here to control?
                current_inf_step = ENC_MHSA_DENSE;
                is_idle = true;
            }
            break;

        case ENC_MHSA_DENSE:
            if ((inst.op == DENSE_BROADCAST_DATA_OP) && (gen_cnt_10b.get_cnt() >= gen_reg_16b_2)) { // No more data to receive, start MACs
                is_idle = false;
                if (compute_in_progress == false){
                    // Note: In ASIC, these would be sequential MACs, but here we are doing them in parallel
                    float result = MAC(NUM_PATCHES+1+EMBEDDING_DEPTH, 128, EMBEDDING_DEPTH, MODEL_PARAM);
                    intermediate_res[189+inst.target_or_sender] = result + params[param_addr_map[SINGLE_PARAMS].addr+ENC_Q_DENSE_BIAS_0FF]; // Q
                    result = MAC(NUM_PATCHES+1+EMBEDDING_DEPTH, 192, EMBEDDING_DEPTH, MODEL_PARAM);
                    intermediate_res[250+inst.target_or_sender] = result + params[param_addr_map[SINGLE_PARAMS].addr+ENC_K_DENSE_BIAS_0FF]; // K 
                    result = MAC(NUM_PATCHES+1+EMBEDDING_DEPTH, 256, EMBEDDING_DEPTH, MODEL_PARAM);
                    intermediate_res[NUM_PATCHES+1+inst.target_or_sender] = result + params[param_addr_map[SINGLE_PARAMS].addr+ENC_V_DENSE_BIAS_0FF]; // V
                }
            }

            if (compute_done) { // Done with all input rows of QKV Dense
                is_idle = true;
            }

            if (compute_done && inst.op == PISTOL_START_OP) {
                current_inf_step = ENC_MHSA_QK_T;
                compute_done = false;
            }
            break;

        case ENC_MHSA_QK_T:
            if ((inst.op == DENSE_BROADCAST_DATA_OP) && (gen_cnt_10b.get_cnt() >= gen_reg_16b_2)) { // No more data to receive, start MACs
                is_idle = false;
                if (compute_in_progress == false){
                    float result = MAC(NUM_PATCHES+1+EMBEDDING_DEPTH+(gen_cnt_10b_2.get_cnt()-1)*NUM_HEADS, 2*(EMBEDDING_DEPTH+NUM_PATCHES+1), NUM_HEADS, INTERMEDIATE_RES); // gen_reg_16b hold the matrix in the Z-stack
                    intermediate_res[2*EMBEDDING_DEPTH+2*(NUM_PATCHES+1)] = result;

                    result = DIV(2*EMBEDDING_DEPTH+2*(NUM_PATCHES+1), param_addr_map[SINGLE_PARAMS].addr+ENC_SQRT_NUM_HEADS_OFF); // /sqrt(NUM_HEADS)
                    intermediate_res[2*EMBEDDING_DEPTH+2*(NUM_PATCHES+1)] = result;
                }
            }
            break;

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

float CiM::MAC(uint16_t in1_start_addr, uint16_t in2_start_addr, uint16_t len, INPUT_TYPE param_type) {
    /* Dot-product between two vectors. The first vector is in the intermediate storage location, and the second is in params storage. */

    float result = 0.0f;
    compute_in_progress = true;
    for (uint16_t i = 0; i < len; ++i) {
        if (param_type == INTERMEDIATE_RES) {result += intermediate_res[in1_start_addr+i] * intermediate_res[in2_start_addr+i]; } // The second vector is an intermediate result
        else { result += intermediate_res[in1_start_addr+i] * params[in2_start_addr+i]; } // The second vector is a model parameter
    }
    compute_in_progress = false;
    compute_done = true;
    return result;
}

float CiM::DIV(uint16_t num_addr, uint16_t den_addr) {
    /* Return division of two inputs. First input is in intermediate storage location, and second is in the params storage. */
    float result = 0.0f;
    compute_in_progress = true; // Using this compute_in_progress signal as it will be used in the CiM (in ASIC, this compute is multi-cycle, so leaving this here for visibility)
    result = intermediate_res[num_addr] / params[den_addr];
    compute_in_progress = false;
}

float CiM::ADD(uint16_t input_addr, uint16_t params_addr) {
    /* Return sum of two inputs. First input is in intermediate storage location, and second is in the params storage. */
    return intermediate_res[input_addr] + params[params_addr];
}

void CiM::LAYERNORM_1ST_HALF(uint16_t input_addr) {
    /* 1st half of Layer normalization of input. Input is in intermediate storage location. Note: All LayerNorms are done over a row of EMBEDDING_DEPTH length. */
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

    // Partial normalization (excludes gamma and beta, which are applied in LAYERNORM_FINAL_NORM() since they need to be applied column-wise)
    for (uint16_t i = 0; i < EMBEDDING_DEPTH; ++i) {
        intermediate_res[input_addr+i] = intermediate_res[input_addr+i] / result2;
    }

    compute_in_progress = false;
}

void CiM::LAYERNORM_2ND_HALF(uint16_t input_addr, float gamma, float beta) {
    /* 2nd half of Layer normalization of input. This applies gamma and beta on each column. */

    compute_in_progress = true;

    // Normalize
    for (uint16_t i = 0; i < EMBEDDING_DEPTH; ++i) {
        intermediate_res[input_addr+i] = gamma * intermediate_res[input_addr+i] + beta;
    }

    compute_in_progress = false;
}

bool CiM::get_is_idle() {
    return is_idle;
}