#include <CiM.hpp>

/*----- NAMESPACE -----*/
using namespace std;

/*----- DECLARATION -----*/
CiM::CiM(const int16_t cim_id) : id(cim_id), gen_cnt_10b(10), gen_cnt_10b_2(10), bytes_rec_cnt(10), bytes_sent_cnt(8) {
    state = RESET_CIM;
}

int CiM::reset(){
    fill(begin(params), end(params), 0); // Reset local params
    fill(begin(intermediate_res), end(intermediate_res), 0); // Reset local intermediate_res
    gen_cnt_10b.reset();
    gen_cnt_10b_2.reset();
    bytes_rec_cnt.reset();
    is_idle = true;
    return 0;
}

int CiM::run(struct ext_signals* ext_sigs, Bus* bus){
    /* Run the CiM FSM */

    // Read bus if new instruction
    struct instruction inst = bus->get_inst();
    if (inst.op != NOP) {
        switch (inst.op){
        case PATCH_LOAD_BROADCAST_OP:
            if (prev_bus_op != PATCH_LOAD_BROADCAST_OP) { bytes_rec_cnt.reset(); };
            intermediate_res[bytes_rec_cnt.get_cnt()] = inst.data[0];
            bytes_rec_cnt.inc();
            if (bytes_rec_cnt.get_cnt() == PATCH_LENGTH_NUM_SAMPLES) { // Received a complete patch, perform part of Dense layer
                float result = MAC(0 /* patch starting address */, 0 /* weight starting address */, PATCH_LENGTH_NUM_SAMPLES, MODEL_PARAM);
                intermediate_res[gen_cnt_10b_2.get_cnt() + PATCH_LENGTH_NUM_SAMPLES] = result + intermediate_res[param_addr_map[SINGLE_PARAMS].addr+PATCH_PROJ_BIAS_OFF];
                gen_cnt_10b_2.inc(); // Increment number of patches received
                bytes_rec_cnt.reset();
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
                bytes_rec_cnt.reset();
            }
            break;

        case DATA_STREAM_OP:
            if (inst.target_or_sender == id) {
                params[bytes_rec_cnt.get_cnt() + gen_reg_16b] = inst.data[0];
                params[bytes_rec_cnt.get_cnt()+1 + gen_reg_16b] = inst.data[1]; // Note: If the length is less than 3, we will be loading junk. That's OK since it will get overwritten
                params[bytes_rec_cnt.get_cnt()+2 + gen_reg_16b] = inst.extra_fields;
            }
            break;

        case DENSE_BROADCAST_START_OP:
        case TRANS_BROADCAST_START_OP:
            if (inst.target_or_sender == id) { // Master controller tells me to start broadcasting some data
                int start_addr = static_cast<int> (inst.data[0]);
                struct instruction new_inst;

                if (inst.op == DENSE_BROADCAST_START_OP) {
                    new_inst = {/*op*/ DENSE_BROADCAST_DATA_OP, /*target_or_sender*/id, /*data*/{intermediate_res[start_addr], intermediate_res[start_addr+1]}, /*extra_fields*/intermediate_res[start_addr+2]};
                } else if (inst.op == TRANS_BROADCAST_START_OP) {
                    new_inst = {/*op*/ TRANS_BROADCAST_DATA_OP, /*target_or_sender*/id, /*data*/{intermediate_res[start_addr], intermediate_res[start_addr+1]}, /*extra_fields*/intermediate_res[start_addr+2]};
                }

                gen_reg_16b = start_addr; // Save address of data to send
                bus->push_inst(new_inst);

                if ((inst.op == TRANS_BROADCAST_DATA_OP) && (current_inf_step == ENC_MHSA_QK_T)) { // Since I'm here, move my own data to the correct location in intermediate_res (do I need to do that in the previous op (QVK dense) as well?)
                    intermediate_res[static_cast<int>(inst.extra_fields)+id] = intermediate_res[gen_reg_16b+id];
                }
            } else {
                gen_reg_16b = static_cast<int> (inst.extra_fields); // Save address where to store data
            }

            if ((current_inf_step == ENC_MHSA_MULT_V) && inst.target_or_sender == 0) { // Count the number of matrices that have been started broadcasting
                gen_cnt_10b_2.inc();
            }

            data_len_reg = inst.data[1]; // Save length of data to send/receive
            bytes_sent_cnt.reset();
            break;

        case DENSE_BROADCAST_DATA_OP:
        case TRANS_BROADCAST_DATA_OP:
            bytes_sent_cnt.inc(3); // Increment data that was sent on the bus

            // Data grab
            if (inst.op == TRANS_BROADCAST_DATA_OP) { // Grab the data that corresponds to my data (note this will also move the data to the correct location for the id that is currently sending)
                if (HAS_MY_DATA(bytes_sent_cnt.get_cnt())) { bytes_rec_cnt.inc(); } // Increment data received
                if (bytes_sent_cnt.get_cnt() <= data_len_reg) {
                    intermediate_res[gen_reg_16b+inst.target_or_sender] = ((bytes_sent_cnt.get_cnt() - id) == 1) ? (inst.extra_fields) : (intermediate_res[gen_reg_16b+inst.target_or_sender]);
                    intermediate_res[gen_reg_16b+inst.target_or_sender] = ((bytes_sent_cnt.get_cnt() - id) == 2) ? (inst.data[1]) : (intermediate_res[gen_reg_16b+inst.target_or_sender]);
                    intermediate_res[gen_reg_16b+inst.target_or_sender] = ((bytes_sent_cnt.get_cnt() - id) == 3) ? (inst.data[0]) : (intermediate_res[gen_reg_16b+inst.target_or_sender]);
                } else {
                    intermediate_res[gen_reg_16b+inst.target_or_sender] = ((data_len_reg - id) == 1) ? (inst.data[0]) : (intermediate_res[gen_reg_16b+inst.target_or_sender]);
                    intermediate_res[gen_reg_16b+inst.target_or_sender] = ((data_len_reg - id) == 2) ? (inst.data[1]) : (intermediate_res[gen_reg_16b+inst.target_or_sender]);
                    intermediate_res[gen_reg_16b+inst.target_or_sender] = ((data_len_reg - id) == 3) ? (inst.extra_fields) : (intermediate_res[gen_reg_16b+inst.target_or_sender]);
                }
            } else if (inst.op == DENSE_BROADCAST_DATA_OP) { // Always move data (even my own) to the correct location to perform operations later
                bytes_rec_cnt.inc(3); // Increment data that was sent on the bus
                intermediate_res[gen_reg_16b+bytes_sent_cnt.get_cnt()-3] = inst.data[0];
                intermediate_res[gen_reg_16b+bytes_sent_cnt.get_cnt()-2] = inst.data[1];
                intermediate_res[gen_reg_16b+bytes_sent_cnt.get_cnt()-1] = inst.extra_fields;
            }

            // Prep new instruction to send
            if (bytes_sent_cnt.get_cnt() < data_len_reg && inst.target_or_sender == id) { // If last time was me and I have more data to send
                struct instruction new_inst = {/*op*/ inst.op, /*target_or_sender*/id, /*data*/{0, 0}, /*extra_fields*/0};
                if ((data_len_reg - bytes_rec_cnt.get_cnt() == 2)) { // Only two bytes left
                    new_inst.data = {intermediate_res[gen_reg_16b+bytes_rec_cnt.get_cnt()], intermediate_res[gen_reg_16b+bytes_rec_cnt.get_cnt()+1]};
                } else if ((data_len_reg - bytes_rec_cnt.get_cnt() == 1)) { // Only one byte left
                    new_inst.data = {intermediate_res[gen_reg_16b+bytes_rec_cnt.get_cnt()], 0};
                } else {
                    new_inst.data = {intermediate_res[gen_reg_16b+bytes_rec_cnt.get_cnt()], intermediate_res[gen_reg_16b+bytes_rec_cnt.get_cnt()+1]};
                    new_inst.extra_fields = intermediate_res[gen_reg_16b+bytes_rec_cnt.get_cnt()+2];
                }
                bus->push_inst(new_inst);
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
        switch (current_inf_step) {
        case CLASS_TOKEN_CONCAT:
            intermediate_res[PATCH_LENGTH_NUM_SAMPLES+NUM_PATCHES] = params[param_addr_map[SINGLE_PARAMS].addr+CLASS_EMB_OFF]; // Move classification token from parameters memory to intermediate storage
            current_inf_step = POS_EMB;
            bytes_rec_cnt.reset();
            break;

        case POS_EMB:
            if (bytes_rec_cnt.get_cnt() < NUM_PATCHES+1) {
                intermediate_res[bytes_rec_cnt.get_cnt()] = ADD(PATCH_LENGTH_NUM_SAMPLES+bytes_rec_cnt.get_cnt(), param_addr_map[POS_EMB_PARAMS].addr+bytes_rec_cnt.get_cnt());
                bytes_rec_cnt.inc();
                if (bytes_rec_cnt.get_cnt() == NUM_PATCHES) { // Done with positional embedding
                    bytes_rec_cnt.reset();
                    current_inf_step = ENC_LAYERNORM_1ST_HALF;
                    if (id == 0) { cout << "CiM: Finished positional embedding" << endl; }
                    is_idle = true; // Indicate to master controller that positional embedding computation is done
                } else { is_idle = false; }
            }
            break;

        case ENC_LAYERNORM_1ST_HALF:
            if (bytes_rec_cnt.get_cnt() == EMB_DEPTH) { // No more data to receive, start LayerNorm
                if (compute_in_progress == false) {
                    gen_reg_16b = 1; // Just a signal to avoid coming here every time FSM runs
                    LAYERNORM_1ST_HALF(0);
                    bytes_rec_cnt.reset();
                    is_idle = false;
                }
            } else if (compute_in_progress == false && gen_reg_16b == 1) { // Done with LayerNorm
                if (id == 0) { cout << "CiM: Finished LayerNorm (1st half)" << endl; }
                is_idle = true;
            } else { // Still collecting data
                is_idle = false;
            }

            if (inst.op == PISTOL_START_OP) {
                current_inf_step = ENC_LAYERNORM_2ND_HALF;
                gen_reg_16b = 0;
            }
            break;

        case ENC_LAYERNORM_2ND_HALF:
            if (bytes_rec_cnt.get_cnt() == NUM_PATCHES+1) { // No more data to receive, start LayerNorm
                if (compute_in_progress == false) {
                    float gamma = params[param_addr_map[SINGLE_PARAMS].addr+ENC_LAYERNORM1_GAMMA_OFF];
                    float beta = params[param_addr_map[SINGLE_PARAMS].addr+ENC_LAYERNORM1_BETA_OFF];
                    gen_reg_16b = 1; // Just a signal to avoid coming here every time FSM runs
                    is_idle = false;
                    LAYERNORM_2ND_HALF(0, gamma, beta);
                    bytes_rec_cnt.reset();
                }
            } else if (compute_in_progress == false && gen_reg_16b == 1) { // Done
                if (id == 0) { cout << "CiM: Finished LayerNorm (2nd half)" << endl; }
                is_idle = true;
            } else { // Still collecting data
                is_idle = false;
            }

            if (inst.op == PISTOL_START_OP) {
                current_inf_step = POST_LAYERNORM_TRANSPOSE_STEP;
                gen_reg_16b = 0;
            }
            break;

        case POST_LAYERNORM_TRANSPOSE_STEP:
            is_idle = bytes_rec_cnt.get_cnt() == EMB_DEPTH;

            if (inst.op == PISTOL_START_OP) {
                current_inf_step = ENC_MHSA_DENSE;
                bytes_rec_cnt.reset();
                gen_cnt_10b_2.reset();
            }
            break;

        case ENC_MHSA_DENSE:
            if (bytes_rec_cnt.get_cnt() >= EMB_DEPTH) { // No more data to receive for this broadcast, start MACs
                if (compute_in_progress == false){
                    // Note: In ASIC, these would be sequential MACs, but here we are doing them in parallel
                    float result = MAC(NUM_PATCHES+1+EMB_DEPTH, 128, EMB_DEPTH, MODEL_PARAM);
                    intermediate_res[189+inst.target_or_sender] = result + params[param_addr_map[SINGLE_PARAMS].addr+ENC_Q_DENSE_BIAS_0FF]; // Q
                    result = MAC(NUM_PATCHES+1+EMB_DEPTH, 192, EMB_DEPTH, MODEL_PARAM);
                    intermediate_res[250+inst.target_or_sender] = result + params[param_addr_map[SINGLE_PARAMS].addr+ENC_K_DENSE_BIAS_0FF]; // K
                    result = MAC(NUM_PATCHES+1+EMB_DEPTH, 256, EMB_DEPTH, MODEL_PARAM);
                    intermediate_res[NUM_PATCHES+1+inst.target_or_sender] = result + params[param_addr_map[SINGLE_PARAMS].addr+ENC_V_DENSE_BIAS_0FF]; // V
                    gen_reg_16b = 1; // Just a signal to avoid coming here every time FSM runs
                    bytes_rec_cnt.reset();
                }
            } else if (compute_in_progress == false && gen_reg_16b == 1){ // Done
                gen_cnt_10b_2.inc();
                if (id == 0 && gen_cnt_10b_2.get_cnt() == EMB_DEPTH) { cout << "CiM: Finished encoder's MHSA Q/K/V Dense" << endl; }
                is_idle = true;
            } else { // Still collecting data
                is_idle = false;
            }

            if (inst.op == PISTOL_START_OP) {
                current_inf_step = ENC_MHSA_Q_TRANSPOSE_STEP;
                gen_reg_16b = 0;
                gen_cnt_10b_2.reset();
            }
            break;

        case ENC_MHSA_Q_TRANSPOSE_STEP:
        case ENC_MHSA_K_TRANSPOSE_STEP:
        case ENC_POST_MHSA_TRANSPOSE_STEP:
            is_idle = bytes_rec_cnt.get_cnt() >= EMB_DEPTH;

            if (inst.op == PISTOL_START_OP) {
                bytes_rec_cnt.reset();
                gen_cnt_10b.reset();
                gen_cnt_10b_2.reset();
                current_inf_step = static_cast<INFERENCE_STEP> (static_cast<int> (current_inf_step) + 1);
            }
            break;

        case ENC_MHSA_QK_T:
            if (bytes_rec_cnt.get_cnt() >= NUM_HEADS) { // No more data to receive, start MACs
                if (compute_in_progress == false){
                    //TODO: Double-check addressing here
                    uint16_t MAC_in1_addr = EMB_DEPTH+NUM_PATCHES+1 + (gen_cnt_10b_2.get_cnt()-1)*NUM_HEADS;
                    uint16_t MAC_in2_addr = 2*(EMB_DEPTH+NUM_PATCHES+1); // Temp storage location of broadcast QK_T clip
                    uint16_t MAC_storage_addr = 2*EMB_DEPTH+3*(NUM_PATCHES+1) + (gen_cnt_10b_2.get_cnt()-1)*(NUM_PATCHES+1) + inst.target_or_sender; // Storage location of MAC result

                    float result = MAC(MAC_in1_addr, MAC_in2_addr, NUM_HEADS, INTERMEDIATE_RES);
                    intermediate_res[MAC_storage_addr] = result;

                    result = DIV(MAC_storage_addr, param_addr_map[SINGLE_PARAMS].addr+ENC_SQRT_NUM_HEADS_OFF); // /sqrt(NUM_HEADS)
                    intermediate_res[MAC_storage_addr] = result;
                    gen_reg_16b = 1; // Just a signal to avoid coming here every time FSM runs
                    bytes_rec_cnt.reset();
                }
            } else if (compute_in_progress == false && gen_reg_16b == 1) { // Done with this MAC
                is_idle = true;
                // Increment counters
                gen_cnt_10b.inc();
                if (gen_cnt_10b.get_cnt() == (NUM_PATCHES+1)) { // All MACs for one matrix are done
                    gen_cnt_10b.reset();
                    gen_cnt_10b_2.inc();
                }

                if (gen_cnt_10b_2.get_cnt() == NUM_HEADS) { // Done with all matrices in the Z-stack
                    if (id == 0) { cout << "CiM: Finished encoder's MHSA QK_T" << endl; }
                    gen_cnt_10b_2.reset();
                    is_idle = false; // If all done, move to next step but don't idle so master doesn't skip a step
                    current_inf_step = ENC_MHSA_SOFTMAX;
                    gen_reg_16b = 0;
                }
            } else { // Still collecting data
                is_idle = false;
                gen_reg_16b = 0;
            }
            break;

        case ENC_MHSA_SOFTMAX:
            if (gen_cnt_10b_2.get_cnt() < NUM_HEADS) {
                if (compute_in_progress == false && gen_reg_16b == 1) { // Done with this matrix in the Z-stack
                    gen_cnt_10b_2.inc();
                    gen_reg_16b = 0;
                } else if (compute_in_progress == false) {
                    uint16_t MAC_storage_addr = 2*EMB_DEPTH + 3*(NUM_PATCHES + 1) + gen_cnt_10b_2.get_cnt()*(NUM_PATCHES+1); // Storage location of MAC result
                    SOFTMAX(/*input addr*/ MAC_storage_addr, /*len*/ NUM_PATCHES+1);
                    gen_reg_16b = 1; // Just a signal to avoid coming here every time FSM runs
                }
            } else {
                if (id == 0) { cout << "CiM: Finished encoder's MHSA softmax" << endl; }
                bytes_rec_cnt.reset();
                is_idle = true;
            }

            if (inst.op == PISTOL_START_OP) {
                current_inf_step = ENC_MHSA_MULT_V;
                gen_cnt_10b.reset();
                gen_cnt_10b_2.reset();
            }
            break;

        case ENC_MHSA_MULT_V:
            if (bytes_rec_cnt.get_cnt() >= (NUM_PATCHES+1)) { // No more data to receive for this given row
                if (IS_MY_MATRIX(gen_cnt_10b_2.get_cnt()-1)) { // Only if matrix being broadcast corresponds to mine (-1 because gen_cnt_10b_2 is incremented when the matrix starts broadcast)
                    if (compute_in_progress == false) {
                        uint16_t MAC_in1_addr = 3*(NUM_PATCHES+1)+2*EMB_DEPTH + (gen_cnt_10b_2.get_cnt()-1)*(NUM_PATCHES+1);
                        float result = MAC(MAC_in1_addr, /*in2 addr*/ NUM_PATCHES+1+EMB_DEPTH, NUM_PATCHES+1, INTERMEDIATE_RES);
                        intermediate_res[2*EMB_DEPTH + NUM_PATCHES+1 + inst.target_or_sender] = result;
                        gen_reg_16b = 1; // Just a signal to avoid coming here every time FSM runs
                        bytes_rec_cnt.reset();
                    }
                } else {
                    is_idle = true;
                }
            } else if (compute_in_progress == false && gen_reg_16b == 1) { // Done with this row in the matrix
                is_idle = true;
            } else { // Still collecting data
                is_idle = false;
                gen_reg_16b = 0;
            }

            if (inst.op == PISTOL_START_OP) {
                current_inf_step = ENC_POST_MHSA_TRANSPOSE_STEP;
                if (id == 0) { cout << "CiM: Finished encoder's V matmul" << endl; }
                bytes_rec_cnt.reset();
                gen_cnt_10b_2.reset();
            }
            break;

        case ENC_POST_MHSA_DENSE_STEP:
            if (bytes_rec_cnt.get_cnt() >= EMB_DEPTH) { // No more data to receive for this broadcast, start MACs
                if (compute_in_progress == false){
                    float result = MAC(NUM_PATCHES+1, 320, EMB_DEPTH, MODEL_PARAM);
                    intermediate_res[NUM_PATCHES+1 + EMB_DEPTH + inst.target_or_sender] = result + params[param_addr_map[SINGLE_PARAMS].addr+ENC_COMB_HEAD_BIAS_OFF];
                    gen_reg_16b = 1; // Just a signal to avoid coming here every time FSM runs
                    bytes_rec_cnt.reset();
                }
            } else if (compute_in_progress == false && gen_reg_16b == 1) { // Done with this MAC
                gen_cnt_10b_2.inc();
                if (id == 0 && gen_cnt_10b_2.get_cnt() == (NUM_PATCHES+1)) { cout << "CiM: Finished encoder's post-MHSA Dense" << endl; }
                is_idle = true;
            } else { // Still collecting data
                is_idle = false;
            }

            if (inst.op == PISTOL_START_OP) {
                current_inf_step = INVALID_INF_STEP; // TODO: Change this step to the next step in the inference pipeline
                gen_reg_16b = 0;
                gen_cnt_10b_2.reset();
            }

            break;

        case INVALID_INF_STEP:
        default:
            is_idle = true;
            break;
        }
        break;

    case INVALID_CIM:
    default:
        cout << "CiM " << id << " controller in an invalid state (" << state << ")! Exiting.\n" << endl;
        exit(-1);
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
    compute_done = true;
    return result;
}

float CiM::ADD(uint16_t input_addr, uint16_t params_addr) {
    /* Return sum of two inputs. First input is in intermediate storage location, and second is in the params storage. */
    return intermediate_res[input_addr] + params[params_addr];
}

void CiM::LAYERNORM_1ST_HALF(uint16_t input_addr) {
    /* 1st half of Layer normalization of input. Input is in intermediate storage location. Note: All LayerNorms are done over a row of EMB_DEPTH length. */
    compute_in_progress = true; // Using this compute_in_progress signal as it will be used in the CiM (in ASIC, this compute is multi-cycle, so leaving this here for visibility)

    float result = 0.0f;
    float result2 = 0.0f;

    // Summation along feature axis
    for (uint16_t i = 0; i < EMB_DEPTH; ++i) {
        result += intermediate_res[input_addr+i];
    }

    result /= EMB_DEPTH; // Mean

    // Subtract and square mean
    for (uint16_t i = 0; i < EMB_DEPTH; ++i) {
        intermediate_res[input_addr+i] -= result; // Subtract
        intermediate_res[input_addr+i] *= intermediate_res[input_addr+i]; // Square
        result2 += intermediate_res[input_addr+i]; // Sum
    }

    result2 /= EMB_DEPTH; // Mean
    result2 += LAYERNORM_EPSILON; // Add epsilon
    result2 = sqrt(result2); // <-- Standard deviation

    // Partial normalization (excludes gamma and beta, which are applied in LAYERNORM_FINAL_NORM() since they need to be applied column-wise)
    for (uint16_t i = 0; i < EMB_DEPTH; ++i) {
        intermediate_res[input_addr+i] = intermediate_res[input_addr+i] / result2;
    }

    compute_in_progress = false;
}

void CiM::LAYERNORM_2ND_HALF(uint16_t input_addr, float gamma, float beta) {
    /* 2nd half of Layer normalization of input. This applies gamma and beta on each column. */

    compute_in_progress = true;

    // Normalize
    for (uint16_t i = 0; i < EMB_DEPTH; ++i) {
        intermediate_res[input_addr+i] = gamma * intermediate_res[input_addr+i] + beta;
    }

    compute_in_progress = false;
}

void CiM::SOFTMAX(uint16_t input_addr, uint16_t len) {
    /* Softmax of input (performed in-place). Input is in intermediate storage location. Note: All Softmax are done over a row of len length. */

    float exp_sum = 0.0f;
    compute_in_progress = true;

    // Exponentiate all elements and sum
    for (uint16_t i = 0; i < len; ++i) {
        intermediate_res[input_addr+i] = exp(intermediate_res[input_addr+i]);
        exp_sum += intermediate_res[input_addr+i];
    }

    // Normalize
    for (uint16_t i = 0; i < len; ++i) {
        intermediate_res[input_addr+i] = intermediate_res[input_addr+i] / exp_sum;
    }

    compute_done = true;
    compute_in_progress = false;
}

bool CiM::get_is_idle() {
    return is_idle;
}