/* TODO:
    - Adjust temporary storage locations for encoder's MHSA intermediate storage as we can overwrite all but the first element of encoder input (which is used in the residual connection) since we will only send the first row of encoder output to MLP head anyways
*/

#include <CiM.hpp>

/*----- NAMESPACE -----*/
using namespace std;

/*----- DECLARATION -----*/
CiM::CiM(const int16_t cim_id) : id(cim_id), gen_cnt_7b(7), gen_cnt_7b_2(7), word_rec_cnt(7), word_snt_cnt(7) {
    reset();
    load_previous_softmax(); // Load in dummy softmax data of previous epochs for verification
}

void CiM::load_previous_softmax() {
    if (id == 0) {
        for (int i = 0; i < (NUM_SAMPLES_OUT_AVG-1); i++) { // Softmax for previous dummy epochs
            std::string filename = std::string(DATA_BASE_DIR)+"/dummy_softmax_" + std::to_string(i) + ".csv";
            rapidcsv::Document csv(filename, rapidcsv::LabelParams(-1, -1));
            std::vector<float> dummy_softmax = csv.GetRow<float>(0);
            for (int j = 0; j < NUM_SLEEP_STAGES; j++) { intermediate_res[mem_map.at(PREV_SOFTMAX_OUTPUT_MEM) + i*NUM_SLEEP_STAGES + j] = (dummy_softmax[j] / NUM_SAMPLES_OUT_AVG); }
        }
    }
}

int CiM::reset(){
    fill(begin(intermediate_res), end(intermediate_res), 0); // Reset local intermediate_res
    is_ready = true;
    compute_in_progress = false;
    computation_result = 0;
    gen_reg_2b = 0;
    tx_addr_reg = 0;
    rx_addr_reg = 0;
    sender_id = 0;
    data_len_reg = 0;
    current_inf_step = CLASS_TOKEN_CONCAT_STEP;
    gen_cnt_7b.reset();
    gen_cnt_7b_2.reset();
    word_rec_cnt.reset();
    word_snt_cnt.reset();
    _neg_exp_cnt = 0;
    _total_exp_cnt = 0;
    _max_exp_input_arg = large_fp_t(0);
    _min_exp_input_arg = large_fp_t(0);
    _compute_process_cnt = 0;
    _num_compute_done = 0;
    load_previous_softmax(); // Load in dummy softmax data of previous epochs for verification
    update_state(IDLE_CIM);
    return 0;
}

int CiM::run(struct ext_signals* ext_sigs, Bus* bus){
    /* Run the CiM FSM */
    if (ext_sigs->master_nrst == false) { reset(); } // Reset if master is reset

    // Check for data overflows
    overflow_check();

    // Update compute process counter
    update_compute_process_cnt();

    // Read bus if new instruction
    struct instruction inst = bus->get_inst();
    switch (inst.op){
    case PATCH_LOAD_BROADCAST_START_OP:
        word_rec_cnt.reset();
        gen_cnt_7b_2.reset();
        break;

    case PATCH_LOAD_BROADCAST_OP:
        intermediate_res[word_rec_cnt.get_cnt()] = inst.data[0];
        word_rec_cnt.inc();
        break;

    case PARAM_STREAM_START_OP:
        rx_addr_reg = inst.data[0]; // Address
        word_rec_cnt.reset();
        break;

    case PARAM_STREAM_OP:
        if (inst.target_or_sender == id) {
            // Can only write to memory one parameter at a time
            if (gen_reg_2b == 0) {
                params[word_rec_cnt.get_cnt() + rx_addr_reg] = inst.data[0];
                gen_reg_2b = 1;
                is_ready = false;
            } else if (gen_reg_2b == 1) {
                params[word_rec_cnt.get_cnt() + rx_addr_reg + 1] = inst.data[1];
                gen_reg_2b = 2;
            } else if (gen_reg_2b == 2) {
                params[word_rec_cnt.get_cnt() + rx_addr_reg + 2] = inst.data[2]; // Note: If the length is less than 3, we will be loading junk. That's OK since it will get overwritten
                gen_reg_2b = 0;
                word_rec_cnt.inc(3);
                is_ready = true;
            }
        }
        break;

    case DENSE_BROADCAST_START_OP:
    case TRANS_BROADCAST_START_OP:
        rx_addr_reg = static_cast<uint16_t> (inst.data[2]); // Save address where to store data
        data_len_reg = inst.data[1]; // Save length of data to send/receive
        sender_id = inst.target_or_sender; // Save sender's id

        if (inst.target_or_sender == id) { // Master controller tells me to start broadcasting some data
            struct instruction new_inst;
            tx_addr_reg = static_cast<uint16_t> (inst.data[0]);
            OP op = (inst.op == DENSE_BROADCAST_START_OP) ? DENSE_BROADCAST_DATA_OP : TRANS_BROADCAST_DATA_OP;
            if (data_len_reg == 1) { new_inst = {op, id, /*data*/{0, 0, intermediate_res[tx_addr_reg]}}; } // Special case to only send one data and have the correct alignment
            else { new_inst = {op, id, /*data*/{intermediate_res[tx_addr_reg], intermediate_res[tx_addr_reg+1], intermediate_res[tx_addr_reg+2]}}; }
            bus->push_inst(new_inst);
        }

        if (current_inf_step == ENC_MHSA_MULT_V_STEP) {
            if (inst.target_or_sender == 0) { gen_cnt_7b_2.inc(); } // Counts matrix
            word_rec_cnt.reset(); // We also want to reset this counter here because the CiM that don't have to multiply anything for this matrix could overflow the counter
        }
        word_snt_cnt.reset();
        break;

    case DENSE_BROADCAST_DATA_OP:
    case TRANS_BROADCAST_DATA_OP:
        // Data grab
        if (inst.op == TRANS_BROADCAST_DATA_OP) { // Grab the data that corresponds to my data (note this will also move the data to the correct location for the id that is currently sending)
            if (HAS_MY_DATA(word_snt_cnt.get_cnt()) || (current_inf_step == MLP_HEAD_DENSE_2_STEP)) { // MLP_HEAD_DENSE_2_STEP is special. Each of CiM's #0-#31 send a single element and every CiM's must grab it, so there is no check to do
                word_rec_cnt.inc();
            }
            if (HAS_MY_DATA(word_snt_cnt.get_cnt())) {
                if ((word_snt_cnt.get_cnt()+3) <= data_len_reg) { // More than 3 word left to receive
                    intermediate_res[rx_addr_reg+inst.target_or_sender] = inst.data[id-word_snt_cnt.get_cnt()];
                } else if (word_snt_cnt.get_cnt() == id) {
                    intermediate_res[rx_addr_reg+inst.target_or_sender] = inst.data[2]; // Only one word left (only possible because some broadcasts are single-word)
                }
                // We'll never get into a situation where we have 2 words left, so no need to check for it.
            }
        } else if (inst.op == DENSE_BROADCAST_DATA_OP) { // Always move data (even my own) to the correct location to perform operations later
            word_rec_cnt.inc(3); // Increment data received
            int num_words_left = data_len_reg - word_snt_cnt.get_cnt();
            if (num_words_left >= 3) {
                intermediate_res[rx_addr_reg+word_snt_cnt.get_cnt()] = inst.data[0];
                intermediate_res[rx_addr_reg+word_snt_cnt.get_cnt()+1] = inst.data[1];
                intermediate_res[rx_addr_reg+word_snt_cnt.get_cnt()+2] = inst.data[2];
            } else if (num_words_left == 2) {
                intermediate_res[rx_addr_reg+word_snt_cnt.get_cnt()] = inst.data[1];
                intermediate_res[rx_addr_reg+word_snt_cnt.get_cnt()+1] = inst.data[2];
            } else if (num_words_left == 1) {
                intermediate_res[rx_addr_reg+word_snt_cnt.get_cnt()] = inst.data[2];
            }
        }

        // Prep new instruction to send
        if (inst.target_or_sender == id) {
            if ((word_snt_cnt.get_cnt()+3) < data_len_reg) { // If last time was me and I have more data to send
                struct instruction new_inst = {inst.op, id, {0, 0, 0}};

                if (data_len_reg - word_snt_cnt.get_cnt() == 5) { // Only two word left
                    new_inst.data = {0, intermediate_res[tx_addr_reg+(word_snt_cnt.get_cnt()+3)], intermediate_res[tx_addr_reg+word_snt_cnt.get_cnt()+4]};
                } else if (data_len_reg - word_snt_cnt.get_cnt() == 4) { // Only one word left
                    new_inst.data = {0, 0, intermediate_res[tx_addr_reg+(word_snt_cnt.get_cnt()+3)]};
                } else {
                    new_inst.data = {intermediate_res[tx_addr_reg+word_snt_cnt.get_cnt()+3], intermediate_res[tx_addr_reg+word_snt_cnt.get_cnt()+4], intermediate_res[tx_addr_reg+word_snt_cnt.get_cnt()+5]};
                }
                bus->push_inst(new_inst);
            } else {
                struct instruction inst = {NOP, id, {0, 0, 0}};
                bus->push_inst(inst);
            }
        }

        word_snt_cnt.inc(3); // Increment data that was sent on the bus
        break;

    case PISTOL_START_OP:
        word_rec_cnt.reset();
        break;

    case INFERENCE_RESULT_OP:
    case NOP:
    default:
        break;
    }

    // Run FSM
    switch (cim_state){
    case IDLE_CIM:
        if (inst.op == PATCH_LOAD_BROADCAST_OP) {
            update_state(PATCH_LOAD_CIM);
        }
        break;

    case RESET_CIM:
        reset();
        break;

    case PATCH_LOAD_CIM:
        if ((compute_in_progress == false) && (word_rec_cnt.get_cnt() == PATCH_LEN) && (gen_reg_2b == 0)) { // Received my complete patch, perform part of Dense layer
            MAC(/* patch starting addr */ 0,/* weight starting addr */ 0, /*len*/ PATCH_LEN, /* bias addr */ param_addr_map[SINGLE_PARAMS].addr+PATCH_PROJ_BIAS_OFF, MODEL_PARAM, LINEAR_ACTIVATION);
            gen_reg_2b = 1;
            word_rec_cnt.reset();
        } else if ((compute_in_progress == false) && (gen_reg_2b == 1)) { // Done computation
            intermediate_res[gen_cnt_7b_2.get_cnt() + mem_map.at(PATCH_MEM)] = computation_result;
            is_ready = true;
            gen_reg_2b = 0;
            gen_cnt_7b_2.inc(); // Increment number of patches received
            if (gen_cnt_7b_2.get_cnt() == NUM_PATCHES) { // Received all patches, automatically start inference
                is_ready = false;
                update_state(INFERENCE_RUNNING_CIM);
                gen_cnt_7b_2.reset();
                verify_computation(PATCH_PROJECTION_VERIF, id, intermediate_res, mem_map.at(PATCH_MEM));
            }
        }
    break;

    case INFERENCE_RUNNING_CIM:
        switch (current_inf_step) {
        case CLASS_TOKEN_CONCAT_STEP:
            intermediate_res[mem_map.at(CLASS_TOKEN_MEM)] = params[param_addr_map[SINGLE_PARAMS].addr+CLASS_TOKEN_OFF]; // Move classification token from parameters memory to intermediate storage
            current_inf_step = POS_EMB_STEP;
            verify_computation(CLASS_TOKEN_VERIF, id, intermediate_res, mem_map.at(CLASS_TOKEN_MEM));
            break;

        case POS_EMB_STEP:
            if ((compute_in_progress == false) && (gen_cnt_7b.get_cnt() < NUM_PATCHES+1)) { // Start computation
                if (gen_reg_2b == 1) { // Save computation result from previous iteration (except for the first iteration)
                    intermediate_res[gen_cnt_7b.get_cnt() + mem_map.at(POS_EMB_MEM)] = computation_result;
                    gen_cnt_7b.inc();
                }
                ADD(mem_map.at(CLASS_TOKEN_MEM)+gen_cnt_7b.get_cnt(), param_addr_map[POS_EMB_PARAMS].addr+gen_cnt_7b.get_cnt(), MODEL_PARAM);
                gen_reg_2b = 1;
            } else if ((compute_in_progress == false) && (gen_cnt_7b.get_cnt() == NUM_PATCHES+1)) { // Done with positional embedding
                intermediate_res[gen_cnt_7b.get_cnt() + mem_map.at(POS_EMB_MEM)] = computation_result; // Save the last result
                current_inf_step = ENC_LAYERNORM_1_1ST_HALF_STEP;
                if (id == 0) { cout << "CiM: Finished positional embedding" << endl; }
                gen_cnt_7b.reset();
                verify_computation(POS_EMB_VERIF, id, intermediate_res, 0);
                is_ready = true;
                if (id == 60) {
                    cout << "CiM: Ready for inference" << endl;
                }
            }
            break;

        case ENC_LAYERNORM_1_1ST_HALF_STEP:
        case ENC_LAYERNORM_2_1ST_HALF_STEP:
        case ENC_LAYERNORM_3_1ST_HALF_STEP:
            if (word_rec_cnt.get_cnt() == EMB_DEPTH && compute_in_progress == false) { // No more data to receive, start LayerNorm
                if (current_inf_step == ENC_LAYERNORM_1_1ST_HALF_STEP) { LAYERNORM_1ST_HALF(mem_map.at(ENC_LN1_1ST_HALF_MEM)); }
                else if (current_inf_step == ENC_LAYERNORM_2_1ST_HALF_STEP) { LAYERNORM_1ST_HALF(mem_map.at(ENC_LN2_1ST_HALF_MEM)); }
                else if (current_inf_step == ENC_LAYERNORM_3_1ST_HALF_STEP) { LAYERNORM_1ST_HALF(mem_map.at(MLP_HEAD_LN_1ST_HALF_MEM)); }
                word_rec_cnt.reset();
            }

            if (inst.op == PISTOL_START_OP) {
                if (id == 0) { cout << "CiM: Finished LayerNorm (1st half)" << endl; }
                current_inf_step = static_cast<INFERENCE_STEP> (static_cast<int> (current_inf_step) + 1);
            }
            break;

        case ENC_LAYERNORM_1_2ND_HALF_STEP:
        case ENC_LAYERNORM_2_2ND_HALF_STEP:
        case ENC_LAYERNORM_3_2ND_HALF_STEP:
            if (((word_rec_cnt.get_cnt() == 1) && (current_inf_step == ENC_LAYERNORM_3_2ND_HALF_STEP)) || (word_rec_cnt.get_cnt() == NUM_PATCHES+1)) { // No more data to receive, start LayerNorm
                if (compute_in_progress == false) {
                    if (current_inf_step == ENC_LAYERNORM_1_2ND_HALF_STEP) {
                        LAYERNORM_2ND_HALF(mem_map.at(ENC_LN1_2ND_HALF_MEM), param_addr_map[SINGLE_PARAMS].addr+ENC_LAYERNORM_1_GAMMA_OFF, param_addr_map[SINGLE_PARAMS].addr+ENC_LAYERNORM_1_BETA_OFF);
                    } else if (current_inf_step == ENC_LAYERNORM_2_2ND_HALF_STEP) {
                        LAYERNORM_2ND_HALF(mem_map.at(ENC_LN2_2ND_HALF_MEM), param_addr_map[SINGLE_PARAMS].addr+ENC_LAYERNORM_2_GAMMA_OFF, param_addr_map[SINGLE_PARAMS].addr+ENC_LAYERNORM_2_BETA_OFF);
                    } else if (current_inf_step == ENC_LAYERNORM_3_2ND_HALF_STEP) {
                        LAYERNORM_2ND_HALF(mem_map.at(MLP_HEAD_LN_2ND_HALF_MEM), param_addr_map[SINGLE_PARAMS].addr+ENC_LAYERNORM_3_GAMMA_OFF, param_addr_map[SINGLE_PARAMS].addr+ENC_LAYERNORM_3_BETA_OFF);
                    }
                    word_rec_cnt.reset();
                }
            }

            if (inst.op == PISTOL_START_OP) {
                if (current_inf_step == ENC_LAYERNORM_1_2ND_HALF_STEP) {
                    current_inf_step = POST_LAYERNORM_TRANSPOSE_STEP;
                    verify_computation(ENC_LAYERNORM1_VERIF, id, intermediate_res, mem_map.at(ENC_LN1_2ND_HALF_MEM));
                } else if (current_inf_step == ENC_LAYERNORM_2_2ND_HALF_STEP) {
                    current_inf_step = ENC_PRE_MLP_TRANSPOSE_STEP;
                    verify_computation(ENC_LAYERNORM2_VERIF, id, intermediate_res, mem_map.at(ENC_LN2_2ND_HALF_MEM));
                } else if (current_inf_step == ENC_LAYERNORM_3_2ND_HALF_STEP) {
                    current_inf_step = MLP_HEAD_PRE_DENSE_1_TRANSPOSE_STEP;
                    verify_computation(MLP_HEAD_LAYERNORM_VERIF, id, intermediate_res, mem_map.at(MLP_HEAD_LN_2ND_HALF_MEM));
                }
                gen_reg_2b = 0;
                if (id == 0) { cout << "CiM: Finished LayerNorm (2nd half)" << endl; }
            }
            break;

        case ENC_MHSA_DENSE_STEP:
        case MLP_DENSE_1_STEP:
        case MLP_DENSE_2_AND_SUM_STEP:
        case MLP_HEAD_DENSE_1_STEP:
        case MLP_HEAD_DENSE_2_STEP:
            if (word_rec_cnt.get_cnt() >= EMB_DEPTH) { // No more data to receive for this broadcast, start MACs
                if ((compute_in_progress == false) && (current_inf_step == ENC_MHSA_DENSE_STEP)) {
                    if (gen_reg_2b == 0) {
                        is_ready = false;
                        MAC(mem_map.at(ENC_QVK_IN_MEM), param_addr_map[ENC_Q_DENSE_PARAMS].addr, EMB_DEPTH, param_addr_map[SINGLE_PARAMS].addr+ENC_Q_DENSE_BIAS_0FF, MODEL_PARAM, LINEAR_ACTIVATION); // Q
                        gen_reg_2b = 1;
                    } else if (gen_reg_2b == 1) {
                        intermediate_res[mem_map.at(ENC_Q_MEM)+sender_id] = computation_result; // Q
                        MAC(mem_map.at(ENC_QVK_IN_MEM), param_addr_map[ENC_K_DENSE_PARAMS].addr, EMB_DEPTH, param_addr_map[SINGLE_PARAMS].addr+ENC_K_DENSE_BIAS_0FF, MODEL_PARAM, LINEAR_ACTIVATION); // K
                        gen_reg_2b = 2;
                    } else if (gen_reg_2b == 2) {
                        intermediate_res[mem_map.at(ENC_K_MEM)+sender_id] = computation_result; // K
                        MAC(mem_map.at(ENC_QVK_IN_MEM), param_addr_map[ENC_V_DENSE_PARAMS].addr, EMB_DEPTH, param_addr_map[SINGLE_PARAMS].addr+ENC_V_DENSE_BIAS_0FF, MODEL_PARAM, LINEAR_ACTIVATION); // V
                        gen_reg_2b = 3;
                        word_rec_cnt.reset();
                    }
                } else if ((compute_in_progress == false) && (current_inf_step == MLP_DENSE_1_STEP)) { // Only least significant MLP_DIM number of CiMs will perform this computation
                    if (id < MLP_DIM) {
                        MAC(mem_map.at(ENC_MLP_IN_MEM), param_addr_map[ENC_MLP_DENSE_1_OR_MLP_HEAD_DENSE_1_PARAMS].addr, EMB_DEPTH, param_addr_map[SINGLE_PARAMS].addr+ENC_MLP_DENSE_1_MLP_HEAD_DENSE_1_BIAS_OFF, MODEL_PARAM, SWISH_ACTIVATION);
                        gen_reg_2b = 3;
                    }
                    word_rec_cnt.reset();
                } else if ((compute_in_progress == false) && (current_inf_step == MLP_HEAD_DENSE_1_STEP) && (id >= MLP_DIM)) { // Only most significant MLP_DIM number of CiMs will perform this computation
                    MAC(mem_map.at(MLP_HEAD_DENSE_1_IN_MEM), param_addr_map[ENC_MLP_DENSE_1_OR_MLP_HEAD_DENSE_1_PARAMS].addr, EMB_DEPTH, param_addr_map[SINGLE_PARAMS].addr+ENC_MLP_DENSE_1_MLP_HEAD_DENSE_1_BIAS_OFF, MODEL_PARAM, SWISH_ACTIVATION);
                    gen_reg_2b = 3;
                    word_rec_cnt.reset();
                }
            } else if ((word_rec_cnt.get_cnt() >= MLP_DIM) && (current_inf_step == MLP_DENSE_2_AND_SUM_STEP || current_inf_step == MLP_HEAD_DENSE_2_STEP)) { // No more data to receive for this broadcast (for MLP's dense #2), start MACs
                if ((compute_in_progress == false) && (current_inf_step == MLP_DENSE_2_AND_SUM_STEP)) {
                    if (gen_reg_2b == 0) {
                        MAC(mem_map.at(ENC_MLP_DENSE2_IN_MEM), param_addr_map[ENC_MLP_DENSE_2_PARAMS].addr, MLP_DIM, param_addr_map[SINGLE_PARAMS].addr+ENC_MLP_DENSE_2_BIAS_OFF, MODEL_PARAM, LINEAR_ACTIVATION);
                        gen_reg_2b = 1;
                    } else if (gen_reg_2b == 1) {
                        intermediate_res[mem_map.at(ENC_MLP_OUT_MEM)+sender_id] = computation_result; // MLP Dense 2
                        ADD(/*enc input*/ sender_id, mem_map.at(ENC_MLP_OUT_MEM)+sender_id, INTERMEDIATE_RES);  // Sum with encoder's input (next step in inference pipeline, but do it now)
                        gen_reg_2b = 3;
                        word_rec_cnt.reset();
                    }
                } else if ((compute_in_progress == false) && (current_inf_step == MLP_HEAD_DENSE_2_STEP) && (id < NUM_SLEEP_STAGES)) {
                    MAC(mem_map.at(MLP_HEAD_DENSE_2_IN_MEM), param_addr_map[MLP_HEAD_DENSE_2_PARAMS].addr, MLP_DIM, param_addr_map[SINGLE_PARAMS].addr+MLP_HEAD_DENSE_2_BIAS_OFF, MODEL_PARAM, LINEAR_ACTIVATION);
                    gen_reg_2b = 3;
                    word_rec_cnt.reset();
                }
            } else if ((compute_in_progress == false) && (gen_reg_2b == 3)){ // Done
                gen_cnt_7b_2.inc();
                if (current_inf_step == ENC_MHSA_DENSE_STEP) { // Encoder's MHSA Q/K/V Dense
                    if ((id == 0) && (gen_cnt_7b_2.get_cnt() == EMB_DEPTH)) { cout << "CiM: Finished encoder's MHSA Q/K/V Dense" << endl; }
                    intermediate_res[mem_map.at(ENC_V_MEM)+sender_id] = computation_result; // V
                } else if (current_inf_step == MLP_DENSE_1_STEP) { // MLP Dense 1
                    if ((id == 0) && (gen_cnt_7b_2.get_cnt() == MLP_DIM)) { cout << "CiM: Finished MLP Dense 1" << endl; }
                    intermediate_res[mem_map.at(ENC_MLP_DENSE1_MEM)+sender_id] = computation_result;
                } else if (current_inf_step == MLP_DENSE_2_AND_SUM_STEP) { // MLP Dense 2 and sum
                    if ((id == 0) && (gen_cnt_7b_2.get_cnt() == 1)) { cout << "CiM: Finished MLP Dense 2" << endl; }
                    intermediate_res[mem_map.at(ENC_MLP_OUT_MEM)+sender_id] = computation_result;
                } else if (current_inf_step == MLP_HEAD_DENSE_1_STEP) { // MLP head's Dense 1
                    if ((id == 32) && (gen_cnt_7b_2.get_cnt() == 1)) { cout << "CiM: Finished MLP head's Dense 1" << endl; }
                    intermediate_res[mem_map.at(MLP_HEAD_DENSE_1_OUT_MEM)+sender_id] = computation_result;
                } else if (current_inf_step == MLP_HEAD_DENSE_2_STEP) { // MLP Dense 2 (softmax)
                    if ((id == 0) && (gen_cnt_7b_2.get_cnt() == 1)) { cout << "CiM: Finished MLP head's Dense 2" << endl; }
                    intermediate_res[mem_map.at(MLP_HEAD_DENSE_2_OUT_MEM)] = computation_result;
                }
                is_ready = true;
                gen_reg_2b = 0;
            }

            if (inst.op == PISTOL_START_OP) {
                if (current_inf_step == ENC_MHSA_DENSE_STEP) {
                    current_inf_step = ENC_MHSA_Q_TRANSPOSE_STEP;
                    verify_computation(ENC_MHSA_DENSE_Q_VERIF, id, intermediate_res, mem_map.at(ENC_Q_MEM));
                    verify_computation(ENC_MHSA_DENSE_K_VERIF, id, intermediate_res, mem_map.at(ENC_K_MEM));
                    verify_computation(ENC_MHSA_DENSE_V_VERIF, id, intermediate_res, mem_map.at(ENC_V_MEM));
                } else if (current_inf_step == MLP_DENSE_1_STEP) {
                    current_inf_step = ENC_POST_DENSE_1_TRANSPOSE_STEP;
                    if (id < MLP_DIM) { verify_computation(ENC_MLP_DENSE1_VERIF, id, intermediate_res, mem_map.at(ENC_MLP_DENSE1_MEM)); }
                } else if (current_inf_step == MLP_DENSE_2_AND_SUM_STEP) {
                    current_inf_step = ENC_LAYERNORM_3_1ST_HALF_STEP;
                    verify_computation(ENC_OUT_VERIF, id, intermediate_res, mem_map.at(ENC_MLP_OUT_MEM));
                } else if (current_inf_step == MLP_HEAD_DENSE_1_STEP) {
                    current_inf_step = MLP_HEAD_PRE_DENSE_2_TRANSPOSE_STEP;
                    if (id >= MLP_DIM) { verify_computation(MLP_HEAD_DENSE_1_VERIF, id-MLP_DIM, intermediate_res, mem_map.at(MLP_HEAD_DENSE_1_OUT_MEM)); }
                } else if (current_inf_step == MLP_HEAD_DENSE_2_STEP) { current_inf_step = MLP_HEAD_PRE_SOFTMAX_TRANSPOSE_STEP; }
                gen_reg_2b = 0;
                gen_cnt_7b_2.reset();
            }
            break;

        case MLP_HEAD_PRE_SOFTMAX_TRANSPOSE_STEP:
            current_inf_step = (word_rec_cnt.get_cnt() == NUM_SLEEP_STAGES) ? MLP_HEAD_SOFTMAX_STEP : MLP_HEAD_PRE_SOFTMAX_TRANSPOSE_STEP;
            break;

        case ENC_MHSA_PRE_SOFTMAX_TRANSPOSE_STEP:
            is_ready = true;
            if (word_rec_cnt.get_cnt() == NUM_PATCHES+1) {  // Count the number of rows received to avoid setting is_ready at the last one to avoid the controller from skipping a state
                gen_cnt_7b_2.inc();
                word_rec_cnt.reset();
            };
            if (gen_cnt_7b_2.get_cnt() == NUM_HEADS || id == 63) {
                is_ready = false;
                gen_cnt_7b_2.reset();
                current_inf_step = static_cast<INFERENCE_STEP> (static_cast<int> (current_inf_step) + 1);
            }
            break;

        case POST_LAYERNORM_TRANSPOSE_STEP:
        case ENC_MHSA_Q_TRANSPOSE_STEP:
        case ENC_MHSA_K_TRANSPOSE_STEP:
        case ENC_POST_MHSA_TRANSPOSE_STEP:
        case ENC_PRE_MLP_TRANSPOSE_STEP:
        case ENC_POST_DENSE_1_TRANSPOSE_STEP:
        case MLP_HEAD_PRE_DENSE_1_TRANSPOSE_STEP:
        case MLP_HEAD_PRE_DENSE_2_TRANSPOSE_STEP:
            if (inst.op == PISTOL_START_OP) {
                word_snt_cnt.reset();
                gen_cnt_7b.reset();
                gen_cnt_7b_2.reset();
                current_inf_step = static_cast<INFERENCE_STEP> (static_cast<int> (current_inf_step) + 1);
            }
            break;

        case ENC_MHSA_QK_T_STEP: {
            uint16_t MAC_storage_addr = mem_map.at(ENC_QK_T_MEM) + gen_cnt_7b_2.get_cnt()*(NUM_PATCHES+1) + sender_id; // Storage location of MAC result
            if (word_rec_cnt.get_cnt() >= NUM_HEADS) { // No more data to receive, start MACs
                if (compute_in_progress == false){
                    if (gen_reg_2b == 0) {
                        uint16_t K_T_addr = mem_map.at(ENC_K_T_MEM) + gen_cnt_7b_2.get_cnt()*NUM_HEADS;
                        uint16_t Q_addr = mem_map.at(ENC_QK_T_IN_MEM); // Temporary storage location for Q's dense broadcast, so it remains the same for all heads
                        MAC(Q_addr, K_T_addr, NUM_HEADS, /*bias addr unused when NO_ACTIVATION*/ 0, INTERMEDIATE_RES, NO_ACTIVATION);
                        gen_reg_2b = 1;
                    } else if (gen_reg_2b == 1) {
                        intermediate_res[MAC_storage_addr] = computation_result;
                        DIV(MAC_storage_addr, param_addr_map[SINGLE_PARAMS].addr+ENC_SQRT_NUM_HEADS_OFF, MODEL_PARAM); // /sqrt(NUM_HEADS)
                        gen_reg_2b = 2; // Just a signal to avoid coming here every time FSM runs
                        word_rec_cnt.reset();
                    }
                }
            } else if (compute_in_progress == false && gen_reg_2b == 2) { // Done with this MAC
                gen_reg_2b = 0;
                intermediate_res[MAC_storage_addr] = computation_result;
                gen_cnt_7b.inc();
                if (gen_cnt_7b.get_cnt() == (NUM_PATCHES+1)) { // All MACs for one matrix are done
                    gen_cnt_7b.reset();
                    gen_cnt_7b_2.inc();
                }
            }

            if (inst.op == PISTOL_START_OP) {
                if (id == 0) { cout << "CiM: Finished encoder's MHSA QK_T" << endl; }
                if (id < NUM_PATCHES+1) { verify_computation(ENC_MHSA_DENSE_QK_T_VERIF, id, intermediate_res, mem_map.at(ENC_QK_T_MEM)); }
                gen_cnt_7b_2.reset();
                current_inf_step = ENC_MHSA_PRE_SOFTMAX_TRANSPOSE_STEP;
            }
            break;
        }
        case MLP_HEAD_SOFTMAX_STEP:
            if ((compute_in_progress == false) && (gen_reg_2b == 0) && (id < NUM_SLEEP_STAGES)) {
                SOFTMAX(/*input addr*/ mem_map.at(MLP_HEAD_SOFTMAX_IN_MEM), /*len*/ NUM_SLEEP_STAGES);
                gen_reg_2b = 1; // Just a signal to avoid coming here every time FSM runs
            } else if (compute_in_progress == false) {
                if (id == 0) { current_inf_step = POST_SOFTMAX_DIVIDE_STEP; }
                else { current_inf_step = INFERENCE_COMPLETE;}
                is_ready = false;
                if (id == 0) { verify_computation(MLP_HEAD_SOFTMAX_VERIF, id, intermediate_res, mem_map.at(MLP_HEAD_SOFTMAX_IN_MEM)); }
                gen_cnt_7b.reset();
                gen_reg_2b = 0;
                if (id == 0) { cout << "CiM: Finished MLP head's Softmax" << endl; }
            }
            break;

        case POST_SOFTMAX_DIVIDE_STEP:
            if ((compute_in_progress == false) && (gen_reg_2b == 0) && (gen_cnt_7b.get_cnt() < NUM_SLEEP_STAGES)) { // Divide all elements by NUM_SAMPLES_OUT_AVG
                DIV(mem_map.at(MLP_HEAD_SOFTMAX_IN_MEM)+gen_cnt_7b.get_cnt(), NUM_SAMPLES_OUT_AVG, IMMEDIATE_VAL);
                gen_reg_2b = 1; // Just a signal to avoid coming here every time FSM runs
            } else if ((gen_cnt_7b.get_cnt() == NUM_SLEEP_STAGES) && (compute_in_progress == false)) {
                cout << "CiM: Finished MLP head's Softmax averaging divide" << endl;
                verify_computation(MLP_HEAD_SOFTMAX_DIV_VERIF, id, intermediate_res, mem_map.at(MLP_HEAD_SOFTMAX_IN_MEM));
                current_inf_step = POST_SOFTMAX_AVERAGING_STEP;
                gen_cnt_7b.reset();
                gen_cnt_7b_2.reset();
            } else if ((compute_in_progress == false) && (gen_reg_2b == 1)) { // Done one division
                intermediate_res[mem_map.at(MLP_HEAD_SOFTMAX_IN_MEM)+gen_cnt_7b.get_cnt()] = computation_result;
                intermediate_res[mem_map.at(SOFTMAX_AVG_SUM_MEM)+gen_cnt_7b.get_cnt()] = computation_result; // Copy there for averaging sum step
                gen_cnt_7b.inc();
                gen_reg_2b = 0;
            }
            break;

        case POST_SOFTMAX_AVERAGING_STEP:
            /* Note:
                - gen_cnt_7b holds the current sleep stage within an epoch's softmax
                - gen_cnt_7b_2 holds the epoch
            */
            if ((compute_in_progress == false) && (gen_cnt_7b_2.get_cnt() < (NUM_SAMPLES_OUT_AVG-1)) && (gen_cnt_7b.get_cnt() <= NUM_SLEEP_STAGES)) {
                uint16_t addr_prev_softmax = mem_map.at(PREV_SOFTMAX_OUTPUT_MEM) + gen_cnt_7b.get_cnt() + gen_cnt_7b_2.get_cnt()*NUM_SLEEP_STAGES;
                uint16_t addr_softmax_divide_sum = mem_map.at(SOFTMAX_AVG_SUM_MEM) + gen_cnt_7b.get_cnt();

                intermediate_res[gen_cnt_7b.get_cnt()] = intermediate_res[addr_prev_softmax]; // Move previous softmax result to intermediate storage
                if (gen_reg_2b == 1) { intermediate_res[mem_map.at(SOFTMAX_AVG_SUM_MEM)+gen_cnt_7b.get_cnt()-1] = computation_result; } // Save previous sum result
                ADD(gen_cnt_7b.get_cnt(), addr_softmax_divide_sum, INTERMEDIATE_RES); // Sum with previous result

                gen_reg_2b = 1;
                gen_cnt_7b.inc();
                if (gen_cnt_7b.get_cnt() == NUM_SLEEP_STAGES) {
                    intermediate_res[mem_map.at(SOFTMAX_AVG_SUM_MEM)+gen_cnt_7b.get_cnt()-1] = computation_result;
                    gen_cnt_7b.reset();
                    gen_cnt_7b_2.inc();
                }
            } else if ((compute_in_progress == false) && (gen_reg_2b == 1)) {
                ARGMAX(/*addr*/ mem_map.at(SOFTMAX_AVG_SUM_MEM), /*len*/ NUM_SLEEP_STAGES); // Start a ARGMAX in the background
                gen_reg_2b = 2;
            } else if ((compute_in_progress == false) && (gen_reg_2b == 2)) {
                gen_cnt_7b.reset();
                gen_cnt_7b_2.reset();
                current_inf_step = RETIRE_SOFTMAX_STEP;
                cout << "CiM: Finished averaging softmax with previous epochs." << endl;
                verify_computation(POST_SOFTMAX_AVG_VERIF, id, intermediate_res, mem_map.at(SOFTMAX_AVG_SUM_MEM));
            }
            break;

        case RETIRE_SOFTMAX_STEP:
            if ((gen_cnt_7b.get_cnt() < NUM_SLEEP_STAGES) && (gen_cnt_7b_2.get_cnt() < (NUM_SAMPLES_OUT_AVG-2))) {
                intermediate_res[mem_map.at(PREV_SOFTMAX_OUTPUT_MEM) + gen_cnt_7b.get_cnt() + NUM_SLEEP_STAGES*(gen_cnt_7b_2.get_cnt()+1)] = intermediate_res[mem_map.at(PREV_SOFTMAX_OUTPUT_MEM) + gen_cnt_7b.get_cnt() + NUM_SLEEP_STAGES*gen_cnt_7b_2.get_cnt()];

                if (gen_cnt_7b_2.get_cnt() == 0) {intermediate_res[mem_map.at(PREV_SOFTMAX_OUTPUT_MEM) + gen_cnt_7b.get_cnt()] = intermediate_res[mem_map.at(MLP_HEAD_SOFTMAX_IN_MEM) + gen_cnt_7b.get_cnt()];}

                gen_cnt_7b.inc();
                if (gen_cnt_7b.get_cnt() == NUM_SLEEP_STAGES) {
                    gen_cnt_7b.reset();
                    gen_cnt_7b_2.inc();
                }
            } else {
                current_inf_step = INFERENCE_COMPLETE;
                is_ready = true;
                gen_cnt_7b.reset();
                gen_cnt_7b_2.reset();
                verify_softmax_storage(intermediate_res, mem_map.at(PREV_SOFTMAX_OUTPUT_MEM));
                if (id == 0) {
                    cout << ">----- CiM STATS -----<" << endl;
                    cout << "CiM #0: Inference complete. Inferred sleep stage: " << computation_result << endl;
                    cout << "CiM #0: Number of exponent operations with negative argument = " << _neg_exp_cnt << "/" << _total_exp_cnt << " (" << 100*_neg_exp_cnt/_total_exp_cnt  << "%)" << endl;
                    cout << "CiM #0: Min./Max. inputs to exponential = " << static_cast<float>(_min_exp_input_arg) << " and " << static_cast<float>(_max_exp_input_arg) << endl;
                }
            }
        break;

        case ENC_MHSA_SOFTMAX_STEP:
            if (gen_cnt_7b_2.get_cnt() < NUM_HEADS && (id < NUM_PATCHES+1)) {
                if (compute_in_progress == false && gen_reg_2b == 1) { // Done with this matrix in the Z-stack
                    gen_cnt_7b_2.inc();
                    gen_reg_2b = 0;
                } else if (compute_in_progress == false) {
                    uint16_t MAC_storage_addr = mem_map.at(ENC_PRE_SOFTMAX_MEM) + gen_cnt_7b_2.get_cnt()*(NUM_PATCHES+1); // Storage location of MAC result
                    SOFTMAX(/*input addr*/ MAC_storage_addr, /*len*/ NUM_PATCHES+1);
                    gen_reg_2b = 1; // Just a signal to avoid coming here every time FSM runs
                }
            } else { is_ready = true; }

            if (inst.op == PISTOL_START_OP) {
                if (id == 0) { cout << "CiM: Finished encoder's MHSA softmax" << endl; }
                if (id < NUM_PATCHES+1) { verify_computation(ENC_SOFTMAX_VERIF, id, intermediate_res, mem_map.at(ENC_PRE_SOFTMAX_MEM)); }
                current_inf_step = ENC_MHSA_MULT_V_STEP;
                gen_cnt_7b.reset();
                gen_cnt_7b_2.reset();
            }
            break;

        case ENC_MHSA_MULT_V_STEP:
            if (word_rec_cnt.get_cnt() >= (NUM_PATCHES+1)) { // No more data to receive for this given row
                if (IS_MY_MATRIX(gen_cnt_7b_2.get_cnt()-1) && (compute_in_progress == false)) { // Only if matrix being broadcast corresponds to mine (-1 because gen_cnt_7b_2 is incremented when the matrix starts broadcast)
                    MAC(mem_map.at(ENC_V_MULT_IN_MEM), /*in2 addr*/ mem_map.at(ENC_V_MEM), NUM_PATCHES+1, /*bias addr unused when NO_ACTIVATION*/ 0, INTERMEDIATE_RES, NO_ACTIVATION);
                    gen_reg_2b = 1; // Just a signal to avoid coming here every time FSM runs
                    word_rec_cnt.reset();
                    if (id >= (NUM_CIM - NUM_HEADS) && sender_id == NUM_PATCHES) { is_ready = false; } // Need to explicitly stop master else it would send pistol_start and we would miss the last row of CiM #56-#63
                }
            } else if (compute_in_progress == false && gen_reg_2b == 1) { // Done with this row in the matrix
                intermediate_res[mem_map.at(ENC_V_MULT_MEM) + gen_cnt_7b.get_cnt()] = computation_result;
                gen_cnt_7b.inc();
                if (gen_cnt_7b.get_cnt() == (NUM_PATCHES+1)) { gen_cnt_7b.reset(); } // gen_cnt_7b counts row in the current matrix
                gen_reg_2b = 0;
            }

            if (inst.op == PISTOL_START_OP) {
                if (id >= (NUM_CIM-NUM_HEADS)) { intermediate_res[mem_map.at(ENC_V_MULT_MEM) + gen_cnt_7b.get_cnt()] = computation_result; }
                current_inf_step = ENC_POST_MHSA_TRANSPOSE_STEP;
                is_ready = true;
                gen_reg_2b = 0;
                verify_computation(ENC_MULT_V_VERIF, id, intermediate_res, mem_map.at(ENC_V_MULT_MEM));
                if (id == 0) { cout << "CiM: Finished encoder's V matmul" << endl; }
                word_rec_cnt.reset();
                gen_cnt_7b_2.reset();
            }
            break;

        case ENC_POST_MHSA_DENSE_AND_INPUT_SUM_STEP:
            if (word_rec_cnt.get_cnt() >= EMB_DEPTH) { // No more data to receive for this broadcast, start MACs
                if ((compute_in_progress == false) && (gen_reg_2b == 0)){ // Start computation
                    MAC(mem_map.at(ENC_DENSE_IN_MEM), param_addr_map[ENC_COMB_HEAD_PARAMS].addr, EMB_DEPTH, param_addr_map[SINGLE_PARAMS].addr+ENC_COMB_HEAD_BIAS_OFF, MODEL_PARAM, LINEAR_ACTIVATION);
                    gen_reg_2b = 1; // Just a signal to avoid coming here every time FSM runs
                } else if ((compute_in_progress == false) && (gen_reg_2b == 1)) {
                    intermediate_res[mem_map.at(ENC_MHSA_OUT_MEM) + sender_id] = computation_result;
                    ADD(mem_map.at(ENC_MHSA_OUT_MEM) + sender_id, sender_id, INTERMEDIATE_RES); // Sum with encoder's input as a residual connection
                    gen_reg_2b = 2;
                    word_rec_cnt.reset();
                }
                if (sender_id == NUM_PATCHES-1) { is_ready = false; }
            } else if (compute_in_progress == false && gen_reg_2b == 2) {
                intermediate_res[mem_map.at(ENC_MHSA_OUT_MEM) + sender_id] = computation_result;
                gen_cnt_7b_2.inc();
                gen_reg_2b = 0;
                is_ready = true;
            }

            if (inst.op == PISTOL_START_OP) {
                verify_computation(ENC_RES_SUM_1_VERIF, id, intermediate_res, mem_map.at(ENC_MHSA_OUT_MEM));
                if (id == 0 && gen_cnt_7b_2.get_cnt() == (NUM_PATCHES+1)) { cout << "CiM: Finished encoder's post-MHSA Dense" << endl; }
                current_inf_step = ENC_LAYERNORM_2_1ST_HALF_STEP; // Start another round of LayerNorm
                gen_reg_2b = 0;
                gen_cnt_7b_2.reset();
            }
            break;

        case INFERENCE_COMPLETE:
            if (inst.op == PISTOL_START_OP) {
                if (id == 0) {
                    struct instruction inst = {/*op*/ INFERENCE_RESULT_OP, /*target_or_sender*/ 0, /*data*/ {computation_result, 0, 0}};
                    bus->push_inst(inst);
                }
                word_rec_cnt.reset();
            }
            update_state(IDLE_CIM);
            break;
        case INVALID_INF_STEP:
        default:
            cout << "Received unknown CiM inference step (" << current_inf_step  << ")! Exiting." << endl;
            exit(-1);
            break;
        }
        break;

    case INVALID_CIM:
    default:
        throw invalid_argument("CiM controller in an invalid state!");
        break;
    }
    return 0;
}

void CiM::update_compute_process_cnt() {
    if (compute_in_progress == true) { _compute_process_cnt++; }
    if (_compute_process_cnt == COMPUTE_CNT_THRESHOLD) {
        _compute_process_cnt = 0;
        compute_in_progress = false;
        _num_compute_done++;
    }
}

void CiM::MAC(uint16_t in1_start_addr, uint16_t in2_start_addr, uint16_t len, uint16_t bias_addr, INPUT_TYPE param_type, ACTIVATION activation) {
    /* Dot-product between two vectors with selectable activation function. */
    if (compute_in_progress == true) { throw runtime_error("Computation already in progress when trying to start MAC!"); }
    compute_in_progress = true;

    compute_temp_fp = large_fp_t { 0 }; // Accumulator
    compute_temp_fp_2 = large_fp_t { params[bias_addr] };

    // MAC
    for (uint16_t i = 0; i < len; i++) {
        if (param_type == INTERMEDIATE_RES) { compute_temp_fp += large_fp_t { intermediate_res[in2_start_addr+i] } * large_fp_t { intermediate_res[in1_start_addr+i] }; }
        else if (param_type == MODEL_PARAM) { compute_temp_fp += large_fp_t { params[in2_start_addr+i] } * large_fp_t { intermediate_res[in1_start_addr+i] }; }
        else {
            cout << "Received unknown parameter type " << param_type << endl;
            exit(-1);
        }
    }

    // Activation
    switch (activation) {
    case LINEAR_ACTIVATION:
        compute_temp_fp += compute_temp_fp_2;
        break;
    case SWISH_ACTIVATION:
        if ((-compute_temp_fp - compute_temp_fp_2) < large_fp_t { 0 }) { _neg_exp_cnt++; }
        _total_exp_cnt++;
        if ((-compute_temp_fp - compute_temp_fp_2) < _min_exp_input_arg) { _min_exp_input_arg = -compute_temp_fp - compute_temp_fp_2; }
        if ((-compute_temp_fp - compute_temp_fp_2) > _max_exp_input_arg) { _max_exp_input_arg = -compute_temp_fp - compute_temp_fp_2; }
        compute_temp_fp = (compute_temp_fp + compute_temp_fp_2) * (large_fp_t { 1 } / (large_fp_t { 1 } + exp(-(compute_temp_fp + compute_temp_fp_2))));
        break;
    case NO_ACTIVATION:
    default:
        break;
    }
    computation_result = static_cast<float> (fix_pt_t { compute_temp_fp });
}

void CiM::DIV(uint16_t num_addr, uint16_t in2, INPUT_TYPE in2_type) {
    /* Return division of two inputs. */
    if (compute_in_progress == true) { throw runtime_error("Computation already in progress when trying to start DIV!"); }
    compute_in_progress = true;

    compute_temp_fp = large_fp_t { 0 }; // Accumulator

    if (in2_type == MODEL_PARAM) { compute_temp_fp = large_fp_t { intermediate_res[num_addr] } / large_fp_t { params[in2] }; } // Second input is a model parameter
    else if (in2_type == IMMEDIATE_VAL) { compute_temp_fp = large_fp_t { intermediate_res[num_addr] } / large_fp_t { in2 }; } // Second input is an immediate value
    else if (in2_type == ADC_INPUT) { compute_temp_fp = large_fp_t { intermediate_res[num_addr] / in2 }; } // Second input is a ADC input (16b)
    else {
        cout << "Received unknown parameter type " << in2_type << endl;
        exit(-1);
    }

    computation_result = static_cast<float> (fix_pt_t { compute_temp_fp });
}

void CiM::ADD(uint16_t in1_addr, uint16_t in2_addr, INPUT_TYPE in2_type) {
    /* Return sum of two inputs. */
    if (compute_in_progress == true) { throw runtime_error("Computation already in progress when trying to start ADD!"); }
    compute_in_progress = true;

    // Get data as would be stored in memory
    compute_temp_fp_2 = large_fp_t { intermediate_res[in1_addr] };

    if (in2_type == INTERMEDIATE_RES) { compute_temp_fp = large_fp_t { intermediate_res[in2_addr] }; }
    else if (in2_type == MODEL_PARAM) { compute_temp_fp = large_fp_t { params[in2_addr] }; }
    else {
        cout << "Received unknown parameter type " << in2_type << endl;
        exit(-1);
    }
    compute_temp_fp += compute_temp_fp_2;
    computation_result = static_cast<float> (fix_pt_t { compute_temp_fp });
}

void CiM::LAYERNORM_1ST_HALF(uint16_t input_addr) {
    /* 1st half of Layer normalization of input. Input is in intermediate storage location. Note: All LayerNorms are done over a row of EMB_DEPTH length. */
    if (compute_in_progress == true) { throw runtime_error("Computation already in progress when trying to start LAYERNORM_1ST_HALF!"); }
    compute_in_progress = true; // Using this compute_in_progress signal as it will be used in the CiM (in ASIC, this compute is multi-cycle, so leaving this here for visibility)

    compute_temp_fp = large_fp_t { 0.0f }; // Mean
    compute_temp_fp_2 = large_fp_t { 0.0f }; // Variance
    compute_temp_fp_3 = large_fp_t { 0.0f }; // Variance

    // Mean
    for (uint16_t i = 0; i < EMB_DEPTH; i++) { compute_temp_fp += large_fp_t { intermediate_res[input_addr+i] }; }
    compute_temp_fp /= EMB_DEPTH;
    verify_result(MEAN, static_cast<float> (compute_temp_fp), intermediate_res, input_addr, EMB_DEPTH, id);

    // Variance
    for (uint16_t i = 0; i < EMB_DEPTH; i++) {
        compute_temp_fp_3 = large_fp_t { intermediate_res[input_addr+i] } - compute_temp_fp; // Subtract
        compute_temp_fp_3 *= compute_temp_fp_3; // Square
        compute_temp_fp_2 += compute_temp_fp_3; // Sum
    }

    compute_temp_fp_2 /= EMB_DEPTH;
    verify_result(VARIANCE, static_cast<float> (compute_temp_fp_2), intermediate_res, input_addr, EMB_DEPTH, id);
    compute_temp_fp_2 = sqrt(compute_temp_fp_2); // Standard deviation
    compute_temp_fp_3 = large_fp_t {1.0f} / compute_temp_fp_2; // Inverse standard deviation (so we can simply multiply instead of divide in the next step)

    // Partial normalization (excludes gamma and beta, which are applied in LAYERNORM_FINAL_NORM() since they need to be applied column-wise)
    for (uint16_t i = 0; i < EMB_DEPTH; i++) {
        intermediate_res[input_addr+i] = static_cast<float> ( fix_pt_t { (large_fp_t { intermediate_res[input_addr+i] } - compute_temp_fp) * compute_temp_fp_3 } );
    }
}

void CiM::LAYERNORM_2ND_HALF(uint16_t input_addr, uint16_t gamma_addr, uint16_t beta_addr) {
    /* 2nd half of Layer normalization of input. This applies gamma and beta on each column. */
    if (compute_in_progress == true) { throw runtime_error("Computation already in progress when trying to start LAYERNORM_2ND_HALF!"); }
    compute_in_progress = true;

    compute_temp_fp_2 = large_fp_t { params[gamma_addr] }; // Gamma
    compute_temp_fp_3 = large_fp_t { params[beta_addr] }; // Beta

    // Normalize
    for (uint16_t i = 0; i < NUM_PATCHES+1; i++) {
        compute_temp_fp = large_fp_t { intermediate_res[input_addr+i] };
        intermediate_res[input_addr+i] = static_cast<float> ( fix_pt_t { (compute_temp_fp_2 * compute_temp_fp + compute_temp_fp_3) } );
    }
}

void CiM::SOFTMAX(uint16_t input_addr, uint16_t len) {
    /* Softmax of input (performed in-place). Input is in intermediate storage location. Note: All Softmax are done over a row of len length. */
    if (compute_in_progress == true) { throw runtime_error("Computation already in progress when trying to start SOFTMAX!"); }
    compute_in_progress = true;

    compute_temp_fp = large_fp_t { 0 };

    // Exponentiate all elements and sum
    for (uint16_t i = 0; i < len; ++i) {
        if (large_fp_t { intermediate_res[input_addr+i] } < large_fp_t { 0 }) { _neg_exp_cnt++; }
        if (large_fp_t { intermediate_res[input_addr+i] } < _min_exp_input_arg) { _min_exp_input_arg = large_fp_t { intermediate_res[input_addr+i] }; }
        if (large_fp_t { intermediate_res[input_addr+i] } > _max_exp_input_arg) { _max_exp_input_arg = large_fp_t { intermediate_res[input_addr+i] }; }
        _total_exp_cnt++;
        intermediate_res[input_addr+i] = static_cast<float> ( fix_pt_t {exp(large_fp_t { intermediate_res[input_addr+i] })});
        compute_temp_fp += large_fp_t { intermediate_res[input_addr+i] };
    }

    // Normalize
    for (uint16_t i = 0; i < len; ++i) {
        intermediate_res[input_addr+i] = static_cast<float> ( fix_pt_t { (large_fp_t { intermediate_res[input_addr+i] } / compute_temp_fp ) });
    }
}

void CiM::ARGMAX(uint16_t input_addr, uint16_t len) {
    /* Index of maximum element of input. Input is in intermediate storage location. Note: All Max are done over a row of len length. */
    if (compute_in_progress == true) { throw runtime_error("Computation already in progress when trying to start MAX!"); }
    compute_in_progress = true;

    compute_temp_fp_3 = large_fp_t { 0 }; // Index of maximum element
    compute_temp_fp = large_fp_t { intermediate_res[input_addr] }; // Value of maximum element

    for (uint16_t i = 1; i < len; i++) {
        compute_temp_fp_2 = large_fp_t { intermediate_res[input_addr+i] };
        if (compute_temp_fp_2 > compute_temp_fp) {
            compute_temp_fp_3 = large_fp_t { i };
            compute_temp_fp = compute_temp_fp_2;
        }
    }

    computation_result = static_cast<float> (fix_pt_t { compute_temp_fp_3 });
}

large_fp_t CiM::exp_approx(large_fp_t input) {
    /* Approximation of exp(x) as used in the ASIC
       Uses the identy exp(x) = 2^(x/ln(2)), float/int exponent splitting and Taylor approximation of the fractional part.
    */

    large_fp_t ln_2 = log(large_fp_t{2});
    large_fp_t input_mapped = input/ln_2;
    large_fp_t input_mapped_fract = input/ln_2 - floor_fpm(input/ln_2);

    // Taylor approximation of the fractional part
    large_fp_t exp_approx_fract = 1 + ln_2*input_mapped_fract + pow(ln_2,large_fp_t{2})/large_fp_t{2}*pow(input_mapped_fract, large_fp_t{2}) + pow(ln_2,large_fp_t{3})/large_fp_t{6}*pow(input_mapped_fract, large_fp_t{3});

    // Integer part
    return pow(large_fp_t{2}, floor_fpm(input_mapped)) * exp_approx_fract;
}

large_fp_t CiM::floor_fpm(large_fp_t input) {
    /* Performs floor() on fpm input */
    return large_fp_t { floor(static_cast<float>(input)) };
}

bool CiM::get_is_ready() {
    return (is_ready & !compute_in_progress);
}

void CiM::update_state(STATE new_state) {
    if (new_state < INVALID_CIM) { cim_state = new_state; }
    else { throw invalid_argument("Received invalid CiM state!"); }
}

void CiM::overflow_check() {
    if (gen_reg_2b > 3) { throw std::overflow_error("CiM: Overflow on gen_reg_2b!"); }
}