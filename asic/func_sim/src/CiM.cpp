/* TODO:
    - Adjust temporary storage locations for encoder's MHSA intermediate storage as we can overwrite all but the first element of encoder input (which is used in the residual connection) since we will only send the first row of encoder output to MLP head anyways
*/

#include <CiM.hpp>

/*----- NAMESPACE -----*/
using namespace std;

/*----- DECLARATION -----*/
CiM::CiM(const int16_t cim_id) : id(cim_id), gen_cnt_10b(10), gen_cnt_10b_2(10), bytes_rec_cnt(13), bytes_sent_cnt(8) {
    cim_state = RESET_CIM;
    reset();
}

int CiM::reset(){
    fill(begin(params), end(params), 0); // Reset local params
    fill(begin(intermediate_res), end(intermediate_res), 0); // Reset local intermediate_res
    is_ready = true;
    gen_cnt_10b.reset();
    gen_cnt_10b_2.reset();
    bytes_rec_cnt.reset();
    compute_process_cnt = 0;
    num_compute_done = 0;
    cim_state = IDLE_CIM;
    return 0;
}

int CiM::run(struct ext_signals* ext_sigs, Bus* bus){
    /* Run the CiM FSM */

    // Update compute process counter
    update_compute_process_cnt();

    // Read bus if new instruction
    struct instruction inst = bus->get_inst();
    if (inst.op != NOP) {
        switch (inst.op){
        case PATCH_LOAD_BROADCAST_START_OP:
            bytes_rec_cnt.reset();
            gen_cnt_10b_2.reset();
            cim_state = PATCH_LOAD_CIM;
            break;

        case PATCH_LOAD_BROADCAST_OP:
            intermediate_res[bytes_rec_cnt.get_cnt()] = inst.data[0];
            bytes_rec_cnt.inc();
            cim_state = PATCH_LOAD_CIM;
            break;

        case DATA_STREAM_START_OP:
            addr_reg = inst.data[0]; // Address
            bytes_rec_cnt.reset();
            break;

        case DATA_STREAM_OP:
            if (inst.target_or_sender == id) {
                params[bytes_rec_cnt.get_cnt() + addr_reg] = inst.data[0];
                params[bytes_rec_cnt.get_cnt()+1 + addr_reg] = inst.data[1]; // Note: If the length is less than 3, we will be loading junk. That's OK since it will get overwritten
                params[bytes_rec_cnt.get_cnt()+2 + addr_reg] = inst.extra_fields;
                bytes_rec_cnt.inc(3);
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

                addr_reg = start_addr; // Save address of data to send
                bus->push_inst(new_inst);

                if ((inst.op == TRANS_BROADCAST_DATA_OP) && (current_inf_step == ENC_MHSA_QK_T)) { // Since I'm here, move my own data to the correct location in intermediate_res (do I need to do that in the previous op (QVK dense) as well?)
                    intermediate_res[static_cast<int>(inst.extra_fields)+id] = intermediate_res[addr_reg+id];
                }
            } else {
                addr_reg = static_cast<int> (inst.extra_fields); // Save address where to store data
            }

            if (current_inf_step == ENC_MHSA_MULT_V) {
                if (inst.target_or_sender == 0) { gen_cnt_10b_2.inc(); }
                bytes_rec_cnt.reset(); // We also want to reset this counter here because the CiM that don't have to multiply anything for this matrix could overflow the counter
            }

            data_len_reg = inst.data[1]; // Save length of data to send/receive
            bytes_sent_cnt.reset();
            break;

        case DENSE_BROADCAST_DATA_OP:
        case TRANS_BROADCAST_DATA_OP:
            bytes_sent_cnt.inc(3); // Increment data that was sent on the bus

            // Data grab
            if (inst.op == TRANS_BROADCAST_DATA_OP) { // Grab the data that corresponds to my data (note this will also move the data to the correct location for the id that is currently sending)
                if (current_inf_step == MLP_HEAD_DENSE_2_STEP) { // This step is special. Each of CiM's #0-#31 send a single element and every CiM's must grab it, so there is no check to do
                    bytes_rec_cnt.inc();
                } else {
                    if (HAS_MY_DATA(bytes_sent_cnt.get_cnt())) { bytes_rec_cnt.inc(); } // Increment data received
                }
                
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
                intermediate_res[addr_reg+bytes_sent_cnt.get_cnt()-3] = inst.data[0];
                intermediate_res[addr_reg+bytes_sent_cnt.get_cnt()-2] = inst.data[1];
                intermediate_res[addr_reg+bytes_sent_cnt.get_cnt()-1] = inst.extra_fields;
            }

            // Prep new instruction to send
            if ((bytes_sent_cnt.get_cnt() < data_len_reg) && (inst.target_or_sender == id)) { // If last time was me and I have more data to send
                struct instruction new_inst = {/*op*/ inst.op, /*target_or_sender*/id, /*data*/{0, 0}, /*extra_fields*/0};
                if ((data_len_reg - bytes_rec_cnt.get_cnt() == 2)) { // Only two bytes left
                    new_inst.data = {intermediate_res[addr_reg+bytes_rec_cnt.get_cnt()], intermediate_res[addr_reg+bytes_rec_cnt.get_cnt()+1]};
                } else if ((data_len_reg - bytes_rec_cnt.get_cnt() == 1)) { // Only one byte left
                    new_inst.data = {intermediate_res[addr_reg+bytes_rec_cnt.get_cnt()], 0};
                } else {
                    new_inst.data = {intermediate_res[addr_reg+bytes_rec_cnt.get_cnt()], intermediate_res[addr_reg+bytes_rec_cnt.get_cnt()+1]};
                    new_inst.extra_fields = intermediate_res[addr_reg+bytes_rec_cnt.get_cnt()+2];
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
    switch (cim_state){
    case IDLE_CIM:
        break;

    case RESET_CIM:
        if (ext_sigs->master_nrst == true) { cim_state = IDLE_CIM; }
        reset();
        break;

    case PATCH_LOAD_CIM:
        if ((compute_in_progress == false) && (bytes_rec_cnt.get_cnt() == PATCH_LEN)) { // Received my complete patch, perform part of Dense layer
            MAC(/* patch starting addr */ 0,/* weight starting addr */ 0, /*len*/ PATCH_LEN, /* bias addr */ param_addr_map[SINGLE_PARAMS].addr+PATCH_PROJ_BIAS_OFF, MODEL_PARAM, LINEAR_ACTIVATION);
            gen_reg_16b = 1;
            bytes_rec_cnt.reset();
        } else if ((compute_in_progress == false) && (gen_reg_16b == 1)) { // Done computation
            intermediate_res[gen_cnt_10b_2.get_cnt() + PATCH_LEN + 1] = computation_result; // +1 to account for classification token
            gen_reg_16b = 0;
            gen_cnt_10b_2.inc(); // Increment number of patches received
            if (gen_cnt_10b_2.get_cnt() == NUM_PATCHES) { // Received all patches, automatically start inference
                is_ready = false;
                cim_state = INFERENCE_RUNNING_CIM;
                gen_cnt_10b_2.reset();
                verify_computation(PATCH_PROJECTION_VERIF, id, intermediate_res, PATCH_LEN+1);
            }
        }
    break;

    case INFERENCE_RUNNING_CIM:
        switch (current_inf_step) {
        case CLASS_TOKEN_CONCAT:
            intermediate_res[PATCH_LEN] = params[param_addr_map[SINGLE_PARAMS].addr+CLASS_EMB_OFF]; // Move classification token from parameters memory to intermediate storage
            current_inf_step = POS_EMB;
            bytes_rec_cnt.reset();
            gen_cnt_10b.reset();
            gen_reg_16b = 0;
            verify_computation(CLASS_TOKEN_VERIF, id, intermediate_res, PATCH_LEN);
            break;
        
        case POS_EMB:
            if ((compute_in_progress == false) && (gen_cnt_10b.get_cnt() < NUM_PATCHES+1)) { // Start computation
                if (gen_reg_16b == 1) { // Save computation result from previous iteration (except for the first iteration)
                    intermediate_res[gen_cnt_10b.get_cnt()] = computation_result;
                    gen_cnt_10b.inc();
                }
                ADD(PATCH_LEN+gen_cnt_10b.get_cnt(), param_addr_map[POS_EMB_PARAMS].addr+gen_cnt_10b.get_cnt(), MODEL_PARAM);
                gen_reg_16b = 1;
            } else if ((compute_in_progress == false) && (gen_cnt_10b.get_cnt() == NUM_PATCHES+1)) { // Done with positional embedding
                intermediate_res[gen_cnt_10b.get_cnt()] = computation_result; // Save the last result
                current_inf_step = ENC_LAYERNORM_1_1ST_HALF_STEP;
                if (id == 0) { cout << "CiM: Finished positional embedding" << endl; }
                gen_cnt_10b.reset();
                verify_computation(POS_EMB_VERIF, id, intermediate_res, 0);
                is_ready = true;
            }
            break;

        case ENC_LAYERNORM_1_1ST_HALF_STEP:
        case ENC_LAYERNORM_2_1ST_HALF_STEP:
        case ENC_LAYERNORM_3_1ST_HALF_STEP:
            if (bytes_rec_cnt.get_cnt() == EMB_DEPTH) { // No more data to receive, start LayerNorm
                if (compute_in_progress == false) {
                    gen_reg_16b = 1; // Just a signal to avoid coming here every time FSM runs
                    LAYERNORM_1ST_HALF(0);
                    bytes_rec_cnt.reset();
                }
            }

            if (inst.op == PISTOL_START_OP) {
                if (id == 0) { cout << "CiM: Finished LayerNorm (1st half)" << endl; }
                if (current_inf_step == ENC_LAYERNORM_1_1ST_HALF_STEP) { current_inf_step = ENC_LAYERNORM_1_2ND_HALF_STEP; }
                else if (current_inf_step == ENC_LAYERNORM_2_1ST_HALF_STEP) { current_inf_step = ENC_LAYERNORM_2_2ND_HALF_STEP; }
                else if (current_inf_step == ENC_LAYERNORM_3_1ST_HALF_STEP) { current_inf_step = ENC_LAYERNORM_3_2ND_HALF_STEP; }
                gen_reg_16b = 0;
                bytes_rec_cnt.reset();
            }
            break;

        case ENC_LAYERNORM_1_2ND_HALF_STEP:
        case ENC_LAYERNORM_2_2ND_HALF_STEP:
        case ENC_LAYERNORM_3_2ND_HALF_STEP:
            if (((bytes_rec_cnt.get_cnt() == 1) && (current_inf_step == ENC_LAYERNORM_3_2ND_HALF_STEP)) || (bytes_rec_cnt.get_cnt() == NUM_PATCHES+1)) { // No more data to receive, start LayerNorm
                if (compute_in_progress == false) {
                    float gamma = 0.0f;
                    float beta = 0.0f;
                    if (current_inf_step == ENC_LAYERNORM_1_2ND_HALF_STEP) {
                        gamma = params[param_addr_map[SINGLE_PARAMS].addr+ENC_LAYERNORM_1_GAMMA_OFF];
                        beta = params[param_addr_map[SINGLE_PARAMS].addr+ENC_LAYERNORM_1_BETA_OFF];
                    } else if (current_inf_step == ENC_LAYERNORM_2_2ND_HALF_STEP) {
                        gamma = params[param_addr_map[SINGLE_PARAMS].addr+ENC_LAYERNORM_2_GAMMA_OFF];
                        beta = params[param_addr_map[SINGLE_PARAMS].addr+ENC_LAYERNORM_2_BETA_OFF];
                    } else if (current_inf_step == ENC_LAYERNORM_3_2ND_HALF_STEP) {
                        gamma = params[param_addr_map[SINGLE_PARAMS].addr+ENC_LAYERNORM_3_GAMMA_OFF];
                        beta = params[param_addr_map[SINGLE_PARAMS].addr+ENC_LAYERNORM_3_BETA_OFF];
                    }
                    gen_reg_16b = 1; // Just a signal to avoid coming here every time FSM runs
                    LAYERNORM_2ND_HALF(0, gamma, beta);
                    bytes_rec_cnt.reset();
                }
            }

            if (inst.op == PISTOL_START_OP) {
                if (current_inf_step == ENC_LAYERNORM_1_2ND_HALF_STEP) { current_inf_step = POST_LAYERNORM_TRANSPOSE_STEP; }
                else if (current_inf_step == ENC_LAYERNORM_2_2ND_HALF_STEP) { current_inf_step = MLP_DENSE_1_STEP; }
                else if (current_inf_step == ENC_LAYERNORM_3_2ND_HALF_STEP) { current_inf_step = MLP_HEAD_DENSE_1_STEP; }
                gen_reg_16b = 0;
                if (id == 0) { cout << "CiM: Finished LayerNorm (2nd half)" << endl; }
                bytes_rec_cnt.reset();
            }
            break;

        case POST_LAYERNORM_TRANSPOSE_STEP:
            if (inst.op == PISTOL_START_OP) {
                current_inf_step = ENC_MHSA_DENSE;
                bytes_rec_cnt.reset();
                gen_cnt_10b_2.reset();
            }
            break;

        case ENC_MHSA_DENSE:
        case MLP_DENSE_1_STEP:
        case MLP_DENSE_2_AND_SUM_STEP:
        case MLP_HEAD_DENSE_1_STEP:
        case MLP_HEAD_DENSE_2_STEP:
            if (bytes_rec_cnt.get_cnt() >= EMB_DEPTH) { // No more data to receive for this broadcast, start MACs
                if ((compute_in_progress == false) && (current_inf_step == ENC_MHSA_DENSE)) {
                    if (gen_reg_16b == 0) {
                        MAC(NUM_PATCHES+1+EMB_DEPTH, 128, param_addr_map[SINGLE_PARAMS].addr+ENC_Q_DENSE_BIAS_0FF, EMB_DEPTH, MODEL_PARAM, LINEAR_ACTIVATION); // Q
                        gen_reg_16b = 1;
                    } else if (gen_reg_16b == 1) {
                        intermediate_res[128+inst.target_or_sender] = computation_result; // Q
                        MAC(NUM_PATCHES+1+EMB_DEPTH, 192, param_addr_map[SINGLE_PARAMS].addr+ENC_K_DENSE_BIAS_0FF, EMB_DEPTH, MODEL_PARAM, LINEAR_ACTIVATION); // K
                        gen_reg_16b = 2;
                    } else if (gen_reg_16b == 2) {
                        intermediate_res[189+inst.target_or_sender] = computation_result; // K
                        MAC(NUM_PATCHES+1+EMB_DEPTH, 256, param_addr_map[SINGLE_PARAMS].addr+ENC_V_DENSE_BIAS_0FF, EMB_DEPTH, MODEL_PARAM, LINEAR_ACTIVATION); // V
                        gen_reg_16b = 3;
                        bytes_rec_cnt.reset();
                    }
                } else if ((compute_in_progress == false) && (current_inf_step == MLP_DENSE_1_STEP) && (id < MLP_DIM)) { // Only least significant MLP_DIM number of CiMs will perform this computation
                    MAC(NUM_PATCHES+1, 384, param_addr_map[SINGLE_PARAMS].addr+ENC_MLP_DENSE_1_MLP_HEAD_DENSE_1_BIAS_OFF, EMB_DEPTH, MODEL_PARAM, SWISH_ACTIVATION);
                    gen_reg_16b = 3;
                    bytes_rec_cnt.reset();
                } else if ((compute_in_progress == false) && (current_inf_step == MLP_HEAD_DENSE_1_STEP) && (id >= MLP_DIM)) { // Only most significant MLP_DIM number of CiMs will perform this computation
                    MAC(0, 384, param_addr_map[SINGLE_PARAMS].addr+ENC_MLP_DENSE_1_MLP_HEAD_DENSE_1_BIAS_OFF, EMB_DEPTH, MODEL_PARAM, SWISH_ACTIVATION);
                    gen_reg_16b = 3;
                    bytes_rec_cnt.reset();
                }
            } else if ((bytes_rec_cnt.get_cnt() >= MLP_DIM) && (current_inf_step == MLP_DENSE_2_AND_SUM_STEP || current_inf_step == MLP_HEAD_DENSE_2_STEP)) { // No more data to receive for this broadcast (for MLP's dense #2), start MACs
                if ((compute_in_progress == false) && (current_inf_step == MLP_DENSE_2_AND_SUM_STEP)) {
                    if (gen_reg_16b == 0) {
                        MAC(NUM_PATCHES+1, 448, param_addr_map[SINGLE_PARAMS].addr+ENC_MLP_DENSE_2_BIAS_OFF, MLP_DIM, MODEL_PARAM, SWISH_ACTIVATION);
                        gen_reg_16b = 1;
                    } else if (gen_reg_16b == 1) {
                        intermediate_res[190+inst.target_or_sender] = computation_result; // MLP Dense 2
                        ADD(/*enc input*/ inst.target_or_sender, 190+inst.target_or_sender, INTERMEDIATE_RES);  // Sum with encoder's input (next step in inference pipeline, but do it now)
                        gen_reg_16b = 3;
                        bytes_rec_cnt.reset();
                    }
                } else if ((compute_in_progress == false) && (current_inf_step == MLP_HEAD_DENSE_2_STEP) && (id < NUM_SLEEP_STAGES)) {
                    MAC(0, 480, param_addr_map[SINGLE_PARAMS].addr+MLP_HEAD_DENSE_2_BIAS_OFF, MLP_DIM, MODEL_PARAM, LINEAR_ACTIVATION);
                    gen_reg_16b = 3;
                    bytes_rec_cnt.reset();
                }
            } else if ((compute_in_progress == false) && (gen_reg_16b == 3)){ // Done
                gen_cnt_10b_2.inc();
                if ((id == 0) && (gen_cnt_10b_2.get_cnt() == EMB_DEPTH) && (current_inf_step == ENC_MHSA_DENSE)) { // Encoder's MHSA Q/K/V Dense
                    cout << "CiM: Finished encoder's MHSA Q/K/V Dense" << endl; 
                    intermediate_res[NUM_PATCHES+1+inst.target_or_sender] = computation_result; // V
                } else if ((id == 0) && (gen_cnt_10b_2.get_cnt() == MLP_DIM) && (current_inf_step == MLP_DENSE_1_STEP)) { // MLP Dense 1 
                    cout << "CiM: Finished MLP Dense 1" << endl;
                    intermediate_res[190+inst.target_or_sender] = computation_result;
                } else if ((id == 0) && (gen_cnt_10b_2.get_cnt() == 1) && (current_inf_step == MLP_DENSE_2_AND_SUM_STEP)) { // MLP Dense 2 and sum 
                    cout << "CiM: Finished MLP Dense 2" << endl;
                    intermediate_res[190+inst.target_or_sender] = computation_result;
                } else if ((id == 32) && (gen_cnt_10b_2.get_cnt() == 1) && (current_inf_step == MLP_HEAD_DENSE_1_STEP)) { // MLP head's Dense 1
                    cout << "CiM: Finished MLP head's Dense 1" << endl; 
                    intermediate_res[125+inst.target_or_sender] = computation_result;
                } else if ((id == 0) && (gen_cnt_10b_2.get_cnt() == 1) && (current_inf_step == MLP_HEAD_DENSE_2_STEP)) { // MLP Dense 2 (softmax)
                    cout << "CiM: Finished MLP head's Dense 2" << endl;
                    intermediate_res[0] = computation_result;
                }
                gen_reg_16b = 0;
            }
            
            if (inst.op == PISTOL_START_OP) {
                if (current_inf_step == ENC_MHSA_DENSE) { current_inf_step = ENC_MHSA_Q_TRANSPOSE_STEP; }
                else if (current_inf_step == MLP_DENSE_1_STEP) { current_inf_step = MLP_DENSE_2_AND_SUM_STEP; }
                else if (current_inf_step == MLP_DENSE_2_AND_SUM_STEP) { current_inf_step = ENC_LAYERNORM_3_1ST_HALF_STEP; }
                else if (current_inf_step == MLP_HEAD_DENSE_1_STEP) { current_inf_step = MLP_HEAD_DENSE_2_STEP; }
                else if (current_inf_step == MLP_HEAD_DENSE_2_STEP) { current_inf_step = MLP_HEAD_PRE_SOFTMAX_TRANSPOSE_STEP; }
                gen_reg_16b = 0;
                gen_cnt_10b_2.reset();
                bytes_rec_cnt.reset();
            }
            break;

        case MLP_HEAD_PRE_SOFTMAX_TRANSPOSE_STEP:
            current_inf_step = (bytes_rec_cnt.get_cnt() == NUM_SLEEP_STAGES) ? MLP_HEAD_SOFTMAX_STEP : MLP_HEAD_PRE_SOFTMAX_TRANSPOSE_STEP;
            break;

        case ENC_MHSA_Q_TRANSPOSE_STEP:
        case ENC_MHSA_K_TRANSPOSE_STEP:
        case ENC_POST_MHSA_TRANSPOSE_STEP:
            if (inst.op == PISTOL_START_OP) {
                bytes_rec_cnt.reset();
                gen_cnt_10b.reset();
                gen_cnt_10b_2.reset();
                current_inf_step = static_cast<INFERENCE_STEP> (static_cast<int> (current_inf_step) + 1);
            }
            break;

        case ENC_MHSA_QK_T: {
            uint16_t MAC_storage_addr = 2*EMB_DEPTH+3*(NUM_PATCHES+1) + (gen_cnt_10b_2.get_cnt()-1)*(NUM_PATCHES+1) + inst.target_or_sender; // Storage location of MAC result
            if (bytes_rec_cnt.get_cnt() >= NUM_HEADS) { // No more data to receive, start MACs
                if (compute_in_progress == false){
                    //TODO: Double-check addressing here
                    if (gen_reg_16b == 0) {
                        uint16_t MAC_in1_addr = EMB_DEPTH+NUM_PATCHES+1 + (gen_cnt_10b_2.get_cnt()-1)*NUM_HEADS;
                        uint16_t MAC_in2_addr = 2*(EMB_DEPTH+NUM_PATCHES+1); // Temp storage location of broadcast QK_T clip

                        MAC(MAC_in1_addr, MAC_in2_addr, /*bias addr unused when NO_ACTIVATION*/ 0, NUM_HEADS, INTERMEDIATE_RES, NO_ACTIVATION);
                        gen_reg_16b = 1;
                    } else if (gen_reg_16b == 1) {
                        intermediate_res[MAC_storage_addr] = computation_result;
                        DIV(MAC_storage_addr, param_addr_map[SINGLE_PARAMS].addr+ENC_SQRT_NUM_HEADS_OFF, MODEL_PARAM); // /sqrt(NUM_HEADS)
                        gen_reg_16b = 2; // Just a signal to avoid coming here every time FSM runs
                        bytes_rec_cnt.reset();
                    }
                }
            } else if (compute_in_progress == false && gen_reg_16b == 2) { // Done with this MAC
                gen_reg_16b = 0;
                intermediate_res[MAC_storage_addr] = computation_result;
                // Increment counters
                gen_cnt_10b.inc();
                if (gen_cnt_10b.get_cnt() == (NUM_PATCHES+1)) { // All MACs for one matrix are done
                    gen_cnt_10b.reset();
                    gen_cnt_10b_2.inc();
                }

                if (gen_cnt_10b_2.get_cnt() == NUM_HEADS) { // Done with all matrices in the Z-stack
                    if (id == 0) { cout << "CiM: Finished encoder's MHSA QK_T" << endl; }
                    gen_cnt_10b_2.reset();
                    bytes_rec_cnt.reset();
                    current_inf_step = ENC_MHSA_SOFTMAX;
                }
            }
            break;
        }
        case MLP_HEAD_SOFTMAX_STEP:
            if ((compute_in_progress == false) && (gen_reg_16b == 0)) {
                SOFTMAX(/*input addr*/ 32, /*len*/ NUM_SLEEP_STAGES);
                gen_reg_16b = 1; // Just a signal to avoid coming here every time FSM runs
            } else if (compute_in_progress == false) {
                current_inf_step = POST_SOFTMAX_AVERAGING_STEP;
                gen_cnt_10b.reset();
                gen_reg_16b = 0;
                if (id == 0) { cout << "CiM: Finished MLP head's Softmax" << endl; }
            }
            break;

        case POST_SOFTMAX_DIVIDE_STEP:
            if ((compute_in_progress == false) && (gen_reg_16b == 0) && (gen_cnt_10b.get_cnt() < NUM_SLEEP_STAGES) && (id == 0)) { // Divide all elements by NUM_SLEEP_STAGES
                DIV(0, NUM_SLEEP_STAGES, IMMEDIATE_VAL);
                gen_reg_16b = 1; // Just a signal to avoid coming here every time FSM runs
            } else if (((gen_cnt_10b.get_cnt() == NUM_SLEEP_STAGES) && (compute_in_progress == false) && (gen_reg_16b == 1)) || (id > 0)) {
                if (id == 0) { cout << "CiM: Finished MLP head's Softmax averaging" << endl; }
                current_inf_step = POST_SOFTMAX_AVERAGING_STEP;
                gen_reg_16b = 0;
            } else if ((compute_in_progress == false) && (gen_reg_16b == 1)) { // Done one division
                intermediate_res[MLP_DIM+gen_cnt_10b.get_cnt()] = computation_result;
                gen_cnt_10b.inc();
                gen_cnt_10b_2.inc();
                gen_reg_16b = 0; // Reset signal
            }
            break;

        case POST_SOFTMAX_AVERAGING_STEP:
            if ((compute_in_progress == false) && (gen_cnt_10b_2.get_cnt() > 0) && (gen_cnt_10b_2.get_cnt() < NUM_SAMPLES_OUT_AVG) && (gen_reg_16b == 0)) { // Store previous sum result
                uint16_t addr_for_sum_result = MLP_DIM + NUM_SLEEP_STAGES + gen_cnt_10b.get_cnt() + NUM_SLEEP_STAGES*gen_cnt_10b_2.get_cnt();
                intermediate_res[addr_for_sum_result] = computation_result;
                if (gen_cnt_10b_2.get_cnt() == NUM_SAMPLES_OUT_AVG-1) {
                    gen_cnt_10b_2.reset();
                    gen_cnt_10b.inc();

                    if (gen_cnt_10b.get_cnt() == NUM_SLEEP_STAGES) { // Done with averaging
                        gen_reg_16b = 1;
                    }
                }
            }
            if ((gen_cnt_10b.get_cnt() < NUM_SLEEP_STAGES) && (id == 0)) {
                if ((compute_in_progress == false) && (gen_cnt_10b_2.get_cnt() < NUM_SAMPLES_OUT_AVG-1)) {
                    uint16_t addr_for_prev_softmax = gen_cnt_10b.get_cnt() + NUM_SLEEP_STAGES*gen_cnt_10b_2.get_cnt();
                    uint16_t addr_current_softmax_divide = MLP_DIM + NUM_SLEEP_STAGES + gen_cnt_10b.get_cnt() + NUM_SLEEP_STAGES*gen_cnt_10b_2.get_cnt();

                    intermediate_res[addr_for_prev_softmax] = prev_softmax_storage[addr_for_prev_softmax]; // Move previous softmax result to intermediate storage
                    ADD(addr_current_softmax_divide, addr_for_prev_softmax, INTERMEDIATE_RES); // Sum with previous result
                    gen_cnt_10b_2.inc();
                }
            } else if ((compute_in_progress == false) && (id == 0) && (gen_reg_16b == 1)) {
                // Start a MAX in the background
                if (gen_cnt_10b_2.get_cnt() == 0) { MAX_INDEX(/*addr*/ MLP_DIM + NUM_SLEEP_STAGES*NUM_SAMPLES_OUT_AVG, /*len*/NUM_SLEEP_STAGES); }

                // Move averages into prev_softmax_storage
                if (gen_cnt_10b_2.get_cnt() < NUM_SLEEP_STAGES) {
                    prev_softmax_storage[gen_cnt_10b_2.get_cnt()+NUM_SLEEP_STAGES] = prev_softmax_storage[gen_cnt_10b_2.get_cnt()]; // Softmax at current epoch - 2
                    prev_softmax_storage[gen_cnt_10b_2.get_cnt()] = intermediate_res[MLP_DIM + NUM_SLEEP_STAGES + gen_cnt_10b_2.get_cnt()]; // Softmax at current epoch - 1
                    gen_cnt_10b_2.inc();
                }
                if (gen_cnt_10b_2.get_cnt() == NUM_SLEEP_STAGES) { gen_reg_16b = 2; }
            } else if ((compute_in_progress == false) && (gen_reg_16b == 2)) {
                float inferred_sleep_stage = computation_result;
                struct instruction inst = {/*op*/ INFERENCE_RESULT_OP, /*target_or_sender*/0, /*data*/{inferred_sleep_stage, 0}, /*extra_fields*/0};
                bus->push_inst(inst);
                current_inf_step = INFERENCE_COMPLETE;
                cout << "CiM: Finished averaging. Inference complete. Inferred sleep stage: " << inferred_sleep_stage << endl;
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
            }

            if (inst.op == PISTOL_START_OP) {
                if (id == 0) { cout << "CiM: Finished encoder's MHSA softmax" << endl; }
                current_inf_step = ENC_MHSA_MULT_V;
                gen_cnt_10b.reset();
                gen_cnt_10b_2.reset();
            }
            break;

        case ENC_MHSA_MULT_V:
            if (bytes_rec_cnt.get_cnt() >= (NUM_PATCHES+1)) { // No more data to receive for this given row
                if (IS_MY_MATRIX(gen_cnt_10b_2.get_cnt()-1) && (compute_in_progress == false)) { // Only if matrix being broadcast corresponds to mine (-1 because gen_cnt_10b_2 is incremented when the matrix starts broadcast)
                    uint16_t MAC_in1_addr = 3*(NUM_PATCHES+1)+2*EMB_DEPTH + (gen_cnt_10b_2.get_cnt()-1)*(NUM_PATCHES+1);
                    MAC(MAC_in1_addr, /*in2 addr*/ NUM_PATCHES+1+EMB_DEPTH, /*bias addr unused when NO_ACTIVATION*/ 0, NUM_PATCHES+1, INTERMEDIATE_RES, NO_ACTIVATION);
                    gen_reg_16b = 1; // Just a signal to avoid coming here every time FSM runs
                    bytes_rec_cnt.reset();
                }
            } else if (compute_in_progress == false && gen_reg_16b == 1) { // Done with this row in the matrix
                intermediate_res[2*EMB_DEPTH + NUM_PATCHES+1 + inst.target_or_sender] = computation_result;
                gen_reg_16b = 0;
            }

            if (inst.op == PISTOL_START_OP) {
                current_inf_step = ENC_POST_MHSA_TRANSPOSE_STEP;
                if (id == 0) { cout << "CiM: Finished encoder's V matmul" << endl; }
                bytes_rec_cnt.reset();
                gen_cnt_10b_2.reset();
            }
            break;

        case ENC_POST_MHSA_DENSE_AND_INPUT_SUM_STEP:
            if (bytes_rec_cnt.get_cnt() >= EMB_DEPTH) { // No more data to receive for this broadcast, start MACs
                if ((compute_in_progress == false) && (gen_reg_16b == 0)){ // Start computation
                    MAC(NUM_PATCHES+1, 320, param_addr_map[SINGLE_PARAMS].addr+ENC_COMB_HEAD_BIAS_OFF, EMB_DEPTH, MODEL_PARAM, LINEAR_ACTIVATION);
                    gen_reg_16b = 1; // Just a signal to avoid coming here every time FSM runs
                } else if ((compute_in_progress == false) && (gen_reg_16b == 1)) { // Done with this MAC
                    intermediate_res[NUM_PATCHES+1 + EMB_DEPTH + inst.target_or_sender] = computation_result; // Store previous result
                    ADD(NUM_PATCHES+1 + EMB_DEPTH + inst.target_or_sender, inst.target_or_sender, INTERMEDIATE_RES); // Sum with encoder's input as a residual connection
                    gen_reg_16b = 2;
                    bytes_rec_cnt.reset();
                }
            } else if (compute_in_progress == false && gen_reg_16b == 2) { // Done with this MAC
                intermediate_res[NUM_PATCHES+1 + EMB_DEPTH + inst.target_or_sender] = computation_result; // Store last result
                gen_cnt_10b_2.inc();
                gen_reg_16b = 0;
            }

            if (inst.op == PISTOL_START_OP) {
                if (id == 0 && gen_cnt_10b_2.get_cnt() == (NUM_PATCHES+1)) { cout << "CiM: Finished encoder's post-MHSA Dense" << endl; }
                current_inf_step = ENC_LAYERNORM_2_1ST_HALF_STEP; // Start another round of LayerNorm
                gen_reg_16b = 0;
                gen_cnt_10b_2.reset();
            }
            break;

        case INFERENCE_COMPLETE:
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
    if (compute_in_progress == true) { compute_process_cnt++; }
    if (compute_process_cnt == COMPUTE_CNT_THRESHOLD) {
        compute_process_cnt = 0;
        compute_in_progress = false;
        num_compute_done++;
    }
}

void CiM::MAC(uint16_t in1_start_addr, uint16_t in2_start_addr, uint16_t len, uint16_t bias_addr, INPUT_TYPE param_type, ACTIVATION activation) {
    /* Dot-product between two vectors. */

    float mac_result = 0.0f;
    float input2 = 0.0f;
    float bias = 0.0f;

    if (compute_in_progress == true) { throw runtime_error("Computation already in progress when trying to start MAC!"); }
    compute_in_progress = true;
    // MAC
    for (uint16_t i = 0; i < len; ++i) {
        if (activation != NO_ACTIVATION)  { bias = params[bias_addr]; }

        // Grab second input
        if (param_type == INTERMEDIATE_RES) { input2 = intermediate_res[in2_start_addr+i]; } 
        else if (param_type == MODEL_PARAM) { input2 = params[in2_start_addr+i]; }
        else {
            cout << "Received unknown parameter type " << param_type << endl;
            exit(-1);
        }

        // Compute MAC
        mac_result += input2 * intermediate_res[in1_start_addr+i];
    }

    // Activation
    switch (activation) {
    case LINEAR_ACTIVATION:
        mac_result += bias;
        break;
    case SWISH_ACTIVATION:
        mac_result = (mac_result + bias) * (1.0f / (1.0f + exp(-(mac_result + bias))));
        break;
    case NO_ACTIVATION:
    default:
        break;
    }
    computation_result = mac_result;
}

void CiM::DIV(uint16_t num_addr, uint16_t in2, INPUT_TYPE in2_type) {
    /* Return division of two inputs. */
    float result = 0.0f;
    if (compute_in_progress == true) { throw runtime_error("Computation already in progress when trying to start DIV!"); }
    compute_in_progress = true;

    if (in2_type == MODEL_PARAM) { // Second input is a model parameter
        result = intermediate_res[num_addr] / params[in2];
    } else if (in2_type == IMMEDIATE_VAL) { // Second input is an immediate value
        result = intermediate_res[num_addr] / in2;
    } else {
        cout << "Received unknown parameter type " << in2_type << endl;
    }
    computation_result = result;
}

void CiM::ADD(uint16_t in1_addr, uint16_t in2_addr, INPUT_TYPE param_type) {
    /* Return sum of two inputs. */
    float results = 0.0f;
    if (compute_in_progress == true) { throw runtime_error("Computation already in progress when trying to start ADD!"); }
    compute_in_progress = true;

    if (param_type == INTERMEDIATE_RES) { results = intermediate_res[in1_addr] + intermediate_res[in2_addr]; }
    else if (param_type == MODEL_PARAM) { results = intermediate_res[in1_addr] + params[in2_addr]; }
    computation_result = results;
}

void CiM::LAYERNORM_1ST_HALF(uint16_t input_addr) {
    /* 1st half of Layer normalization of input. Input is in intermediate storage location. Note: All LayerNorms are done over a row of EMB_DEPTH length. */
    if (compute_in_progress == true) { throw runtime_error("Computation already in progress when trying to start LAYERNORM_1ST_HALF!"); }
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
}

void CiM::LAYERNORM_2ND_HALF(uint16_t input_addr, float gamma, float beta) {
    /* 2nd half of Layer normalization of input. This applies gamma and beta on each column. */
    if (compute_in_progress == true) { throw runtime_error("Computation already in progress when trying to start LAYERNORM_2ND_HALF!"); }
    compute_in_progress = true;

    // Normalize
    for (uint16_t i = 0; i < EMB_DEPTH; ++i) {
        intermediate_res[input_addr+i] = gamma * intermediate_res[input_addr+i] + beta;
    }
}

void CiM::SOFTMAX(uint16_t input_addr, uint16_t len) {
    /* Softmax of input (performed in-place). Input is in intermediate storage location. Note: All Softmax are done over a row of len length. */

    float exp_sum = 0.0f;
    if (compute_in_progress == true) { throw runtime_error("Computation already in progress when trying to start SOFTMAX!"); }
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
}

void CiM::MAX_INDEX(uint16_t input_addr, uint16_t len) {
    /* Index of maximum element of input. Input is in intermediate storage location. Note: All Max are done over a row of len length. */

    float max_index = 0.0f;
    float max_val = 0.0f;
    if (compute_in_progress == true) { throw runtime_error("Computation already in progress when trying to start MAX!"); }
    compute_in_progress = true;

    // Find max
    for (uint16_t i = 0; i < len; ++i) {
        if (intermediate_res[input_addr+i] > max_val) { 
            max_index = i;
            max_val = intermediate_res[input_addr+i];
        }
    }
    computation_result = max_index;
}

bool CiM::get_is_ready() {
    return (is_ready & !compute_in_progress);
}