#ifndef CIM_H
#define CIM_H

#include <iostream>
#include <map>

#include <Misc.hpp>
#include <Param_Layer_Mapping.hpp>
#include <Compute_Verification.hpp>

/*----- DEFINE -----*/
#define CIM_PARAMS_STORAGE_SIZE_KB 2112
#define CIM_INT_RES_SIZE_KB 3340
#define CIM_PREV_SOFTMAX_SIZE_KB (NUM_SLEEP_STAGES * (NUM_SAMPLES_OUT_AVG-1) * sizeof(float))
#define COMPUTE_CNT_THRESHOLD 5 // Used to simulate the delay in the computation to match the real hardware

#define HAS_MY_DATA(x) ((id >= (x-3)) && (id < (x))) // Determines whether the current broadcast transaction contains data I need
#define IS_MY_MATRIX(x) ((id >= (x)*NUM_HEADS) && (id < (x+1)*NUM_HEADS)) // Returns whether a given count corresponds to my matrix (for the *V matmul in encoder's MHSA)

/*----- CLASS -----*/
class CiM {
    private:
        enum ACTIVATION {
            NO_ACTIVATION, // Used for simple matrix multiplies
            LINEAR_ACTIVATION,
            SWISH_ACTIVATION,
            SOFTMAX_ACTIVATION
        };

        enum INPUT_TYPE { // Type of input for a given computation
            MODEL_PARAM,
            INTERMEDIATE_RES,
            IMMEDIATE_VAL
        };

        enum STATE {
            IDLE_CIM,
            RESET_CIM,
            PATCH_LOAD_CIM,
            INFERENCE_RUNNING_CIM,
            INVALID_CIM
        };

        enum INFERENCE_STEP {
            CLASS_TOKEN_CONCAT_STEP,
            POS_EMB_STEP,
            ENC_LAYERNORM_1_1ST_HALF_STEP,
            ENC_LAYERNORM_1_2ND_HALF_STEP,
            ENC_LAYERNORM_2_1ST_HALF_STEP,
            ENC_LAYERNORM_2_2ND_HALF_STEP,
            ENC_LAYERNORM_3_1ST_HALF_STEP,
            ENC_LAYERNORM_3_2ND_HALF_STEP,
            POST_LAYERNORM_TRANSPOSE_STEP,
            ENC_MHSA_DENSE_STEP,
            ENC_MHSA_Q_TRANSPOSE_STEP,
            ENC_MHSA_K_TRANSPOSE_STEP,
            ENC_MHSA_QK_T_STEP,
            ENC_MHSA_PRE_SOFTMAX_TRANSPOSE_STEP,
            ENC_MHSA_SOFTMAX_STEP,
            ENC_MHSA_MULT_V_STEP,
            ENC_POST_MHSA_TRANSPOSE_STEP,
            ENC_POST_MHSA_DENSE_AND_INPUT_SUM_STEP,
            ENC_PRE_MLP_TRANSPOSE_STEP,
            MLP_DENSE_1_STEP,
            ENC_POST_DENSE_1_TRANSPOSE_STEP,
            MLP_DENSE_2_AND_SUM_STEP,
            MLP_HEAD_PRE_DENSE_1_TRANSPOSE_STEP,
            MLP_HEAD_DENSE_1_STEP,
            MLP_HEAD_PRE_DENSE_2_TRANSPOSE_STEP,
            MLP_HEAD_DENSE_2_STEP,
            MLP_HEAD_PRE_SOFTMAX_TRANSPOSE_STEP,
            MLP_HEAD_SOFTMAX_STEP,
            POST_SOFTMAX_DIVIDE_STEP,
            POST_SOFTMAX_AVERAGING_STEP,
            RETIRE_SOFTMAX_STEP,
            INFERENCE_COMPLETE,
            INVALID_INF_STEP = -1
        };

        enum DATA {
            PATCH_MEM,
            CLASS_TOKEN_MEM,
            POS_EMB_MEM,
            ENC_LN1_1ST_HALF_MEM,
            ENC_LN1_2ND_HALF_MEM,
            ENC_QVK_IN_MEM,
            ENC_Q_MEM,
            ENC_K_MEM,
            ENC_V_MEM,
            ENC_K_T_MEM,
            ENC_QK_T_IN_MEM,
            ENC_QK_T_MEM,
            ENC_PRE_SOFTMAX_MEM,
            ENC_V_MULT_IN_MEM,
            ENC_V_MULT_MEM,
            ENC_DENSE_IN_MEM,
            ENC_MHSA_OUT_MEM,
            ENC_LN2_1ST_HALF_MEM,
            ENC_LN2_2ND_HALF_MEM,
            ENC_MLP_IN_MEM,
            ENC_MLP_DENSE1_MEM,
            ENC_MLP_DENSE2_IN_MEM,
            ENC_MLP_OUT_MEM,
            MLP_HEAD_LN_1ST_HALF_MEM,
            MLP_HEAD_LN_2ND_HALF_MEM,
            MLP_HEAD_DENSE_1_IN_MEM,
            MLP_HEAD_DENSE_1_OUT_MEM,
            MLP_HEAD_DENSE_2_IN_MEM,
            MLP_HEAD_DENSE_2_OUT_MEM,
            MLP_HEAD_SOFTMAX_IN_MEM,
            PREV_SOFTMAX_STORAGE,
            SOFTMAX_AVG_SUM_MEM
        };

        const std::map<DATA, uint16_t> mem_map = {
            {PATCH_MEM,                 PATCH_LEN+1},
            {CLASS_TOKEN_MEM,           PATCH_LEN},
            {POS_EMB_MEM,               0},
            {ENC_LN1_1ST_HALF_MEM,      NUM_PATCHES+1},
            {ENC_LN1_2ND_HALF_MEM,      NUM_PATCHES+1+EMB_DEPTH},
            {ENC_QVK_IN_MEM,            NUM_PATCHES+1+EMB_DEPTH},
            {ENC_Q_MEM,                 2*EMB_DEPTH+NUM_PATCHES+1},
            {ENC_K_MEM,                 2*(EMB_DEPTH+NUM_PATCHES+1)},
            {ENC_V_MEM,                 NUM_PATCHES+1},
            {ENC_K_T_MEM,               2*EMB_DEPTH+NUM_PATCHES+1},
            {ENC_QK_T_IN_MEM,           3*EMB_DEPTH+NUM_PATCHES+1},
            {ENC_QK_T_MEM,              3*EMB_DEPTH+2*(NUM_PATCHES+1)},
            {ENC_PRE_SOFTMAX_MEM,       2*(EMB_DEPTH+NUM_PATCHES+1)},
            {ENC_V_MULT_IN_MEM,         NUM_PATCHES+1+EMB_DEPTH},
            {ENC_V_MULT_MEM,            2*EMB_DEPTH+NUM_PATCHES+1},
            {ENC_DENSE_IN_MEM,          NUM_PATCHES+1+EMB_DEPTH},
            {ENC_MHSA_OUT_MEM,          2*EMB_DEPTH+NUM_PATCHES+1},
            {ENC_LN2_1ST_HALF_MEM,      NUM_PATCHES+1},
            {ENC_LN2_2ND_HALF_MEM,      NUM_PATCHES+1+EMB_DEPTH},
            {ENC_MLP_IN_MEM,            NUM_PATCHES+1},
            {ENC_MLP_DENSE1_MEM,        NUM_PATCHES+1+EMB_DEPTH},
            {ENC_MLP_DENSE2_IN_MEM,     NUM_PATCHES+1},
            {ENC_MLP_OUT_MEM,           2*EMB_DEPTH+NUM_PATCHES+2},
            {MLP_HEAD_LN_1ST_HALF_MEM,  0},
            {MLP_HEAD_LN_2ND_HALF_MEM,  EMB_DEPTH},
            {MLP_HEAD_DENSE_1_IN_MEM,   EMB_DEPTH},
            {MLP_HEAD_DENSE_1_OUT_MEM,  2*EMB_DEPTH},
            {MLP_HEAD_DENSE_2_IN_MEM,   EMB_DEPTH},
            {MLP_HEAD_DENSE_2_OUT_MEM,  2*EMB_DEPTH},
            {MLP_HEAD_SOFTMAX_IN_MEM,   MLP_DIM},
            {PREV_SOFTMAX_STORAGE,      0}, // Note this is in prev_softmax_storage, and only for CiM #0
            {SOFTMAX_AVG_SUM_MEM,       2*EMB_DEPTH}
        };

        bool compute_in_progress = false; // Used by compute element to notify CiM controller when is in progress
        bool is_ready = true; // Signal that CiM can use to override the compute_in_progress signal and force the master to wait before sending the next instruction
        uint16_t id; // ID of the CiM
        uint16_t gen_reg_16b; // General-purpose register
        uint16_t tx_addr_reg; // Record the address of the data sent on the bus
        uint16_t rx_addr_reg; // Record the address of the data received on the bus
        uint16_t sender_id; // Record the id of an instruction's sender at a start of broadcast
        uint16_t data_len_reg; // General-purpose register used to record len of data sent/received on the bus
        float compute_temp; // Temporary register used to store intermediate results during computation
        float compute_temp_2; // Temporary register used to store intermediate results during computation
        uint16_t _compute_process_cnt; // [Not in ASIC] Counter used to track the progress of the current computation (used to simulate the delay in the computation to match the real hardware)
        uint16_t _num_compute_done; // [Not in ASIC] Counter used to track the number of computations done in a given inference step
        float computation_result; // Used to store the result of the computation
        float params[CIM_PARAMS_STORAGE_SIZE_KB / sizeof(float)];
        float intermediate_res[CIM_INT_RES_SIZE_KB / sizeof(float)];
        float prev_softmax_storage[CIM_PREV_SOFTMAX_SIZE_KB / sizeof(float)]; // In ASIC, this should only be synthesized for CiM #0

        STATE cim_state;
        INFERENCE_STEP current_inf_step = CLASS_TOKEN_CONCAT_STEP;
        OP prev_bus_op;
        Counter gen_cnt_10b;
        Counter gen_cnt_10b_2;
        Counter bytes_rec_cnt; // Tracks the # of bytes received from the bus that are relevant to me
        Counter bytes_sent_cnt; // Tracks the # of bytes sent to the bus

        void update_compute_process_cnt();
        void ADD(uint16_t in1_addr, uint16_t in2_addr, INPUT_TYPE param_type);
        void DIV(uint16_t num_addr, uint16_t in2, INPUT_TYPE in2_type);
        void LAYERNORM_1ST_HALF(uint16_t input_addr);
        void LAYERNORM_2ND_HALF(uint16_t input_addr, uint16_t gamma_addr, uint16_t beta_addr);
        void MAC(uint16_t in1_start_addr, uint16_t in2_start_addr, uint16_t len, uint16_t bias_addr, INPUT_TYPE param_type, ACTIVATION activation);
        void SOFTMAX(uint16_t input_addr, uint16_t len);
        void ARGMAX(uint16_t input_addr, uint16_t len);
        void update_state(STATE new_state);

    public:
        CiM() : id(-1), gen_cnt_10b(10), gen_cnt_10b_2(10), bytes_rec_cnt(8), bytes_sent_cnt(8) {}
        CiM(const int16_t cim_id);
        bool get_is_ready();
        int reset();
        int run(struct ext_signals* ext_sigs, Bus* bus);
};

#endif //CIM_H
