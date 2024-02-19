#ifndef CIM_H
#define CIM_H

#include <iostream>

#include <Misc.hpp>
#include <Param_Layer_Mapping.hpp>

/*----- DEFINE -----*/
#define CIM_PARAMS_STORAGE_SIZE_KB 3072
#define CIM_INT_RES_SIZE_KB 3328
#define CIM_PREV_SOFTMAX_SIZE_KB (NUM_SLEEP_STAGES * NUM_SAMPLES_OUT_AVG * sizeof(float))
#define COMPUTE_CNT_THRESHOLD 10 // Used to simulate the delay in the computation to match the real hardware

#define HAS_MY_DATA(x) ((id >= (x-3)) && (id < x)) // Determines whether the current broadcast transaction contains data I need
#define IS_MY_MATRIX(x) ((id >= x*NUM_HEADS) && (id < (x+1)*NUM_HEADS)) // Returns whether a given count corresponds to my matrix (for the *V matmul in encoder's MHSA)

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
            INVALID_CIM = -1
        };

        enum INFERENCE_STEP {
            CLASS_TOKEN_CONCAT,
            POS_EMB,
            ENC_LAYERNORM_1_1ST_HALF_STEP,
            ENC_LAYERNORM_1_2ND_HALF_STEP,
            ENC_LAYERNORM_2_1ST_HALF_STEP,
            ENC_LAYERNORM_2_2ND_HALF_STEP,
            ENC_LAYERNORM_3_1ST_HALF_STEP,
            ENC_LAYERNORM_3_2ND_HALF_STEP,
            POST_LAYERNORM_TRANSPOSE_STEP,
            ENC_MHSA_DENSE,
            ENC_MHSA_Q_TRANSPOSE_STEP,
            ENC_MHSA_K_TRANSPOSE_STEP,
            ENC_MHSA_QK_T,
            ENC_MHSA_SOFTMAX,
            ENC_MHSA_MULT_V,
            ENC_POST_MHSA_TRANSPOSE_STEP,
            ENC_POST_MHSA_DENSE_AND_INPUT_SUM_STEP,
            MLP_DENSE_1_STEP,
            MLP_DENSE_2_AND_SUM_STEP,
            MLP_HEAD_DENSE_1_STEP,
            MLP_HEAD_DENSE_2_STEP,
            MLP_HEAD_PRE_SOFTMAX_TRANSPOSE_STEP,
            MLP_HEAD_SOFTMAX_STEP,
            INFERENCE_COMPLETE,
            POST_SOFTMAX_DIVIDE_STEP,
            POST_SOFTMAX_AVERAGING_STEP,
            INVALID_INF_STEP = -1
        };

        bool is_idle = true; // Used by Ctrl to know whether CiMs have completed a given inference step. To avoid routing NUM_CIM bits to ctrl, these signals will be daisy-chained ANDed from one CiM to the next... or could just always use CiM #63 as a proxy for all CiM...TBD
        bool compute_in_progress = false; // Used by compute element to notify CiM controller when is in progress
        uint16_t id; // ID of the CiM
        uint16_t gen_reg_16b; // General-purpose register
        uint16_t addr_reg; // General-purpose register used to record the address of the data sent/received on the bus
        uint16_t data_len_reg; // General-purpose register used to record len of data sent/received on the bus
        uint16_t compute_process_cnt; // Counter used to track the progress of the current computation (used to simulate the delay in the computation to match the real hardware)
        uint16_t num_compute_done; // Counter used to track the number of computations done in a given inference step
        float computation_result; // Used to store the result of the computation
        float params[CIM_PARAMS_STORAGE_SIZE_KB / sizeof(float)];
        float intermediate_res[CIM_INT_RES_SIZE_KB / sizeof(float)];
        float prev_softmax_storage[CIM_PREV_SOFTMAX_SIZE_KB / sizeof(float)]; // In ASIC, this should only be synthesized for CiM #0

        STATE cim_state;
        INFERENCE_STEP current_inf_step = CLASS_TOKEN_CONCAT;
        OP prev_bus_op;
        Counter gen_cnt_10b;
        Counter gen_cnt_10b_2;
        Counter bytes_rec_cnt; // Tracks the # of bytes received from the bus that are relevant to me
        Counter bytes_sent_cnt; // Tracks the # of bytes sent to the bus

    public:
        CiM() : id(-1), gen_cnt_10b(10), gen_cnt_10b_2(10), bytes_rec_cnt(8), bytes_sent_cnt(8) {}
        CiM(const int16_t cim_id);
        bool get_is_compute_done();
        int reset();
        int run(struct ext_signals* ext_sigs, Bus* bus);
        void update_compute_process_cnt();
        void ADD(uint16_t in1_addr, uint16_t in2_addr, INPUT_TYPE param_type);
        void DIV(uint16_t num_addr, uint16_t in2, INPUT_TYPE in2_type);
        void LAYERNORM_1ST_HALF(uint16_t input_addr);
        void LAYERNORM_2ND_HALF(uint16_t input_addr, float gamma, float beta);
        void MAC(uint16_t in1_start_addr, uint16_t in2_start_addr, uint16_t len, uint16_t bias_addr, INPUT_TYPE param_type, ACTIVATION activation);
        void SOFTMAX(uint16_t input_addr, uint16_t len);
        void MAX_INDEX(uint16_t input_addr, uint16_t len);
};

#endif //CIM_H
