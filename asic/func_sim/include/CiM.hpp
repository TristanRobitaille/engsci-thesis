#ifndef CIM_H
#define CIM_H

#include <CiM_Compute.hpp>
#include <Misc.hpp>
#include <Memory_Map.hpp>
#include <Compute_Verification.hpp>

/*----- DEFINE -----*/
#define HAS_MY_DATA(x) ((id >= (x)) && (id < (x+3))) // Determines whether the current broadcast transaction contains data I need
#define IS_MY_MATRIX(x) ((id >= (x)*NUM_HEADS) && (id < (x+1)*NUM_HEADS)) // Returns whether a given count corresponds to my matrix (for the *V matmul in encoder's MHSA)

/*----- CLASS -----*/
class CiM : public CiM_Compute {
    private:
        /*----- ENUM -----*/
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
            POST_SOFTMAX_ARGMAX_STEP,
            RETIRE_SOFTMAX_STEP,
            INFERENCE_COMPLETE,
            INVALID_INF_STEP = -1
        };

        /*----- PRIVATE VARIABLES -----*/
        bool is_ready = true; // Signal that CiM can use to override the compute_in_progress signal and force the master to wait before sending the next instruction
        uint16_t id; // ID of the CiM
        uint16_t gen_reg_2b; // General-purpose register
        uint16_t tx_addr_reg; // Record the address of the data sent on the bus
        uint16_t rx_addr_reg; // Record the address of the data received on the bus
        uint16_t sender_id; // Record the id of an instruction's sender at a start of broadcast
        uint16_t data_len_reg; // General-purpose register used to record len of data sent/received on the bus
        uint16_t _softmax_max_index;
        DATA_WIDTH data_width;

        STATE cim_state;
        INFERENCE_STEP current_inf_step;
        Counter gen_cnt_7b;
        Counter gen_cnt_7b_2;
        Counter word_rec_cnt; // Tracks the # of words received from the bus that are relevant to me
        Counter word_snt_cnt; // Tracks the # of words sent to the bus

        void update_state(STATE new_state);
        void load_previous_softmax();
        void overflow_check();

    public:
        CiM() : id(-1), gen_cnt_7b(7), gen_cnt_7b_2(7), word_rec_cnt(7), word_snt_cnt(7) {}
        CiM(const int16_t cim_id);
        int reset();
        int run(struct ext_signals* ext_sigs, Bus* bus);
        bool get_is_ready();
        uint32_t get_softmax_max_index();
};

#endif //CIM_H
