#ifndef CIM_H
#define CIM_H

#include <CiM_Compute.hpp>
#include <Misc.hpp>
#include <Param_Layer_Mapping.hpp>
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
            SOFTMAX_AVG_SUM_MEM,
            PREV_SOFTMAX_OUTPUT_MEM
        };

        /*----- MAP -----*/
        const std::map<DATA, uint16_t> mem_map = {
            {PATCH_MEM,                 PATCH_LEN+DOUBLE_WIDTH},
            {CLASS_TOKEN_MEM,           PATCH_LEN},
            {POS_EMB_MEM,               0},
            {ENC_LN1_1ST_HALF_MEM,      DOUBLE_WIDTH*(NUM_PATCHES+1)},
            {ENC_LN1_2ND_HALF_MEM,      DOUBLE_WIDTH*(NUM_PATCHES+1+EMB_DEPTH)},
            {ENC_QVK_IN_MEM,            DOUBLE_WIDTH*(2*(NUM_PATCHES+1)+2*EMB_DEPTH)},
            {ENC_Q_MEM,                 DOUBLE_WIDTH*(2*(NUM_PATCHES+1)+3*EMB_DEPTH)},
            {ENC_K_MEM,                 DOUBLE_WIDTH*(3*(NUM_PATCHES+1)+3*EMB_DEPTH)},
            {ENC_V_MEM,                 NUM_PATCHES+1},
            {ENC_K_T_MEM,               DOUBLE_WIDTH*(EMB_DEPTH)+EMB_DEPTH+NUM_PATCHES+1},
            {ENC_QK_T_IN_MEM,           DOUBLE_WIDTH*(EMB_DEPTH)+2*EMB_DEPTH+NUM_PATCHES+1},
            {ENC_QK_T_MEM,              DOUBLE_WIDTH*(EMB_DEPTH)+2*EMB_DEPTH+2*(NUM_PATCHES+1)}, // This address + NUM_HEADS*(NUM_PATCHES+1) is the determining factor in the total memory requirement
            {ENC_PRE_SOFTMAX_MEM,       DOUBLE_WIDTH*(EMB_DEPTH)+NUM_PATCHES+1},
            {ENC_V_MULT_IN_MEM,         DOUBLE_WIDTH*(EMB_DEPTH)+8*EMB_DEPTH+NUM_PATCHES+1},
            {ENC_V_MULT_MEM,            DOUBLE_WIDTH*(EMB_DEPTH)+9*EMB_DEPTH+NUM_PATCHES+1},
            {ENC_DENSE_IN_MEM,          DOUBLE_WIDTH*(NUM_PATCHES+1+EMB_DEPTH)},
            {ENC_MHSA_OUT_MEM,          DOUBLE_WIDTH*(2*EMB_DEPTH+NUM_PATCHES+1)},
            {ENC_LN2_1ST_HALF_MEM,      DOUBLE_WIDTH*(NUM_PATCHES+1)},
            {ENC_LN2_2ND_HALF_MEM,      DOUBLE_WIDTH*(NUM_PATCHES+1+EMB_DEPTH)},
            {ENC_MLP_IN_MEM,            DOUBLE_WIDTH*(NUM_PATCHES+1)},
            {ENC_MLP_DENSE1_MEM,        DOUBLE_WIDTH*(NUM_PATCHES+1+EMB_DEPTH)},
            {ENC_MLP_DENSE2_IN_MEM,     DOUBLE_WIDTH*(NUM_PATCHES+1)},
            {ENC_MLP_OUT_MEM,           DOUBLE_WIDTH*(3*EMB_DEPTH+NUM_PATCHES+2)},
            {MLP_HEAD_LN_1ST_HALF_MEM,  DOUBLE_WIDTH*(0)},
            {MLP_HEAD_LN_2ND_HALF_MEM,  DOUBLE_WIDTH*(EMB_DEPTH)},
            {MLP_HEAD_DENSE_1_IN_MEM,   DOUBLE_WIDTH*(EMB_DEPTH)},
            {MLP_HEAD_DENSE_1_OUT_MEM,  DOUBLE_WIDTH*(2*EMB_DEPTH)},
            {MLP_HEAD_DENSE_2_IN_MEM,   DOUBLE_WIDTH*(EMB_DEPTH)},
            {MLP_HEAD_DENSE_2_OUT_MEM,  DOUBLE_WIDTH*(2*EMB_DEPTH)},
            {MLP_HEAD_SOFTMAX_IN_MEM,   DOUBLE_WIDTH*(MLP_DIM)},
            {PREV_SOFTMAX_OUTPUT_MEM,   866}, // Only relevant for CiM #0 //FIXME
            {SOFTMAX_AVG_SUM_MEM,       DOUBLE_WIDTH*(2*EMB_DEPTH)}
        };

        /*----- PRIVATE VARIABLES -----*/
        bool is_ready = true; // Signal that CiM can use to override the compute_in_progress signal and force the master to wait before sending the next instruction
        uint16_t id; // ID of the CiM
        uint16_t gen_reg_2b; // General-purpose register
        uint16_t tx_addr_reg; // Record the address of the data sent on the bus
        uint16_t rx_addr_reg; // Record the address of the data received on the bus
        uint16_t sender_id; // Record the id of an instruction's sender at a start of broadcast
        uint16_t data_len_reg; // General-purpose register used to record len of data sent/received on the bus
        uint32_t softmax_max_index;
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
