#ifndef CIM_H
#define CIM_H

#include <iostream>

#include <Misc.hpp>
#include <Param_Layer_Mapping.hpp>

/*----- DEFINE -----*/
#define CIM_PARAMS_STORAGE_SIZE_KB 3072
#define CIM_INT_RES_SIZE_KB 3328

/*----- CLASS -----*/
class CiM {
    private:
        enum STATE {
            IDLE_CIM,
            RESET_CIM,
            INFERENCE_RUNNING_CIM,
            INVALID_CIM = -1
        };

        enum INFERENCE_STEP {
            CLASS_TOKEN_CONCAT,
            POS_EMB,
            ENC_LAYERNORM_1ST_HALF,
            ENC_LAYERNORM_2ND_HALF,
            ENC_MHSA_DENSE,
            ENC_MHSA_QK_T,
            ENC_MHSA_SOFTMAX,
            ENC_MHSA_MULT_V,
            INVALID_INF_STEP = -1
        };

        bool is_idle = true; // Used by Ctrl to know whether CiMs have completed a given inference step. To avoid routing NUM_CIM bits to ctrl, these signals will be daisy-chained ANDed from one CiM to the next... or could just always use CiM #63 as a proxy for all CiM...TBD
        bool compute_in_progress = false; // Used by compute element to notify CiM controller when is in progress
        bool compute_done = false; // Used by compute element to notify CiM controller when is done
        int16_t id; // ID of the CiM
        int16_t gen_reg_16b; // General-purpose register
        int16_t gen_reg_16b_2; // General-purpose register
        float params[CIM_PARAMS_STORAGE_SIZE_KB / sizeof(float)];
        float intermediate_res[CIM_INT_RES_SIZE_KB / sizeof(float)];

        STATE state;
        INFERENCE_STEP current_inf_step = CLASS_TOKEN_CONCAT;
        OP prev_bus_op;
        Counter gen_cnt_10b;
        Counter gen_cnt_10b_2;

    public:
        CiM() : id(-1), gen_cnt_10b(10), gen_cnt_10b_2(10) {}
        CiM(const int16_t cim_id);
        bool get_is_idle();
        int reset();
        int run(struct ext_signals* ext_sigs, Bus* bus);
        float ADD(uint16_t input_addr, uint16_t params_addr);
        float DIV(uint16_t num_addr, uint16_t den_addr);
        void LAYERNORM_1ST_HALF(uint16_t input_addr);
        void LAYERNORM_2ND_HALF(uint16_t input_addr, float gamma, float beta);
        float MAC(uint16_t input_start_addr, uint16_t params_start_addr, uint16_t len, INPUT_TYPE param_type);
        void SOFTMAX(uint16_t input_addr, uint16_t len);
};

#endif //CIM_H
