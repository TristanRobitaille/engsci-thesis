#ifndef CIM_CENTRALIZED_H
#define CIM_CENTRALIZED_H

#include <CiM_Compute.hpp>
#include <Memory_Map.hpp>
#include <../include/highfive/H5File.hpp>

/*----- CLASS -----*/
class CiM_Centralized : public CiM_Compute {
    private:
        /*----- ENUM -----*/
        enum STATE {
            IDLE_CIM,
            INFERENCE_RUNNING,
            INVALID_CIM
        };

        enum MACOROP {
            MAC_OP,
            DIV_OP,
            ADD_OP
        };

        enum INFERENCE_STEP {
            PATCH_PROJ_STEP,
            CLASS_TOKEN_CONCAT_STEP,
            POS_EMB_STEP,
            ENC_LAYERNORM_1_1ST_HALF_STEP,
            ENC_LAYERNORM_1_2ND_HALF_STEP,
            POS_EMB_COMPRESSION_STEP,
            ENC_MHSA_Q_STEP,
            ENC_MHSA_K_STEP,
            ENC_MHSA_V_STEP,
            ENC_MHSA_QK_T_STEP,
            ENC_MHSA_SOFTMAX_STEP,
            ENC_MHSA_MULT_V_STEP,
            ENC_POST_MHSA_DENSE_AND_INPUT_SUM_STEP,
            ENC_LAYERNORM_2_1ST_HALF_STEP,
            ENC_LAYERNORM_2_2ND_HALF_STEP,
            MLP_DENSE_1_STEP,
            MLP_DENSE_2_AND_SUM_STEP,
            MLP_HEAD_SOFTMAX_STEP,
            ENC_LAYERNORM_3_1ST_HALF_STEP,
            ENC_LAYERNORM_3_2ND_HALF_STEP,
            MLP_HEAD_DENSE_1_STEP,
            MLP_HEAD_DENSE_2_STEP,
            INVALID_STEP
        };

        /*----- PRIVATE VARIABLES -----*/
        bool generic_done;
        STATE cim_state = INVALID_CIM;
        INFERENCE_STEP current_inf_step = INVALID_STEP;
        SYSTEM_STATE system_state;
        Counter gen_cnt_4b;
        Counter gen_cnt_7b;
        Counter gen_cnt_9b;
        MACOROP mac_or_div;
        MACOROP mac_or_add;

        void load_params_from_h5(const std::string params_filepath);
        void load_eeg_from_h5(const std::string eeg_filepath, uint16_t clip_index);
        void load_previous_softmax(const std::string prev_softmax_base_filepath);

    public:
        CiM_Centralized(const std::string params_filepath);
        int reset();
        SYSTEM_STATE run(struct ext_signals* ext_sigs, string softmax_base_filepath, string eeg_filepath, uint16_t clip_index);
};

#endif //CIM_CENTRALIZED_H