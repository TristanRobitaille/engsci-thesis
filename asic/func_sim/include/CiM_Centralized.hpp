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

        enum INFERENCE_STEP {
            PATCH_PROJ_STEP,
            CLASS_TOKEN_CONCAT_STEP,
            POS_EMB_STEP,
            ENC_LAYERNORM_1_STEP,
            INVALID_STEP
        };

        /*----- PRIVATE VARIABLES -----*/
        STATE cim_state = INVALID_CIM;
        INFERENCE_STEP current_inf_step = INVALID_STEP;
        SYSTEM_STATE system_state;
        Counter gen_cnt_7b;
        Counter gen_cnt_7b_2;

        void load_params_from_h5(const std::string params_filepath);
        void load_eeg_from_h5(const std::string eeg_filepath, uint16_t clip_index);
        void load_previous_softmax(const std::string prev_softmax_base_filepath);

    public:
        CiM_Centralized(const std::string params_filepath);
        int reset();
        SYSTEM_STATE run(struct ext_signals* ext_sigs, string softmax_base_filepath, string eeg_filepath, uint16_t clip_index);
};

#endif //CIM_CENTRALIZED_H