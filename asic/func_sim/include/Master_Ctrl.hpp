#ifndef MASTER_CTRL_H
#define MASTER_CTRL_H

#include <iostream>
#include <../include/highfive/H5File.hpp>

#include <Misc.hpp>
#include <Param_Layer_Mapping.hpp>

/*----- DEFINE -----*/
#define CENTRALIZED_STORAGE_WEIGHTS_KB 2048

/*----- CLASS -----*/
class Counter;

class Master_ctrl {
    private:
        enum state {
            IDLE,
            PARAM_LOAD,
            SIGNAL_LOAD,
            INFERENCE_RUNNING,
            RESET,
            INVALID = -1
        };

        struct parameters {
            std::vector<float> class_emb[64];
            std::vector<std::vector<float>> pos_emb[64][61];
        };

        float storage[CENTRALIZED_STORAGE_WEIGHTS_KB / sizeof(float)];
        Counter gen_cnt_8b;
        Counter gen_cnt_10b;
        state state;

        // EEG file
        std::vector<float> eeg_ds;
        std::vector<float>::iterator eeg; 
        
        // Parameters
        int params_curr_layer = PATCH_PROJ_DENSE_KERNEL; // Keeps track of which layer of parameters we are sending
        int params_cim_cnt = -1; // Keeps track of current CiM to which we send parameters
        int params_data_cnt = 0; // Keeps track of data element we've sent to current CiM
        HighFive::File params_file;
        struct parameters params;

        int broadcast_inst(struct instruction);

    public:
        Master_ctrl(const std::string eeg_filepath, const std::string params_filepath);
        int reset();
        system_state run(struct ext_signals* ext_sigs, Bus* bus);
        int start_signal_load();
        struct instruction param_to_send();
};

#endif //MASTER_CTRL_H