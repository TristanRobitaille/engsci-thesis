#ifndef MASTER_CTRL_H
#define MASTER_CTRL_H

#include <iostream>
#include <../include/highfive/H5File.hpp>
#include <fmt/core.h>

#include <CiM.hpp>
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
            // Patch projection Dense
            std::array<std::array<float, EMBEDDING_DEPTH>, PATCH_LENGTH_NUM_SAMPLES> patch_proj_kernel;
            std::array<float, EMBEDDING_DEPTH> patch_proj_bias;

            std::array<float, EMBEDDING_DEPTH> class_emb; // Classification token embedding
            std::array<std::array<float, EMBEDDING_DEPTH>, NUM_PATCHES+1> pos_emb; // Positional embeddding

            // Encoders
            std::array<std::array<std::array<float, EMBEDDING_DEPTH>, 2>, NUM_ENCODERS> enc_layer_norm_gamma;
            std::array<std::array<std::array<float, EMBEDDING_DEPTH>, 2>, NUM_ENCODERS> enc_layer_norm_beta;
        };

        float storage[CENTRALIZED_STORAGE_WEIGHTS_KB / sizeof(float)];
        Counter gen_cnt_8b;
        Counter gen_cnt_10b;
        state state;

        // EEG file
        std::vector<float> eeg_ds;
        std::vector<float>::iterator eeg;

        // Parameters
        PARAM_NAMES params_curr_layer = PATCH_PROJ_KERNEL_PARAMS; // Keeps track of which layer of parameters we are sending
        int params_cim_cnt = -1; // Keeps track of current CiM to which we send parameters
        int params_data_cnt = -1; // Keeps track of data element we've sent to current CiM
        struct parameters params;

        int load_params_from_h5(const std::string params_filepath);

    public:
        Master_ctrl(const std::string eeg_filepath, const std::string params_filepath);
        int reset();
        SYSTEM_STATE run(struct ext_signals* ext_sigs, Bus* bus, CiM cims[]);
        int start_signal_load();
        struct instruction param_to_send();
};

#endif //MASTER_CTRL_H