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
        enum STATE {
            IDLE,
            PARAM_LOAD,
            SIGNAL_LOAD,
            INFERENCE_RUNNING,
            BROADCAST_MANAGEMENT,
            RESET,
            INVALID = -1
        };

        enum HIGH_LEVEL_INFERENCE_STEP {
            PRE_LAYERNORM_TRANSPOSE_STEP,
        };

        struct parameters {
            // Patch projection Dense
            PatchProjKernel_t patch_proj_kernel;
            EmbDepthVect_t patch_proj_bias;

            EmbDepthVect_t class_emb; // Classification token embedding
            PosEmb_t pos_emb; // Positional embeddding

            // Encoders
            EncEmbDepthVect2_t enc_layernorm_gamma;
            EncEmbDepthVect2_t enc_layernorm_beta;
            EncEmbDepthMat_t enc_mhsa_Q_kernel;
            EncEmbDepthMat_t enc_mhsa_K_kernel;
            EncEmbDepthMat_t enc_mhsa_V_kernel;
            EncEmbDepthVect_t enc_mhsa_Q_bias;
            EncEmbDepthVect_t enc_mhsa_K_bias;
            EncEmbDepthVect_t enc_mhsa_V_bias;
            EncEmbDepthVect_t enc_mhsa_combine_bias;
            EncEmbDepthMat_t enc_mhsa_combine_kernel;;
            EncEmbDepthMat2_t enc_mlp_dense_kernel;
            EncEmbDepthVect2_t enc_mlp_dense_bias;
        };

        float storage[CENTRALIZED_STORAGE_WEIGHTS_KB / sizeof(float)];
        Counter gen_cnt_8b;
        Counter gen_cnt_10b;
        uint16_t gen_reg_16b;
        STATE state;
        HIGH_LEVEL_INFERENCE_STEP high_level_inf_step = PRE_LAYERNORM_TRANSPOSE_STEP;

        // EEG file
        std::vector<float> eeg_ds;
        std::vector<float>::iterator eeg;

        // Parameters
        PARAM_NAME params_curr_layer = PATCH_PROJ_KERNEL_PARAMS; // Keeps track of which layer of parameters we are sending
        int params_cim_cnt = -1; // Keeps track of current CiM to which we send parameters
        int params_data_cnt = -1; // Keeps track of data element we've sent to current CiM
        struct parameters params;

        int load_params_from_h5(const std::string params_filepath);
        void update_inst_with_params(PARAM_NAME param_name, struct instruction* inst);

    public:
        Master_ctrl(const std::string eeg_filepath, const std::string params_filepath);
        int reset();
        SYSTEM_STATE run(struct ext_signals* ext_sigs, Bus* bus, CiM cims[]);
        int start_signal_load();
        struct instruction param_to_send();
};

#endif //MASTER_CTRL_H