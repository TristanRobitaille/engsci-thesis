#ifndef MASTER_CTRL_H
#define MASTER_CTRL_H

#include <iostream>
#include <../include/highfive/H5File.hpp>
#include <armadillo>

#include <CiM.hpp>
#include <Misc.hpp>
#include <Param_Layer_Mapping.hpp>

/*----- CLASS -----*/
class Counter;

class Master_Ctrl {
    private:
        enum STATE {
            IDLE,
            PARAM_LOAD,
            SIGNAL_LOAD,
            INFERENCE_RUNNING,
            BROADCAST_MANAGEMENT,
            WAITING_FOR_CIM_COMPLETION,
            RESET,
        };

        enum HIGH_LEVEL_INFERENCE_STEP {
            PRE_LAYERNORM_1_TRANS_STEP,
            INTRA_LAYERNORM_1_TRANS_STEP, // After the first half of transpose (row to column) for LayerNorm 1
            POST_LAYERNORM_1_TRANS_STEP, // After the second half of transpose (column to row) for LayerNorm 1 and final normalization with gamma/beta
            ENC_MHSA_DENSE_STEP, // Perform all three dense operations (Q, K, V) for the Multi-Head Self-Attention
            ENC_MHSA_Q_TRANS_STEP, // Transpose (row to column) for encoder MHSA's Q
            ENC_MHSA_K_TRANS_STEP, // Transpose (row to column) for encoder MHSA's K
            ENC_MHSA_QK_T_STEP, // QK_T multiplication for encoder MHSA
            ENC_MHSA_PRE_SOFTMAX_TRANS_STEP, // Transpose (column to row) for the pre-softmax operation for encoder MHSA
            ENC_MHSA_SOFTMAX_STEP, // Softmax operation for encoder MHSA
            ENC_MHSA_V_MULT_STEP, // Multiplication of softmax with V for encoder MHSA,
            ENC_MHSA_POST_V_TRANS_STEP, // Transpose following the V multiplication for encoder MHSA
            ENC_MHSA_POST_V_DENSE_STEP, // Perform the tranpose dense operation for the encoder's MLP
            PRE_LAYERNORM_2_TRANS_STEP, // Transpose (row to column) for the first half of LayerNorm 2
            INTRA_LAYERNORM_2_TRANS_STEP, // Transpose (column to row) for the second half of LayerNorm 2 and final normalization with gamma/beta
            ENC_PRE_MLP_TRANSPOSE_STEP, // Transpose (column to row) for the encoder's MLP
            ENC_MLP_DENSE_1_STEP, // Perform the first dense operation for the encoder's MLP
            ENC_MLP_DENSE_2_TRANSPOSE_STEP, // Transpose (row to column) for the second dense operation for the encoder's MLP
            ENC_MLP_DENSE_2_AND_SUM_STEP, // Perform the second dense operation for the encoder's MLP and the sum with the encoder's input (residual connection)
            PRE_LAYERNORM_3_TRANS_STEP, // Transpose (row to column) for the first half of LayerNorm 3
            INTRA_LAYERNORM_3_TRANS_STEP, // Transpose (column to row) for the second half of LayerNorm 3 and final normalization with gamma/beta
            PRE_MLP_HEAD_DENSE_TRANS_STEP, // Transpose (column to row) for the MLP head
            MLP_HEAD_DENSE_1_STEP, // Perform the first dense operation for the MLP head
            PRE_MLP_HEAD_DENSE_2_TRANS_STEP, // Transpose (row to column) for the final softmax operation for the MLP head
            MLP_HEAD_DENSE_2_STEP, // Final softmax operation for the MLP head
            MLP_HEAD_SOFTMAX_TRANS_STEP, // Transpose (row to column) for the final softmax operation to be done on CiM #0
            SOFTMAX_AVERAGING, // Give time to CiM to finish averaging after softmax
            INFERENCE_FINISHED
       };

        struct broadcast_op_info {
            OP op;
            float tx_addr;
            float len;
            float rx_addr;
            uint16_t num_cims; // Number of CiMs that will need to send data
        };

        struct parameters {
            // Patch projection Dense
            PatchProjKernel_t patch_proj_kernel;
            EmbDepthVect_t patch_proj_bias;

            EmbDepthVect_t class_emb; // Classification token embedding
            PosEmb_t pos_emb; // Positional embeddding

            // Encoders
            EncEmbDepthVect3_t layernorm_gamma; // Includes MLP head's LayerNorm
            EncEmbDepthVect3_t layernorm_beta; // Includes MLP head's LayerNorm
            EncEmbDepthMat_t enc_mhsa_Q_kernel;
            EncEmbDepthMat_t enc_mhsa_K_kernel;
            EncEmbDepthMat_t enc_mhsa_V_kernel;
            EmbDepthVect_t enc_mhsa_Q_bias;
            EmbDepthVect_t enc_mhsa_K_bias;
            EmbDepthVect_t enc_mhsa_V_bias;
            EmbDepthVect_t enc_mhsa_combine_bias;
            EncEmbDepthMat_t enc_mhsa_combine_kernel;;
            EmbDepthxMlpDimMat_t enc_mlp_dense_1_kernel;
            MlpDimVect_t enc_mlp_dense_1_bias;
            EncMlpDimxEmbDepthMat_t enc_mlp_dense_2_kernel;
            EmbDepthVect_t enc_mlp_dense_2_bias;
            float enc_mhsa_sqrt_num_heads;

            // MLP head
            MlpDimVect_t mlp_head_dense_1_bias;
            EmbDepthxMlpDimMat_t mlp_head_dense_1_kernel;
            NumSleepStagesxMlpDimMat_t mlp_head_dense_2_kernel;
            NumSleepStagesVect_t mlp_head_dense_2_bias;
        };

        const std::map<HIGH_LEVEL_INFERENCE_STEP, broadcast_op_info> broadcast_ops = {
            {PRE_LAYERNORM_1_TRANS_STEP,        {/*op*/ TRANS_BROADCAST_START_OP, /*tx addr*/ 0,                                /*tx len*/ NUM_PATCHES+1,  /*rx addr*/ NUM_PATCHES+1,                  /*num cims*/ NUM_CIM}},
            {INTRA_LAYERNORM_1_TRANS_STEP,      {/*op*/ TRANS_BROADCAST_START_OP, /*tx addr*/ NUM_PATCHES+1,                    /*tx len*/ EMB_DEPTH,      /*rx addr*/ NUM_PATCHES+1+EMB_DEPTH,        /*num cims*/ NUM_PATCHES+1}},
            {POST_LAYERNORM_1_TRANS_STEP,       {/*op*/ TRANS_BROADCAST_START_OP, /*tx addr*/ NUM_PATCHES+1+EMB_DEPTH,          /*tx len*/ NUM_PATCHES+1,  /*rx addr*/ 2*EMB_DEPTH+3*(NUM_PATCHES+1),  /*num cims*/ NUM_CIM}},
            {ENC_MHSA_DENSE_STEP,               {/*op*/ DENSE_BROADCAST_START_OP, /*tx addr*/ 2*EMB_DEPTH+3*(NUM_PATCHES+1),    /*tx len*/ EMB_DEPTH,      /*rx addr*/ NUM_PATCHES+1+EMB_DEPTH,        /*num cims*/ NUM_PATCHES+1}},
            {ENC_MHSA_Q_TRANS_STEP,             {/*op*/ TRANS_BROADCAST_START_OP, /*tx addr*/ 2*EMB_DEPTH+NUM_PATCHES+1,        /*tx len*/ NUM_PATCHES+1,  /*rx addr*/ NUM_PATCHES+1+EMB_DEPTH,        /*num cims*/ NUM_CIM}},
            {ENC_MHSA_K_TRANS_STEP,             {/*op*/ TRANS_BROADCAST_START_OP, /*tx addr*/ 2*(EMB_DEPTH+NUM_PATCHES+1),      /*tx len*/ NUM_PATCHES+1,  /*rx addr*/ 2*EMB_DEPTH+NUM_PATCHES+1,      /*num cims*/ NUM_CIM}},
            {ENC_MHSA_QK_T_STEP,                {/*op*/ DENSE_BROADCAST_START_OP, /*tx addr*/ NUM_PATCHES+1+EMB_DEPTH,          /*tx len*/ NUM_HEADS,      /*rx addr*/ 3*EMB_DEPTH+NUM_PATCHES+1,      /*num cims*/ NUM_PATCHES+1}}, // Will need to call this NUM_HEADS times to go through the Z-stack of the Q and K matrices
            {ENC_MHSA_PRE_SOFTMAX_TRANS_STEP,   {/*op*/ TRANS_BROADCAST_START_OP, /*tx addr*/ 3*EMB_DEPTH+2*(NUM_PATCHES+1),    /*tx len*/ NUM_PATCHES+1,  /*rx addr*/ 2*(EMB_DEPTH+NUM_PATCHES+1),    /*num cims*/ NUM_PATCHES+1}}, // Will need to call this NUM_HEADS times to go through the Z-stack of the Q and K matrices
            {ENC_MHSA_V_MULT_STEP,              {/*op*/ DENSE_BROADCAST_START_OP, /*tx addr*/ 2*(EMB_DEPTH+NUM_PATCHES+1),      /*tx len*/ NUM_PATCHES+1,  /*rx addr*/ NUM_PATCHES+1+EMB_DEPTH,        /*num cims*/ NUM_PATCHES+1}},  // Will need to call this NUM_HEADS times to go through the Z-stack of the QK_T matrices
            {ENC_MHSA_POST_V_TRANS_STEP,        {/*op*/ TRANS_BROADCAST_START_OP, /*tx addr*/ 2*EMB_DEPTH+NUM_PATCHES+1,        /*tx len*/ NUM_PATCHES+1,  /*rx addr*/ NUM_PATCHES+1,                  /*num cims*/ NUM_CIM}},
            {ENC_MHSA_POST_V_DENSE_STEP,        {/*op*/ DENSE_BROADCAST_START_OP, /*tx addr*/ NUM_PATCHES+1,                    /*tx len*/ EMB_DEPTH,      /*rx addr*/ NUM_PATCHES+1+EMB_DEPTH,        /*num cims*/ NUM_PATCHES+1}},
            {PRE_LAYERNORM_2_TRANS_STEP,        {/*op*/ TRANS_BROADCAST_START_OP, /*tx addr*/ 2*EMB_DEPTH+NUM_PATCHES+1,        /*tx len*/ NUM_PATCHES+1,  /*rx addr*/ NUM_PATCHES+1,                  /*num cims*/ NUM_CIM}},
            {INTRA_LAYERNORM_2_TRANS_STEP,      {/*op*/ TRANS_BROADCAST_START_OP, /*tx addr*/ NUM_PATCHES+1,                    /*tx len*/ EMB_DEPTH,      /*rx addr*/ NUM_PATCHES+1+EMB_DEPTH,        /*num cims*/ NUM_PATCHES+1}},
            {ENC_PRE_MLP_TRANSPOSE_STEP,        {/*op*/ TRANS_BROADCAST_START_OP, /*tx addr*/ NUM_PATCHES+1+EMB_DEPTH,          /*tx len*/ NUM_PATCHES+1,  /*rx addr*/ 2*EMB_DEPTH+NUM_PATCHES+1,      /*num cims*/ NUM_CIM}},
            {ENC_MLP_DENSE_1_STEP,              {/*op*/ DENSE_BROADCAST_START_OP, /*tx addr*/ 2*EMB_DEPTH+NUM_PATCHES+1,        /*tx len*/ EMB_DEPTH,      /*rx addr*/ NUM_PATCHES+1,                  /*num cims*/ NUM_CIM}},
            {ENC_MLP_DENSE_2_TRANSPOSE_STEP,    {/*op*/ TRANS_BROADCAST_START_OP, /*tx addr*/ NUM_PATCHES+1+EMB_DEPTH,          /*tx len*/ 1,              /*rx addr*/ NUM_PATCHES+1,                  /*num cims*/ MLP_DIM}}, // Only CiMs 0 to 31 will need to send data (and only their first data since we will only use the first row in the subsequent step)
            {ENC_MLP_DENSE_2_AND_SUM_STEP,      {/*op*/ DENSE_BROADCAST_START_OP, /*tx addr*/ NUM_PATCHES+1,                    /*tx len*/ MLP_DIM,        /*rx addr*/ NUM_PATCHES+1,                  /*num cims*/ 1}}, // Since we will select only the top row to send to MLP head, we only need CiM 0 to send data
            {PRE_LAYERNORM_3_TRANS_STEP,        {/*op*/ TRANS_BROADCAST_START_OP, /*tx addr*/ 2*EMB_DEPTH+NUM_PATCHES+2,        /*tx len*/ 1,              /*rx addr*/ 0,                              /*num cims*/ NUM_CIM}},
            {INTRA_LAYERNORM_3_TRANS_STEP,      {/*op*/ TRANS_BROADCAST_START_OP, /*tx addr*/ 0,                                /*tx len*/ EMB_DEPTH,      /*rx addr*/ EMB_DEPTH,                      /*num cims*/ 1}}, // Since we will select only the top row to send to MLP head, we only need CiM 0 to send data
            {PRE_MLP_HEAD_DENSE_TRANS_STEP,     {/*op*/ TRANS_BROADCAST_START_OP, /*tx addr*/ EMB_DEPTH,                        /*tx len*/ 1,              /*rx addr*/ 0,                              /*num cims*/ EMB_DEPTH}},
            {MLP_HEAD_DENSE_1_STEP,             {/*op*/ DENSE_BROADCAST_START_OP, /*tx addr*/ 0,                                /*tx len*/ EMB_DEPTH,      /*rx addr*/ EMB_DEPTH,                      /*num cims*/ 1}}, // Since we will select only the top row to send to MLP head, we only need CiM 0 to send data
            {PRE_MLP_HEAD_DENSE_2_TRANS_STEP,   {/*op*/ TRANS_BROADCAST_START_OP, /*tx addr*/ 2*EMB_DEPTH,                      /*tx len*/ 1,              /*rx addr*/ 0,                              /*num cims*/ MLP_DIM}}, // Only CiMs 32 to 63 will need to send data
            {MLP_HEAD_DENSE_2_STEP,             {/*op*/ DENSE_BROADCAST_START_OP, /*tx addr*/ MLP_DIM,                          /*tx len*/ MLP_DIM,        /*rx addr*/ EMB_DEPTH,                      /*num cims*/ 1}}, // Since we will select only the top row to send to MLP head, we only need CiM 0 to send data
            {MLP_HEAD_SOFTMAX_TRANS_STEP,       {/*op*/ TRANS_BROADCAST_START_OP, /*tx addr*/ 2*EMB_DEPTH,                      /*tx len*/ 1,              /*rx addr*/ MLP_DIM,                        /*num cims*/ NUM_SLEEP_STAGES}}
        };

        const std::map<HIGH_LEVEL_INFERENCE_STEP, int> num_necessary_idles = {
            {PRE_LAYERNORM_1_TRANS_STEP,        EMB_DEPTH},
            {INTRA_LAYERNORM_1_TRANS_STEP,      NUM_PATCHES+1},
            {POST_LAYERNORM_1_TRANS_STEP,       EMB_DEPTH},
            {ENC_MHSA_DENSE_STEP,               NUM_PATCHES+1},
            {ENC_MHSA_Q_TRANS_STEP,             NUM_PATCHES+1},
            {ENC_MHSA_K_TRANS_STEP,             NUM_PATCHES+1},
            {ENC_MHSA_QK_T_STEP,                NUM_HEADS},
            {ENC_MHSA_PRE_SOFTMAX_TRANS_STEP,   NUM_PATCHES+1},
            {ENC_MHSA_SOFTMAX_STEP,             NUM_PATCHES+1},
            {ENC_MHSA_V_MULT_STEP,              NUM_HEADS},
            {ENC_MHSA_POST_V_TRANS_STEP,        NUM_PATCHES+1},
            {ENC_MHSA_POST_V_DENSE_STEP,        NUM_PATCHES+1},
            {PRE_LAYERNORM_2_TRANS_STEP,        EMB_DEPTH},
            {INTRA_LAYERNORM_2_TRANS_STEP,      NUM_PATCHES+1},
            {ENC_PRE_MLP_TRANSPOSE_STEP,        EMB_DEPTH},
            {ENC_MLP_DENSE_1_STEP,              MLP_DIM},
            {ENC_MLP_DENSE_2_TRANSPOSE_STEP,    MLP_DIM},
            {ENC_MLP_DENSE_2_AND_SUM_STEP,      MLP_DIM},
            {PRE_LAYERNORM_3_TRANS_STEP,        EMB_DEPTH},
            {INTRA_LAYERNORM_3_TRANS_STEP,      1},
            {PRE_MLP_HEAD_DENSE_TRANS_STEP,     EMB_DEPTH},
            {MLP_HEAD_DENSE_1_STEP,             MLP_DIM},
            {PRE_MLP_HEAD_DENSE_2_TRANS_STEP,   EMB_DEPTH},
            {MLP_HEAD_DENSE_2_STEP,             NUM_SLEEP_STAGES},
            {MLP_HEAD_SOFTMAX_TRANS_STEP,       1},
            {SOFTMAX_AVERAGING,                 1},
            {INFERENCE_FINISHED,                1}
        };

        bool all_cims_ready = true;
        Counter gen_cnt_7b;
        Counter gen_cnt_7b_2;
        uint16_t gen_reg_16b = 0;
        uint16_t gen_reg_16b_2 = 0;
        uint16_t gen_reg_16b_3 = 0;
        STATE state;
        HIGH_LEVEL_INFERENCE_STEP high_level_inf_step = PRE_LAYERNORM_1_TRANS_STEP;

        // EEG file
        std::vector<float> eeg_ds;
        std::vector<float>::iterator eeg;

        // Parameters
        PARAM_NAME params_curr_layer = PATCH_PROJ_KERNEL_PARAMS; // Keeps track of which layer of parameters we are sending
        struct parameters params;

        int load_params_from_h5(const std::string params_filepath);
        void update_inst_with_params(PARAM_NAME param_name, struct instruction* inst);

    public:
        Master_Ctrl(const std::string eeg_filepath, const std::string params_filepath);
        int reset();
        SYSTEM_STATE run(struct ext_signals* ext_sigs, Bus* bus, std::vector<CiM> cims);
        int start_signal_load(Bus* bus);
        struct instruction param_to_send();
        int prepare_for_broadcast(broadcast_op_info op_info, Bus* bus);
};

#endif //MASTER_CTRL_H