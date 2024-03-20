/* Initialize struct of Master. To be called inside Master module */

initial begin
    broadcast_ops[PRE_LAYERNORM_1_TRANS_STEP]       = '{op: TRANS_BROADCAST_START_OP,   tx_addr: $clog2(TEMP_RES_STORAGE_SIZE_CIM)'(0),                             len: $clog2(NUM_CIMS+1)'(NUM_PATCHES+1),    rx_addr: $clog2(TEMP_RES_STORAGE_SIZE_CIM)'(NUM_PATCHES+1),                 num_cim: $clog2(NUM_CIMS+1)'(NUM_CIMS)};
    broadcast_ops[INTRA_LAYERNORM_1_TRANS_STEP]     = '{op: TRANS_BROADCAST_START_OP,   tx_addr: $clog2(TEMP_RES_STORAGE_SIZE_CIM)'(NUM_PATCHES+1),                 len: $clog2(NUM_CIMS+1)'(EMB_DEPTH),        rx_addr: $clog2(TEMP_RES_STORAGE_SIZE_CIM)'(NUM_PATCHES+1+EMB_DEPTH),       num_cim: $clog2(NUM_CIMS+1)'(NUM_PATCHES+1)};
    broadcast_ops[POST_LAYERNORM_1_TRANS_STEP]      = '{op: TRANS_BROADCAST_START_OP,   tx_addr: $clog2(TEMP_RES_STORAGE_SIZE_CIM)'(NUM_PATCHES+1+EMB_DEPTH),       len: $clog2(NUM_CIMS+1)'(NUM_PATCHES+1),    rx_addr: $clog2(TEMP_RES_STORAGE_SIZE_CIM)'(2*EMB_DEPTH+3*(NUM_PATCHES+1)), num_cim: $clog2(NUM_CIMS+1)'(NUM_CIMS)};
    broadcast_ops[ENC_MHSA_DENSE_STEP]              = '{op: DENSE_BROADCAST_START_OP,   tx_addr: $clog2(TEMP_RES_STORAGE_SIZE_CIM)'(2*EMB_DEPTH+3*(NUM_PATCHES+1)), len: $clog2(NUM_CIMS+1)'(EMB_DEPTH),        rx_addr: $clog2(TEMP_RES_STORAGE_SIZE_CIM)'(NUM_PATCHES+1+EMB_DEPTH),       num_cim: $clog2(NUM_CIMS+1)'(NUM_PATCHES+1)};
    broadcast_ops[ENC_MHSA_Q_TRANS_STEP]            = '{op: TRANS_BROADCAST_START_OP,   tx_addr: $clog2(TEMP_RES_STORAGE_SIZE_CIM)'(2*EMB_DEPTH+NUM_PATCHES+1),     len: $clog2(NUM_CIMS+1)'(NUM_PATCHES+1),    rx_addr: $clog2(TEMP_RES_STORAGE_SIZE_CIM)'(NUM_PATCHES+1+EMB_DEPTH),       num_cim: $clog2(NUM_CIMS+1)'(NUM_CIMS)};
    broadcast_ops[ENC_MHSA_K_TRANS_STEP]            = '{op: TRANS_BROADCAST_START_OP,   tx_addr: $clog2(TEMP_RES_STORAGE_SIZE_CIM)'(2*(EMB_DEPTH+NUM_PATCHES+1)),   len: $clog2(NUM_CIMS+1)'(NUM_PATCHES+1),    rx_addr: $clog2(TEMP_RES_STORAGE_SIZE_CIM)'(2*EMB_DEPTH+NUM_PATCHES+1),     num_cim: $clog2(NUM_CIMS+1)'(NUM_CIMS)};
    broadcast_ops[ENC_MHSA_QK_T_STEP]               = '{op: DENSE_BROADCAST_START_OP,   tx_addr: $clog2(TEMP_RES_STORAGE_SIZE_CIM)'(NUM_PATCHES+1+EMB_DEPTH),       len: $clog2(NUM_CIMS+1)'(NUM_HEADS),        rx_addr: $clog2(TEMP_RES_STORAGE_SIZE_CIM)'(3*EMB_DEPTH+NUM_PATCHES+1),     num_cim: $clog2(NUM_CIMS+1)'(NUM_PATCHES+1)};
    broadcast_ops[ENC_MHSA_PRE_SOFTMAX_TRANS_STEP]  = '{op: TRANS_BROADCAST_START_OP,   tx_addr: $clog2(TEMP_RES_STORAGE_SIZE_CIM)'(3*EMB_DEPTH+2*(NUM_PATCHES+1)), len: $clog2(NUM_CIMS+1)'(NUM_PATCHES+1),    rx_addr: $clog2(TEMP_RES_STORAGE_SIZE_CIM)'(2*(EMB_DEPTH+NUM_PATCHES+1)),   num_cim: $clog2(NUM_CIMS+1)'(NUM_PATCHES+1)};
    broadcast_ops[ENC_MHSA_SOFTMAX_STEP]            = '{op: NOP,                        tx_addr: $clog2(TEMP_RES_STORAGE_SIZE_CIM)'(0),                             len: $clog2(NUM_CIMS+1)'(0),                rx_addr: $clog2(TEMP_RES_STORAGE_SIZE_CIM)'(0),                             num_cim: $clog2(NUM_CIMS+1)'(0)}; // Dummy step. Not ran.    
    broadcast_ops[ENC_MHSA_V_MULT_STEP]             = '{op: DENSE_BROADCAST_START_OP,   tx_addr: $clog2(TEMP_RES_STORAGE_SIZE_CIM)'(2*(EMB_DEPTH+NUM_PATCHES+1)),   len: $clog2(NUM_CIMS+1)'(NUM_PATCHES+1),    rx_addr: $clog2(TEMP_RES_STORAGE_SIZE_CIM)'(NUM_PATCHES+1+EMB_DEPTH),       num_cim: $clog2(NUM_CIMS+1)'(NUM_PATCHES+1)};
    broadcast_ops[ENC_MHSA_POST_V_TRANS_STEP]       = '{op: TRANS_BROADCAST_START_OP,   tx_addr: $clog2(TEMP_RES_STORAGE_SIZE_CIM)'(2*EMB_DEPTH+NUM_PATCHES+1),     len: $clog2(NUM_CIMS+1)'(NUM_PATCHES+1),    rx_addr: $clog2(TEMP_RES_STORAGE_SIZE_CIM)'(NUM_PATCHES+1),                 num_cim: $clog2(NUM_CIMS+1)'(NUM_CIMS)};
    broadcast_ops[ENC_MHSA_POST_V_DENSE_STEP]       = '{op: DENSE_BROADCAST_START_OP,   tx_addr: $clog2(TEMP_RES_STORAGE_SIZE_CIM)'(NUM_PATCHES+1),                 len: $clog2(NUM_CIMS+1)'(EMB_DEPTH),        rx_addr: $clog2(TEMP_RES_STORAGE_SIZE_CIM)'(NUM_PATCHES+1+EMB_DEPTH),       num_cim: $clog2(NUM_CIMS+1)'(NUM_PATCHES+1)};
    broadcast_ops[PRE_LAYERNORM_2_TRANS_STEP]       = '{op: TRANS_BROADCAST_START_OP,   tx_addr: $clog2(TEMP_RES_STORAGE_SIZE_CIM)'(2*EMB_DEPTH+NUM_PATCHES+1),     len: $clog2(NUM_CIMS+1)'(NUM_PATCHES+1),    rx_addr: $clog2(TEMP_RES_STORAGE_SIZE_CIM)'(NUM_PATCHES+1),                 num_cim: $clog2(NUM_CIMS+1)'(NUM_CIMS)};
    broadcast_ops[INTRA_LAYERNORM_2_TRANS_STEP]     = '{op: TRANS_BROADCAST_START_OP,   tx_addr: $clog2(TEMP_RES_STORAGE_SIZE_CIM)'(NUM_PATCHES+1),                 len: $clog2(NUM_CIMS+1)'(EMB_DEPTH),        rx_addr: $clog2(TEMP_RES_STORAGE_SIZE_CIM)'(NUM_PATCHES+1+EMB_DEPTH),       num_cim: $clog2(NUM_CIMS+1)'(NUM_PATCHES+1)};
    broadcast_ops[ENC_PRE_MLP_TRANSPOSE_STEP]       = '{op: TRANS_BROADCAST_START_OP,   tx_addr: $clog2(TEMP_RES_STORAGE_SIZE_CIM)'(NUM_PATCHES+1+EMB_DEPTH),       len: $clog2(NUM_CIMS+1)'(NUM_PATCHES+1),    rx_addr: $clog2(TEMP_RES_STORAGE_SIZE_CIM)'(2*EMB_DEPTH+NUM_PATCHES+1),     num_cim: $clog2(NUM_CIMS+1)'(NUM_CIMS)};
    broadcast_ops[ENC_MLP_DENSE_1_STEP]             = '{op: DENSE_BROADCAST_START_OP,   tx_addr: $clog2(TEMP_RES_STORAGE_SIZE_CIM)'(2*EMB_DEPTH+NUM_PATCHES+1),     len: $clog2(NUM_CIMS+1)'(EMB_DEPTH),        rx_addr: $clog2(TEMP_RES_STORAGE_SIZE_CIM)'(NUM_PATCHES+1),                 num_cim: $clog2(NUM_CIMS+1)'(NUM_CIMS)};
    broadcast_ops[ENC_MLP_DENSE_2_TRANSPOSE_STEP]   = '{op: TRANS_BROADCAST_START_OP,   tx_addr: $clog2(TEMP_RES_STORAGE_SIZE_CIM)'(NUM_PATCHES+1+EMB_DEPTH),       len: $clog2(NUM_CIMS+1)'(1),                rx_addr: $clog2(TEMP_RES_STORAGE_SIZE_CIM)'(NUM_PATCHES+1),                 num_cim: $clog2(NUM_CIMS+1)'(MLP_DIM)};
    broadcast_ops[ENC_MLP_DENSE_2_AND_SUM_STEP]     = '{op: DENSE_BROADCAST_START_OP,   tx_addr: $clog2(TEMP_RES_STORAGE_SIZE_CIM)'(NUM_PATCHES+1),                 len: $clog2(NUM_CIMS+1)'(MLP_DIM),          rx_addr: $clog2(TEMP_RES_STORAGE_SIZE_CIM)'(NUM_PATCHES+1),                 num_cim: $clog2(NUM_CIMS+1)'(1)};
    broadcast_ops[PRE_LAYERNORM_3_TRANS_STEP]       = '{op: TRANS_BROADCAST_START_OP,   tx_addr: $clog2(TEMP_RES_STORAGE_SIZE_CIM)'(2*EMB_DEPTH+NUM_PATCHES+2),     len: $clog2(NUM_CIMS+1)'(1),                rx_addr: $clog2(TEMP_RES_STORAGE_SIZE_CIM)'(0),                             num_cim: $clog2(NUM_CIMS+1)'(NUM_CIMS)};
    broadcast_ops[INTRA_LAYERNORM_3_TRANS_STEP]     = '{op: TRANS_BROADCAST_START_OP,   tx_addr: $clog2(TEMP_RES_STORAGE_SIZE_CIM)'(0),                             len: $clog2(NUM_CIMS+1)'(EMB_DEPTH),        rx_addr: $clog2(TEMP_RES_STORAGE_SIZE_CIM)'(EMB_DEPTH),                     num_cim: $clog2(NUM_CIMS+1)'(1)};
    broadcast_ops[PRE_MLP_HEAD_DENSE_TRANS_STEP]    = '{op: TRANS_BROADCAST_START_OP,   tx_addr: $clog2(TEMP_RES_STORAGE_SIZE_CIM)'(EMB_DEPTH),                     len: $clog2(NUM_CIMS+1)'(1),                rx_addr: $clog2(TEMP_RES_STORAGE_SIZE_CIM)'(0),                             num_cim: $clog2(NUM_CIMS+1)'(EMB_DEPTH)};
    broadcast_ops[MLP_HEAD_DENSE_1_STEP]            = '{op: DENSE_BROADCAST_START_OP,   tx_addr: $clog2(TEMP_RES_STORAGE_SIZE_CIM)'(0),                             len: $clog2(NUM_CIMS+1)'(EMB_DEPTH),        rx_addr: $clog2(TEMP_RES_STORAGE_SIZE_CIM)'(EMB_DEPTH),                     num_cim: $clog2(NUM_CIMS+1)'(1)};

    broadcast_ops[PRE_MLP_HEAD_DENSE_2_TRANS_STEP]  = '{op: TRANS_BROADCAST_START_OP,   tx_addr: $clog2(TEMP_RES_STORAGE_SIZE_CIM)'(2*EMB_DEPTH),                   len: $clog2(NUM_CIMS+1)'(1),                rx_addr: $clog2(TEMP_RES_STORAGE_SIZE_CIM)'(0),                             num_cim: $clog2(NUM_CIMS+1)'(MLP_DIM)};
    broadcast_ops[MLP_HEAD_DENSE_2_STEP]            = '{op: DENSE_BROADCAST_START_OP,   tx_addr: $clog2(TEMP_RES_STORAGE_SIZE_CIM)'(MLP_DIM),                       len: $clog2(NUM_CIMS+1)'(MLP_DIM),          rx_addr: $clog2(TEMP_RES_STORAGE_SIZE_CIM)'(EMB_DEPTH),                     num_cim: $clog2(NUM_CIMS+1)'(1)};
    broadcast_ops[MLP_HEAD_SOFTMAX_TRANS_STEP]      = '{op: TRANS_BROADCAST_START_OP,   tx_addr: $clog2(TEMP_RES_STORAGE_SIZE_CIM)'(2*EMB_DEPTH),                   len: $clog2(NUM_CIMS+1)'(1),                rx_addr: $clog2(TEMP_RES_STORAGE_SIZE_CIM)'(MLP_DIM),                       num_cim: $clog2(NUM_CIMS+1)'(NUM_SLEEP_STAGES)};
end

// Listed here are the rows of the external memory that each parameter is stored in (each row is 64 bits wide)
initial begin
    ext_mem_param_addr_map[PATCH_PROJ_KERNEL_EXT_MEM]               = $clog2(NUM_PARAMS/64)'(0);
    ext_mem_param_addr_map[PATCH_PROJ_BIAS_EXT_MEM]                 = $clog2(NUM_PARAMS/64)'(478);
    ext_mem_param_addr_map[POS_EMB_EXT_MEM]                         = $clog2(NUM_PARAMS/64)'(64);
    ext_mem_param_addr_map[ENC_Q_DENSE_KERNEL_EXT_MEM]              = $clog2(NUM_PARAMS/64)'(125);
    ext_mem_param_addr_map[ENC_Q_DENSE_BIAS_EXT_MEM]                = $clog2(NUM_PARAMS/64)'(479);
    ext_mem_param_addr_map[ENC_K_DENSE_KERNEL_EXT_MEM]              = $clog2(NUM_PARAMS/64)'(189);
    ext_mem_param_addr_map[ENC_K_DENSE_BIAS_EXT_MEM]                = $clog2(NUM_PARAMS/64)'(480);
    ext_mem_param_addr_map[ENC_V_DENSE_KERNEL_EXT_MEM]              = $clog2(NUM_PARAMS/64)'(253);
    ext_mem_param_addr_map[ENC_V_DENSE_BIAS_EXT_MEM]                = $clog2(NUM_PARAMS/64)'(481);
    ext_mem_param_addr_map[ENC_COMB_HEAD_KERNEL_EXT_MEM]            = $clog2(NUM_PARAMS/64)'(317);
    ext_mem_param_addr_map[ENC_COMB_HEAD_BIAS_EXT_MEM]              = $clog2(NUM_PARAMS/64)'(482);
    ext_mem_param_addr_map[MLP_HEAD_DENSE_1_KERNEL_EXT_MEM]         = $clog2(NUM_PARAMS/64)'(381); // Note: Offset by MLP_DIM
    ext_mem_param_addr_map[MLP_HEAD_DENSE_1_BIAS_EXT_MEM]           = $clog2(NUM_PARAMS/64)'(483);
    ext_mem_param_addr_map[MLP_DENSE_2_KERNEL_EXT_MEM]              = $clog2(NUM_PARAMS/64)'(445);
    ext_mem_param_addr_map[MLP_DENSE_1_BIAS_EXT_MEM]                = $clog2(NUM_PARAMS/64)'(491); // Note: Offset by MLP_DIM
    ext_mem_param_addr_map[MLP_DENSE_1_KERNEL_EXT_MEM]              = $clog2(NUM_PARAMS/64)'(381);
    ext_mem_param_addr_map[MLP_DENSE_2_BIAS_EXT_MEM]                = $clog2(NUM_PARAMS/64)'(484);
    ext_mem_param_addr_map[CLASS_EMB_EXT_MEM]                       = $clog2(NUM_PARAMS/64)'(477);
    ext_mem_param_addr_map[ENC_LAYERNORM_1_BETA_EXT_MEM]            = $clog2(NUM_PARAMS/64)'(485);
    ext_mem_param_addr_map[ENC_LAYERNORM_1_GAMMA_EXT_MEM]           = $clog2(NUM_PARAMS/64)'(486);
    ext_mem_param_addr_map[ENC_LAYERNORM_2_BETA_EXT_MEM]            = $clog2(NUM_PARAMS/64)'(487);
    ext_mem_param_addr_map[ENC_LAYERNORM_2_GAMMA_EXT_MEM]           = $clog2(NUM_PARAMS/64)'(488);
    ext_mem_param_addr_map[MLP_HEAD_LAYERNORM_BETA_EXT_MEM]         = $clog2(NUM_PARAMS/64)'(489);
    ext_mem_param_addr_map[MLP_HEAD_LAYERNORM_GAMMA_EXT_MEM]        = $clog2(NUM_PARAMS/64)'(490);
    ext_mem_param_addr_map[MLP_HEAD_DENSE_SOFTMAX_KERNEL_EXT_MEM]   = $clog2(NUM_PARAMS/64)'(492);
    ext_mem_param_addr_map[MLP_HEAD_DENSE_SOFTMAX_BIAS_EXT_MEM]     = $clog2(NUM_PARAMS/64)'(483); // Note: Offset by MLP_DIM
    ext_mem_param_addr_map[SQRT_NUM_HEAD_EXT_MEM]                   = $clog2(NUM_PARAMS/64)'(483); // Note: Offset by MLP_DIM + NUM_SLEEP_STAGES
end
