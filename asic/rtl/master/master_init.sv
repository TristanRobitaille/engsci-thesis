/* INITIALIZE STRUCTS OF MASTER */

initial begin
    param_addr_map[PATCH_PROJ_KERNEL_PARAMS]                            = '{addr: $clog2(PARAMS_STORAGE_SIZE_CIM)'(0),    len: $clog2(NUM_CIMS+1)'(PATCH_LEN),      num_rec: $clog2(NUM_CIMS+1)'(NUM_CIMS)};
    param_addr_map[POS_EMB_PARAMS]                                      = '{addr: $clog2(PARAMS_STORAGE_SIZE_CIM)'(1*64), len: $clog2(NUM_CIMS+1)'(NUM_PATCHES+1),  num_rec: $clog2(NUM_CIMS+1)'(NUM_CIMS)};
    param_addr_map[ENC_Q_DENSE_KERNEL_PARAMS]                           = '{addr: $clog2(PARAMS_STORAGE_SIZE_CIM)'(128),  len: $clog2(NUM_CIMS+1)'(EMB_DEPTH),      num_rec: $clog2(NUM_CIMS+1)'(NUM_CIMS)};
    param_addr_map[ENC_K_DENSE_KERNEL_PARAMS]                           = '{addr: $clog2(PARAMS_STORAGE_SIZE_CIM)'(192),  len: $clog2(NUM_CIMS+1)'(EMB_DEPTH),      num_rec: $clog2(NUM_CIMS+1)'(NUM_CIMS)};
    param_addr_map[ENC_V_DENSE_KERNEL_PARAMS]                           = '{addr: $clog2(PARAMS_STORAGE_SIZE_CIM)'(256),  len: $clog2(NUM_CIMS+1)'(EMB_DEPTH),      num_rec: $clog2(NUM_CIMS+1)'(NUM_CIMS)};
    param_addr_map[ENC_COMB_HEAD_KERNEL_PARAMS]                         = '{addr: $clog2(PARAMS_STORAGE_SIZE_CIM)'(320),  len: $clog2(NUM_CIMS+1)'(EMB_DEPTH),      num_rec: $clog2(NUM_CIMS+1)'(NUM_CIMS)};
    param_addr_map[ENC_MLP_DENSE_1_OR_MLP_HEAD_DENSE_1_KERNEL_PARAMS]   = '{addr: $clog2(PARAMS_STORAGE_SIZE_CIM)'(384),  len: $clog2(NUM_CIMS+1)'(EMB_DEPTH),      num_rec: $clog2(NUM_CIMS+1)'(EMB_DEPTH)}; // The number of rec is EMB_DEPTH, but individually it is MLP_DIM
    param_addr_map[ENC_MLP_DENSE_2_KERNEL_PARAMS]                       = '{addr: $clog2(PARAMS_STORAGE_SIZE_CIM)'(448),  len: $clog2(NUM_CIMS+1)'(MLP_DIM),        num_rec: $clog2(NUM_CIMS+1)'(EMB_DEPTH)};
    param_addr_map[MLP_HEAD_DENSE_2_KERNEL_PARAMS]                      = '{addr: $clog2(PARAMS_STORAGE_SIZE_CIM)'(480),  len: $clog2(NUM_CIMS+1)'(MLP_DIM),        num_rec: $clog2(NUM_CIMS+1)'(NUM_SLEEP_STAGES)};
    param_addr_map[SINGLE_PARAMS]                                       = '{addr: $clog2(PARAMS_STORAGE_SIZE_CIM)'(8*64), len: $clog2(NUM_CIMS+1)'(16),             num_rec: $clog2(NUM_CIMS+1)'(NUM_CIMS)};
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
