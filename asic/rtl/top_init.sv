`ifndef _top_init_svh_
`define _top_init_svh_

`include "../types.svh"

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

`endif
