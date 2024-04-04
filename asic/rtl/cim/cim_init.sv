/* Initialize struct of CiM. To be called inside CiM module */
`ifndef _cim_init_sv_
`define _cim_init_sv_

initial begin
    mem_map[PATCH_MEM]                  = $clog2(TEMP_RES_STORAGE_SIZE_CIM)'(PATCH_LEN+'d1);
    mem_map[CLASS_TOKEN_MEM]            = $clog2(TEMP_RES_STORAGE_SIZE_CIM)'(PATCH_LEN);
    mem_map[POS_EMB_MEM]                = $clog2(TEMP_RES_STORAGE_SIZE_CIM)'('d0);
    mem_map[ENC_LN1_1ST_HALF_MEM]       = $clog2(TEMP_RES_STORAGE_SIZE_CIM)'(NUM_PATCHES+1);
    mem_map[ENC_LN1_2ND_HALF_MEM]       = $clog2(TEMP_RES_STORAGE_SIZE_CIM)'(NUM_PATCHES+1+EMB_DEPTH);
    mem_map[ENC_QVK_IN_MEM]             = $clog2(TEMP_RES_STORAGE_SIZE_CIM)'(NUM_PATCHES+1+EMB_DEPTH);
    mem_map[ENC_Q_MEM]                  = $clog2(TEMP_RES_STORAGE_SIZE_CIM)'(2*EMB_DEPTH+NUM_PATCHES+1);
    mem_map[ENC_K_MEM]                  = $clog2(TEMP_RES_STORAGE_SIZE_CIM)'(2*(EMB_DEPTH+NUM_PATCHES+1));
    mem_map[ENC_V_MEM]                  = $clog2(TEMP_RES_STORAGE_SIZE_CIM)'(NUM_PATCHES+1);
    mem_map[ENC_K_T_MEM]                = $clog2(TEMP_RES_STORAGE_SIZE_CIM)'(2*EMB_DEPTH+NUM_PATCHES+1);
    mem_map[ENC_QK_T_IN_MEM]            = $clog2(TEMP_RES_STORAGE_SIZE_CIM)'(3*EMB_DEPTH+NUM_PATCHES+1);
    mem_map[ENC_QK_T_MEM]               = $clog2(TEMP_RES_STORAGE_SIZE_CIM)'(3*EMB_DEPTH+2*(NUM_PATCHES+1));
    mem_map[ENC_PRE_SOFTMAX_MEM]        = $clog2(TEMP_RES_STORAGE_SIZE_CIM)'(2*(EMB_DEPTH+NUM_PATCHES+1));
    mem_map[ENC_V_MULT_IN_MEM]          = $clog2(TEMP_RES_STORAGE_SIZE_CIM)'(NUM_PATCHES+1+EMB_DEPTH);
    mem_map[ENC_V_MULT_MEM]             = $clog2(TEMP_RES_STORAGE_SIZE_CIM)'(2*EMB_DEPTH+NUM_PATCHES+1);
    mem_map[ENC_DENSE_IN_MEM]           = $clog2(TEMP_RES_STORAGE_SIZE_CIM)'(NUM_PATCHES+1+EMB_DEPTH);
    mem_map[ENC_MHSA_OUT_MEM]           = $clog2(TEMP_RES_STORAGE_SIZE_CIM)'(2*EMB_DEPTH+NUM_PATCHES+1);
    mem_map[ENC_LN2_1ST_HALF_MEM]       = $clog2(TEMP_RES_STORAGE_SIZE_CIM)'(NUM_PATCHES+1);
    mem_map[ENC_LN2_2ND_HALF_MEM]       = $clog2(TEMP_RES_STORAGE_SIZE_CIM)'(NUM_PATCHES+1+EMB_DEPTH);
    mem_map[ENC_MLP_IN_MEM]             = $clog2(TEMP_RES_STORAGE_SIZE_CIM)'(NUM_PATCHES+1);
    mem_map[ENC_MLP_DENSE1_MEM]         = $clog2(TEMP_RES_STORAGE_SIZE_CIM)'(NUM_PATCHES+1+EMB_DEPTH);
    mem_map[ENC_MLP_DENSE2_IN_MEM]      = $clog2(TEMP_RES_STORAGE_SIZE_CIM)'(NUM_PATCHES+1);
    mem_map[ENC_MLP_OUT_MEM]            = $clog2(TEMP_RES_STORAGE_SIZE_CIM)'(3*EMB_DEPTH+NUM_PATCHES+2);
    mem_map[MLP_HEAD_LN_1ST_HALF_MEM]   = $clog2(TEMP_RES_STORAGE_SIZE_CIM)'(0);
    mem_map[MLP_HEAD_LN_2ND_HALF_MEM]   = $clog2(TEMP_RES_STORAGE_SIZE_CIM)'(EMB_DEPTH);
    mem_map[MLP_HEAD_DENSE_1_IN_MEM]    = $clog2(TEMP_RES_STORAGE_SIZE_CIM)'(EMB_DEPTH);
    mem_map[MLP_HEAD_DENSE_1_OUT_MEM]   = $clog2(TEMP_RES_STORAGE_SIZE_CIM)'(2*EMB_DEPTH);
    mem_map[MLP_HEAD_DENSE_2_IN_MEM]    = $clog2(TEMP_RES_STORAGE_SIZE_CIM)'(EMB_DEPTH);
    mem_map[MLP_HEAD_DENSE_2_OUT_MEM]   = $clog2(TEMP_RES_STORAGE_SIZE_CIM)'(2*EMB_DEPTH);
    mem_map[MLP_HEAD_SOFTMAX_IN_MEM]    = $clog2(TEMP_RES_STORAGE_SIZE_CIM)'(MLP_DIM);
    mem_map[PREV_SOFTMAX_OUTPUT_MEM]    = $clog2(TEMP_RES_STORAGE_SIZE_CIM)'(836); // Only relevant for CiM #0
    mem_map[SOFTMAX_AVG_SUM_MEM]        = $clog2(TEMP_RES_STORAGE_SIZE_CIM)'(2*EMB_DEPTH);
end
`endif
