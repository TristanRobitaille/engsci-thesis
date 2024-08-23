`ifndef _top_init_svh_
`define _top_init_svh_

`include "types.svh"

const ParamInfo_t param_addr_map[10] = '{ 
    {/*addr*/ $clog2(PARAMS_STORAGE_SIZE_CIM)'(1*64),   /*len*/ $clog2(NUM_CIMS+1)'(NUM_PATCHES+1), /*num_rec*/ $clog2(NUM_CIMS+1)'(NUM_CIMS)},
    {/*addr*/ $clog2(PARAMS_STORAGE_SIZE_CIM)'(0),      /*len*/ $clog2(NUM_CIMS+1)'(PATCH_LEN),     /*num_rec*/ $clog2(NUM_CIMS+1)'(NUM_CIMS)},
    {/*addr*/ $clog2(PARAMS_STORAGE_SIZE_CIM)'(2*64),   /*len*/ $clog2(NUM_CIMS+1)'(EMB_DEPTH),     /*num_rec*/ $clog2(NUM_CIMS+1)'(NUM_CIMS)},
    {/*addr*/ $clog2(PARAMS_STORAGE_SIZE_CIM)'(3*64),   /*len*/ $clog2(NUM_CIMS+1)'(EMB_DEPTH),     /*num_rec*/ $clog2(NUM_CIMS+1)'(NUM_CIMS)},
    {/*addr*/ $clog2(PARAMS_STORAGE_SIZE_CIM)'(4*64),   /*len*/ $clog2(NUM_CIMS+1)'(EMB_DEPTH),     /*num_rec*/ $clog2(NUM_CIMS+1)'(NUM_CIMS)},
    {/*addr*/ $clog2(PARAMS_STORAGE_SIZE_CIM)'(5*64),   /*len*/ $clog2(NUM_CIMS+1)'(EMB_DEPTH),     /*num_rec*/ $clog2(NUM_CIMS+1)'(NUM_CIMS)},
    {/*addr*/ $clog2(PARAMS_STORAGE_SIZE_CIM)'(6*64),   /*len*/ $clog2(NUM_CIMS+1)'(EMB_DEPTH),     /*num_rec*/ $clog2(NUM_CIMS+1)'(EMB_DEPTH)}, // The number of rec is EMB_DEPTH, but individually it is MLP_DIM
    {/*addr*/ $clog2(PARAMS_STORAGE_SIZE_CIM)'(7*64),   /*len*/ $clog2(NUM_CIMS+1)'(MLP_DIM),       /*num_rec*/ $clog2(NUM_CIMS+1)'(EMB_DEPTH)},
    {/*addr*/ $clog2(PARAMS_STORAGE_SIZE_CIM)'(480),    /*len*/ $clog2(NUM_CIMS+1)'(MLP_DIM),       /*num_rec*/ $clog2(NUM_CIMS+1)'(NUM_SLEEP_STAGES)},
    {/*addr*/ $clog2(PARAMS_STORAGE_SIZE_CIM)'(8*64),   /*len*/ $clog2(NUM_CIMS+1)'(16),            /*num_rec*/ $clog2(NUM_CIMS+1)'(NUM_CIMS)}
};

`endif
