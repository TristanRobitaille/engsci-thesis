/* Initialize struct of CiM. To be called inside CiM module */
`ifndef _cim_init_sv_
`define _cim_init_sv_

parameter logic [$clog2(TEMP_RES_STORAGE_SIZE_CIM)-1:0] mem_map[32] = {
    $clog2(TEMP_RES_STORAGE_SIZE_CIM)'(PATCH_LEN+1),
    $clog2(TEMP_RES_STORAGE_SIZE_CIM)'(PATCH_LEN),
    $clog2(TEMP_RES_STORAGE_SIZE_CIM)'(0),
    $clog2(TEMP_RES_STORAGE_SIZE_CIM)'(NUM_PATCHES+1),
    $clog2(TEMP_RES_STORAGE_SIZE_CIM)'(NUM_PATCHES+1+EMB_DEPTH),
    $clog2(TEMP_RES_STORAGE_SIZE_CIM)'(NUM_PATCHES+1+EMB_DEPTH),
    $clog2(TEMP_RES_STORAGE_SIZE_CIM)'(2*EMB_DEPTH+NUM_PATCHES+1),
    $clog2(TEMP_RES_STORAGE_SIZE_CIM)'(2*(EMB_DEPTH+NUM_PATCHES+1)),
    $clog2(TEMP_RES_STORAGE_SIZE_CIM)'(NUM_PATCHES+1),
    $clog2(TEMP_RES_STORAGE_SIZE_CIM)'(2*EMB_DEPTH+NUM_PATCHES+1),
    $clog2(TEMP_RES_STORAGE_SIZE_CIM)'(3*EMB_DEPTH+NUM_PATCHES+1),
    $clog2(TEMP_RES_STORAGE_SIZE_CIM)'(3*EMB_DEPTH+2*(NUM_PATCHES+1)),
    $clog2(TEMP_RES_STORAGE_SIZE_CIM)'(2*(EMB_DEPTH+NUM_PATCHES+1)),
    $clog2(TEMP_RES_STORAGE_SIZE_CIM)'(NUM_PATCHES+1+EMB_DEPTH),
    $clog2(TEMP_RES_STORAGE_SIZE_CIM)'(2*EMB_DEPTH+NUM_PATCHES+1),
    $clog2(TEMP_RES_STORAGE_SIZE_CIM)'(NUM_PATCHES+1+EMB_DEPTH),
    $clog2(TEMP_RES_STORAGE_SIZE_CIM)'(2*EMB_DEPTH+NUM_PATCHES+1),
    $clog2(TEMP_RES_STORAGE_SIZE_CIM)'(NUM_PATCHES+1),
    $clog2(TEMP_RES_STORAGE_SIZE_CIM)'(NUM_PATCHES+1+EMB_DEPTH),
    $clog2(TEMP_RES_STORAGE_SIZE_CIM)'(NUM_PATCHES+1),
    $clog2(TEMP_RES_STORAGE_SIZE_CIM)'(NUM_PATCHES+1+EMB_DEPTH),
    $clog2(TEMP_RES_STORAGE_SIZE_CIM)'(NUM_PATCHES+1),
    $clog2(TEMP_RES_STORAGE_SIZE_CIM)'(3*EMB_DEPTH+NUM_PATCHES+2),
    $clog2(TEMP_RES_STORAGE_SIZE_CIM)'(0),
    $clog2(TEMP_RES_STORAGE_SIZE_CIM)'(EMB_DEPTH),
    $clog2(TEMP_RES_STORAGE_SIZE_CIM)'(EMB_DEPTH),
    $clog2(TEMP_RES_STORAGE_SIZE_CIM)'(2*EMB_DEPTH),
    $clog2(TEMP_RES_STORAGE_SIZE_CIM)'(EMB_DEPTH),
    $clog2(TEMP_RES_STORAGE_SIZE_CIM)'(2*EMB_DEPTH),
    $clog2(TEMP_RES_STORAGE_SIZE_CIM)'(MLP_DIM),
    $clog2(TEMP_RES_STORAGE_SIZE_CIM)'(2*EMB_DEPTH),
    $clog2(TEMP_RES_STORAGE_SIZE_CIM)'(836)
};

`endif