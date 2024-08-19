`ifndef _defines_vh_
`define _defines_vh_

package Defines;
    // Constants
    localparam int CIM_PARAMS_BANK_SIZE_NUM_WORD    = 15872; // Need 2 banks
    localparam int CIM_INT_RES_BANK_SIZE_NUM_WORD   = 14336; // Need 4 banks
    localparam int CIM_PARAMS_NUM_BANKS             = 2;
    localparam int CIM_INT_RES_NUM_BANKS            = 4;
    localparam int N_STO_INT_RES                    = 9;
    localparam int N_STO_PARAMS                     = 9;

    // Types
    typedef logic [$clog2(CIM_PARAMS_NUM_BANKS*CIM_PARAMS_BANK_SIZE_NUM_WORD)-1:0]      TempResAddr_t;
    typedef logic [$clog2(CIM_INT_RES_NUM_BANKS*CIM_INT_RES_BANK_SIZE_NUM_WORD)-1:0]    ParamAddr_t;
    typedef logic [N_STO_PARAMS-1:0]                                                    Param_t;
    typedef logic [N_STO_INT_RES-1:0]                                                   IntResSingle_t;
    typedef logic [2*N_STO_INT_RES-1:0]                                                 IntResDouble_t;
endpackage

`endif // _defines_vh_
