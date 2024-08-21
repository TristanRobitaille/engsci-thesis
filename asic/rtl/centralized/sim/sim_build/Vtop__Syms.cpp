// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Symbol table implementation internals

#include "Vtop__Syms.h"
#include "Vtop.h"
#include "Vtop___024root.h"
#include "Vtop___024unit.h"
#include "Vtop_Defines.h"

// FUNCTIONS
Vtop__Syms::~Vtop__Syms()
{

    // Tear down scope hierarchy
    __Vhier.remove(0, &__Vscope_cim_centralized_tb);
    __Vhier.remove(&__Vscope_cim_centralized_tb, &__Vscope_cim_centralized_tb__cim_centralized);
    __Vhier.remove(&__Vscope_cim_centralized_tb__cim_centralized, &__Vscope_cim_centralized_tb__cim_centralized__cnt_4b);
    __Vhier.remove(&__Vscope_cim_centralized_tb__cim_centralized, &__Vscope_cim_centralized_tb__cim_centralized__cnt_7b);
    __Vhier.remove(&__Vscope_cim_centralized_tb__cim_centralized, &__Vscope_cim_centralized_tb__cim_centralized__cnt_9b);

}

Vtop__Syms::Vtop__Syms(VerilatedContext* contextp, const char* namep, Vtop* modelp)
    : VerilatedSyms{contextp}
    // Setup internal state of the Syms class
    , __Vm_modelp{modelp}
    // Setup module instances
    , TOP{this, namep}
    , TOP__Defines{this, Verilated::catName(namep, "Defines")}
{
    // Configure time unit / time precision
    _vm_contextp__->timeunit(-9);
    _vm_contextp__->timeprecision(-11);
    // Setup each module's pointers to their submodules
    TOP.__PVT__Defines = &TOP__Defines;
    // Setup each module's pointer back to symbol table (for public functions)
    TOP.__Vconfigure(true);
    TOP__Defines.__Vconfigure(true);
    // Setup scopes
    __Vscope_Defines.configure(this, name(), "Defines", "Defines", -9, VerilatedScope::SCOPE_OTHER);
    __Vscope_TOP.configure(this, name(), "TOP", "TOP", 0, VerilatedScope::SCOPE_OTHER);
    __Vscope_cim_centralized_tb.configure(this, name(), "cim_centralized_tb", "cim_centralized_tb", -9, VerilatedScope::SCOPE_MODULE);
    __Vscope_cim_centralized_tb__cim_centralized.configure(this, name(), "cim_centralized_tb.cim_centralized", "cim_centralized", -9, VerilatedScope::SCOPE_MODULE);
    __Vscope_cim_centralized_tb__cim_centralized__cnt_4b.configure(this, name(), "cim_centralized_tb.cim_centralized.cnt_4b", "cnt_4b", -9, VerilatedScope::SCOPE_MODULE);
    __Vscope_cim_centralized_tb__cim_centralized__cnt_7b.configure(this, name(), "cim_centralized_tb.cim_centralized.cnt_7b", "cnt_7b", -9, VerilatedScope::SCOPE_MODULE);
    __Vscope_cim_centralized_tb__cim_centralized__cnt_9b.configure(this, name(), "cim_centralized_tb.cim_centralized.cnt_9b", "cnt_9b", -9, VerilatedScope::SCOPE_MODULE);

    // Set up scope hierarchy
    __Vhier.add(0, &__Vscope_cim_centralized_tb);
    __Vhier.add(&__Vscope_cim_centralized_tb, &__Vscope_cim_centralized_tb__cim_centralized);
    __Vhier.add(&__Vscope_cim_centralized_tb__cim_centralized, &__Vscope_cim_centralized_tb__cim_centralized__cnt_4b);
    __Vhier.add(&__Vscope_cim_centralized_tb__cim_centralized, &__Vscope_cim_centralized_tb__cim_centralized__cnt_7b);
    __Vhier.add(&__Vscope_cim_centralized_tb__cim_centralized, &__Vscope_cim_centralized_tb__cim_centralized__cnt_9b);

    // Setup export functions
    for (int __Vfinal = 0; __Vfinal < 2; ++__Vfinal) {
        __Vscope_Defines.varInsert(__Vfinal,"CIM_INT_RES_SIZE_NUM_ELEM", const_cast<void*>(static_cast<const void*>(&(TOP__Defines.CIM_INT_RES_SIZE_NUM_ELEM))), true, VLVT_UINT32,VLVD_NODIR|VLVF_PUB_RW|VLVF_DPI_CLAY,1 ,31,0);
        __Vscope_Defines.varInsert(__Vfinal,"CIM_PARAMS_STORAGE_SIZE_NUM_ELEM", const_cast<void*>(static_cast<const void*>(&(TOP__Defines.CIM_PARAMS_STORAGE_SIZE_NUM_ELEM))), true, VLVT_UINT32,VLVD_NODIR|VLVF_PUB_RW|VLVF_DPI_CLAY,1 ,31,0);
        __Vscope_Defines.varInsert(__Vfinal,"N_STO_INT_RES", const_cast<void*>(static_cast<const void*>(&(TOP__Defines.N_STO_INT_RES))), true, VLVT_UINT32,VLVD_NODIR|VLVF_PUB_RW|VLVF_DPI_CLAY,1 ,31,0);
        __Vscope_Defines.varInsert(__Vfinal,"N_STO_PARAMS", const_cast<void*>(static_cast<const void*>(&(TOP__Defines.N_STO_PARAMS))), true, VLVT_UINT32,VLVD_NODIR|VLVF_PUB_RW|VLVF_DPI_CLAY,1 ,31,0);
        __Vscope_TOP.varInsert(__Vfinal,"clk", &(TOP.clk), false, VLVT_UINT8,VLVD_IN|VLVF_PUB_RW,0);
        __Vscope_TOP.varInsert(__Vfinal,"rst_n", &(TOP.rst_n), false, VLVT_UINT8,VLVD_IN|VLVF_PUB_RW,0);
        __Vscope_cim_centralized_tb.varInsert(__Vfinal,"clk", &(TOP.cim_centralized_tb__DOT__clk), false, VLVT_UINT8,VLVD_NODIR|VLVF_PUB_RW,0);
        __Vscope_cim_centralized_tb.varInsert(__Vfinal,"rst_n", &(TOP.cim_centralized_tb__DOT__rst_n), false, VLVT_UINT8,VLVD_NODIR|VLVF_PUB_RW,0);
        __Vscope_cim_centralized_tb__cim_centralized.varInsert(__Vfinal,"clk", &(TOP.cim_centralized_tb__DOT__cim_centralized__DOT__clk), false, VLVT_UINT8,VLVD_NODIR|VLVF_PUB_RW,0);
        __Vscope_cim_centralized_tb__cim_centralized.varInsert(__Vfinal,"cnt_4b_cnt", &(TOP.cim_centralized_tb__DOT__cim_centralized__DOT__cnt_4b_cnt), false, VLVT_UINT8,VLVD_NODIR|VLVF_PUB_RW,1 ,3,0);
        __Vscope_cim_centralized_tb__cim_centralized.varInsert(__Vfinal,"cnt_4b_inc", &(TOP.cim_centralized_tb__DOT__cim_centralized__DOT__cnt_4b_inc), false, VLVT_UINT8,VLVD_NODIR|VLVF_PUB_RW,0);
        __Vscope_cim_centralized_tb__cim_centralized.varInsert(__Vfinal,"cnt_4b_rst_n", &(TOP.cim_centralized_tb__DOT__cim_centralized__DOT__cnt_4b_rst_n), false, VLVT_UINT8,VLVD_NODIR|VLVF_PUB_RW,0);
        __Vscope_cim_centralized_tb__cim_centralized.varInsert(__Vfinal,"cnt_7b_cnt", &(TOP.cim_centralized_tb__DOT__cim_centralized__DOT__cnt_7b_cnt), false, VLVT_UINT8,VLVD_NODIR|VLVF_PUB_RW,1 ,6,0);
        __Vscope_cim_centralized_tb__cim_centralized.varInsert(__Vfinal,"cnt_7b_inc", &(TOP.cim_centralized_tb__DOT__cim_centralized__DOT__cnt_7b_inc), false, VLVT_UINT8,VLVD_NODIR|VLVF_PUB_RW,0);
        __Vscope_cim_centralized_tb__cim_centralized.varInsert(__Vfinal,"cnt_7b_rst_n", &(TOP.cim_centralized_tb__DOT__cim_centralized__DOT__cnt_7b_rst_n), false, VLVT_UINT8,VLVD_NODIR|VLVF_PUB_RW,0);
        __Vscope_cim_centralized_tb__cim_centralized.varInsert(__Vfinal,"cnt_9b_cnt", &(TOP.cim_centralized_tb__DOT__cim_centralized__DOT__cnt_9b_cnt), false, VLVT_UINT16,VLVD_NODIR|VLVF_PUB_RW,1 ,8,0);
        __Vscope_cim_centralized_tb__cim_centralized.varInsert(__Vfinal,"cnt_9b_inc", &(TOP.cim_centralized_tb__DOT__cim_centralized__DOT__cnt_9b_inc), false, VLVT_UINT8,VLVD_NODIR|VLVF_PUB_RW,0);
        __Vscope_cim_centralized_tb__cim_centralized.varInsert(__Vfinal,"cnt_9b_rst_n", &(TOP.cim_centralized_tb__DOT__cim_centralized__DOT__cnt_9b_rst_n), false, VLVT_UINT8,VLVD_NODIR|VLVF_PUB_RW,0);
        __Vscope_cim_centralized_tb__cim_centralized.varInsert(__Vfinal,"int_res", &(TOP.cim_centralized_tb__DOT__cim_centralized__DOT__int_res), false, VLVT_UINT16,VLVD_NODIR|VLVF_PUB_RW,2 ,8,0 ,57115,0);
        __Vscope_cim_centralized_tb__cim_centralized.varInsert(__Vfinal,"param_addr", &(TOP.cim_centralized_tb__DOT__cim_centralized__DOT__param_addr), false, VLVT_UINT16,VLVD_NODIR|VLVF_PUB_RW,1 ,15,0);
        __Vscope_cim_centralized_tb__cim_centralized.varInsert(__Vfinal,"params", &(TOP.cim_centralized_tb__DOT__cim_centralized__DOT__params), false, VLVT_UINT16,VLVD_NODIR|VLVF_PUB_RW,2 ,8,0 ,31647,0);
        __Vscope_cim_centralized_tb__cim_centralized.varInsert(__Vfinal,"rst_n", &(TOP.cim_centralized_tb__DOT__cim_centralized__DOT__rst_n), false, VLVT_UINT8,VLVD_NODIR|VLVF_PUB_RW,0);
        __Vscope_cim_centralized_tb__cim_centralized.varInsert(__Vfinal,"temp_res_addr", &(TOP.cim_centralized_tb__DOT__cim_centralized__DOT__temp_res_addr), false, VLVT_UINT16,VLVD_NODIR|VLVF_PUB_RW,1 ,14,0);
        __Vscope_cim_centralized_tb__cim_centralized__cnt_4b.varInsert(__Vfinal,"MODE", const_cast<void*>(static_cast<const void*>(&(TOP.cim_centralized_tb__DOT__cim_centralized__DOT__cnt_4b__DOT__MODE))), true, VLVT_UINT32,VLVD_NODIR|VLVF_PUB_RW|VLVF_DPI_CLAY,1 ,31,0);
        __Vscope_cim_centralized_tb__cim_centralized__cnt_4b.varInsert(__Vfinal,"WIDTH", const_cast<void*>(static_cast<const void*>(&(TOP.cim_centralized_tb__DOT__cim_centralized__DOT__cnt_4b__DOT__WIDTH))), true, VLVT_UINT32,VLVD_NODIR|VLVF_PUB_RW|VLVF_DPI_CLAY,1 ,31,0);
        __Vscope_cim_centralized_tb__cim_centralized__cnt_4b.varInsert(__Vfinal,"clk", &(TOP.cim_centralized_tb__DOT__cim_centralized__DOT__cnt_4b__DOT__clk), false, VLVT_UINT8,VLVD_NODIR|VLVF_PUB_RW,0);
        __Vscope_cim_centralized_tb__cim_centralized__cnt_4b.varInsert(__Vfinal,"cnt", &(TOP.cim_centralized_tb__DOT__cim_centralized__DOT__cnt_4b__DOT__cnt), false, VLVT_UINT8,VLVD_NODIR|VLVF_PUB_RW,1 ,3,0);
        __Vscope_cim_centralized_tb__cim_centralized__cnt_4b.varInsert(__Vfinal,"inc", &(TOP.cim_centralized_tb__DOT__cim_centralized__DOT__cnt_4b__DOT__inc), false, VLVT_UINT8,VLVD_NODIR|VLVF_PUB_RW,0);
        __Vscope_cim_centralized_tb__cim_centralized__cnt_4b.varInsert(__Vfinal,"inc_prev", &(TOP.cim_centralized_tb__DOT__cim_centralized__DOT__cnt_4b__DOT__inc_prev), false, VLVT_UINT8,VLVD_NODIR|VLVF_PUB_RW,0);
        __Vscope_cim_centralized_tb__cim_centralized__cnt_4b.varInsert(__Vfinal,"rst_n", &(TOP.cim_centralized_tb__DOT__cim_centralized__DOT__cnt_4b__DOT__rst_n), false, VLVT_UINT8,VLVD_NODIR|VLVF_PUB_RW,0);
        __Vscope_cim_centralized_tb__cim_centralized__cnt_7b.varInsert(__Vfinal,"MODE", const_cast<void*>(static_cast<const void*>(&(TOP.cim_centralized_tb__DOT__cim_centralized__DOT__cnt_7b__DOT__MODE))), true, VLVT_UINT32,VLVD_NODIR|VLVF_PUB_RW|VLVF_DPI_CLAY,1 ,31,0);
        __Vscope_cim_centralized_tb__cim_centralized__cnt_7b.varInsert(__Vfinal,"WIDTH", const_cast<void*>(static_cast<const void*>(&(TOP.cim_centralized_tb__DOT__cim_centralized__DOT__cnt_7b__DOT__WIDTH))), true, VLVT_UINT32,VLVD_NODIR|VLVF_PUB_RW|VLVF_DPI_CLAY,1 ,31,0);
        __Vscope_cim_centralized_tb__cim_centralized__cnt_7b.varInsert(__Vfinal,"clk", &(TOP.cim_centralized_tb__DOT__cim_centralized__DOT__cnt_7b__DOT__clk), false, VLVT_UINT8,VLVD_NODIR|VLVF_PUB_RW,0);
        __Vscope_cim_centralized_tb__cim_centralized__cnt_7b.varInsert(__Vfinal,"cnt", &(TOP.cim_centralized_tb__DOT__cim_centralized__DOT__cnt_7b__DOT__cnt), false, VLVT_UINT8,VLVD_NODIR|VLVF_PUB_RW,1 ,6,0);
        __Vscope_cim_centralized_tb__cim_centralized__cnt_7b.varInsert(__Vfinal,"inc", &(TOP.cim_centralized_tb__DOT__cim_centralized__DOT__cnt_7b__DOT__inc), false, VLVT_UINT8,VLVD_NODIR|VLVF_PUB_RW,0);
        __Vscope_cim_centralized_tb__cim_centralized__cnt_7b.varInsert(__Vfinal,"inc_prev", &(TOP.cim_centralized_tb__DOT__cim_centralized__DOT__cnt_7b__DOT__inc_prev), false, VLVT_UINT8,VLVD_NODIR|VLVF_PUB_RW,0);
        __Vscope_cim_centralized_tb__cim_centralized__cnt_7b.varInsert(__Vfinal,"rst_n", &(TOP.cim_centralized_tb__DOT__cim_centralized__DOT__cnt_7b__DOT__rst_n), false, VLVT_UINT8,VLVD_NODIR|VLVF_PUB_RW,0);
        __Vscope_cim_centralized_tb__cim_centralized__cnt_9b.varInsert(__Vfinal,"MODE", const_cast<void*>(static_cast<const void*>(&(TOP.cim_centralized_tb__DOT__cim_centralized__DOT__cnt_9b__DOT__MODE))), true, VLVT_UINT32,VLVD_NODIR|VLVF_PUB_RW|VLVF_DPI_CLAY,1 ,31,0);
        __Vscope_cim_centralized_tb__cim_centralized__cnt_9b.varInsert(__Vfinal,"WIDTH", const_cast<void*>(static_cast<const void*>(&(TOP.cim_centralized_tb__DOT__cim_centralized__DOT__cnt_9b__DOT__WIDTH))), true, VLVT_UINT32,VLVD_NODIR|VLVF_PUB_RW|VLVF_DPI_CLAY,1 ,31,0);
        __Vscope_cim_centralized_tb__cim_centralized__cnt_9b.varInsert(__Vfinal,"clk", &(TOP.cim_centralized_tb__DOT__cim_centralized__DOT__cnt_9b__DOT__clk), false, VLVT_UINT8,VLVD_NODIR|VLVF_PUB_RW,0);
        __Vscope_cim_centralized_tb__cim_centralized__cnt_9b.varInsert(__Vfinal,"cnt", &(TOP.cim_centralized_tb__DOT__cim_centralized__DOT__cnt_9b__DOT__cnt), false, VLVT_UINT16,VLVD_NODIR|VLVF_PUB_RW,1 ,8,0);
        __Vscope_cim_centralized_tb__cim_centralized__cnt_9b.varInsert(__Vfinal,"inc", &(TOP.cim_centralized_tb__DOT__cim_centralized__DOT__cnt_9b__DOT__inc), false, VLVT_UINT8,VLVD_NODIR|VLVF_PUB_RW,0);
        __Vscope_cim_centralized_tb__cim_centralized__cnt_9b.varInsert(__Vfinal,"inc_prev", &(TOP.cim_centralized_tb__DOT__cim_centralized__DOT__cnt_9b__DOT__inc_prev), false, VLVT_UINT8,VLVD_NODIR|VLVF_PUB_RW,0);
        __Vscope_cim_centralized_tb__cim_centralized__cnt_9b.varInsert(__Vfinal,"rst_n", &(TOP.cim_centralized_tb__DOT__cim_centralized__DOT__cnt_9b__DOT__rst_n), false, VLVT_UINT8,VLVD_NODIR|VLVF_PUB_RW,0);
    }
}
