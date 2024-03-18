// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Symbol table implementation internals

#include "Vtop__pch.h"
#include "Vtop.h"
#include "Vtop___024root.h"
#include "Vtop___024unit.h"

// FUNCTIONS
Vtop__Syms::~Vtop__Syms()
{

    // Tear down scope hierarchy
    __Vhier.remove(0, &__Vscope___024unit);
    __Vhier.remove(0, &__Vscope_cim);
    __Vhier.remove(&__Vscope_cim, &__Vscope_cim__gen_cnt_7b_2_inst);
    __Vhier.remove(&__Vscope_cim, &__Vscope_cim__gen_cnt_7b_inst);
    __Vhier.remove(&__Vscope_cim, &__Vscope_cim__word_rec_cnt_inst);
    __Vhier.remove(&__Vscope_cim, &__Vscope_cim__word_snt_cnt_inst);

}

Vtop__Syms::Vtop__Syms(VerilatedContext* contextp, const char* namep, Vtop* modelp)
    : VerilatedSyms{contextp}
    // Setup internal state of the Syms class
    , __Vm_modelp{modelp}
    , __Vm_threadPoolp{static_cast<VlThreadPool*>(contextp->threadPoolp())}
    // Setup module instances
    , TOP{this, namep}
    , TOP____024unit{this, Verilated::catName(namep, "$unit")}
{
        // Check resources
        Verilated::stackCheck(25);
    // Configure time unit / time precision
    _vm_contextp__->timeunit(-9);
    _vm_contextp__->timeprecision(-12);
    // Setup each module's pointers to their submodules
    TOP.__PVT____024unit = &TOP____024unit;
    // Setup each module's pointer back to symbol table (for public functions)
    TOP.__Vconfigure(true);
    TOP____024unit.__Vconfigure(true);
    // Setup scopes
    __Vscope_TOP.configure(this, name(), "TOP", "TOP", 0, VerilatedScope::SCOPE_OTHER);
    __Vscope___024unit.configure(this, name(), "\\$unit ", "\\$unit ", -9, VerilatedScope::SCOPE_PACKAGE);
    __Vscope_cim.configure(this, name(), "cim", "cim", -9, VerilatedScope::SCOPE_MODULE);
    __Vscope_cim__gen_cnt_7b_2_inst.configure(this, name(), "cim.gen_cnt_7b_2_inst", "gen_cnt_7b_2_inst", -9, VerilatedScope::SCOPE_MODULE);
    __Vscope_cim__gen_cnt_7b_inst.configure(this, name(), "cim.gen_cnt_7b_inst", "gen_cnt_7b_inst", -9, VerilatedScope::SCOPE_MODULE);
    __Vscope_cim__word_rec_cnt_inst.configure(this, name(), "cim.word_rec_cnt_inst", "word_rec_cnt_inst", -9, VerilatedScope::SCOPE_MODULE);
    __Vscope_cim__word_snt_cnt_inst.configure(this, name(), "cim.word_snt_cnt_inst", "word_snt_cnt_inst", -9, VerilatedScope::SCOPE_MODULE);

    // Set up scope hierarchy
    __Vhier.add(0, &__Vscope___024unit);
    __Vhier.add(0, &__Vscope_cim);
    __Vhier.add(&__Vscope_cim, &__Vscope_cim__gen_cnt_7b_2_inst);
    __Vhier.add(&__Vscope_cim, &__Vscope_cim__gen_cnt_7b_inst);
    __Vhier.add(&__Vscope_cim, &__Vscope_cim__word_rec_cnt_inst);
    __Vhier.add(&__Vscope_cim, &__Vscope_cim__word_snt_cnt_inst);

    // Setup export functions
    for (int __Vfinal = 0; __Vfinal < 2; ++__Vfinal) {
        __Vscope_TOP.varInsert(__Vfinal,"clk", &(TOP.clk), false, VLVT_UINT8,VLVD_IN|VLVF_PUB_RW,0);
        __Vscope_TOP.varInsert(__Vfinal,"is_ready", &(TOP.is_ready), false, VLVT_UINT8,VLVD_OUT|VLVF_PUB_RW,0);
        __Vscope_TOP.varInsert(__Vfinal,"rst_n", &(TOP.rst_n), false, VLVT_UINT8,VLVD_IN|VLVF_PUB_RW,0);
        __Vscope___024unit.varInsert(__Vfinal,"BUS_OP_WIDTH", const_cast<void*>(static_cast<const void*>(&(TOP____024unit.BUS_OP_WIDTH))), true, VLVT_UINT32,VLVD_NODIR|VLVF_PUB_RW,1 ,31,0);
        __Vscope___024unit.varInsert(__Vfinal,"EEG_SAMPLE_DEPTH", const_cast<void*>(static_cast<const void*>(&(TOP____024unit.EEG_SAMPLE_DEPTH))), true, VLVT_UINT32,VLVD_NODIR|VLVF_PUB_RW,1 ,31,0);
        __Vscope___024unit.varInsert(__Vfinal,"EMB_DEPTH", const_cast<void*>(static_cast<const void*>(&(TOP____024unit.EMB_DEPTH))), true, VLVT_UINT32,VLVD_NODIR|VLVF_PUB_RW,1 ,31,0);
        __Vscope___024unit.varInsert(__Vfinal,"MLP_DIM", const_cast<void*>(static_cast<const void*>(&(TOP____024unit.MLP_DIM))), true, VLVT_UINT32,VLVD_NODIR|VLVF_PUB_RW,1 ,31,0);
        __Vscope___024unit.varInsert(__Vfinal,"NUM_CIMS", const_cast<void*>(static_cast<const void*>(&(TOP____024unit.NUM_CIMS))), true, VLVT_UINT32,VLVD_NODIR|VLVF_PUB_RW,1 ,31,0);
        __Vscope___024unit.varInsert(__Vfinal,"NUM_HEADS", const_cast<void*>(static_cast<const void*>(&(TOP____024unit.NUM_HEADS))), true, VLVT_UINT32,VLVD_NODIR|VLVF_PUB_RW,1 ,31,0);
        __Vscope___024unit.varInsert(__Vfinal,"NUM_PARAMS", const_cast<void*>(static_cast<const void*>(&(TOP____024unit.NUM_PARAMS))), true, VLVT_UINT32,VLVD_NODIR|VLVF_PUB_RW,1 ,31,0);
        __Vscope___024unit.varInsert(__Vfinal,"NUM_PATCHES", const_cast<void*>(static_cast<const void*>(&(TOP____024unit.NUM_PATCHES))), true, VLVT_UINT32,VLVD_NODIR|VLVF_PUB_RW,1 ,31,0);
        __Vscope___024unit.varInsert(__Vfinal,"NUM_SLEEP_STAGES", const_cast<void*>(static_cast<const void*>(&(TOP____024unit.NUM_SLEEP_STAGES))), true, VLVT_UINT32,VLVD_NODIR|VLVF_PUB_RW,1 ,31,0);
        __Vscope___024unit.varInsert(__Vfinal,"N_COMP", const_cast<void*>(static_cast<const void*>(&(TOP____024unit.N_COMP))), true, VLVT_UINT32,VLVD_NODIR|VLVF_PUB_RW,1 ,31,0);
        __Vscope___024unit.varInsert(__Vfinal,"N_STORAGE", const_cast<void*>(static_cast<const void*>(&(TOP____024unit.N_STORAGE))), true, VLVT_UINT32,VLVD_NODIR|VLVF_PUB_RW,1 ,31,0);
        __Vscope___024unit.varInsert(__Vfinal,"PARAMS_STORAGE_SIZE_CIM", const_cast<void*>(static_cast<const void*>(&(TOP____024unit.PARAMS_STORAGE_SIZE_CIM))), true, VLVT_UINT32,VLVD_NODIR|VLVF_PUB_RW,1 ,31,0);
        __Vscope___024unit.varInsert(__Vfinal,"PATCH_LEN", const_cast<void*>(static_cast<const void*>(&(TOP____024unit.PATCH_LEN))), true, VLVT_UINT32,VLVD_NODIR|VLVF_PUB_RW,1 ,31,0);
        __Vscope___024unit.varInsert(__Vfinal,"Q", const_cast<void*>(static_cast<const void*>(&(TOP____024unit.Q))), true, VLVT_UINT32,VLVD_NODIR|VLVF_PUB_RW,1 ,31,0);
        __Vscope___024unit.varInsert(__Vfinal,"TEMP_RES_STORAGE_SIZE_CIM", const_cast<void*>(static_cast<const void*>(&(TOP____024unit.TEMP_RES_STORAGE_SIZE_CIM))), true, VLVT_UINT32,VLVD_NODIR|VLVF_PUB_RW,1 ,31,0);
        __Vscope_cim.varInsert(__Vfinal,"cim_state", &(TOP.cim__DOT__cim_state), false, VLVT_UINT8,VLVD_NODIR|VLVF_PUB_RW,1 ,2,0);
        __Vscope_cim.varInsert(__Vfinal,"clk", &(TOP.cim__DOT__clk), false, VLVT_UINT8,VLVD_NODIR|VLVF_PUB_RW,0);
        __Vscope_cim.varInsert(__Vfinal,"computation_result", &(TOP.cim__DOT__computation_result), false, VLVT_UINT32,VLVD_NODIR|VLVF_PUB_RW,1 ,21,0);
        __Vscope_cim.varInsert(__Vfinal,"compute_in_progress", &(TOP.cim__DOT__compute_in_progress), false, VLVT_UINT8,VLVD_NODIR|VLVF_PUB_RW,0);
        __Vscope_cim.varInsert(__Vfinal,"compute_temp", &(TOP.cim__DOT__compute_temp), false, VLVT_UINT32,VLVD_NODIR|VLVF_PUB_RW,1 ,21,0);
        __Vscope_cim.varInsert(__Vfinal,"compute_temp_2", &(TOP.cim__DOT__compute_temp_2), false, VLVT_UINT32,VLVD_NODIR|VLVF_PUB_RW,1 ,21,0);
        __Vscope_cim.varInsert(__Vfinal,"compute_temp_3", &(TOP.cim__DOT__compute_temp_3), false, VLVT_UINT32,VLVD_NODIR|VLVF_PUB_RW,1 ,21,0);
        __Vscope_cim.varInsert(__Vfinal,"current_inf_step", &(TOP.cim__DOT__current_inf_step), false, VLVT_UINT8,VLVD_NODIR|VLVF_PUB_RW,1 ,5,0);
        __Vscope_cim.varInsert(__Vfinal,"data_len", &(TOP.cim__DOT__data_len), false, VLVT_UINT8,VLVD_NODIR|VLVF_PUB_RW,1 ,6,0);
        __Vscope_cim.varInsert(__Vfinal,"gen_cnt_3b", &(TOP.cim__DOT__gen_cnt_3b), false, VLVT_UINT8,VLVD_NODIR|VLVF_PUB_RW,1 ,2,0);
        __Vscope_cim.varInsert(__Vfinal,"gen_cnt_7b_2_cnt", &(TOP.cim__DOT__gen_cnt_7b_2_cnt), false, VLVT_UINT8,VLVD_NODIR|VLVF_PUB_RW,1 ,6,0);
        __Vscope_cim.varInsert(__Vfinal,"gen_cnt_7b_2_inc", &(TOP.cim__DOT__gen_cnt_7b_2_inc), false, VLVT_UINT8,VLVD_NODIR|VLVF_PUB_RW,1 ,6,0);
        __Vscope_cim.varInsert(__Vfinal,"gen_cnt_7b_2_rst_n", &(TOP.cim__DOT__gen_cnt_7b_2_rst_n), false, VLVT_UINT8,VLVD_NODIR|VLVF_PUB_RW,0);
        __Vscope_cim.varInsert(__Vfinal,"gen_cnt_7b_cnt", &(TOP.cim__DOT__gen_cnt_7b_cnt), false, VLVT_UINT8,VLVD_NODIR|VLVF_PUB_RW,1 ,6,0);
        __Vscope_cim.varInsert(__Vfinal,"gen_cnt_7b_inc", &(TOP.cim__DOT__gen_cnt_7b_inc), false, VLVT_UINT8,VLVD_NODIR|VLVF_PUB_RW,1 ,6,0);
        __Vscope_cim.varInsert(__Vfinal,"gen_cnt_7b_rst_n", &(TOP.cim__DOT__gen_cnt_7b_rst_n), false, VLVT_UINT8,VLVD_NODIR|VLVF_PUB_RW,0);
        __Vscope_cim.varInsert(__Vfinal,"intermediate_res", &(TOP.cim__DOT__intermediate_res), false, VLVT_UINT16,VLVD_NODIR|VLVF_PUB_RW,2 ,15,0 ,847,0);
        __Vscope_cim.varInsert(__Vfinal,"is_ready", &(TOP.cim__DOT__is_ready), false, VLVT_UINT8,VLVD_NODIR|VLVF_PUB_RW,0);
        __Vscope_cim.varInsert(__Vfinal,"params", &(TOP.cim__DOT__params), false, VLVT_UINT16,VLVD_NODIR|VLVF_PUB_RW,2 ,15,0 ,527,0);
        __Vscope_cim.varInsert(__Vfinal,"rst_n", &(TOP.cim__DOT__rst_n), false, VLVT_UINT8,VLVD_NODIR|VLVF_PUB_RW,0);
        __Vscope_cim.varInsert(__Vfinal,"rx_addr", &(TOP.cim__DOT__rx_addr), false, VLVT_UINT16,VLVD_NODIR|VLVF_PUB_RW,1 ,9,0);
        __Vscope_cim.varInsert(__Vfinal,"sender_id", &(TOP.cim__DOT__sender_id), false, VLVT_UINT8,VLVD_NODIR|VLVF_PUB_RW,1 ,6,0);
        __Vscope_cim.varInsert(__Vfinal,"tx_addr", &(TOP.cim__DOT__tx_addr), false, VLVT_UINT16,VLVD_NODIR|VLVF_PUB_RW,1 ,9,0);
        __Vscope_cim.varInsert(__Vfinal,"word_rec_cnt", &(TOP.cim__DOT__word_rec_cnt), false, VLVT_UINT8,VLVD_NODIR|VLVF_PUB_RW,1 ,6,0);
        __Vscope_cim.varInsert(__Vfinal,"word_rec_cnt_inc", &(TOP.cim__DOT__word_rec_cnt_inc), false, VLVT_UINT8,VLVD_NODIR|VLVF_PUB_RW,1 ,6,0);
        __Vscope_cim.varInsert(__Vfinal,"word_rec_cnt_rst_n", &(TOP.cim__DOT__word_rec_cnt_rst_n), false, VLVT_UINT8,VLVD_NODIR|VLVF_PUB_RW,0);
        __Vscope_cim.varInsert(__Vfinal,"word_snt_cnt", &(TOP.cim__DOT__word_snt_cnt), false, VLVT_UINT8,VLVD_NODIR|VLVF_PUB_RW,1 ,6,0);
        __Vscope_cim.varInsert(__Vfinal,"word_snt_cnt_inc", &(TOP.cim__DOT__word_snt_cnt_inc), false, VLVT_UINT8,VLVD_NODIR|VLVF_PUB_RW,1 ,6,0);
        __Vscope_cim.varInsert(__Vfinal,"word_snt_cnt_rst_n", &(TOP.cim__DOT__word_snt_cnt_rst_n), false, VLVT_UINT8,VLVD_NODIR|VLVF_PUB_RW,0);
        __Vscope_cim__gen_cnt_7b_2_inst.varInsert(__Vfinal,"MODE", const_cast<void*>(static_cast<const void*>(&(TOP.cim__DOT__gen_cnt_7b_2_inst__DOT__MODE))), true, VLVT_UINT32,VLVD_NODIR|VLVF_PUB_RW|VLVF_DPI_CLAY,1 ,31,0);
        __Vscope_cim__gen_cnt_7b_2_inst.varInsert(__Vfinal,"WIDTH", const_cast<void*>(static_cast<const void*>(&(TOP.cim__DOT__gen_cnt_7b_2_inst__DOT__WIDTH))), true, VLVT_UINT32,VLVD_NODIR|VLVF_PUB_RW|VLVF_DPI_CLAY,1 ,31,0);
        __Vscope_cim__gen_cnt_7b_2_inst.varInsert(__Vfinal,"clk", &(TOP.cim__DOT__gen_cnt_7b_2_inst__DOT__clk), false, VLVT_UINT8,VLVD_NODIR|VLVF_PUB_RW,0);
        __Vscope_cim__gen_cnt_7b_2_inst.varInsert(__Vfinal,"cnt", &(TOP.cim__DOT__gen_cnt_7b_2_inst__DOT__cnt), false, VLVT_UINT8,VLVD_NODIR|VLVF_PUB_RW,1 ,6,0);
        __Vscope_cim__gen_cnt_7b_2_inst.varInsert(__Vfinal,"inc", &(TOP.cim__DOT__gen_cnt_7b_2_inst__DOT__inc), false, VLVT_UINT8,VLVD_NODIR|VLVF_PUB_RW,1 ,6,0);
        __Vscope_cim__gen_cnt_7b_2_inst.varInsert(__Vfinal,"inc_prev", &(TOP.cim__DOT__gen_cnt_7b_2_inst__DOT__inc_prev), false, VLVT_UINT8,VLVD_NODIR|VLVF_PUB_RW,1 ,6,0);
        __Vscope_cim__gen_cnt_7b_2_inst.varInsert(__Vfinal,"rst_n", &(TOP.cim__DOT__gen_cnt_7b_2_inst__DOT__rst_n), false, VLVT_UINT8,VLVD_NODIR|VLVF_PUB_RW,0);
        __Vscope_cim__gen_cnt_7b_inst.varInsert(__Vfinal,"MODE", const_cast<void*>(static_cast<const void*>(&(TOP.cim__DOT__gen_cnt_7b_inst__DOT__MODE))), true, VLVT_UINT32,VLVD_NODIR|VLVF_PUB_RW|VLVF_DPI_CLAY,1 ,31,0);
        __Vscope_cim__gen_cnt_7b_inst.varInsert(__Vfinal,"WIDTH", const_cast<void*>(static_cast<const void*>(&(TOP.cim__DOT__gen_cnt_7b_inst__DOT__WIDTH))), true, VLVT_UINT32,VLVD_NODIR|VLVF_PUB_RW|VLVF_DPI_CLAY,1 ,31,0);
        __Vscope_cim__gen_cnt_7b_inst.varInsert(__Vfinal,"clk", &(TOP.cim__DOT__gen_cnt_7b_inst__DOT__clk), false, VLVT_UINT8,VLVD_NODIR|VLVF_PUB_RW,0);
        __Vscope_cim__gen_cnt_7b_inst.varInsert(__Vfinal,"cnt", &(TOP.cim__DOT__gen_cnt_7b_inst__DOT__cnt), false, VLVT_UINT8,VLVD_NODIR|VLVF_PUB_RW,1 ,6,0);
        __Vscope_cim__gen_cnt_7b_inst.varInsert(__Vfinal,"inc", &(TOP.cim__DOT__gen_cnt_7b_inst__DOT__inc), false, VLVT_UINT8,VLVD_NODIR|VLVF_PUB_RW,1 ,6,0);
        __Vscope_cim__gen_cnt_7b_inst.varInsert(__Vfinal,"inc_prev", &(TOP.cim__DOT__gen_cnt_7b_inst__DOT__inc_prev), false, VLVT_UINT8,VLVD_NODIR|VLVF_PUB_RW,1 ,6,0);
        __Vscope_cim__gen_cnt_7b_inst.varInsert(__Vfinal,"rst_n", &(TOP.cim__DOT__gen_cnt_7b_inst__DOT__rst_n), false, VLVT_UINT8,VLVD_NODIR|VLVF_PUB_RW,0);
        __Vscope_cim__word_rec_cnt_inst.varInsert(__Vfinal,"MODE", const_cast<void*>(static_cast<const void*>(&(TOP.cim__DOT__word_rec_cnt_inst__DOT__MODE))), true, VLVT_UINT32,VLVD_NODIR|VLVF_PUB_RW|VLVF_DPI_CLAY,1 ,31,0);
        __Vscope_cim__word_rec_cnt_inst.varInsert(__Vfinal,"WIDTH", const_cast<void*>(static_cast<const void*>(&(TOP.cim__DOT__word_rec_cnt_inst__DOT__WIDTH))), true, VLVT_UINT32,VLVD_NODIR|VLVF_PUB_RW|VLVF_DPI_CLAY,1 ,31,0);
        __Vscope_cim__word_rec_cnt_inst.varInsert(__Vfinal,"clk", &(TOP.cim__DOT__word_rec_cnt_inst__DOT__clk), false, VLVT_UINT8,VLVD_NODIR|VLVF_PUB_RW,0);
        __Vscope_cim__word_rec_cnt_inst.varInsert(__Vfinal,"cnt", &(TOP.cim__DOT__word_rec_cnt_inst__DOT__cnt), false, VLVT_UINT8,VLVD_NODIR|VLVF_PUB_RW,1 ,6,0);
        __Vscope_cim__word_rec_cnt_inst.varInsert(__Vfinal,"inc", &(TOP.cim__DOT__word_rec_cnt_inst__DOT__inc), false, VLVT_UINT8,VLVD_NODIR|VLVF_PUB_RW,1 ,6,0);
        __Vscope_cim__word_rec_cnt_inst.varInsert(__Vfinal,"inc_prev", &(TOP.cim__DOT__word_rec_cnt_inst__DOT__inc_prev), false, VLVT_UINT8,VLVD_NODIR|VLVF_PUB_RW,1 ,6,0);
        __Vscope_cim__word_rec_cnt_inst.varInsert(__Vfinal,"rst_n", &(TOP.cim__DOT__word_rec_cnt_inst__DOT__rst_n), false, VLVT_UINT8,VLVD_NODIR|VLVF_PUB_RW,0);
        __Vscope_cim__word_snt_cnt_inst.varInsert(__Vfinal,"MODE", const_cast<void*>(static_cast<const void*>(&(TOP.cim__DOT__word_snt_cnt_inst__DOT__MODE))), true, VLVT_UINT32,VLVD_NODIR|VLVF_PUB_RW|VLVF_DPI_CLAY,1 ,31,0);
        __Vscope_cim__word_snt_cnt_inst.varInsert(__Vfinal,"WIDTH", const_cast<void*>(static_cast<const void*>(&(TOP.cim__DOT__word_snt_cnt_inst__DOT__WIDTH))), true, VLVT_UINT32,VLVD_NODIR|VLVF_PUB_RW|VLVF_DPI_CLAY,1 ,31,0);
        __Vscope_cim__word_snt_cnt_inst.varInsert(__Vfinal,"clk", &(TOP.cim__DOT__word_snt_cnt_inst__DOT__clk), false, VLVT_UINT8,VLVD_NODIR|VLVF_PUB_RW,0);
        __Vscope_cim__word_snt_cnt_inst.varInsert(__Vfinal,"cnt", &(TOP.cim__DOT__word_snt_cnt_inst__DOT__cnt), false, VLVT_UINT8,VLVD_NODIR|VLVF_PUB_RW,1 ,6,0);
        __Vscope_cim__word_snt_cnt_inst.varInsert(__Vfinal,"inc", &(TOP.cim__DOT__word_snt_cnt_inst__DOT__inc), false, VLVT_UINT8,VLVD_NODIR|VLVF_PUB_RW,1 ,6,0);
        __Vscope_cim__word_snt_cnt_inst.varInsert(__Vfinal,"inc_prev", &(TOP.cim__DOT__word_snt_cnt_inst__DOT__inc_prev), false, VLVT_UINT8,VLVD_NODIR|VLVF_PUB_RW,1 ,6,0);
        __Vscope_cim__word_snt_cnt_inst.varInsert(__Vfinal,"rst_n", &(TOP.cim__DOT__word_snt_cnt_inst__DOT__rst_n), false, VLVT_UINT8,VLVD_NODIR|VLVF_PUB_RW,0);
    }
}
