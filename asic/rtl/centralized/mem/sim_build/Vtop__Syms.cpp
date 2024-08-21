// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Symbol table implementation internals

#include "Vtop__Syms.h"
#include "Vtop.h"
#include "Vtop___024root.h"
#include "Vtop___024unit.h"
#include "Vtop_Defines.h"
#include "Vtop_MemoryAccessSignals__Tz1_TBz2.h"
#include "Vtop_MemoryAccessSignals__Tz3_TBz4.h"
#include "Vtop_MemoryAccessSignals__Tz1_TBz9.h"

// FUNCTIONS
Vtop__Syms::~Vtop__Syms()
{

    // Tear down scope hierarchy
    __Vhier.remove(0, &__Vscope_mem_tb);
    __Vhier.remove(&__Vscope_mem_tb, &__Vscope_mem_tb__int_res);
    __Vhier.remove(&__Vscope_mem_tb, &__Vscope_mem_tb__int_res_read_sig);
    __Vhier.remove(&__Vscope_mem_tb, &__Vscope_mem_tb__int_res_write_sig);
    __Vhier.remove(&__Vscope_mem_tb, &__Vscope_mem_tb__param_read_sig);
    __Vhier.remove(&__Vscope_mem_tb, &__Vscope_mem_tb__param_write_sig);
    __Vhier.remove(&__Vscope_mem_tb, &__Vscope_mem_tb__params);
    __Vhier.remove(&__Vscope_mem_tb__int_res, &__Vscope_mem_tb__int_res__int_res_0);
    __Vhier.remove(&__Vscope_mem_tb__int_res, &__Vscope_mem_tb__int_res__int_res_0_read);
    __Vhier.remove(&__Vscope_mem_tb__int_res, &__Vscope_mem_tb__int_res__int_res_0_write);
    __Vhier.remove(&__Vscope_mem_tb__int_res, &__Vscope_mem_tb__int_res__int_res_1);
    __Vhier.remove(&__Vscope_mem_tb__int_res, &__Vscope_mem_tb__int_res__int_res_1_read);
    __Vhier.remove(&__Vscope_mem_tb__int_res, &__Vscope_mem_tb__int_res__int_res_1_write);
    __Vhier.remove(&__Vscope_mem_tb__int_res, &__Vscope_mem_tb__int_res__int_res_2);
    __Vhier.remove(&__Vscope_mem_tb__int_res, &__Vscope_mem_tb__int_res__int_res_2_read);
    __Vhier.remove(&__Vscope_mem_tb__int_res, &__Vscope_mem_tb__int_res__int_res_2_write);
    __Vhier.remove(&__Vscope_mem_tb__int_res, &__Vscope_mem_tb__int_res__int_res_3);
    __Vhier.remove(&__Vscope_mem_tb__int_res, &__Vscope_mem_tb__int_res__int_res_3_read);
    __Vhier.remove(&__Vscope_mem_tb__int_res, &__Vscope_mem_tb__int_res__int_res_3_write);
    __Vhier.remove(&__Vscope_mem_tb__params, &__Vscope_mem_tb__params__params_0);
    __Vhier.remove(&__Vscope_mem_tb__params, &__Vscope_mem_tb__params__params_0_read);
    __Vhier.remove(&__Vscope_mem_tb__params, &__Vscope_mem_tb__params__params_0_write);
    __Vhier.remove(&__Vscope_mem_tb__params, &__Vscope_mem_tb__params__params_1);
    __Vhier.remove(&__Vscope_mem_tb__params, &__Vscope_mem_tb__params__params_1_read);
    __Vhier.remove(&__Vscope_mem_tb__params, &__Vscope_mem_tb__params__params_1_write);

}

Vtop__Syms::Vtop__Syms(VerilatedContext* contextp, const char* namep, Vtop* modelp)
    : VerilatedSyms{contextp}
    // Setup internal state of the Syms class
    , __Vm_modelp{modelp}
    // Setup module instances
    , TOP{this, namep}
    , TOP__Defines{this, Verilated::catName(namep, "Defines")}
    , TOP__mem_tb__DOT__int_res__DOT__int_res_0_read{this, Verilated::catName(namep, "mem_tb.int_res.int_res_0_read")}
    , TOP__mem_tb__DOT__int_res__DOT__int_res_0_write{this, Verilated::catName(namep, "mem_tb.int_res.int_res_0_write")}
    , TOP__mem_tb__DOT__int_res__DOT__int_res_1_read{this, Verilated::catName(namep, "mem_tb.int_res.int_res_1_read")}
    , TOP__mem_tb__DOT__int_res__DOT__int_res_1_write{this, Verilated::catName(namep, "mem_tb.int_res.int_res_1_write")}
    , TOP__mem_tb__DOT__int_res__DOT__int_res_2_read{this, Verilated::catName(namep, "mem_tb.int_res.int_res_2_read")}
    , TOP__mem_tb__DOT__int_res__DOT__int_res_2_write{this, Verilated::catName(namep, "mem_tb.int_res.int_res_2_write")}
    , TOP__mem_tb__DOT__int_res__DOT__int_res_3_read{this, Verilated::catName(namep, "mem_tb.int_res.int_res_3_read")}
    , TOP__mem_tb__DOT__int_res__DOT__int_res_3_write{this, Verilated::catName(namep, "mem_tb.int_res.int_res_3_write")}
    , TOP__mem_tb__DOT__int_res_read_sig{this, Verilated::catName(namep, "mem_tb.int_res_read_sig")}
    , TOP__mem_tb__DOT__int_res_write_sig{this, Verilated::catName(namep, "mem_tb.int_res_write_sig")}
    , TOP__mem_tb__DOT__param_read_sig{this, Verilated::catName(namep, "mem_tb.param_read_sig")}
    , TOP__mem_tb__DOT__param_write_sig{this, Verilated::catName(namep, "mem_tb.param_write_sig")}
    , TOP__mem_tb__DOT__params__DOT__params_0_read{this, Verilated::catName(namep, "mem_tb.params.params_0_read")}
    , TOP__mem_tb__DOT__params__DOT__params_0_write{this, Verilated::catName(namep, "mem_tb.params.params_0_write")}
    , TOP__mem_tb__DOT__params__DOT__params_1_read{this, Verilated::catName(namep, "mem_tb.params.params_1_read")}
    , TOP__mem_tb__DOT__params__DOT__params_1_write{this, Verilated::catName(namep, "mem_tb.params.params_1_write")}
{
    // Configure time unit / time precision
    _vm_contextp__->timeunit(-9);
    _vm_contextp__->timeprecision(-11);
    // Setup each module's pointers to their submodules
    TOP.__PVT__Defines = &TOP__Defines;
    TOP.__PVT__mem_tb__DOT__int_res__DOT__int_res_0_read = &TOP__mem_tb__DOT__int_res__DOT__int_res_0_read;
    TOP.__PVT__mem_tb__DOT__int_res__DOT__int_res_0_write = &TOP__mem_tb__DOT__int_res__DOT__int_res_0_write;
    TOP.__PVT__mem_tb__DOT__int_res__DOT__int_res_1_read = &TOP__mem_tb__DOT__int_res__DOT__int_res_1_read;
    TOP.__PVT__mem_tb__DOT__int_res__DOT__int_res_1_write = &TOP__mem_tb__DOT__int_res__DOT__int_res_1_write;
    TOP.__PVT__mem_tb__DOT__int_res__DOT__int_res_2_read = &TOP__mem_tb__DOT__int_res__DOT__int_res_2_read;
    TOP.__PVT__mem_tb__DOT__int_res__DOT__int_res_2_write = &TOP__mem_tb__DOT__int_res__DOT__int_res_2_write;
    TOP.__PVT__mem_tb__DOT__int_res__DOT__int_res_3_read = &TOP__mem_tb__DOT__int_res__DOT__int_res_3_read;
    TOP.__PVT__mem_tb__DOT__int_res__DOT__int_res_3_write = &TOP__mem_tb__DOT__int_res__DOT__int_res_3_write;
    TOP.__PVT__mem_tb__DOT__int_res_read_sig = &TOP__mem_tb__DOT__int_res_read_sig;
    TOP.__PVT__mem_tb__DOT__int_res_write_sig = &TOP__mem_tb__DOT__int_res_write_sig;
    TOP.__PVT__mem_tb__DOT__param_read_sig = &TOP__mem_tb__DOT__param_read_sig;
    TOP.__PVT__mem_tb__DOT__param_write_sig = &TOP__mem_tb__DOT__param_write_sig;
    TOP.__PVT__mem_tb__DOT__params__DOT__params_0_read = &TOP__mem_tb__DOT__params__DOT__params_0_read;
    TOP.__PVT__mem_tb__DOT__params__DOT__params_0_write = &TOP__mem_tb__DOT__params__DOT__params_0_write;
    TOP.__PVT__mem_tb__DOT__params__DOT__params_1_read = &TOP__mem_tb__DOT__params__DOT__params_1_read;
    TOP.__PVT__mem_tb__DOT__params__DOT__params_1_write = &TOP__mem_tb__DOT__params__DOT__params_1_write;
    // Setup each module's pointer back to symbol table (for public functions)
    TOP.__Vconfigure(true);
    TOP__Defines.__Vconfigure(true);
    TOP__mem_tb__DOT__int_res__DOT__int_res_0_read.__Vconfigure(true);
    TOP__mem_tb__DOT__int_res__DOT__int_res_0_write.__Vconfigure(false);
    TOP__mem_tb__DOT__int_res__DOT__int_res_1_read.__Vconfigure(false);
    TOP__mem_tb__DOT__int_res__DOT__int_res_1_write.__Vconfigure(false);
    TOP__mem_tb__DOT__int_res__DOT__int_res_2_read.__Vconfigure(false);
    TOP__mem_tb__DOT__int_res__DOT__int_res_2_write.__Vconfigure(false);
    TOP__mem_tb__DOT__int_res__DOT__int_res_3_read.__Vconfigure(false);
    TOP__mem_tb__DOT__int_res__DOT__int_res_3_write.__Vconfigure(false);
    TOP__mem_tb__DOT__int_res_read_sig.__Vconfigure(true);
    TOP__mem_tb__DOT__int_res_write_sig.__Vconfigure(false);
    TOP__mem_tb__DOT__param_read_sig.__Vconfigure(true);
    TOP__mem_tb__DOT__param_write_sig.__Vconfigure(false);
    TOP__mem_tb__DOT__params__DOT__params_0_read.__Vconfigure(false);
    TOP__mem_tb__DOT__params__DOT__params_0_write.__Vconfigure(false);
    TOP__mem_tb__DOT__params__DOT__params_1_read.__Vconfigure(false);
    TOP__mem_tb__DOT__params__DOT__params_1_write.__Vconfigure(false);
    // Setup scopes
    __Vscope_Defines.configure(this, name(), "Defines", "Defines", -9, VerilatedScope::SCOPE_OTHER);
    __Vscope_TOP.configure(this, name(), "TOP", "TOP", 0, VerilatedScope::SCOPE_OTHER);
    __Vscope_mem_tb.configure(this, name(), "mem_tb", "mem_tb", -9, VerilatedScope::SCOPE_MODULE);
    __Vscope_mem_tb__int_res.configure(this, name(), "mem_tb.int_res", "int_res", -9, VerilatedScope::SCOPE_MODULE);
    __Vscope_mem_tb__int_res__int_res_0.configure(this, name(), "mem_tb.int_res.int_res_0", "int_res_0", -9, VerilatedScope::SCOPE_MODULE);
    __Vscope_mem_tb__int_res__int_res_0_read.configure(this, name(), "mem_tb.int_res.int_res_0_read", "int_res_0_read", -9, VerilatedScope::SCOPE_MODULE);
    __Vscope_mem_tb__int_res__int_res_0_write.configure(this, name(), "mem_tb.int_res.int_res_0_write", "int_res_0_write", -9, VerilatedScope::SCOPE_MODULE);
    __Vscope_mem_tb__int_res__int_res_1.configure(this, name(), "mem_tb.int_res.int_res_1", "int_res_1", -9, VerilatedScope::SCOPE_MODULE);
    __Vscope_mem_tb__int_res__int_res_1_read.configure(this, name(), "mem_tb.int_res.int_res_1_read", "int_res_1_read", -9, VerilatedScope::SCOPE_MODULE);
    __Vscope_mem_tb__int_res__int_res_1_write.configure(this, name(), "mem_tb.int_res.int_res_1_write", "int_res_1_write", -9, VerilatedScope::SCOPE_MODULE);
    __Vscope_mem_tb__int_res__int_res_2.configure(this, name(), "mem_tb.int_res.int_res_2", "int_res_2", -9, VerilatedScope::SCOPE_MODULE);
    __Vscope_mem_tb__int_res__int_res_2_read.configure(this, name(), "mem_tb.int_res.int_res_2_read", "int_res_2_read", -9, VerilatedScope::SCOPE_MODULE);
    __Vscope_mem_tb__int_res__int_res_2_write.configure(this, name(), "mem_tb.int_res.int_res_2_write", "int_res_2_write", -9, VerilatedScope::SCOPE_MODULE);
    __Vscope_mem_tb__int_res__int_res_3.configure(this, name(), "mem_tb.int_res.int_res_3", "int_res_3", -9, VerilatedScope::SCOPE_MODULE);
    __Vscope_mem_tb__int_res__int_res_3_read.configure(this, name(), "mem_tb.int_res.int_res_3_read", "int_res_3_read", -9, VerilatedScope::SCOPE_MODULE);
    __Vscope_mem_tb__int_res__int_res_3_write.configure(this, name(), "mem_tb.int_res.int_res_3_write", "int_res_3_write", -9, VerilatedScope::SCOPE_MODULE);
    __Vscope_mem_tb__int_res_read_sig.configure(this, name(), "mem_tb.int_res_read_sig", "int_res_read_sig", -9, VerilatedScope::SCOPE_MODULE);
    __Vscope_mem_tb__int_res_write_sig.configure(this, name(), "mem_tb.int_res_write_sig", "int_res_write_sig", -9, VerilatedScope::SCOPE_MODULE);
    __Vscope_mem_tb__param_read_sig.configure(this, name(), "mem_tb.param_read_sig", "param_read_sig", -9, VerilatedScope::SCOPE_MODULE);
    __Vscope_mem_tb__param_write_sig.configure(this, name(), "mem_tb.param_write_sig", "param_write_sig", -9, VerilatedScope::SCOPE_MODULE);
    __Vscope_mem_tb__params.configure(this, name(), "mem_tb.params", "params", -9, VerilatedScope::SCOPE_MODULE);
    __Vscope_mem_tb__params__params_0.configure(this, name(), "mem_tb.params.params_0", "params_0", -9, VerilatedScope::SCOPE_MODULE);
    __Vscope_mem_tb__params__params_0_read.configure(this, name(), "mem_tb.params.params_0_read", "params_0_read", -9, VerilatedScope::SCOPE_MODULE);
    __Vscope_mem_tb__params__params_0_write.configure(this, name(), "mem_tb.params.params_0_write", "params_0_write", -9, VerilatedScope::SCOPE_MODULE);
    __Vscope_mem_tb__params__params_1.configure(this, name(), "mem_tb.params.params_1", "params_1", -9, VerilatedScope::SCOPE_MODULE);
    __Vscope_mem_tb__params__params_1_read.configure(this, name(), "mem_tb.params.params_1_read", "params_1_read", -9, VerilatedScope::SCOPE_MODULE);
    __Vscope_mem_tb__params__params_1_write.configure(this, name(), "mem_tb.params.params_1_write", "params_1_write", -9, VerilatedScope::SCOPE_MODULE);

    // Set up scope hierarchy
    __Vhier.add(0, &__Vscope_mem_tb);
    __Vhier.add(&__Vscope_mem_tb, &__Vscope_mem_tb__int_res);
    __Vhier.add(&__Vscope_mem_tb, &__Vscope_mem_tb__int_res_read_sig);
    __Vhier.add(&__Vscope_mem_tb, &__Vscope_mem_tb__int_res_write_sig);
    __Vhier.add(&__Vscope_mem_tb, &__Vscope_mem_tb__param_read_sig);
    __Vhier.add(&__Vscope_mem_tb, &__Vscope_mem_tb__param_write_sig);
    __Vhier.add(&__Vscope_mem_tb, &__Vscope_mem_tb__params);
    __Vhier.add(&__Vscope_mem_tb__int_res, &__Vscope_mem_tb__int_res__int_res_0);
    __Vhier.add(&__Vscope_mem_tb__int_res, &__Vscope_mem_tb__int_res__int_res_0_read);
    __Vhier.add(&__Vscope_mem_tb__int_res, &__Vscope_mem_tb__int_res__int_res_0_write);
    __Vhier.add(&__Vscope_mem_tb__int_res, &__Vscope_mem_tb__int_res__int_res_1);
    __Vhier.add(&__Vscope_mem_tb__int_res, &__Vscope_mem_tb__int_res__int_res_1_read);
    __Vhier.add(&__Vscope_mem_tb__int_res, &__Vscope_mem_tb__int_res__int_res_1_write);
    __Vhier.add(&__Vscope_mem_tb__int_res, &__Vscope_mem_tb__int_res__int_res_2);
    __Vhier.add(&__Vscope_mem_tb__int_res, &__Vscope_mem_tb__int_res__int_res_2_read);
    __Vhier.add(&__Vscope_mem_tb__int_res, &__Vscope_mem_tb__int_res__int_res_2_write);
    __Vhier.add(&__Vscope_mem_tb__int_res, &__Vscope_mem_tb__int_res__int_res_3);
    __Vhier.add(&__Vscope_mem_tb__int_res, &__Vscope_mem_tb__int_res__int_res_3_read);
    __Vhier.add(&__Vscope_mem_tb__int_res, &__Vscope_mem_tb__int_res__int_res_3_write);
    __Vhier.add(&__Vscope_mem_tb__params, &__Vscope_mem_tb__params__params_0);
    __Vhier.add(&__Vscope_mem_tb__params, &__Vscope_mem_tb__params__params_0_read);
    __Vhier.add(&__Vscope_mem_tb__params, &__Vscope_mem_tb__params__params_0_write);
    __Vhier.add(&__Vscope_mem_tb__params, &__Vscope_mem_tb__params__params_1);
    __Vhier.add(&__Vscope_mem_tb__params, &__Vscope_mem_tb__params__params_1_read);
    __Vhier.add(&__Vscope_mem_tb__params, &__Vscope_mem_tb__params__params_1_write);

    // Setup export functions
    for (int __Vfinal = 0; __Vfinal < 2; ++__Vfinal) {
        __Vscope_Defines.varInsert(__Vfinal,"CIM_INT_RES_BANK_SIZE_NUM_WORD", const_cast<void*>(static_cast<const void*>(&(TOP__Defines.CIM_INT_RES_BANK_SIZE_NUM_WORD))), true, VLVT_UINT32,VLVD_NODIR|VLVF_PUB_RW|VLVF_DPI_CLAY,1 ,31,0);
        __Vscope_Defines.varInsert(__Vfinal,"CIM_INT_RES_NUM_BANKS", const_cast<void*>(static_cast<const void*>(&(TOP__Defines.CIM_INT_RES_NUM_BANKS))), true, VLVT_UINT32,VLVD_NODIR|VLVF_PUB_RW|VLVF_DPI_CLAY,1 ,31,0);
        __Vscope_Defines.varInsert(__Vfinal,"CIM_PARAMS_BANK_SIZE_NUM_WORD", const_cast<void*>(static_cast<const void*>(&(TOP__Defines.CIM_PARAMS_BANK_SIZE_NUM_WORD))), true, VLVT_UINT32,VLVD_NODIR|VLVF_PUB_RW|VLVF_DPI_CLAY,1 ,31,0);
        __Vscope_Defines.varInsert(__Vfinal,"CIM_PARAMS_NUM_BANKS", const_cast<void*>(static_cast<const void*>(&(TOP__Defines.CIM_PARAMS_NUM_BANKS))), true, VLVT_UINT32,VLVD_NODIR|VLVF_PUB_RW|VLVF_DPI_CLAY,1 ,31,0);
        __Vscope_Defines.varInsert(__Vfinal,"N_STO_INT_RES", const_cast<void*>(static_cast<const void*>(&(TOP__Defines.N_STO_INT_RES))), true, VLVT_UINT32,VLVD_NODIR|VLVF_PUB_RW|VLVF_DPI_CLAY,1 ,31,0);
        __Vscope_Defines.varInsert(__Vfinal,"N_STO_PARAMS", const_cast<void*>(static_cast<const void*>(&(TOP__Defines.N_STO_PARAMS))), true, VLVT_UINT32,VLVD_NODIR|VLVF_PUB_RW|VLVF_DPI_CLAY,1 ,31,0);
        __Vscope_TOP.varInsert(__Vfinal,"clk", &(TOP.clk), false, VLVT_UINT8,VLVD_IN|VLVF_PUB_RW,0);
        __Vscope_TOP.varInsert(__Vfinal,"int_res_chip_en", &(TOP.int_res_chip_en), false, VLVT_UINT8,VLVD_IN|VLVF_PUB_RW,0);
        __Vscope_TOP.varInsert(__Vfinal,"int_res_read_addr", &(TOP.int_res_read_addr), false, VLVT_UINT16,VLVD_IN|VLVF_PUB_RW,1 ,15,0);
        __Vscope_TOP.varInsert(__Vfinal,"int_res_read_data", &(TOP.int_res_read_data), false, VLVT_UINT32,VLVD_OUT|VLVF_PUB_RW,1 ,17,0);
        __Vscope_TOP.varInsert(__Vfinal,"int_res_read_data_width", &(TOP.int_res_read_data_width), false, VLVT_UINT8,VLVD_IN|VLVF_PUB_RW,0);
        __Vscope_TOP.varInsert(__Vfinal,"int_res_read_en", &(TOP.int_res_read_en), false, VLVT_UINT8,VLVD_IN|VLVF_PUB_RW,0);
        __Vscope_TOP.varInsert(__Vfinal,"int_res_write_addr", &(TOP.int_res_write_addr), false, VLVT_UINT16,VLVD_IN|VLVF_PUB_RW,1 ,15,0);
        __Vscope_TOP.varInsert(__Vfinal,"int_res_write_data", &(TOP.int_res_write_data), false, VLVT_UINT32,VLVD_IN|VLVF_PUB_RW,1 ,17,0);
        __Vscope_TOP.varInsert(__Vfinal,"int_res_write_data_width", &(TOP.int_res_write_data_width), false, VLVT_UINT8,VLVD_IN|VLVF_PUB_RW,0);
        __Vscope_TOP.varInsert(__Vfinal,"int_res_write_en", &(TOP.int_res_write_en), false, VLVT_UINT8,VLVD_IN|VLVF_PUB_RW,0);
        __Vscope_TOP.varInsert(__Vfinal,"param_chip_en", &(TOP.param_chip_en), false, VLVT_UINT8,VLVD_IN|VLVF_PUB_RW,0);
        __Vscope_TOP.varInsert(__Vfinal,"param_read_addr", &(TOP.param_read_addr), false, VLVT_UINT16,VLVD_IN|VLVF_PUB_RW,1 ,14,0);
        __Vscope_TOP.varInsert(__Vfinal,"param_read_data", &(TOP.param_read_data), false, VLVT_UINT16,VLVD_OUT|VLVF_PUB_RW,1 ,8,0);
        __Vscope_TOP.varInsert(__Vfinal,"param_read_data_width", &(TOP.param_read_data_width), false, VLVT_UINT8,VLVD_IN|VLVF_PUB_RW,0);
        __Vscope_TOP.varInsert(__Vfinal,"param_read_en", &(TOP.param_read_en), false, VLVT_UINT8,VLVD_IN|VLVF_PUB_RW,0);
        __Vscope_TOP.varInsert(__Vfinal,"param_write_addr", &(TOP.param_write_addr), false, VLVT_UINT16,VLVD_IN|VLVF_PUB_RW,1 ,14,0);
        __Vscope_TOP.varInsert(__Vfinal,"param_write_data", &(TOP.param_write_data), false, VLVT_UINT16,VLVD_IN|VLVF_PUB_RW,1 ,8,0);
        __Vscope_TOP.varInsert(__Vfinal,"param_write_data_width", &(TOP.param_write_data_width), false, VLVT_UINT8,VLVD_IN|VLVF_PUB_RW,0);
        __Vscope_TOP.varInsert(__Vfinal,"param_write_en", &(TOP.param_write_en), false, VLVT_UINT8,VLVD_IN|VLVF_PUB_RW,0);
        __Vscope_TOP.varInsert(__Vfinal,"rst_n", &(TOP.rst_n), false, VLVT_UINT8,VLVD_IN|VLVF_PUB_RW,0);
        __Vscope_mem_tb.varInsert(__Vfinal,"clk", &(TOP.mem_tb__DOT__clk), false, VLVT_UINT8,VLVD_NODIR|VLVF_PUB_RW,0);
        __Vscope_mem_tb.varInsert(__Vfinal,"int_res_chip_en", &(TOP.mem_tb__DOT__int_res_chip_en), false, VLVT_UINT8,VLVD_NODIR|VLVF_PUB_RW,0);
        __Vscope_mem_tb.varInsert(__Vfinal,"int_res_read_addr", &(TOP.mem_tb__DOT__int_res_read_addr), false, VLVT_UINT16,VLVD_NODIR|VLVF_PUB_RW,1 ,15,0);
        __Vscope_mem_tb.varInsert(__Vfinal,"int_res_read_data", &(TOP.mem_tb__DOT__int_res_read_data), false, VLVT_UINT32,VLVD_NODIR|VLVF_PUB_RW,1 ,17,0);
        __Vscope_mem_tb.varInsert(__Vfinal,"int_res_read_data_width", &(TOP.mem_tb__DOT__int_res_read_data_width), false, VLVT_UINT8,VLVD_NODIR|VLVF_PUB_RW,0);
        __Vscope_mem_tb.varInsert(__Vfinal,"int_res_read_en", &(TOP.mem_tb__DOT__int_res_read_en), false, VLVT_UINT8,VLVD_NODIR|VLVF_PUB_RW,0);
        __Vscope_mem_tb.varInsert(__Vfinal,"int_res_write_addr", &(TOP.mem_tb__DOT__int_res_write_addr), false, VLVT_UINT16,VLVD_NODIR|VLVF_PUB_RW,1 ,15,0);
        __Vscope_mem_tb.varInsert(__Vfinal,"int_res_write_data", &(TOP.mem_tb__DOT__int_res_write_data), false, VLVT_UINT32,VLVD_NODIR|VLVF_PUB_RW,1 ,17,0);
        __Vscope_mem_tb.varInsert(__Vfinal,"int_res_write_data_width", &(TOP.mem_tb__DOT__int_res_write_data_width), false, VLVT_UINT8,VLVD_NODIR|VLVF_PUB_RW,0);
        __Vscope_mem_tb.varInsert(__Vfinal,"int_res_write_en", &(TOP.mem_tb__DOT__int_res_write_en), false, VLVT_UINT8,VLVD_NODIR|VLVF_PUB_RW,0);
        __Vscope_mem_tb.varInsert(__Vfinal,"param_chip_en", &(TOP.mem_tb__DOT__param_chip_en), false, VLVT_UINT8,VLVD_NODIR|VLVF_PUB_RW,0);
        __Vscope_mem_tb.varInsert(__Vfinal,"param_read_addr", &(TOP.mem_tb__DOT__param_read_addr), false, VLVT_UINT16,VLVD_NODIR|VLVF_PUB_RW,1 ,14,0);
        __Vscope_mem_tb.varInsert(__Vfinal,"param_read_data", &(TOP.mem_tb__DOT__param_read_data), false, VLVT_UINT16,VLVD_NODIR|VLVF_PUB_RW,1 ,8,0);
        __Vscope_mem_tb.varInsert(__Vfinal,"param_read_data_width", &(TOP.mem_tb__DOT__param_read_data_width), false, VLVT_UINT8,VLVD_NODIR|VLVF_PUB_RW,0);
        __Vscope_mem_tb.varInsert(__Vfinal,"param_read_en", &(TOP.mem_tb__DOT__param_read_en), false, VLVT_UINT8,VLVD_NODIR|VLVF_PUB_RW,0);
        __Vscope_mem_tb.varInsert(__Vfinal,"param_write_addr", &(TOP.mem_tb__DOT__param_write_addr), false, VLVT_UINT16,VLVD_NODIR|VLVF_PUB_RW,1 ,14,0);
        __Vscope_mem_tb.varInsert(__Vfinal,"param_write_data", &(TOP.mem_tb__DOT__param_write_data), false, VLVT_UINT16,VLVD_NODIR|VLVF_PUB_RW,1 ,8,0);
        __Vscope_mem_tb.varInsert(__Vfinal,"param_write_data_width", &(TOP.mem_tb__DOT__param_write_data_width), false, VLVT_UINT8,VLVD_NODIR|VLVF_PUB_RW,0);
        __Vscope_mem_tb.varInsert(__Vfinal,"param_write_en", &(TOP.mem_tb__DOT__param_write_en), false, VLVT_UINT8,VLVD_NODIR|VLVF_PUB_RW,0);
        __Vscope_mem_tb.varInsert(__Vfinal,"rst_n", &(TOP.mem_tb__DOT__rst_n), false, VLVT_UINT8,VLVD_NODIR|VLVF_PUB_RW,0);
        __Vscope_mem_tb__int_res.varInsert(__Vfinal,"bank_read_current", &(TOP.mem_tb__DOT__int_res__DOT__bank_read_current), false, VLVT_UINT8,VLVD_NODIR|VLVF_PUB_RW,1 ,1,0);
        __Vscope_mem_tb__int_res.varInsert(__Vfinal,"bank_read_prev", &(TOP.mem_tb__DOT__int_res__DOT__bank_read_prev), false, VLVT_UINT8,VLVD_NODIR|VLVF_PUB_RW,1 ,1,0);
        __Vscope_mem_tb__int_res.varInsert(__Vfinal,"bank_write_current", &(TOP.mem_tb__DOT__int_res__DOT__bank_write_current), false, VLVT_UINT8,VLVD_NODIR|VLVF_PUB_RW,1 ,1,0);
        __Vscope_mem_tb__int_res.varInsert(__Vfinal,"clk", &(TOP.mem_tb__DOT__int_res__DOT__clk), false, VLVT_UINT8,VLVD_NODIR|VLVF_PUB_RW,0);
        __Vscope_mem_tb__int_res.varInsert(__Vfinal,"read_base_addr", &(TOP.mem_tb__DOT__int_res__DOT__read_base_addr), false, VLVT_UINT16,VLVD_NODIR|VLVF_PUB_RW,1 ,13,0);
        __Vscope_mem_tb__int_res.varInsert(__Vfinal,"read_data_width_prev", &(TOP.mem_tb__DOT__int_res__DOT__read_data_width_prev), false, VLVT_UINT8,VLVD_NODIR|VLVF_PUB_RW,0);
        __Vscope_mem_tb__int_res.varInsert(__Vfinal,"rst_n", &(TOP.mem_tb__DOT__int_res__DOT__rst_n), false, VLVT_UINT8,VLVD_NODIR|VLVF_PUB_RW,0);
        __Vscope_mem_tb__int_res.varInsert(__Vfinal,"write_base_addr", &(TOP.mem_tb__DOT__int_res__DOT__write_base_addr), false, VLVT_UINT16,VLVD_NODIR|VLVF_PUB_RW,1 ,13,0);
        __Vscope_mem_tb__int_res__int_res_0.varInsert(__Vfinal,"CEN", &(TOP.mem_tb__DOT__int_res__DOT__int_res_0__DOT__CEN), false, VLVT_UINT8,VLVD_NODIR|VLVF_PUB_RW,0);
        __Vscope_mem_tb__int_res__int_res_0.varInsert(__Vfinal,"DEPTH", const_cast<void*>(static_cast<const void*>(&(TOP.mem_tb__DOT__int_res__DOT__int_res_0__DOT__DEPTH))), true, VLVT_UINT32,VLVD_NODIR|VLVF_PUB_RW|VLVF_DPI_CLAY,1 ,31,0);
        __Vscope_mem_tb__int_res__int_res_0.varInsert(__Vfinal,"WEN", &(TOP.mem_tb__DOT__int_res__DOT__int_res_0__DOT__WEN), false, VLVT_UINT8,VLVD_NODIR|VLVF_PUB_RW,0);
        __Vscope_mem_tb__int_res__int_res_0.varInsert(__Vfinal,"clk", &(TOP.mem_tb__DOT__int_res__DOT__int_res_0__DOT__clk), false, VLVT_UINT8,VLVD_NODIR|VLVF_PUB_RW,0);
        __Vscope_mem_tb__int_res__int_res_0.varInsert(__Vfinal,"memory", &(TOP.mem_tb__DOT__int_res__DOT__int_res_0__DOT__memory), false, VLVT_UINT16,VLVD_NODIR|VLVF_PUB_RW,2 ,8,0 ,14335,0);
        __Vscope_mem_tb__int_res__int_res_0.varInsert(__Vfinal,"rst_n", &(TOP.mem_tb__DOT__int_res__DOT__int_res_0__DOT__rst_n), false, VLVT_UINT8,VLVD_NODIR|VLVF_PUB_RW,0);
        __Vscope_mem_tb__int_res__int_res_0_read.varInsert(__Vfinal,"addr", &(TOP__mem_tb__DOT__int_res__DOT__int_res_0_read.addr), false, VLVT_UINT16,VLVD_NODIR|VLVF_PUB_RW,1 ,13,0);
        __Vscope_mem_tb__int_res__int_res_0_read.varInsert(__Vfinal,"chip_en", &(TOP__mem_tb__DOT__int_res__DOT__int_res_0_read.chip_en), false, VLVT_UINT8,VLVD_NODIR|VLVF_PUB_RW,0);
        __Vscope_mem_tb__int_res__int_res_0_read.varInsert(__Vfinal,"data", &(TOP__mem_tb__DOT__int_res__DOT__int_res_0_read.data), false, VLVT_UINT16,VLVD_NODIR|VLVF_PUB_RW,1 ,8,0);
        __Vscope_mem_tb__int_res__int_res_0_read.varInsert(__Vfinal,"data_width", &(TOP__mem_tb__DOT__int_res__DOT__int_res_0_read.data_width), false, VLVT_UINT8,VLVD_NODIR|VLVF_PUB_RW,0);
        __Vscope_mem_tb__int_res__int_res_0_read.varInsert(__Vfinal,"en", &(TOP__mem_tb__DOT__int_res__DOT__int_res_0_read.en), false, VLVT_UINT8,VLVD_NODIR|VLVF_PUB_RW,0);
        __Vscope_mem_tb__int_res__int_res_0_write.varInsert(__Vfinal,"addr", &(TOP__mem_tb__DOT__int_res__DOT__int_res_0_write.addr), false, VLVT_UINT16,VLVD_NODIR|VLVF_PUB_RW,1 ,13,0);
        __Vscope_mem_tb__int_res__int_res_0_write.varInsert(__Vfinal,"chip_en", &(TOP__mem_tb__DOT__int_res__DOT__int_res_0_write.chip_en), false, VLVT_UINT8,VLVD_NODIR|VLVF_PUB_RW,0);
        __Vscope_mem_tb__int_res__int_res_0_write.varInsert(__Vfinal,"data", &(TOP__mem_tb__DOT__int_res__DOT__int_res_0_write.data), false, VLVT_UINT16,VLVD_NODIR|VLVF_PUB_RW,1 ,8,0);
        __Vscope_mem_tb__int_res__int_res_0_write.varInsert(__Vfinal,"data_width", &(TOP__mem_tb__DOT__int_res__DOT__int_res_0_write.data_width), false, VLVT_UINT8,VLVD_NODIR|VLVF_PUB_RW,0);
        __Vscope_mem_tb__int_res__int_res_0_write.varInsert(__Vfinal,"en", &(TOP__mem_tb__DOT__int_res__DOT__int_res_0_write.en), false, VLVT_UINT8,VLVD_NODIR|VLVF_PUB_RW,0);
        __Vscope_mem_tb__int_res__int_res_1.varInsert(__Vfinal,"CEN", &(TOP.mem_tb__DOT__int_res__DOT__int_res_1__DOT__CEN), false, VLVT_UINT8,VLVD_NODIR|VLVF_PUB_RW,0);
        __Vscope_mem_tb__int_res__int_res_1.varInsert(__Vfinal,"DEPTH", const_cast<void*>(static_cast<const void*>(&(TOP.mem_tb__DOT__int_res__DOT__int_res_1__DOT__DEPTH))), true, VLVT_UINT32,VLVD_NODIR|VLVF_PUB_RW|VLVF_DPI_CLAY,1 ,31,0);
        __Vscope_mem_tb__int_res__int_res_1.varInsert(__Vfinal,"WEN", &(TOP.mem_tb__DOT__int_res__DOT__int_res_1__DOT__WEN), false, VLVT_UINT8,VLVD_NODIR|VLVF_PUB_RW,0);
        __Vscope_mem_tb__int_res__int_res_1.varInsert(__Vfinal,"clk", &(TOP.mem_tb__DOT__int_res__DOT__int_res_1__DOT__clk), false, VLVT_UINT8,VLVD_NODIR|VLVF_PUB_RW,0);
        __Vscope_mem_tb__int_res__int_res_1.varInsert(__Vfinal,"memory", &(TOP.mem_tb__DOT__int_res__DOT__int_res_1__DOT__memory), false, VLVT_UINT16,VLVD_NODIR|VLVF_PUB_RW,2 ,8,0 ,14335,0);
        __Vscope_mem_tb__int_res__int_res_1.varInsert(__Vfinal,"rst_n", &(TOP.mem_tb__DOT__int_res__DOT__int_res_1__DOT__rst_n), false, VLVT_UINT8,VLVD_NODIR|VLVF_PUB_RW,0);
        __Vscope_mem_tb__int_res__int_res_1_read.varInsert(__Vfinal,"addr", &(TOP__mem_tb__DOT__int_res__DOT__int_res_1_read.addr), false, VLVT_UINT16,VLVD_NODIR|VLVF_PUB_RW,1 ,13,0);
        __Vscope_mem_tb__int_res__int_res_1_read.varInsert(__Vfinal,"chip_en", &(TOP__mem_tb__DOT__int_res__DOT__int_res_1_read.chip_en), false, VLVT_UINT8,VLVD_NODIR|VLVF_PUB_RW,0);
        __Vscope_mem_tb__int_res__int_res_1_read.varInsert(__Vfinal,"data", &(TOP__mem_tb__DOT__int_res__DOT__int_res_1_read.data), false, VLVT_UINT16,VLVD_NODIR|VLVF_PUB_RW,1 ,8,0);
        __Vscope_mem_tb__int_res__int_res_1_read.varInsert(__Vfinal,"data_width", &(TOP__mem_tb__DOT__int_res__DOT__int_res_1_read.data_width), false, VLVT_UINT8,VLVD_NODIR|VLVF_PUB_RW,0);
        __Vscope_mem_tb__int_res__int_res_1_read.varInsert(__Vfinal,"en", &(TOP__mem_tb__DOT__int_res__DOT__int_res_1_read.en), false, VLVT_UINT8,VLVD_NODIR|VLVF_PUB_RW,0);
        __Vscope_mem_tb__int_res__int_res_1_write.varInsert(__Vfinal,"addr", &(TOP__mem_tb__DOT__int_res__DOT__int_res_1_write.addr), false, VLVT_UINT16,VLVD_NODIR|VLVF_PUB_RW,1 ,13,0);
        __Vscope_mem_tb__int_res__int_res_1_write.varInsert(__Vfinal,"chip_en", &(TOP__mem_tb__DOT__int_res__DOT__int_res_1_write.chip_en), false, VLVT_UINT8,VLVD_NODIR|VLVF_PUB_RW,0);
        __Vscope_mem_tb__int_res__int_res_1_write.varInsert(__Vfinal,"data", &(TOP__mem_tb__DOT__int_res__DOT__int_res_1_write.data), false, VLVT_UINT16,VLVD_NODIR|VLVF_PUB_RW,1 ,8,0);
        __Vscope_mem_tb__int_res__int_res_1_write.varInsert(__Vfinal,"data_width", &(TOP__mem_tb__DOT__int_res__DOT__int_res_1_write.data_width), false, VLVT_UINT8,VLVD_NODIR|VLVF_PUB_RW,0);
        __Vscope_mem_tb__int_res__int_res_1_write.varInsert(__Vfinal,"en", &(TOP__mem_tb__DOT__int_res__DOT__int_res_1_write.en), false, VLVT_UINT8,VLVD_NODIR|VLVF_PUB_RW,0);
        __Vscope_mem_tb__int_res__int_res_2.varInsert(__Vfinal,"CEN", &(TOP.mem_tb__DOT__int_res__DOT__int_res_2__DOT__CEN), false, VLVT_UINT8,VLVD_NODIR|VLVF_PUB_RW,0);
        __Vscope_mem_tb__int_res__int_res_2.varInsert(__Vfinal,"DEPTH", const_cast<void*>(static_cast<const void*>(&(TOP.mem_tb__DOT__int_res__DOT__int_res_2__DOT__DEPTH))), true, VLVT_UINT32,VLVD_NODIR|VLVF_PUB_RW|VLVF_DPI_CLAY,1 ,31,0);
        __Vscope_mem_tb__int_res__int_res_2.varInsert(__Vfinal,"WEN", &(TOP.mem_tb__DOT__int_res__DOT__int_res_2__DOT__WEN), false, VLVT_UINT8,VLVD_NODIR|VLVF_PUB_RW,0);
        __Vscope_mem_tb__int_res__int_res_2.varInsert(__Vfinal,"clk", &(TOP.mem_tb__DOT__int_res__DOT__int_res_2__DOT__clk), false, VLVT_UINT8,VLVD_NODIR|VLVF_PUB_RW,0);
        __Vscope_mem_tb__int_res__int_res_2.varInsert(__Vfinal,"memory", &(TOP.mem_tb__DOT__int_res__DOT__int_res_2__DOT__memory), false, VLVT_UINT16,VLVD_NODIR|VLVF_PUB_RW,2 ,8,0 ,14335,0);
        __Vscope_mem_tb__int_res__int_res_2.varInsert(__Vfinal,"rst_n", &(TOP.mem_tb__DOT__int_res__DOT__int_res_2__DOT__rst_n), false, VLVT_UINT8,VLVD_NODIR|VLVF_PUB_RW,0);
        __Vscope_mem_tb__int_res__int_res_2_read.varInsert(__Vfinal,"addr", &(TOP__mem_tb__DOT__int_res__DOT__int_res_2_read.addr), false, VLVT_UINT16,VLVD_NODIR|VLVF_PUB_RW,1 ,13,0);
        __Vscope_mem_tb__int_res__int_res_2_read.varInsert(__Vfinal,"chip_en", &(TOP__mem_tb__DOT__int_res__DOT__int_res_2_read.chip_en), false, VLVT_UINT8,VLVD_NODIR|VLVF_PUB_RW,0);
        __Vscope_mem_tb__int_res__int_res_2_read.varInsert(__Vfinal,"data", &(TOP__mem_tb__DOT__int_res__DOT__int_res_2_read.data), false, VLVT_UINT16,VLVD_NODIR|VLVF_PUB_RW,1 ,8,0);
        __Vscope_mem_tb__int_res__int_res_2_read.varInsert(__Vfinal,"data_width", &(TOP__mem_tb__DOT__int_res__DOT__int_res_2_read.data_width), false, VLVT_UINT8,VLVD_NODIR|VLVF_PUB_RW,0);
        __Vscope_mem_tb__int_res__int_res_2_read.varInsert(__Vfinal,"en", &(TOP__mem_tb__DOT__int_res__DOT__int_res_2_read.en), false, VLVT_UINT8,VLVD_NODIR|VLVF_PUB_RW,0);
        __Vscope_mem_tb__int_res__int_res_2_write.varInsert(__Vfinal,"addr", &(TOP__mem_tb__DOT__int_res__DOT__int_res_2_write.addr), false, VLVT_UINT16,VLVD_NODIR|VLVF_PUB_RW,1 ,13,0);
        __Vscope_mem_tb__int_res__int_res_2_write.varInsert(__Vfinal,"chip_en", &(TOP__mem_tb__DOT__int_res__DOT__int_res_2_write.chip_en), false, VLVT_UINT8,VLVD_NODIR|VLVF_PUB_RW,0);
        __Vscope_mem_tb__int_res__int_res_2_write.varInsert(__Vfinal,"data", &(TOP__mem_tb__DOT__int_res__DOT__int_res_2_write.data), false, VLVT_UINT16,VLVD_NODIR|VLVF_PUB_RW,1 ,8,0);
        __Vscope_mem_tb__int_res__int_res_2_write.varInsert(__Vfinal,"data_width", &(TOP__mem_tb__DOT__int_res__DOT__int_res_2_write.data_width), false, VLVT_UINT8,VLVD_NODIR|VLVF_PUB_RW,0);
        __Vscope_mem_tb__int_res__int_res_2_write.varInsert(__Vfinal,"en", &(TOP__mem_tb__DOT__int_res__DOT__int_res_2_write.en), false, VLVT_UINT8,VLVD_NODIR|VLVF_PUB_RW,0);
        __Vscope_mem_tb__int_res__int_res_3.varInsert(__Vfinal,"CEN", &(TOP.mem_tb__DOT__int_res__DOT__int_res_3__DOT__CEN), false, VLVT_UINT8,VLVD_NODIR|VLVF_PUB_RW,0);
        __Vscope_mem_tb__int_res__int_res_3.varInsert(__Vfinal,"DEPTH", const_cast<void*>(static_cast<const void*>(&(TOP.mem_tb__DOT__int_res__DOT__int_res_3__DOT__DEPTH))), true, VLVT_UINT32,VLVD_NODIR|VLVF_PUB_RW|VLVF_DPI_CLAY,1 ,31,0);
        __Vscope_mem_tb__int_res__int_res_3.varInsert(__Vfinal,"WEN", &(TOP.mem_tb__DOT__int_res__DOT__int_res_3__DOT__WEN), false, VLVT_UINT8,VLVD_NODIR|VLVF_PUB_RW,0);
        __Vscope_mem_tb__int_res__int_res_3.varInsert(__Vfinal,"clk", &(TOP.mem_tb__DOT__int_res__DOT__int_res_3__DOT__clk), false, VLVT_UINT8,VLVD_NODIR|VLVF_PUB_RW,0);
        __Vscope_mem_tb__int_res__int_res_3.varInsert(__Vfinal,"memory", &(TOP.mem_tb__DOT__int_res__DOT__int_res_3__DOT__memory), false, VLVT_UINT16,VLVD_NODIR|VLVF_PUB_RW,2 ,8,0 ,14335,0);
        __Vscope_mem_tb__int_res__int_res_3.varInsert(__Vfinal,"rst_n", &(TOP.mem_tb__DOT__int_res__DOT__int_res_3__DOT__rst_n), false, VLVT_UINT8,VLVD_NODIR|VLVF_PUB_RW,0);
        __Vscope_mem_tb__int_res__int_res_3_read.varInsert(__Vfinal,"addr", &(TOP__mem_tb__DOT__int_res__DOT__int_res_3_read.addr), false, VLVT_UINT16,VLVD_NODIR|VLVF_PUB_RW,1 ,13,0);
        __Vscope_mem_tb__int_res__int_res_3_read.varInsert(__Vfinal,"chip_en", &(TOP__mem_tb__DOT__int_res__DOT__int_res_3_read.chip_en), false, VLVT_UINT8,VLVD_NODIR|VLVF_PUB_RW,0);
        __Vscope_mem_tb__int_res__int_res_3_read.varInsert(__Vfinal,"data", &(TOP__mem_tb__DOT__int_res__DOT__int_res_3_read.data), false, VLVT_UINT16,VLVD_NODIR|VLVF_PUB_RW,1 ,8,0);
        __Vscope_mem_tb__int_res__int_res_3_read.varInsert(__Vfinal,"data_width", &(TOP__mem_tb__DOT__int_res__DOT__int_res_3_read.data_width), false, VLVT_UINT8,VLVD_NODIR|VLVF_PUB_RW,0);
        __Vscope_mem_tb__int_res__int_res_3_read.varInsert(__Vfinal,"en", &(TOP__mem_tb__DOT__int_res__DOT__int_res_3_read.en), false, VLVT_UINT8,VLVD_NODIR|VLVF_PUB_RW,0);
        __Vscope_mem_tb__int_res__int_res_3_write.varInsert(__Vfinal,"addr", &(TOP__mem_tb__DOT__int_res__DOT__int_res_3_write.addr), false, VLVT_UINT16,VLVD_NODIR|VLVF_PUB_RW,1 ,13,0);
        __Vscope_mem_tb__int_res__int_res_3_write.varInsert(__Vfinal,"chip_en", &(TOP__mem_tb__DOT__int_res__DOT__int_res_3_write.chip_en), false, VLVT_UINT8,VLVD_NODIR|VLVF_PUB_RW,0);
        __Vscope_mem_tb__int_res__int_res_3_write.varInsert(__Vfinal,"data", &(TOP__mem_tb__DOT__int_res__DOT__int_res_3_write.data), false, VLVT_UINT16,VLVD_NODIR|VLVF_PUB_RW,1 ,8,0);
        __Vscope_mem_tb__int_res__int_res_3_write.varInsert(__Vfinal,"data_width", &(TOP__mem_tb__DOT__int_res__DOT__int_res_3_write.data_width), false, VLVT_UINT8,VLVD_NODIR|VLVF_PUB_RW,0);
        __Vscope_mem_tb__int_res__int_res_3_write.varInsert(__Vfinal,"en", &(TOP__mem_tb__DOT__int_res__DOT__int_res_3_write.en), false, VLVT_UINT8,VLVD_NODIR|VLVF_PUB_RW,0);
        __Vscope_mem_tb__int_res_read_sig.varInsert(__Vfinal,"addr", &(TOP__mem_tb__DOT__int_res_read_sig.addr), false, VLVT_UINT16,VLVD_NODIR|VLVF_PUB_RW,1 ,15,0);
        __Vscope_mem_tb__int_res_read_sig.varInsert(__Vfinal,"chip_en", &(TOP__mem_tb__DOT__int_res_read_sig.chip_en), false, VLVT_UINT8,VLVD_NODIR|VLVF_PUB_RW,0);
        __Vscope_mem_tb__int_res_read_sig.varInsert(__Vfinal,"data", &(TOP__mem_tb__DOT__int_res_read_sig.data), false, VLVT_UINT32,VLVD_NODIR|VLVF_PUB_RW,1 ,17,0);
        __Vscope_mem_tb__int_res_read_sig.varInsert(__Vfinal,"data_width", &(TOP__mem_tb__DOT__int_res_read_sig.data_width), false, VLVT_UINT8,VLVD_NODIR|VLVF_PUB_RW,0);
        __Vscope_mem_tb__int_res_read_sig.varInsert(__Vfinal,"en", &(TOP__mem_tb__DOT__int_res_read_sig.en), false, VLVT_UINT8,VLVD_NODIR|VLVF_PUB_RW,0);
        __Vscope_mem_tb__int_res_write_sig.varInsert(__Vfinal,"addr", &(TOP__mem_tb__DOT__int_res_write_sig.addr), false, VLVT_UINT16,VLVD_NODIR|VLVF_PUB_RW,1 ,15,0);
        __Vscope_mem_tb__int_res_write_sig.varInsert(__Vfinal,"chip_en", &(TOP__mem_tb__DOT__int_res_write_sig.chip_en), false, VLVT_UINT8,VLVD_NODIR|VLVF_PUB_RW,0);
        __Vscope_mem_tb__int_res_write_sig.varInsert(__Vfinal,"data", &(TOP__mem_tb__DOT__int_res_write_sig.data), false, VLVT_UINT32,VLVD_NODIR|VLVF_PUB_RW,1 ,17,0);
        __Vscope_mem_tb__int_res_write_sig.varInsert(__Vfinal,"data_width", &(TOP__mem_tb__DOT__int_res_write_sig.data_width), false, VLVT_UINT8,VLVD_NODIR|VLVF_PUB_RW,0);
        __Vscope_mem_tb__int_res_write_sig.varInsert(__Vfinal,"en", &(TOP__mem_tb__DOT__int_res_write_sig.en), false, VLVT_UINT8,VLVD_NODIR|VLVF_PUB_RW,0);
        __Vscope_mem_tb__param_read_sig.varInsert(__Vfinal,"addr", &(TOP__mem_tb__DOT__param_read_sig.addr), false, VLVT_UINT16,VLVD_NODIR|VLVF_PUB_RW,1 ,14,0);
        __Vscope_mem_tb__param_read_sig.varInsert(__Vfinal,"chip_en", &(TOP__mem_tb__DOT__param_read_sig.chip_en), false, VLVT_UINT8,VLVD_NODIR|VLVF_PUB_RW,0);
        __Vscope_mem_tb__param_read_sig.varInsert(__Vfinal,"data", &(TOP__mem_tb__DOT__param_read_sig.data), false, VLVT_UINT16,VLVD_NODIR|VLVF_PUB_RW,1 ,8,0);
        __Vscope_mem_tb__param_read_sig.varInsert(__Vfinal,"data_width", &(TOP__mem_tb__DOT__param_read_sig.data_width), false, VLVT_UINT8,VLVD_NODIR|VLVF_PUB_RW,0);
        __Vscope_mem_tb__param_read_sig.varInsert(__Vfinal,"en", &(TOP__mem_tb__DOT__param_read_sig.en), false, VLVT_UINT8,VLVD_NODIR|VLVF_PUB_RW,0);
        __Vscope_mem_tb__param_write_sig.varInsert(__Vfinal,"addr", &(TOP__mem_tb__DOT__param_write_sig.addr), false, VLVT_UINT16,VLVD_NODIR|VLVF_PUB_RW,1 ,14,0);
        __Vscope_mem_tb__param_write_sig.varInsert(__Vfinal,"chip_en", &(TOP__mem_tb__DOT__param_write_sig.chip_en), false, VLVT_UINT8,VLVD_NODIR|VLVF_PUB_RW,0);
        __Vscope_mem_tb__param_write_sig.varInsert(__Vfinal,"data", &(TOP__mem_tb__DOT__param_write_sig.data), false, VLVT_UINT16,VLVD_NODIR|VLVF_PUB_RW,1 ,8,0);
        __Vscope_mem_tb__param_write_sig.varInsert(__Vfinal,"data_width", &(TOP__mem_tb__DOT__param_write_sig.data_width), false, VLVT_UINT8,VLVD_NODIR|VLVF_PUB_RW,0);
        __Vscope_mem_tb__param_write_sig.varInsert(__Vfinal,"en", &(TOP__mem_tb__DOT__param_write_sig.en), false, VLVT_UINT8,VLVD_NODIR|VLVF_PUB_RW,0);
        __Vscope_mem_tb__params.varInsert(__Vfinal,"clk", &(TOP.mem_tb__DOT__params__DOT__clk), false, VLVT_UINT8,VLVD_NODIR|VLVF_PUB_RW,0);
        __Vscope_mem_tb__params.varInsert(__Vfinal,"params_0_read_en_prev", &(TOP.mem_tb__DOT__params__DOT__params_0_read_en_prev), false, VLVT_UINT8,VLVD_NODIR|VLVF_PUB_RW,0);
        __Vscope_mem_tb__params.varInsert(__Vfinal,"params_1_read_en_prev", &(TOP.mem_tb__DOT__params__DOT__params_1_read_en_prev), false, VLVT_UINT8,VLVD_NODIR|VLVF_PUB_RW,0);
        __Vscope_mem_tb__params.varInsert(__Vfinal,"rst_n", &(TOP.mem_tb__DOT__params__DOT__rst_n), false, VLVT_UINT8,VLVD_NODIR|VLVF_PUB_RW,0);
        __Vscope_mem_tb__params__params_0.varInsert(__Vfinal,"CEN", &(TOP.mem_tb__DOT__params__DOT__params_0__DOT__CEN), false, VLVT_UINT8,VLVD_NODIR|VLVF_PUB_RW,0);
        __Vscope_mem_tb__params__params_0.varInsert(__Vfinal,"DEPTH", const_cast<void*>(static_cast<const void*>(&(TOP.mem_tb__DOT__params__DOT__params_0__DOT__DEPTH))), true, VLVT_UINT32,VLVD_NODIR|VLVF_PUB_RW|VLVF_DPI_CLAY,1 ,31,0);
        __Vscope_mem_tb__params__params_0.varInsert(__Vfinal,"WEN", &(TOP.mem_tb__DOT__params__DOT__params_0__DOT__WEN), false, VLVT_UINT8,VLVD_NODIR|VLVF_PUB_RW,0);
        __Vscope_mem_tb__params__params_0.varInsert(__Vfinal,"clk", &(TOP.mem_tb__DOT__params__DOT__params_0__DOT__clk), false, VLVT_UINT8,VLVD_NODIR|VLVF_PUB_RW,0);
        __Vscope_mem_tb__params__params_0.varInsert(__Vfinal,"memory", &(TOP.mem_tb__DOT__params__DOT__params_0__DOT__memory), false, VLVT_UINT16,VLVD_NODIR|VLVF_PUB_RW,2 ,8,0 ,15871,0);
        __Vscope_mem_tb__params__params_0.varInsert(__Vfinal,"rst_n", &(TOP.mem_tb__DOT__params__DOT__params_0__DOT__rst_n), false, VLVT_UINT8,VLVD_NODIR|VLVF_PUB_RW,0);
        __Vscope_mem_tb__params__params_0_read.varInsert(__Vfinal,"addr", &(TOP__mem_tb__DOT__params__DOT__params_0_read.addr), false, VLVT_UINT16,VLVD_NODIR|VLVF_PUB_RW,1 ,13,0);
        __Vscope_mem_tb__params__params_0_read.varInsert(__Vfinal,"chip_en", &(TOP__mem_tb__DOT__params__DOT__params_0_read.chip_en), false, VLVT_UINT8,VLVD_NODIR|VLVF_PUB_RW,0);
        __Vscope_mem_tb__params__params_0_read.varInsert(__Vfinal,"data", &(TOP__mem_tb__DOT__params__DOT__params_0_read.data), false, VLVT_UINT16,VLVD_NODIR|VLVF_PUB_RW,1 ,8,0);
        __Vscope_mem_tb__params__params_0_read.varInsert(__Vfinal,"data_width", &(TOP__mem_tb__DOT__params__DOT__params_0_read.data_width), false, VLVT_UINT8,VLVD_NODIR|VLVF_PUB_RW,0);
        __Vscope_mem_tb__params__params_0_read.varInsert(__Vfinal,"en", &(TOP__mem_tb__DOT__params__DOT__params_0_read.en), false, VLVT_UINT8,VLVD_NODIR|VLVF_PUB_RW,0);
        __Vscope_mem_tb__params__params_0_write.varInsert(__Vfinal,"addr", &(TOP__mem_tb__DOT__params__DOT__params_0_write.addr), false, VLVT_UINT16,VLVD_NODIR|VLVF_PUB_RW,1 ,13,0);
        __Vscope_mem_tb__params__params_0_write.varInsert(__Vfinal,"chip_en", &(TOP__mem_tb__DOT__params__DOT__params_0_write.chip_en), false, VLVT_UINT8,VLVD_NODIR|VLVF_PUB_RW,0);
        __Vscope_mem_tb__params__params_0_write.varInsert(__Vfinal,"data", &(TOP__mem_tb__DOT__params__DOT__params_0_write.data), false, VLVT_UINT16,VLVD_NODIR|VLVF_PUB_RW,1 ,8,0);
        __Vscope_mem_tb__params__params_0_write.varInsert(__Vfinal,"data_width", &(TOP__mem_tb__DOT__params__DOT__params_0_write.data_width), false, VLVT_UINT8,VLVD_NODIR|VLVF_PUB_RW,0);
        __Vscope_mem_tb__params__params_0_write.varInsert(__Vfinal,"en", &(TOP__mem_tb__DOT__params__DOT__params_0_write.en), false, VLVT_UINT8,VLVD_NODIR|VLVF_PUB_RW,0);
        __Vscope_mem_tb__params__params_1.varInsert(__Vfinal,"CEN", &(TOP.mem_tb__DOT__params__DOT__params_1__DOT__CEN), false, VLVT_UINT8,VLVD_NODIR|VLVF_PUB_RW,0);
        __Vscope_mem_tb__params__params_1.varInsert(__Vfinal,"DEPTH", const_cast<void*>(static_cast<const void*>(&(TOP.mem_tb__DOT__params__DOT__params_1__DOT__DEPTH))), true, VLVT_UINT32,VLVD_NODIR|VLVF_PUB_RW|VLVF_DPI_CLAY,1 ,31,0);
        __Vscope_mem_tb__params__params_1.varInsert(__Vfinal,"WEN", &(TOP.mem_tb__DOT__params__DOT__params_1__DOT__WEN), false, VLVT_UINT8,VLVD_NODIR|VLVF_PUB_RW,0);
        __Vscope_mem_tb__params__params_1.varInsert(__Vfinal,"clk", &(TOP.mem_tb__DOT__params__DOT__params_1__DOT__clk), false, VLVT_UINT8,VLVD_NODIR|VLVF_PUB_RW,0);
        __Vscope_mem_tb__params__params_1.varInsert(__Vfinal,"memory", &(TOP.mem_tb__DOT__params__DOT__params_1__DOT__memory), false, VLVT_UINT16,VLVD_NODIR|VLVF_PUB_RW,2 ,8,0 ,15871,0);
        __Vscope_mem_tb__params__params_1.varInsert(__Vfinal,"rst_n", &(TOP.mem_tb__DOT__params__DOT__params_1__DOT__rst_n), false, VLVT_UINT8,VLVD_NODIR|VLVF_PUB_RW,0);
        __Vscope_mem_tb__params__params_1_read.varInsert(__Vfinal,"addr", &(TOP__mem_tb__DOT__params__DOT__params_1_read.addr), false, VLVT_UINT16,VLVD_NODIR|VLVF_PUB_RW,1 ,13,0);
        __Vscope_mem_tb__params__params_1_read.varInsert(__Vfinal,"chip_en", &(TOP__mem_tb__DOT__params__DOT__params_1_read.chip_en), false, VLVT_UINT8,VLVD_NODIR|VLVF_PUB_RW,0);
        __Vscope_mem_tb__params__params_1_read.varInsert(__Vfinal,"data", &(TOP__mem_tb__DOT__params__DOT__params_1_read.data), false, VLVT_UINT16,VLVD_NODIR|VLVF_PUB_RW,1 ,8,0);
        __Vscope_mem_tb__params__params_1_read.varInsert(__Vfinal,"data_width", &(TOP__mem_tb__DOT__params__DOT__params_1_read.data_width), false, VLVT_UINT8,VLVD_NODIR|VLVF_PUB_RW,0);
        __Vscope_mem_tb__params__params_1_read.varInsert(__Vfinal,"en", &(TOP__mem_tb__DOT__params__DOT__params_1_read.en), false, VLVT_UINT8,VLVD_NODIR|VLVF_PUB_RW,0);
        __Vscope_mem_tb__params__params_1_write.varInsert(__Vfinal,"addr", &(TOP__mem_tb__DOT__params__DOT__params_1_write.addr), false, VLVT_UINT16,VLVD_NODIR|VLVF_PUB_RW,1 ,13,0);
        __Vscope_mem_tb__params__params_1_write.varInsert(__Vfinal,"chip_en", &(TOP__mem_tb__DOT__params__DOT__params_1_write.chip_en), false, VLVT_UINT8,VLVD_NODIR|VLVF_PUB_RW,0);
        __Vscope_mem_tb__params__params_1_write.varInsert(__Vfinal,"data", &(TOP__mem_tb__DOT__params__DOT__params_1_write.data), false, VLVT_UINT16,VLVD_NODIR|VLVF_PUB_RW,1 ,8,0);
        __Vscope_mem_tb__params__params_1_write.varInsert(__Vfinal,"data_width", &(TOP__mem_tb__DOT__params__DOT__params_1_write.data_width), false, VLVT_UINT8,VLVD_NODIR|VLVF_PUB_RW,0);
        __Vscope_mem_tb__params__params_1_write.varInsert(__Vfinal,"en", &(TOP__mem_tb__DOT__params__DOT__params_1_write.en), false, VLVT_UINT8,VLVD_NODIR|VLVF_PUB_RW,0);
    }
}
