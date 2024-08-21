// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Symbol table internal header
//
// Internal details; most calling programs do not need this header,
// unless using verilator public meta comments.

#ifndef VERILATED_VTOP__SYMS_H_
#define VERILATED_VTOP__SYMS_H_  // guard

#include "verilated.h"

// INCLUDE MODEL CLASS

#include "Vtop.h"

// INCLUDE MODULE CLASSES
#include "Vtop___024root.h"
#include "Vtop___024unit.h"
#include "Vtop_Defines.h"
#include "Vtop_MemoryAccessSignals__Tz1_TBz2.h"
#include "Vtop_MemoryAccessSignals__Tz3_TBz4.h"
#include "Vtop_MemoryAccessSignals__Tz1_TBz9.h"

// DPI TYPES for DPI Export callbacks (Internal use)

// SYMS CLASS (contains all model state)
class alignas(VL_CACHE_LINE_BYTES)Vtop__Syms final : public VerilatedSyms {
  public:
    // INTERNAL STATE
    Vtop* const __Vm_modelp;
    bool __Vm_activity = false;  ///< Used by trace routines to determine change occurred
    uint32_t __Vm_baseCode = 0;  ///< Used by trace routines when tracing multiple models
    VlDeleter __Vm_deleter;
    bool __Vm_didInit = false;

    // MODULE INSTANCE STATE
    Vtop___024root                 TOP;
    Vtop_Defines                   TOP__Defines;
    Vtop_MemoryAccessSignals__Tz1_TBz9 TOP__mem_tb__DOT__int_res__DOT__int_res_0_read;
    Vtop_MemoryAccessSignals__Tz1_TBz9 TOP__mem_tb__DOT__int_res__DOT__int_res_0_write;
    Vtop_MemoryAccessSignals__Tz1_TBz9 TOP__mem_tb__DOT__int_res__DOT__int_res_1_read;
    Vtop_MemoryAccessSignals__Tz1_TBz9 TOP__mem_tb__DOT__int_res__DOT__int_res_1_write;
    Vtop_MemoryAccessSignals__Tz1_TBz9 TOP__mem_tb__DOT__int_res__DOT__int_res_2_read;
    Vtop_MemoryAccessSignals__Tz1_TBz9 TOP__mem_tb__DOT__int_res__DOT__int_res_2_write;
    Vtop_MemoryAccessSignals__Tz1_TBz9 TOP__mem_tb__DOT__int_res__DOT__int_res_3_read;
    Vtop_MemoryAccessSignals__Tz1_TBz9 TOP__mem_tb__DOT__int_res__DOT__int_res_3_write;
    Vtop_MemoryAccessSignals__Tz3_TBz4 TOP__mem_tb__DOT__int_res_read_sig;
    Vtop_MemoryAccessSignals__Tz3_TBz4 TOP__mem_tb__DOT__int_res_write_sig;
    Vtop_MemoryAccessSignals__Tz1_TBz2 TOP__mem_tb__DOT__param_read_sig;
    Vtop_MemoryAccessSignals__Tz1_TBz2 TOP__mem_tb__DOT__param_write_sig;
    Vtop_MemoryAccessSignals__Tz1_TBz9 TOP__mem_tb__DOT__params__DOT__params_0_read;
    Vtop_MemoryAccessSignals__Tz1_TBz9 TOP__mem_tb__DOT__params__DOT__params_0_write;
    Vtop_MemoryAccessSignals__Tz1_TBz9 TOP__mem_tb__DOT__params__DOT__params_1_read;
    Vtop_MemoryAccessSignals__Tz1_TBz9 TOP__mem_tb__DOT__params__DOT__params_1_write;

    // SCOPE NAMES
    VerilatedScope __Vscope_Defines;
    VerilatedScope __Vscope_TOP;
    VerilatedScope __Vscope_mem_tb;
    VerilatedScope __Vscope_mem_tb__int_res;
    VerilatedScope __Vscope_mem_tb__int_res__int_res_0;
    VerilatedScope __Vscope_mem_tb__int_res__int_res_0_read;
    VerilatedScope __Vscope_mem_tb__int_res__int_res_0_write;
    VerilatedScope __Vscope_mem_tb__int_res__int_res_1;
    VerilatedScope __Vscope_mem_tb__int_res__int_res_1_read;
    VerilatedScope __Vscope_mem_tb__int_res__int_res_1_write;
    VerilatedScope __Vscope_mem_tb__int_res__int_res_2;
    VerilatedScope __Vscope_mem_tb__int_res__int_res_2_read;
    VerilatedScope __Vscope_mem_tb__int_res__int_res_2_write;
    VerilatedScope __Vscope_mem_tb__int_res__int_res_3;
    VerilatedScope __Vscope_mem_tb__int_res__int_res_3_read;
    VerilatedScope __Vscope_mem_tb__int_res__int_res_3_write;
    VerilatedScope __Vscope_mem_tb__int_res_read_sig;
    VerilatedScope __Vscope_mem_tb__int_res_write_sig;
    VerilatedScope __Vscope_mem_tb__param_read_sig;
    VerilatedScope __Vscope_mem_tb__param_write_sig;
    VerilatedScope __Vscope_mem_tb__params;
    VerilatedScope __Vscope_mem_tb__params__params_0;
    VerilatedScope __Vscope_mem_tb__params__params_0_read;
    VerilatedScope __Vscope_mem_tb__params__params_0_write;
    VerilatedScope __Vscope_mem_tb__params__params_1;
    VerilatedScope __Vscope_mem_tb__params__params_1_read;
    VerilatedScope __Vscope_mem_tb__params__params_1_write;

    // SCOPE HIERARCHY
    VerilatedHierarchy __Vhier;

    // CONSTRUCTORS
    Vtop__Syms(VerilatedContext* contextp, const char* namep, Vtop* modelp);
    ~Vtop__Syms();

    // METHODS
    const char* name() { return TOP.name(); }
};

#endif  // guard
