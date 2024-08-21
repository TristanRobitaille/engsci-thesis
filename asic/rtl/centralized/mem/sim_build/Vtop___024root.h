// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Design internal header
// See Vtop.h for the primary calling header

#ifndef VERILATED_VTOP___024ROOT_H_
#define VERILATED_VTOP___024ROOT_H_  // guard

#include "verilated.h"
class Vtop_Defines;
class Vtop_MemoryAccessSignals__Tz1_TBz2;
class Vtop_MemoryAccessSignals__Tz1_TBz9;
class Vtop_MemoryAccessSignals__Tz3_TBz4;


class Vtop__Syms;

class alignas(VL_CACHE_LINE_BYTES) Vtop___024root final : public VerilatedModule {
  public:
    // CELLS
    Vtop_Defines* __PVT__Defines;
    Vtop_MemoryAccessSignals__Tz1_TBz2* __PVT__mem_tb__DOT__param_read_sig;
    Vtop_MemoryAccessSignals__Tz1_TBz2* __PVT__mem_tb__DOT__param_write_sig;
    Vtop_MemoryAccessSignals__Tz3_TBz4* __PVT__mem_tb__DOT__int_res_read_sig;
    Vtop_MemoryAccessSignals__Tz3_TBz4* __PVT__mem_tb__DOT__int_res_write_sig;
    Vtop_MemoryAccessSignals__Tz1_TBz9* __PVT__mem_tb__DOT__params__DOT__params_0_read;
    Vtop_MemoryAccessSignals__Tz1_TBz9* __PVT__mem_tb__DOT__params__DOT__params_0_write;
    Vtop_MemoryAccessSignals__Tz1_TBz9* __PVT__mem_tb__DOT__params__DOT__params_1_read;
    Vtop_MemoryAccessSignals__Tz1_TBz9* __PVT__mem_tb__DOT__params__DOT__params_1_write;
    Vtop_MemoryAccessSignals__Tz1_TBz9* __PVT__mem_tb__DOT__int_res__DOT__int_res_0_read;
    Vtop_MemoryAccessSignals__Tz1_TBz9* __PVT__mem_tb__DOT__int_res__DOT__int_res_0_write;
    Vtop_MemoryAccessSignals__Tz1_TBz9* __PVT__mem_tb__DOT__int_res__DOT__int_res_1_read;
    Vtop_MemoryAccessSignals__Tz1_TBz9* __PVT__mem_tb__DOT__int_res__DOT__int_res_1_write;
    Vtop_MemoryAccessSignals__Tz1_TBz9* __PVT__mem_tb__DOT__int_res__DOT__int_res_2_read;
    Vtop_MemoryAccessSignals__Tz1_TBz9* __PVT__mem_tb__DOT__int_res__DOT__int_res_2_write;
    Vtop_MemoryAccessSignals__Tz1_TBz9* __PVT__mem_tb__DOT__int_res__DOT__int_res_3_read;
    Vtop_MemoryAccessSignals__Tz1_TBz9* __PVT__mem_tb__DOT__int_res__DOT__int_res_3_write;

    // DESIGN SPECIFIC STATE
    // Anonymous structures to workaround compiler member-count bugs
    struct {
        VL_IN8(clk,0,0);
        VL_IN8(rst_n,0,0);
        VL_IN8(param_read_en,0,0);
        VL_IN8(param_write_en,0,0);
        VL_IN8(param_chip_en,0,0);
        VL_IN8(param_read_data_width,0,0);
        VL_IN8(param_write_data_width,0,0);
        VL_IN8(int_res_read_en,0,0);
        VL_IN8(int_res_write_en,0,0);
        VL_IN8(int_res_chip_en,0,0);
        VL_IN8(int_res_read_data_width,0,0);
        VL_IN8(int_res_write_data_width,0,0);
        CData/*0:0*/ mem_tb__DOT__clk;
        CData/*0:0*/ mem_tb__DOT__rst_n;
        CData/*0:0*/ mem_tb__DOT__param_read_en;
        CData/*0:0*/ mem_tb__DOT__param_write_en;
        CData/*0:0*/ mem_tb__DOT__param_chip_en;
        CData/*0:0*/ mem_tb__DOT__param_read_data_width;
        CData/*0:0*/ mem_tb__DOT__param_write_data_width;
        CData/*0:0*/ mem_tb__DOT__int_res_read_en;
        CData/*0:0*/ mem_tb__DOT__int_res_write_en;
        CData/*0:0*/ mem_tb__DOT__int_res_chip_en;
        CData/*0:0*/ mem_tb__DOT__int_res_read_data_width;
        CData/*0:0*/ mem_tb__DOT__int_res_write_data_width;
        CData/*0:0*/ mem_tb__DOT__params__DOT__clk;
        CData/*0:0*/ mem_tb__DOT__params__DOT__rst_n;
        CData/*0:0*/ mem_tb__DOT__params__DOT__params_0_read_en_prev;
        CData/*0:0*/ mem_tb__DOT__params__DOT__params_1_read_en_prev;
        CData/*0:0*/ mem_tb__DOT__params__DOT__params_0__DOT__clk;
        CData/*0:0*/ mem_tb__DOT__params__DOT__params_0__DOT__rst_n;
        CData/*0:0*/ mem_tb__DOT__params__DOT__params_0__DOT__WEN;
        CData/*0:0*/ mem_tb__DOT__params__DOT__params_0__DOT__CEN;
        CData/*0:0*/ mem_tb__DOT__params__DOT__params_1__DOT__clk;
        CData/*0:0*/ mem_tb__DOT__params__DOT__params_1__DOT__rst_n;
        CData/*0:0*/ mem_tb__DOT__params__DOT__params_1__DOT__WEN;
        CData/*0:0*/ mem_tb__DOT__params__DOT__params_1__DOT__CEN;
        CData/*0:0*/ mem_tb__DOT__int_res__DOT__clk;
        CData/*0:0*/ mem_tb__DOT__int_res__DOT__rst_n;
        CData/*1:0*/ mem_tb__DOT__int_res__DOT__bank_read_current;
        CData/*1:0*/ mem_tb__DOT__int_res__DOT__bank_read_prev;
        CData/*1:0*/ mem_tb__DOT__int_res__DOT__bank_write_current;
        CData/*0:0*/ mem_tb__DOT__int_res__DOT__read_data_width_prev;
        CData/*0:0*/ mem_tb__DOT__int_res__DOT__int_res_0__DOT__clk;
        CData/*0:0*/ mem_tb__DOT__int_res__DOT__int_res_0__DOT__rst_n;
        CData/*0:0*/ mem_tb__DOT__int_res__DOT__int_res_0__DOT__WEN;
        CData/*0:0*/ mem_tb__DOT__int_res__DOT__int_res_0__DOT__CEN;
        CData/*0:0*/ mem_tb__DOT__int_res__DOT__int_res_1__DOT__clk;
        CData/*0:0*/ mem_tb__DOT__int_res__DOT__int_res_1__DOT__rst_n;
        CData/*0:0*/ mem_tb__DOT__int_res__DOT__int_res_1__DOT__WEN;
        CData/*0:0*/ mem_tb__DOT__int_res__DOT__int_res_1__DOT__CEN;
        CData/*0:0*/ mem_tb__DOT__int_res__DOT__int_res_2__DOT__clk;
        CData/*0:0*/ mem_tb__DOT__int_res__DOT__int_res_2__DOT__rst_n;
        CData/*0:0*/ mem_tb__DOT__int_res__DOT__int_res_2__DOT__WEN;
        CData/*0:0*/ mem_tb__DOT__int_res__DOT__int_res_2__DOT__CEN;
        CData/*0:0*/ mem_tb__DOT__int_res__DOT__int_res_3__DOT__clk;
        CData/*0:0*/ mem_tb__DOT__int_res__DOT__int_res_3__DOT__rst_n;
        CData/*0:0*/ mem_tb__DOT__int_res__DOT__int_res_3__DOT__WEN;
        CData/*0:0*/ mem_tb__DOT__int_res__DOT__int_res_3__DOT__CEN;
        CData/*0:0*/ __Vtrigprevexpr___TOP__clk__0;
        CData/*0:0*/ __VactContinue;
        VL_IN16(param_read_addr,14,0);
        VL_IN16(param_write_addr,14,0);
        VL_IN16(param_write_data,8,0);
        VL_OUT16(param_read_data,8,0);
    };
    struct {
        VL_IN16(int_res_read_addr,15,0);
        VL_IN16(int_res_write_addr,15,0);
        SData/*14:0*/ mem_tb__DOT__param_read_addr;
        SData/*14:0*/ mem_tb__DOT__param_write_addr;
        SData/*8:0*/ mem_tb__DOT__param_write_data;
        SData/*8:0*/ mem_tb__DOT__param_read_data;
        SData/*15:0*/ mem_tb__DOT__int_res_read_addr;
        SData/*15:0*/ mem_tb__DOT__int_res_write_addr;
        SData/*8:0*/ mem_tb__DOT__params__DOT__params_0__DOT____Vlvbound_h4cea1195__0;
        SData/*8:0*/ mem_tb__DOT__params__DOT__params_1__DOT____Vlvbound_h4cea1195__0;
        SData/*13:0*/ mem_tb__DOT__int_res__DOT__read_base_addr;
        SData/*13:0*/ mem_tb__DOT__int_res__DOT__write_base_addr;
        SData/*8:0*/ mem_tb__DOT__int_res__DOT__int_res_0__DOT____Vlvbound_h203f7518__0;
        SData/*8:0*/ mem_tb__DOT__int_res__DOT__int_res_1__DOT____Vlvbound_h203f7518__0;
        SData/*8:0*/ mem_tb__DOT__int_res__DOT__int_res_2__DOT____Vlvbound_h203f7518__0;
        SData/*8:0*/ mem_tb__DOT__int_res__DOT__int_res_3__DOT____Vlvbound_h203f7518__0;
        VL_IN(int_res_write_data,17,0);
        VL_OUT(int_res_read_data,17,0);
        IData/*17:0*/ mem_tb__DOT__int_res_write_data;
        IData/*17:0*/ mem_tb__DOT__int_res_read_data;
        IData/*31:0*/ __VstlIterCount;
        IData/*31:0*/ __VicoIterCount;
        IData/*31:0*/ __VactIterCount;
        VlUnpacked<SData/*8:0*/, 15872> mem_tb__DOT__params__DOT__params_0__DOT__memory;
        VlUnpacked<SData/*8:0*/, 15872> mem_tb__DOT__params__DOT__params_1__DOT__memory;
        VlUnpacked<SData/*8:0*/, 14336> mem_tb__DOT__int_res__DOT__int_res_0__DOT__memory;
        VlUnpacked<SData/*8:0*/, 14336> mem_tb__DOT__int_res__DOT__int_res_1__DOT__memory;
        VlUnpacked<SData/*8:0*/, 14336> mem_tb__DOT__int_res__DOT__int_res_2__DOT__memory;
        VlUnpacked<SData/*8:0*/, 14336> mem_tb__DOT__int_res__DOT__int_res_3__DOT__memory;
    };
    VlTriggerVec<1> __VstlTriggered;
    VlTriggerVec<1> __VicoTriggered;
    VlTriggerVec<1> __VactTriggered;
    VlTriggerVec<1> __VnbaTriggered;

    // INTERNAL VARIABLES
    Vtop__Syms* const vlSymsp;

    // PARAMETERS
    static constexpr IData/*31:0*/ mem_tb__DOT__params__DOT__params_0__DOT__DEPTH = 0x00003e00U;
    static constexpr IData/*31:0*/ mem_tb__DOT__params__DOT__params_1__DOT__DEPTH = 0x00003e00U;
    static constexpr IData/*31:0*/ mem_tb__DOT__int_res__DOT__int_res_0__DOT__DEPTH = 0x00003800U;
    static constexpr IData/*31:0*/ mem_tb__DOT__int_res__DOT__int_res_1__DOT__DEPTH = 0x00003800U;
    static constexpr IData/*31:0*/ mem_tb__DOT__int_res__DOT__int_res_2__DOT__DEPTH = 0x00003800U;
    static constexpr IData/*31:0*/ mem_tb__DOT__int_res__DOT__int_res_3__DOT__DEPTH = 0x00003800U;

    // CONSTRUCTORS
    Vtop___024root(Vtop__Syms* symsp, const char* v__name);
    ~Vtop___024root();
    VL_UNCOPYABLE(Vtop___024root);

    // INTERNAL METHODS
    void __Vconfigure(bool first);
};


#endif  // guard
